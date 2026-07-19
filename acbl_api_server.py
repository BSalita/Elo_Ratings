from __future__ import annotations

import gc
import json
import os
import pathlib
import tempfile
import time
from datetime import datetime, timezone

import duckdb
import polars as pl
import psutil

from elo_common import (
    CHESS_DISPLAY_MEAN,
    CHESS_DISPLAY_SD,
    ELO_TITLE_SQL_CASE,
    title_from_elo_expr,
    zscore_chess_sql,
)
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

DATA_ROOT = pathlib.Path(__file__).resolve().parent / "data"
API_SOURCE_PATH = pathlib.Path(__file__).resolve()
API_PROCESS_STARTED_AT = datetime.now(timezone.utc)
# Bump when deploying memory/toggle fixes so /health confirms the running build.
API_BUILD_TAG = "2026-07-19-detail-online-filter"

# Module-level caches to avoid re-reading parquet files on every request.
# Keys are source paths; values are the cached objects.
_SCHEMA_CACHE: dict[str, dict] = {}
_FRAME_CACHE: dict[str, pl.DataFrame] = {}
_FRAME_CACHE_TIMES: dict[str, float] = {}
# Full-dataset Date bounds captured once at frame load (avoids per-request scans).
_FRAME_DATE_BOUNDS: dict[str, tuple[datetime | None, datetime | None]] = {}

import threading as _threading
_DB_LOCK = _threading.Lock()
_FRAME_LOCK = _threading.Lock()
# Serialize report/detail handlers so Club↔Tournament toggles cannot overlap two
# full-frame loads (~13 GB Club + reload) and OOM the container.
_REPORT_LOCK = _threading.Lock()
_DB_CON: duckdb.DuckDBPyConnection | None = None

app = FastAPI(title="ACBL Elo API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_text(path: pathlib.Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def _detect_cgroup_limits() -> dict:
    """Best-effort container limits from cgroups (v2 first, then v1)."""
    from streamlitlib.memory_usage import get_memory_metrics

    memory_limit_bytes: int | None = None
    cpu_limit_cores: float | None = None

    metrics = get_memory_metrics()
    if metrics.cgroup_limit_bytes is not None and metrics.cgroup_limit_bytes > 0:
        memory_limit_bytes = int(metrics.cgroup_limit_bytes)

    # cgroup v2 cpu
    cpu_max_v2 = _read_text(pathlib.Path("/sys/fs/cgroup/cpu.max"))
    if cpu_max_v2:
        parts = cpu_max_v2.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                quota = int(parts[0])
                period = int(parts[1])
                if quota > 0 and period > 0:
                    cpu_limit_cores = quota / period
            except ValueError:
                cpu_limit_cores = None

    if cpu_limit_cores is None:
        quota_v1 = _read_text(pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"))
        period_v1 = _read_text(pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us"))
        if quota_v1 and period_v1:
            try:
                quota = int(quota_v1)
                period = int(period_v1)
                if quota > 0 and period > 0:
                    cpu_limit_cores = quota / period
            except ValueError:
                pass

    return {
        "memory_limit_bytes": memory_limit_bytes,
        "cpu_limit_cores": cpu_limit_cores,
    }


def _recommended_threads() -> int:
    limits = _detect_cgroup_limits()
    host_cpus = int(os.cpu_count() or 4)
    if limits["cpu_limit_cores"] is not None:
        cpu_budget = max(1, int(limits["cpu_limit_cores"]))
    else:
        cpu_budget = host_cpus
    # Keep conservative default while respecting container CPU budget.
    return max(4, cpu_budget // 2) if cpu_budget >= 8 else max(1, cpu_budget)


def _cgroup_memory_limit_bytes() -> int | None:
    """Container memory cap from cgroup, or None if unknown/unlimited."""
    limits = _detect_cgroup_limits()
    limit = limits.get("memory_limit_bytes")
    if limit is None or limit <= 0:
        return None
    return int(limit)


def _memory_budget_bytes() -> int:
    """Memory budget for cache/DuckDB sizing — never host RAM when cgroup is unknown."""
    limit = _cgroup_memory_limit_bytes()
    if limit is not None:
        return limit
    env_gb = os.getenv("ACBL_CONTAINER_MEMORY_GB", "").strip()
    if env_gb:
        try:
            return max(int(1 * 1024 ** 3), int(float(env_gb) * 1024 ** 3))
        except ValueError:
            pass
    # Conservative default for unknown limits (Docker without cgroup v2 memory.max).
    return int(32 * 1024 ** 3)


def _container_memory_bytes() -> int:
    """Total memory budget: cgroup limit if present, else conservative default."""
    return _memory_budget_bytes()


DUAL_FRAME_CACHE_MIN_BYTES = 40 * 1024 ** 3
_DUAL_FRAME_CACHE_LOGGED = False


def _dual_frame_cache_enabled() -> bool:
    """Keep Club + Tournament frames resident when the container has enough RAM.

    Requires a *known* cgroup limit >= 40 GB. Never uses host RAM as a proxy
    (that mis-detects Docker limits and OOM-kills the process).
    """
    override = os.getenv("ACBL_DUAL_FRAME_CACHE", "").strip().lower()
    if override in ("0", "false", "no", "off"):
        return False

    limit = _cgroup_memory_limit_bytes()
    if limit is None:
        if override in ("1", "true", "yes", "on"):
            print(
                "[acbl-api] ACBL_DUAL_FRAME_CACHE=1 but cgroup limit unknown; "
                "using single-frame cache",
                flush=True,
            )
        return False

    if limit < DUAL_FRAME_CACHE_MIN_BYTES:
        if override in ("1", "true", "yes", "on"):
            print(
                f"[acbl-api] ACBL_DUAL_FRAME_CACHE=1 but cgroup limit "
                f"{limit / 1024 ** 3:.0f} GB < 40 GB; using single-frame cache",
                flush=True,
            )
        return False

    if override in ("1", "true", "yes", "on"):
        return True
    return limit >= DUAL_FRAME_CACHE_MIN_BYTES


def _estimated_frame_bytes(club_or_tournament: str) -> int:
    """Conservative in-memory size for a dataset not yet cached."""
    source_path, _ = _parquet_source_for(club_or_tournament)
    cached = _FRAME_CACHE.get(source_path)
    if cached is not None:
        try:
            return int(cached.estimated_size())
        except Exception:
            pass
    if club_or_tournament.lower() == "club":
        # Shrunk resident frame ~13 GB; streaming collect peaks higher before shrink.
        return int(22 * 1024 ** 3)
    return int(5 * 1024 ** 3)


def _dual_frame_cache_safe_for_load(club_or_tournament: str) -> bool:
    """True when dual-frame is on and cgroup has room to load this dataset too."""
    if not _dual_frame_cache_enabled():
        return False
    limit = _cgroup_memory_limit_bytes()
    if limit is None:
        return False
    source_path, _ = _parquet_source_for(club_or_tournament)
    if source_path in _FRAME_CACHE:
        return True
    other_cached = sum(
        int(f.estimated_size())
        for p, f in _FRAME_CACHE.items()
        if p != source_path and f is not None
    )
    load_bytes = _estimated_frame_bytes(club_or_tournament)
    # Extra headroom for Polars streaming collect peak during a cold load.
    load_peak = int(6 * 1024 ** 3) if club_or_tournament.lower() == "club" else 0
    query_headroom = int(4 * 1024 ** 3)
    projected = other_cached + load_bytes + load_peak + query_headroom
    return projected <= int(limit * 0.85)


def _evict_frames_except(keep_source_path: str) -> None:
    evicted: list[str] = []
    for old_path in list(_FRAME_CACHE.keys()):
        if old_path == keep_source_path:
            continue
        old_frame = _FRAME_CACHE.pop(old_path, None)
        _FRAME_CACHE_TIMES.pop(old_path, None)
        _FRAME_DATE_BOUNDS.pop(old_path, None)
        if old_frame is not None:
            del old_frame
        evicted.append(pathlib.Path(old_path).name)
    if evicted:
        keep_label = (
            pathlib.Path(keep_source_path).name if keep_source_path else "(none)"
        )
        print(
            f"[acbl-api] evicted frame cache {evicted} "
            f"(keeping {keep_label}, mem {_cgroup_mem_summary()})",
            flush=True,
        )
        gc.collect()
        _reset_duckdb_connection()
        for _ in range(4):
            gc.collect()
            _malloc_trim()
            time.sleep(0.25)
        print(
            f"[acbl-api] post-evict mem {_cgroup_mem_summary()}",
            flush=True,
        )


def _evict_other_frames(keep_source_path: str, club_or_tournament: str) -> None:
    """Drop any cached frames other than the one we're about to use.

    When dual-frame caching is enabled and cgroup headroom allows, both Club
    and Tournament stay resident once loaded. A *cold* Club load always evicts
    other frames first: streaming collect peaks well above the shrunk frame
    size and OOM-kills the process if Tournament is still resident.
    """
    if keep_source_path in _FRAME_CACHE:
        return

    others_cached = any(p != keep_source_path for p in _FRAME_CACHE)
    if club_or_tournament.lower() == "club" and others_cached:
        print(
            "[acbl-api] cold club load: evicting other frames before load "
            f"(mem {_cgroup_mem_summary()})",
            flush=True,
        )
        _evict_frames_except(keep_source_path)
        if not _wait_for_cgroup_headroom_after_evict(club_or_tournament):
            _raise_insufficient_memory(club_or_tournament, "after cold club eviction")
        return

    if _dual_frame_cache_enabled() and _dual_frame_cache_safe_for_load(club_or_tournament):
        return
    if _dual_frame_cache_enabled() and _FRAME_CACHE:
        print(
            f"[acbl-api] dual-frame over budget for {club_or_tournament} "
            f"(cached {_cached_frame_bytes() / 1024 ** 3:.1f} GB, limit "
            f"{(_cgroup_memory_limit_bytes() or 0) / 1024 ** 3:.0f} GB); evicting other frames",
            flush=True,
        )
    _evict_frames_except(keep_source_path)
    if others_cached:
        if not _wait_for_cgroup_headroom_after_evict(club_or_tournament):
            _raise_insufficient_memory(club_or_tournament, "after eviction")


def _cached_frame_bytes() -> int:
    """Best-effort in-memory size of all resident Polars frame caches."""
    total = 0
    for frame in _FRAME_CACHE.values():
        if frame is None:
            continue
        try:
            total += int(frame.estimated_size())
        except Exception:
            pass
    return total


def _duckdb_memory_limit_bytes() -> int:
    """Hard cap for DuckDB's buffer manager.

    DuckDB defaults its ``memory_limit`` to ~80% of detected RAM. With the
    resident Polars frame (~19 GB for Club on prod) cached alongside, that
    default overcommits the container and triggers OOM on the next query.
    Budget from cgroup limit minus the live cached frame (or a conservative
    club estimate before first load). Override at deploy with
    ``DUCKDB_MEMORY_LIMIT_GB``.
    """
    override = os.getenv("DUCKDB_MEMORY_LIMIT_GB", "").strip()
    if override:
        try:
            return max(int(0.5 * 1024 ** 3), int(float(override) * 1024 ** 3))
        except ValueError:
            pass
    total = _memory_budget_bytes()
    frame_bytes = _cached_frame_bytes()
    if frame_bytes <= 0:
        frame_bytes = int(19.5 * 1024 ** 3)
    runtime_overhead = int(2 * 1024 ** 3)
    reserve = frame_bytes + runtime_overhead
    headroom = total - reserve
    floor = int(1 * 1024 ** 3)
    if headroom <= floor:
        return floor
    # Give DuckDB at most 40% of remaining headroom, capped at 4 GB.
    cap = min(int(4 * 1024 ** 3), max(floor, int(headroom * 0.4)))
    return min(headroom, cap)


def _duckdb_temp_dir() -> str:
    """Writable directory DuckDB can spill to when a query exceeds the cap."""
    override = os.getenv("DUCKDB_TEMP_DIR", "").strip()
    if override:
        return override
    return str(pathlib.Path(tempfile.gettempdir()) / "acbl_duckdb_spill")


def _r2_enabled() -> bool:
    return bool(os.getenv("R2_BUCKET", "").strip())


def _r2_storage_options() -> dict:
    bucket = os.getenv("R2_BUCKET", "").strip()
    endpoint = os.getenv("R2_ENDPOINT", "").strip()
    access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
    region = os.getenv("R2_REGION", "auto").strip() or "auto"

    if not bucket or not endpoint or not access_key or not secret_key:
        raise ValueError("R2 env vars are incomplete. Set bucket, endpoint, access key, and secret key.")

    return {
        "aws_region": region,
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
        "aws_endpoint_url": endpoint,
    }


# In-memory cache for shrinkage sidecars (one per club|tournament). Loaded
# lazily on first /acbl/report request and reused across calls.
_SHRINKAGE_META_CACHE: dict[str, dict | None] = {}

# Default Bayesian shrinkage prior weight (sessions equivalent). The Streamlit
# /acbl/report client may override via the prior_sessions query parameter.
SHRINKAGE_DEFAULT_PRIOR_SESSIONS = 50

# Elite skill gate. The leaderboard's headline Elo is a strong *within-field*
# predictor, but a player/pair who only dominates weak fields can still reach
# the top ranks (the "Zubatch" failure mode). We therefore gate the top ranks
# by a field-INDEPENDENT skill signal: the pool z-score of card play
# (DD_Tricks_Diff) + par bidding (Par_Suit, Par_Contract). Players/pairs below
# ``SKILL_GATE_DEFAULT_Z`` are excluded from the leaderboard (their Elo can't
# be trusted as 'elite' because their absolute card play is only average).
# A field-relative "tested against stronger cohorts" gate was prototyped and
# rejected: a strong individual is by construction rated above their field
# mean, so it is structurally unobservable from this data. Set the
# ``min_skill_z`` query param <= SKILL_GATE_DISABLED to turn the gate off.
SKILL_GATE_DEFAULT_Z = 0.5
SKILL_GATE_DISABLED = -90.0


def _skill_gate_clause(min_skill_z: float, *, col: str = "Skill_Z") -> str:
    """WHERE clause body for the skill gate (empty string if disabled).

    NULL skill (no card-play data) fails the gate by design: no evidence of
    competitiveness => not elite."""
    if min_skill_z is None or min_skill_z <= SKILL_GATE_DISABLED:
        return ""
    return f"WHERE {col} >= {float(min_skill_z)!r}"


def _skill_z_sql(prefix: str, stats_alias: str = "ss") -> str:
    """Mean of per-metric pool z-scores: card play + two par-bidding rates.

    ``prefix`` is the CTE alias holding the per-entity aggregates (e.g. pwq)."""
    return (
        f"( ({prefix}.DD_Tricks_Diff_Avg - {stats_alias}.m_dd) / NULLIF({stats_alias}.s_dd, 0)"
        f" + ({prefix}.Par_Suit_Rate - {stats_alias}.m_ps) / NULLIF({stats_alias}.s_ps, 0)"
        f" + ({prefix}.Par_Contract_Rate - {stats_alias}.m_pc) / NULLIF({stats_alias}.s_pc, 0) ) / 3.0"
    )


_SKILL_STATS_CTE_TEMPLATE = """
    {name} AS (
      SELECT AVG(DD_Tricks_Diff_Avg) AS m_dd, STDDEV_POP(DD_Tricks_Diff_Avg) AS s_dd,
             AVG(Par_Suit_Rate) AS m_ps, STDDEV_POP(Par_Suit_Rate) AS s_ps,
             AVG(Par_Contract_Rate) AS m_pc, STDDEV_POP(Par_Contract_Rate) AS s_pc
      FROM {source}
    )"""


def _shrinkage_sidecar_search_paths(filename: str) -> list[pathlib.Path]:
    """Return ordered local filesystem search paths for the shrinkage sidecar JSON.

    Looks in (a) the API's bundled ``data/`` directory (where deployment
    artifacts live alongside the parquets), then (b) the canonical
    ``e:/bridge/data/acbl/`` source of truth used by
    ``acbl_elo_ratings_create.py``. An optional ``ACBL_SHRINKAGE_DIR``
    environment variable can override the second location.
    """
    candidates = [DATA_ROOT / filename]
    override = os.getenv("ACBL_SHRINKAGE_DIR", "").strip()
    if override:
        candidates.append(pathlib.Path(override) / filename)
    candidates.append(pathlib.Path("e:/bridge/data/acbl") / filename)
    return candidates


def _load_shrinkage_meta_from_r2(filename: str) -> dict | None:
    """Fetch the shrinkage sidecar JSON from R2 (S3-compatible) storage.

    Uses the same ``R2_*`` env vars as :func:`_r2_storage_options`. Returns
    ``None`` on any failure (missing key, network error, JSON parse error,
    missing optional dependency) so the caller can fall back to local paths.
    """
    if not _r2_enabled():
        return None

    try:
        import boto3
        from botocore.config import Config as _BotoConfig
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:
        return None

    bucket = os.getenv("R2_BUCKET", "").strip()
    prefix = os.getenv("R2_PREFIX", "data").strip().strip("/")
    key = f"{prefix}/{filename}" if prefix else filename

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("R2_ENDPOINT", "").strip() or None,
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip() or None,
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip() or None,
            region_name=os.getenv("R2_REGION", "auto").strip() or "auto",
            config=_BotoConfig(signature_version="s3v4"),
        )
        obj = s3.get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read()
        return json.loads(body.decode("utf-8"))
    except (BotoCoreError, ClientError, json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def _load_shrinkage_meta(club_or_tournament: str) -> dict | None:
    """Load the shrinkage sidecar JSON written by acbl_elo_ratings_create.py.

    Returns the parsed dict, or None when the sidecar is unavailable (e.g.
    first deployment after the math change but before the recompute has
    rebuilt the lookup parquets). In that case the report falls back to
    Published == Raw so the API still works.

    Lookup order:

    1. R2 (S3-compatible) at ``s3://$R2_BUCKET/$R2_PREFIX/<filename>`` when
       ``R2_BUCKET`` is set. This is the canonical source when parquets live in R2.
    2. Local filesystem candidates (DATA_ROOT, ACBL_SHRINKAGE_DIR override,
       e:/bridge/data/acbl) for local dev.

    Cached at module level.
    """
    key = club_or_tournament.lower()
    if key in _SHRINKAGE_META_CACHE:
        return _SHRINKAGE_META_CACHE[key]

    filename = f"acbl_{key}_elo_shrinkage.json"

    if _r2_enabled():
        meta = _load_shrinkage_meta_from_r2(filename)
        if meta is not None:
            _SHRINKAGE_META_CACHE[key] = meta
            return meta

    for path in _shrinkage_sidecar_search_paths(filename):
        if not path.exists():
            continue
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        _SHRINKAGE_META_CACHE[key] = meta
        return meta

    _SHRINKAGE_META_CACHE[key] = None
    return None


def _shrinkage_anchor(meta: dict | None, kind: str) -> float | None:
    """Pull the prior anchor (median Elo of established subset) for ``kind``.

    ``kind`` is "player" or "pair".
    Returns None when the sidecar is missing or the kind's entry is empty.
    """
    if meta is None:
        return None
    section = meta.get(kind) if isinstance(meta, dict) else None
    if not isinstance(section, dict):
        return None
    anchor = section.get("prior_anchor")
    if anchor is None:
        return None
    try:
        return float(anchor)
    except (TypeError, ValueError):
        return None


def _published_elo_sql(raw_col: str, sessions_col: str,
                       prior_anchor: float | None, prior_sessions: int) -> str:
    """Return a SQL expression that wraps ``raw_col`` with Bayesian shrinkage.

    When the prior anchor is unavailable or ``prior_sessions`` is 0, the
    expression collapses to ``raw_col`` so Published == Raw is preserved.
    Otherwise:

        Published = (n * raw + prior_sessions * prior_anchor)
                    / (n + prior_sessions)

    The result is cast to INTEGER (the existing display convention).
    """
    if prior_anchor is None or prior_sessions <= 0:
        return f"CAST(COALESCE({raw_col}, 0) AS INTEGER)"
    return (
        f"CAST(ROUND("
        f"(CAST({sessions_col} AS DOUBLE) * COALESCE({raw_col}, 0) "
        f"+ {float(prior_sessions)} * {float(prior_anchor)}) "
        f"/ NULLIF((CAST({sessions_col} AS DOUBLE) + {float(prior_sessions)}), 0)"
        f") AS INTEGER)"
    )


def _parquet_source_for(club_or_tournament: str) -> tuple[str, dict | None]:
    filename = f"acbl_{club_or_tournament.lower()}_elo_ratings.parquet"
    if _r2_enabled():
        bucket = os.getenv("R2_BUCKET", "").strip()
        prefix = os.getenv("R2_PREFIX", "data").strip().strip("/")
        key = f"{prefix}/{filename}" if prefix else filename
        return f"s3://{bucket}/{key}", _r2_storage_options()

    file_path = DATA_ROOT.joinpath(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")
    return str(file_path), None


def load_elo_ratings_schema_map(club_or_tournament: str) -> dict:
    source_path, storage_options = _parquet_source_for(club_or_tournament)
    if source_path in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[source_path]
    df0 = pl.read_parquet(source_path, n_rows=0, storage_options=storage_options)
    _SCHEMA_CACHE[source_path] = df0.schema
    return df0.schema


def _malloc_trim() -> None:
    """Prod glibc to return freed pages to the OS.

    On Linux this releases per-request allocation slack that the
    allocator otherwise keeps in its arenas forever. No-ops on Windows
    (no libc.so.6) and on platforms whose malloc lacks malloc_trim.
    """
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _load_headroom_bytes(club_or_tournament: str) -> int:
    """RAM required to start a cold parquet load without OOM."""
    peak = int(8 * 1024 ** 3) if club_or_tournament.lower() == "club" else int(3 * 1024 ** 3)
    return _estimated_frame_bytes(club_or_tournament) + peak


def _raise_insufficient_memory(club_or_tournament: str, phase: str) -> None:
    used = _cgroup_memory_used_bytes()
    limit = _cgroup_memory_limit_bytes() or _memory_budget_bytes()
    need = _load_headroom_bytes(club_or_tournament)
    free = (limit - used) if used is not None else None
    free_str = f"{free / 1024 ** 3:.1f} GB" if free is not None else "unknown"
    detail = (
        f"Insufficient memory to load {club_or_tournament} ({phase}): "
        f"need ~{need / 1024 ** 3:.0f} GB free, "
        f"have {free_str} free of {limit / 1024 ** 3:.0f} GB limit "
        f"({_cgroup_mem_summary()}). Retry in 30 seconds."
    )
    print(f"[acbl-api] {detail}", flush=True)
    raise HTTPException(status_code=503, detail=detail, headers={"Retry-After": "30"})


def _cgroup_memory_used_bytes() -> int | None:
    try:
        return int(pathlib.Path("/sys/fs/cgroup/memory.current").read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _cgroup_mem_summary() -> str:
    try:
        used = _cgroup_memory_used_bytes()
        if used is None:
            return "n/a"
        limit_raw = pathlib.Path("/sys/fs/cgroup/memory.max").read_text(encoding="utf-8").strip()
        if limit_raw.isdigit():
            limit = int(limit_raw)
            if limit > 0:
                return f"{used / 1024 ** 3:.2f}/{limit / 1024 ** 3:.2f} GB ({100 * used / limit:.0f}%)"
        return f"{used / 1024 ** 3:.2f} GB"
    except Exception:
        return "n/a"


def _wait_for_cgroup_headroom_after_evict(club_or_tournament: str) -> bool:
    """Poll cgroup usage until enough RAM is free to start a cold parquet load."""
    limit = _cgroup_memory_limit_bytes()
    if limit is None:
        return True
    load_budget = _load_headroom_bytes(club_or_tournament)
    max_used = max(int(2 * 1024 ** 3), int(limit - load_budget))
    deadline = time.perf_counter() + 60.0
    while time.perf_counter() < deadline:
        used = _cgroup_memory_used_bytes()
        if used is None or used <= max_used:
            print(
                f"[acbl-api] cgroup ready for {club_or_tournament} load "
                f"(target <= {max_used / 1024 ** 3:.1f} GB, mem {_cgroup_mem_summary()})",
                flush=True,
            )
            return True
        time.sleep(0.25)
    print(
        f"[acbl-api] cgroup headroom wait timed out before {club_or_tournament} load "
        f"(target <= {max_used / 1024 ** 3:.1f} GB, mem {_cgroup_mem_summary()})",
        flush=True,
    )
    return False


def _reset_duckdb_connection() -> None:
    """Close DuckDB so buffer-pool memory is released on dataset switch."""
    global _DB_CON
    with _DB_LOCK:
        if _DB_CON is None:
            return
        try:
            try:
                _DB_CON.unregister("self")
            except Exception:
                pass
            _DB_CON.close()
        except Exception:
            pass
        _DB_CON = None
    gc.collect()
    _malloc_trim()


def _dtype_shrink_exprs(schema: dict) -> list[pl.Expr]:
    """Polars expressions that down-cast wasteful dtypes.

    Three categories of waste live in the raw parquet:

    1. ``Player_Name_*`` and ``Player_ID_*`` are String columns whose values
       repeat across millions of board-level rows (the same ~500K names and
       ~1M ids appear over and over in 58M rows on the club parquet).
       Converting to Categorical (dictionary-encoded) collapses them to
       roughly one Int32 code per row plus a small dictionary.
    2. ``session_id`` is Int64 on club (down-casts to Int32) and a compound
       String on tournament (down-casts to Categorical).
    3. The float columns are already Float32 so we leave them alone.

    DuckDB consumes polars Categorical columns natively (dictionary-encoded
    VARCHAR), so existing SQL such as ``Player_ID_N || '-' || Player_ID_S``
    and ``WHERE Player_ID_{pos} IS NOT NULL`` keep working unchanged.

    Net effect: club parquet drops from ~15.7 GB to ~13.1 GB in-memory;
    tournament parquet from ~4.4 GB to ~3.6 GB. Returned as expressions so
    callers can splice them into a lazy plan and avoid materializing the
    bigger pre-cast frame during the streaming collect.
    """
    exprs: list[pl.Expr] = []
    for p in "NESW":
        name_col = f"Player_Name_{p}"
        id_col = f"Player_ID_{p}"
        if name_col in schema and schema[name_col] == pl.Utf8:
            exprs.append(pl.col(name_col).cast(pl.Categorical))
        if id_col in schema and schema[id_col] == pl.Utf8:
            exprs.append(pl.col(id_col).cast(pl.Categorical))
    if "session_id" in schema:
        sid_dtype = schema["session_id"]
        if sid_dtype == pl.Utf8:
            exprs.append(pl.col("session_id").cast(pl.Categorical))
        elif sid_dtype == pl.Int64:
            exprs.append(pl.col("session_id").cast(pl.Int32, strict=False))
    return exprs


def _shrink_frame_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    """Eager variant of :func:`_dtype_shrink_exprs` for already-materialized
    frames (used by the smoke-test harness; the production load path applies
    the casts inside the lazy plan)."""
    exprs = _dtype_shrink_exprs(df.schema)
    return df.with_columns(exprs) if exprs else df


def _record_frame_date_bounds(source_path: str, full_df: pl.DataFrame) -> None:
    if "Date" not in full_df.columns or full_df.is_empty():
        _FRAME_DATE_BOUNDS.pop(source_path, None)
        return
    dmin, dmax = full_df.select(
        pl.col("Date").min().alias("min"),
        pl.col("Date").max().alias("max"),
    ).row(0)
    _FRAME_DATE_BOUNDS[source_path] = (dmin, dmax)


def _frame_date_range(source_path: str, date_from: datetime | None = None) -> str:
    bounds = _FRAME_DATE_BOUNDS.get(source_path)
    if not bounds or bounds[0] is None or bounds[1] is None:
        return ""
    dmin, dmax = bounds
    if date_from is not None and dmin is not None and date_from > dmin:
        dmin = date_from
    return f"{str(dmin)[:10]} to {str(dmax)[:10]}"


def _load_full_frame(club_or_tournament: str) -> pl.DataFrame:
    """Load the full parquet once and cache it at module level.

    On first access for a given dataset (Club or Tournament), evicts the other
    cached frame unless dual-frame caching is enabled (cgroup >= 40 GB).
    Down-casts wasteful String / Int64 columns to Categorical / Int32.
    """
    global _DUAL_FRAME_CACHE_LOGGED
    source_path, storage_options = _parquet_source_for(club_or_tournament)
    with _FRAME_LOCK:
        if source_path in _FRAME_CACHE:
            return _FRAME_CACHE[source_path]

        if not _DUAL_FRAME_CACHE_LOGGED:
            mode = "dual-frame" if _dual_frame_cache_enabled() else "single-frame"
            limit_gb = (_cgroup_memory_limit_bytes() or 0) / 1024 ** 3
            print(
                f"[acbl-api] frame cache mode={mode} "
                f"(cgroup limit {limit_gb:.0f} GB or unknown, mem {_cgroup_mem_summary()})",
                flush=True,
            )
            _DUAL_FRAME_CACHE_LOGGED = True

        _evict_other_frames(source_path, club_or_tournament)

        used = _cgroup_memory_used_bytes()
        limit = _cgroup_memory_limit_bytes() or _memory_budget_bytes()
        need = _load_headroom_bytes(club_or_tournament)
        if used is not None and (limit - used) < need:
            _raise_insufficient_memory(club_or_tournament, "before parquet read")

        print(
            f"[acbl-api] loading {club_or_tournament} parquet "
            f"({pathlib.Path(source_path).name}, mem {_cgroup_mem_summary()})",
            flush=True,
        )
        t0 = time.perf_counter()
        schema_map = load_elo_ratings_schema_map(club_or_tournament)
        lf = pl.scan_parquet(source_path, storage_options=storage_options)

        if "Date" in schema_map:
            if schema_map["Date"] == pl.Utf8:
                parsed_dt = pl.coalesce(
                    [
                        pl.col("Date").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False),
                        pl.col("Date").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False),
                        pl.col("Date").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.f", strict=False),
                        pl.col("Date").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S", strict=False),
                        pl.col("Date").str.strptime(pl.Date, format="%Y-%m-%d", strict=False).cast(pl.Datetime, strict=False),
                    ]
                )
                lf = lf.with_columns(parsed_dt.alias("Date"))
            else:
                lf = lf.with_columns(pl.col("Date").cast(pl.Datetime, strict=False).alias("Date"))

        shrink_exprs = _dtype_shrink_exprs(dict(schema_map))
        if shrink_exprs:
            lf = lf.with_columns(shrink_exprs)

        full_df = lf.collect(engine="streaming")
        _FRAME_CACHE[source_path] = full_df
        _FRAME_CACHE_TIMES[source_path] = time.time()
        _record_frame_date_bounds(source_path, full_df)
        _malloc_trim()
        print(
            f"[acbl-api] loaded {club_or_tournament} "
            f"({full_df.height} rows, {time.perf_counter() - t0:.1f}s, mem {_cgroup_mem_summary()})",
            flush=True,
        )
        return full_df


def load_elo_ratings(club_or_tournament: str, columns: list[str] | None = None, date_from: datetime | None = None) -> pl.DataFrame:
    """Return a (possibly filtered) view of the cached full frame."""
    full_df = _load_full_frame(club_or_tournament)
    schema_map = load_elo_ratings_schema_map(club_or_tournament)
    df = full_df

    if columns:
        columns = list(dict.fromkeys(columns))
        if date_from is not None and "Date" in schema_map and "Date" not in columns:
            columns = ["Date", *columns]
        valid = [c for c in columns if c in df.columns]
        if valid:
            df = df.select(valid)

    if date_from is not None and "Date" in df.columns:
        df = df.filter(pl.col("Date") >= pl.lit(date_from))

    return df


def _get_db_connection() -> duckdb.DuckDBPyConnection:
    """Return a long-lived DuckDB connection, creating it on first call."""
    global _DB_CON
    if _DB_CON is not None:
        return _DB_CON
    with _DB_LOCK:
        if _DB_CON is not None:
            return _DB_CON
        con = duckdb.connect()
        con.execute(f"PRAGMA threads={_recommended_threads()};")
        con.execute("PRAGMA preserve_insertion_order=false;")
        # Cap the buffer manager so DuckDB never overcommits the container
        # alongside the resident Polars frame, and allow spill-to-disk so a
        # large aggregation degrades gracefully instead of OOM-killing the
        # process. See _duckdb_memory_limit_bytes for the budgeting rationale.
        mem_limit_bytes = _duckdb_memory_limit_bytes()
        con.execute(f"PRAGMA memory_limit='{mem_limit_bytes}B';")
        try:
            temp_dir = _duckdb_temp_dir()
            pathlib.Path(temp_dir).mkdir(parents=True, exist_ok=True)
            con.execute(f"PRAGMA temp_directory='{temp_dir}';")
            con.execute("PRAGMA max_temp_directory_size='32GB';")
        except Exception:
            # Spill is a safety net; if the temp dir can't be created we still
            # run with the in-memory cap (queries just can't exceed it).
            pass
        _DB_CON = con
    return _DB_CON


def _filter_valid_percentages_acbl(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty() or "Pct_NS" not in df.columns:
        return df
    pct_ns = pl.col("Pct_NS").cast(pl.Float64, strict=False)
    return df.filter(pct_ns.is_null() | ((pct_ns >= 0.0) & (pct_ns <= 1.0)))


def _duckdb_timestamp_literal(dt: datetime) -> str:
    return f"TIMESTAMP '{dt.strftime('%Y-%m-%d %H:%M:%S')}'"


def _self_filter_sql_clauses(
    full_df: pl.DataFrame,
    date_from: datetime | None,
    online_filter: str,
) -> list[str]:
    clauses: list[str] = []
    if date_from is not None and "Date" in full_df.columns:
        clauses.append(f"Date >= {_duckdb_timestamp_literal(date_from)}")
    if online_filter == "Local Only" and "is_virtual_game" in full_df.columns:
        clauses.append("is_virtual_game = false")
    elif online_filter == "Online Only" and "is_virtual_game" in full_df.columns:
        clauses.append("is_virtual_game IS NULL")
    if "Pct_NS" in full_df.columns:
        clauses.append("(Pct_NS IS NULL OR (Pct_NS >= 0 AND Pct_NS <= 1))")
    return clauses


def _prepare_self_view(
    con: duckdb.DuckDBPyConnection,
    full_df: pl.DataFrame,
    source_path: str,
    date_from: datetime | None,
    online_filter: str,
) -> tuple[int | None, str]:
    """Register the cached full frame and expose filtered rows as temp view ``self``.

    Avoids per-request Polars ``select``/``filter`` copies of the ~19 GB club
    frame; DuckDB reads columns lazily from the registered Arrow buffer.
    Row counts and date ranges come from load-time metadata (no COUNT scans).
    """
    where_clauses = _self_filter_sql_clauses(full_df, date_from, online_filter)
    where_sql = " AND ".join(where_clauses) if where_clauses else "TRUE"
    with _DB_LOCK:
        try:
            con.execute("DROP VIEW IF EXISTS self")
        except Exception:
            pass
        try:
            con.unregister("_full")
        except Exception:
            pass
        con.register("_full", full_df)
        con.execute(f"CREATE TEMP VIEW self AS SELECT * FROM _full WHERE {where_sql}")
    input_rows = full_df.height if not where_clauses else None
    date_range = _frame_date_range(source_path, date_from)
    return input_rows, date_range


def _teardown_self_view(con: duckdb.DuckDBPyConnection) -> None:
    with _DB_LOCK:
        try:
            con.execute("DROP VIEW IF EXISTS self")
        except Exception:
            pass
        try:
            con.unregister("_full")
        except Exception:
            pass


def get_elo_column_names(elo_rating_type: str) -> dict:
    if elo_rating_type == "Current Rating (End of Session)":
        return {"player_pattern": "Elo_R_{pos}", "pair_ns": "Elo_R_NS", "pair_ew": "Elo_R_EW"}
    if elo_rating_type == "Rating at Start of Session":
        return {"player_pattern": "Elo_R_{pos}_Before", "pair_ns": "Elo_R_NS_Before", "pair_ew": "Elo_R_EW_Before"}
    if elo_rating_type == "Rating at Event Start":
        return {"player_pattern": "Elo_R_{pos}_EventStart", "pair_ns": "Elo_R_NS_EventStart", "pair_ew": "Elo_R_EW_EventStart"}
    if elo_rating_type == "Rating at Event End":
        return {"player_pattern": "Elo_R_{pos}_EventEnd", "pair_ns": "Elo_R_NS_EventEnd", "pair_ew": "Elo_R_EW_EventEnd"}
    if elo_rating_type == "Expected Rating":
        return {"player_pattern": None, "pair_ns": "Elo_E_Pair_NS", "pair_ew": "Elo_E_Pair_EW"}
    return {"player_pattern": "Elo_R_{pos}", "pair_ns": "Elo_R_NS", "pair_ew": "Elo_R_EW"}


def _required_columns_for_mode(rating_type: str, elo_rating_type: str) -> list[str]:
    cols = {
        "Date", "session_id", "is_virtual_game", "Pct_NS", "Round", "Board",
        "DD_Tricks_Diff", "Is_Par_Suit", "Is_Par_Contract", "Is_Sacrifice",
        "Pair_Number_NS", "Pair_Number_EW",
    }
    for p in "NESW":
        cols.update({f"Player_ID_{p}", f"Player_Name_{p}", f"MasterPoints_{p}"})
    elo_cols = get_elo_column_names(elo_rating_type)
    if rating_type == "Players":
        player_pat = elo_cols.get("player_pattern")
        if player_pat:
            for p in "NESW":
                cols.add(player_pat.format(pos=p))
    else:
        pair_ns = elo_cols.get("pair_ns")
        pair_ew = elo_cols.get("pair_ew")
        if pair_ns:
            cols.add(pair_ns)
        if pair_ew:
            cols.add(pair_ew)
        player_pat = elo_cols.get("player_pattern")
        if player_pat:
            for p in "NESW":
                cols.add(player_pat.format(pos=p))
    return sorted(cols)


def _required_columns_for_detail(rating_type: str, elo_rating_type: str) -> list[str]:
    # is_virtual_game required so Local/Online filters match the leaderboard report.
    cols = {"Date", "session_id", "Pct_NS", "Round", "Board", "is_virtual_game"}
    for p in "NESW":
        cols.update({f"Player_ID_{p}", f"Player_Name_{p}"})

    elo_cols = get_elo_column_names(elo_rating_type)
    if rating_type == "Players":
        player_pat = elo_cols.get("player_pattern")
        if player_pat:
            for p in "NESW":
                cols.add(player_pat.format(pos=p))
                cols.add(f"Elo_R_{p}_Before")
        # Field-mix indicator for drill-down (added in the field-relative Elo
        # rewrite; tolerated as missing on legacy parquets).
        cols.update({"Field_Stdev_NS_Player", "Field_Stdev_EW_Player"})
    else:
        pair_ns = elo_cols.get("pair_ns")
        pair_ew = elo_cols.get("pair_ew")
        if pair_ns:
            cols.add(pair_ns)
        if pair_ew:
            cols.add(pair_ew)
        cols.update({"Elo_R_NS_Before", "Elo_R_EW_Before"})
        cols.update({"Field_Stdev_NS", "Field_Stdev_EW"})
    return sorted(cols)


def _server_runtime_info() -> dict:
    vm = psutil.virtual_memory()
    sm = psutil.swap_memory()
    proc = psutil.Process(os.getpid())
    proc_mem = proc.memory_info()
    limits = _detect_cgroup_limits()

    mem_limit_bytes = limits["memory_limit_bytes"] if limits["memory_limit_bytes"] is not None else int(vm.total)
    ram_used_bytes = int(proc_mem.rss)
    ram_percent = (ram_used_bytes / mem_limit_bytes * 100.0) if mem_limit_bytes > 0 else 0.0

    if limits["cpu_limit_cores"] is not None:
        cpu_count = max(1, int(limits["cpu_limit_cores"]))
        cpu_limit_cores = round(float(limits["cpu_limit_cores"]), 2)
    else:
        cpu_count = int(os.cpu_count() or 0)
        cpu_limit_cores = None

    threads = _recommended_threads()

    cached_frames = {}
    for src, cached_time in _FRAME_CACHE_TIMES.items():
        frame = _FRAME_CACHE.get(src)
        rows = len(frame) if frame is not None else 0
        cached_frames[pathlib.Path(src).name] = {
            "rows": rows,
            "cached_at": datetime.fromtimestamp(cached_time, tz=timezone.utc).isoformat(),
            "age_seconds": round(time.time() - cached_time, 1),
        }

    return {
        "api_process_started_at": API_PROCESS_STARTED_AT.isoformat(),
        "api_uptime_seconds": round(time.time() - API_PROCESS_STARTED_AT.timestamp(), 3),
        "api_source_file": str(API_SOURCE_PATH),
        "api_source_mtime": datetime.fromtimestamp(API_SOURCE_PATH.stat().st_mtime, tz=timezone.utc).isoformat(),
        # Container-aware metrics (preferred for cgroup memory limits)
        "ram_used_gb": round(ram_used_bytes / (1024 ** 3), 2),
        "ram_total_gb": round(mem_limit_bytes / (1024 ** 3), 2),
        "ram_percent": round(ram_percent, 1),
        "cpu_count": cpu_count,
        "cpu_limit_cores": cpu_limit_cores,
        "threads": threads,
        # Host metrics (diagnostic only)
        "host_ram_total_gb": round(vm.total / (1024 ** 3), 2),
        "host_ram_percent": round(vm.percent, 1),
        "process_rss_gb": round(proc_mem.rss / (1024 ** 3), 2),
        "swap_used_gb": round(sm.used / (1024 ** 3), 2),
        "swap_total_gb": round(sm.total / (1024 ** 3), 2),
        "swap_percent": round(sm.percent, 1),
        "swap_enabled": bool(sm.total > 0),
        "frame_cache": cached_frames,
        "dual_frame_cache": _dual_frame_cache_enabled(),
        "cgroup_limit_gb": round((_cgroup_memory_limit_bytes() or 0) / (1024 ** 3), 2) or None,
        "cached_frame_gb": round(_cached_frame_bytes() / (1024 ** 3), 2),
        "duckdb_memory_limit_gb": round(_duckdb_memory_limit_bytes() / (1024 ** 3), 2),
    }



def _detail_chess_calibration(df: pl.DataFrame, cols: list[str]) -> tuple[float | None, float | None]:
    """Population mean / stdev over the per-board running-Elo columns.

    Used to z-score the detail grids onto the same chess scale as the
    leaderboards. The basis here is the per-board running rating (all seats /
    sides, full frame) rather than the leaderboard's per-entity Latest, so a
    given entity's number can differ slightly between the two views; both are
    anchored at CHESS_DISPLAY_MEAN with the same title bands.
    """
    present = [c for c in cols if c in df.columns]
    if not present:
        return (None, None)
    stacked = pl.concat([df.get_column(c).cast(pl.Float64) for c in present])
    stacked = stacked.drop_nulls().drop_nans()
    if stacked.len() < 2:
        return (None, None)
    sd = float(stacked.std(ddof=0))
    return (float(stacked.mean()), sd if sd > 0 else None)


def _standardize_detail(detail: pl.DataFrame, pop_mean: float | None, pop_sd: float | None) -> pl.DataFrame:
    """Z-score Elo_Before/After onto the chess scale and add a Title column.

    No-op (leaves native values) when calibration is unavailable so the grid
    never errors on legacy parquets.
    """
    if detail.is_empty() or pop_mean is None or pop_sd is None or pop_sd <= 0:
        return detail

    def _aff(col: str) -> pl.Expr:
        return (
            (pl.lit(CHESS_DISPLAY_MEAN) + (pl.col(col).cast(pl.Float64) - pop_mean) / pop_sd * CHESS_DISPLAY_SD)
            .clip(0.0, 3500.0)
            .round(0)
            .cast(pl.Int32, strict=False)
            .alias(col)
        )

    elo_cols = [c for c in ("Elo_Before", "Elo_After") if c in detail.columns]
    if not elo_cols:
        return detail
    detail = detail.with_columns([_aff(c) for c in elo_cols])
    if "Elo_After" in detail.columns:
        detail = detail.with_columns(title_from_elo_expr("Elo_After", "Title"))
    if "Elo_Before" in detail.columns and "Elo_After" in detail.columns:
        detail = detail.with_columns((pl.col("Elo_After") - pl.col("Elo_Before")).alias("Elo_Delta"))
    return detail


def _build_player_detail(df: pl.DataFrame, player_id: str, elo_rating_type: str) -> pl.DataFrame:
    elo_columns = get_elo_column_names(elo_rating_type)
    frames: list[pl.DataFrame] = []
    for pos in "NESW":
        elo_col = elo_columns["player_pattern"].format(pos=pos) if elo_columns["player_pattern"] else None
        if elo_col is None or elo_col not in df.columns:
            continue
        partner_pos_map = {"N": "S", "S": "N", "E": "W", "W": "E"}
        partner = partner_pos_map[pos]
        opp1, opp2 = (("E", "W") if pos in ("N", "S") else ("N", "S"))
        cols_to_select = [pl.col("Date"), pl.col("session_id").alias("Session")]
        if "Round" in df.columns:
            cols_to_select.append(pl.col("Round"))
        if "Board" in df.columns:
            cols_to_select.append(pl.col("Board"))
        is_ns = pos in ("N", "S")
        if "Pct_NS" in df.columns:
            pct_expr = (pl.col("Pct_NS").cast(pl.Float64) * 100).round(1) if is_ns else ((1 - pl.col("Pct_NS").cast(pl.Float64)) * 100).round(1)
        else:
            pct_expr = pl.lit(None, dtype=pl.Float64)
        before_col = f"Elo_R_{pos}_Before" if elo_rating_type in ("Current Rating (End of Session)", "Rating at Start of Session") else None
        cols_to_select += [
            pl.lit(pos).alias("Seat"),
            pl.col(f"Player_Name_{partner}").alias("Partner"),
            (pl.col(f"Player_Name_{opp1}") + " - " + pl.col(f"Player_Name_{opp2}")).alias("Opponents"),
            pct_expr.alias("Pct"),
        ]
        if before_col and before_col in df.columns:
            cols_to_select.append(pl.col(before_col).cast(pl.Float64).round(0).cast(pl.Int32, strict=False).alias("Elo_Before"))
        cols_to_select.append(pl.col(elo_col).cast(pl.Float64).round(0).cast(pl.Int32, strict=False).alias("Elo_After"))
        # Per-board same-direction field stdev (added in field-relative Elo
        # rewrite). Tolerated as missing on legacy parquets.
        field_stdev_col = "Field_Stdev_NS_Player" if is_ns else "Field_Stdev_EW_Player"
        if field_stdev_col in df.columns:
            cols_to_select.append(pl.col(field_stdev_col).cast(pl.Float64).round(1).alias("Field_Stdev"))
        # Match leaderboard SQL: only boards with a usable Elo contribute to Sessions.
        frames.append(
            df.filter(
                (pl.col(f"Player_ID_{pos}") == player_id)
                & pl.col(elo_col).is_not_null()
                & (~pl.col(elo_col).is_nan())
            ).select(cols_to_select)
        )

    if not frames:
        return pl.DataFrame()
    detail = pl.concat(frames, how="diagonal")
    sort_cols = ["Date", "Session"]
    sort_desc = [True, True]
    if "Round" in detail.columns:
        sort_cols.append("Round")
        sort_desc.append(False)
    if "Board" in detail.columns:
        sort_cols.append("Board")
        sort_desc.append(False)
    detail = detail.sort(sort_cols, descending=sort_desc)
    if "Elo_Before" in detail.columns and "Elo_After" in detail.columns:
        detail = detail.with_columns((pl.col("Elo_After") - pl.col("Elo_Before")).alias("Elo_Delta"))
    pattern = elo_columns["player_pattern"]
    after_cols = [pattern.format(pos=p) for p in "NESW"] if pattern else []
    pop_mean, pop_sd = _detail_chess_calibration(df, after_cols)
    detail = _standardize_detail(detail, pop_mean, pop_sd)
    return detail


def _build_pair_detail(df: pl.DataFrame, pair_ids: str, elo_rating_type: str) -> pl.DataFrame:
    if "-" not in pair_ids:
        return pl.DataFrame()
    player_a, player_b = pair_ids.split("-", 1)
    elo_columns = get_elo_column_names(elo_rating_type)
    frames: list[pl.DataFrame] = []
    for side, id1_col, id2_col in [("NS", "Player_ID_N", "Player_ID_S"), ("EW", "Player_ID_E", "Player_ID_W")]:
        pair_elo_col = elo_columns.get(f"pair_{side.lower()}")
        if not pair_elo_col or pair_elo_col not in df.columns:
            continue
        # Order-independent match: min_horizontal/max_horizontal on Categorical
        # columns compare by physical dictionary code, not string value, so they
        # disagree with the lexical CASE WHEN ordering used to build Pair_IDs in
        # the leaderboard SQL and return zero rows.
        # Match leaderboard SQL: only boards with a usable pair Elo contribute
        # to Sessions (avoids Session History counting more sessions than the grid).
        side_df = df.filter(
            (
                ((pl.col(id1_col) == player_a) & (pl.col(id2_col) == player_b))
                | ((pl.col(id1_col) == player_b) & (pl.col(id2_col) == player_a))
            )
            & pl.col(pair_elo_col).is_not_null()
            & (~pl.col(pair_elo_col).is_nan())
        )
        if side_df.is_empty():
            continue
        opp_side = "EW" if side == "NS" else "NS"
        opp1, opp2 = (("E", "W") if opp_side == "EW" else ("N", "S"))
        cols_to_select = [pl.col("Date"), pl.col("session_id").alias("Session")]
        if "Round" in df.columns:
            cols_to_select.append(pl.col("Round"))
        if "Board" in df.columns:
            cols_to_select.append(pl.col("Board"))
        is_ns = side == "NS"
        if "Pct_NS" in df.columns:
            pct_expr = (pl.col("Pct_NS").cast(pl.Float64) * 100).round(1) if is_ns else ((1 - pl.col("Pct_NS").cast(pl.Float64)) * 100).round(1)
        else:
            pct_expr = pl.lit(None, dtype=pl.Float64)
        before_col = f"Elo_R_{side}_Before" if elo_rating_type in ("Current Rating (End of Session)", "Rating at Start of Session") else None
        cols_to_select += [
            pl.lit(side).alias("Side"),
            (pl.col(f"Player_Name_{opp1}") + " - " + pl.col(f"Player_Name_{opp2}")).alias("Opponents"),
            pct_expr.alias("Pct"),
        ]
        if before_col and before_col in df.columns:
            cols_to_select.append(pl.col(before_col).cast(pl.Float64).round(0).cast(pl.Int32, strict=False).alias("Elo_Before"))
        cols_to_select.append(pl.col(pair_elo_col).cast(pl.Float64).round(0).cast(pl.Int32, strict=False).alias("Elo_After"))
        # Per-board same-direction field stdev (added in field-relative Elo
        # rewrite). Tolerated as missing on legacy parquets.
        field_stdev_col = f"Field_Stdev_{side}"
        if field_stdev_col in df.columns:
            cols_to_select.append(pl.col(field_stdev_col).cast(pl.Float64).round(1).alias("Field_Stdev"))
        frames.append(side_df.select(cols_to_select))

    if not frames:
        return pl.DataFrame()
    detail = pl.concat(frames, how="diagonal")
    sort_cols = ["Date", "Session"]
    sort_desc = [True, True]
    if "Round" in detail.columns:
        sort_cols.append("Round")
        sort_desc.append(False)
    if "Board" in detail.columns:
        sort_cols.append("Board")
        sort_desc.append(False)
    detail = detail.sort(sort_cols, descending=sort_desc)
    if "Elo_Before" in detail.columns and "Elo_After" in detail.columns:
        detail = detail.with_columns((pl.col("Elo_After") - pl.col("Elo_Before")).alias("Elo_Delta"))
    after_cols = [c for c in (elo_columns.get("pair_ns"), elo_columns.get("pair_ew")) if c]
    pop_mean, pop_sd = _detail_chess_calibration(df, after_cols)
    detail = _standardize_detail(detail, pop_mean, pop_sd)
    return detail


def _rating_agg_expr(rating_method: str, value_col: str, *, has_round: bool) -> str:
    """SQL aggregate expression for the chosen rating method.

    ``Latest`` uses ``LAST(... ORDER BY Date, session_id, [Round,] Board)`` so the
    result is deterministically the player's/pair's most recent end-of-board
    Elo (DuckDB's bare ``LAST`` is non-deterministic in parallel execution).
    ``Round`` is dropped from the ordering for parquets that don't have it
    (the tournament parquet doesn't expose Round).
    ``Avg`` and ``Max`` are straightforward.
    """
    if rating_method == "Avg":
        return f"AVG({value_col})"
    if rating_method == "Max":
        return f"MAX({value_col})"
    if rating_method == "Latest":
        order_cols = "Date, session_id, Round, Board" if has_round else "Date, session_id, Board"
        return f"LAST({value_col} ORDER BY {order_cols})"
    raise ValueError(f"Invalid rating_method: {rating_method!r}")


# Quality_Score composite: arithmetic mean of four rank columns over the
# qualifying pool — pure Elo plus three field-independent bridge-quality
# metrics. Pulls down players whose Elo is high but whose actual card play
# (DD_Tricks_Diff) and bidding (Par_Suit_Rate, Par_Contract_Rate) are weak,
# which is the Zubatch failure mode. Arithmetic mean (not geometric) so a
# small-sample specialist with rank=1 in one metric but rank=46K in Elo
# doesn't get artificially elevated.
_QUALITY_SCORE_EXPR = (
    "(CAST(Player_Elo_Rank AS DOUBLE) + CAST(Par_Suit_Rank AS DOUBLE) "
    "+ CAST(Par_Contract_Rank AS DOUBLE) + CAST(DD_Tricks_Diff_Rank AS DOUBLE)) / 4.0"
)
_QUALITY_SCORE_EXPR_PAIR = (
    "(CAST(Pair_Elo_Rank AS DOUBLE) + CAST(Par_Suit_Rank AS DOUBLE) "
    "+ CAST(Par_Contract_Rank AS DOUBLE) + CAST(DD_Tricks_Diff_Rank AS DOUBLE)) / 4.0"
)


def generate_top_players_sql(
    top_n: int,
    min_sessions: int,
    rating_method: str,
    elo_rating_type: str,
    *,
    prior_anchor: float | None = None,
    prior_sessions: int = 0,
    has_round: bool = True,
    min_skill_z: float = SKILL_GATE_DEFAULT_Z,
) -> str:
    """Build the top-players ranking SQL.

    Emits ``Player_Elo_Raw`` (the original aggregate) and ``Player_Elo_Published``
    (Bayesian-shrunk toward ``prior_anchor`` with weight ``prior_sessions``).
    The existing ``Player_Elo_Score`` is kept for backward compatibility and
    is aliased to ``Player_Elo_Published``.

    Rows are ordered by ``Player_Elo_Rank`` (pure Bayesian-shrunk Elo).
    ``Quality_Rank`` — the average of Player_Elo_Rank, Par_Suit_Rank,
    Par_Contract_Rank, and DD_Tricks_Diff_Rank — is emitted as a sidecar
    column (after Sessions_Played) so users can spot players whose Elo
    position significantly outruns their field-independent quality metrics
    (the Zubatch failure mode); sorting by Quality_Rank in the grid is one
    click away.
    """
    rating_expr = _rating_agg_expr(rating_method, "Elo_R_Player", has_round=has_round)
    elo_cols = get_elo_column_names(elo_rating_type)
    player_pattern = elo_cols.get("player_pattern")
    if not player_pattern:
        raise ValueError(f"Player ratings not available for {elo_rating_type}")
    round_col = "Round" if has_round else "NULL AS Round"
    union_parts = []
    for pos in "NESW":
        elo_col = player_pattern.format(pos=pos)
        union_parts.append(
            f"""
            SELECT Date, session_id, {round_col}, Board, Player_ID_{pos} AS Player_ID, Player_Name_{pos} AS Player_Name,
                   MasterPoints_{pos} AS MasterPoints, {elo_col} AS Elo_R_Player,
                   Is_Par_Suit, Is_Par_Contract, Is_Sacrifice, DD_Tricks_Diff
            FROM self
            WHERE Player_ID_{pos} IS NOT NULL AND {elo_col} IS NOT NULL AND NOT isnan({elo_col})
            """
        )
    published_sql = _published_elo_sql(
        raw_col="Player_Elo_Raw_Float",
        sessions_col="Sessions_Played",
        prior_anchor=prior_anchor,
        prior_sessions=prior_sessions,
    )
    return f"""
    WITH player_positions AS (
      {' UNION ALL '.join(union_parts)}
    ),
    player_aggregates AS (
      SELECT
        Player_ID,
        LAST(Player_Name ORDER BY Date, session_id) AS Player_Name,
        MAX(MasterPoints) AS MasterPoints,
        {rating_expr} AS Player_Elo_Raw_Float,
        COUNT(DISTINCT session_id) AS Sessions_Played,
        AVG(CAST(Is_Par_Suit AS INTEGER)) AS Par_Suit_Rate,
        AVG(CAST(Is_Par_Contract AS INTEGER)) AS Par_Contract_Rate,
        AVG(CAST(Is_Sacrifice AS INTEGER)) AS Sacrifice_Rate,
        AVG(DD_Tricks_Diff) AS DD_Tricks_Diff_Avg
      FROM player_positions
      GROUP BY Player_ID
      HAVING COUNT(DISTINCT session_id) >= {min_sessions}
    ),
    player_with_published AS (
      SELECT
        Player_ID, Player_Name, MasterPoints,
        CAST(COALESCE(Player_Elo_Raw_Float, 0) AS INTEGER) AS Player_Elo_Raw,
        {published_sql} AS Player_Elo_Published,
        Sessions_Played, Par_Suit_Rate, Par_Contract_Rate, Sacrifice_Rate, DD_Tricks_Diff_Avg
      FROM player_aggregates
    ),
    player_with_ranks AS (
      SELECT
        *,
        CAST(ROW_NUMBER() OVER (ORDER BY Player_Elo_Published DESC, MasterPoints DESC, Player_ID ASC) AS INTEGER) AS Player_Elo_Rank,
        CAST(RANK() OVER (ORDER BY MasterPoints DESC) AS INTEGER) AS MasterPoint_Rank,
        CAST(RANK() OVER (ORDER BY Par_Suit_Rate DESC) AS INTEGER) AS Par_Suit_Rank,
        CAST(RANK() OVER (ORDER BY Par_Contract_Rate DESC) AS INTEGER) AS Par_Contract_Rank,
        CAST(RANK() OVER (ORDER BY Sacrifice_Rate DESC) AS INTEGER) AS Sacrifice_Rank,
        CAST(RANK() OVER (ORDER BY DD_Tricks_Diff_Avg DESC NULLS LAST) AS INTEGER) AS DD_Tricks_Diff_Rank
      FROM player_with_published
    ),
    player_with_quality AS (
      SELECT
        *,
        {_QUALITY_SCORE_EXPR} AS Quality_Score,
        CAST(RANK() OVER (ORDER BY {_QUALITY_SCORE_EXPR} ASC) AS INTEGER) AS Quality_Rank
      FROM player_with_ranks
    ),
    elo_stats AS (
      SELECT AVG(CAST(Player_Elo_Published AS DOUBLE)) AS elo_mean,
             STDDEV_POP(CAST(Player_Elo_Published AS DOUBLE)) AS elo_sd
      FROM player_with_published
    ),{_SKILL_STATS_CTE_TEMPLATE.format(name="skill_stats", source="player_with_published")},
    player_scaled AS (
      SELECT pwq.*,
        {zscore_chess_sql("Player_Elo_Published", "elo_mean", "elo_sd")} AS Player_Elo_Pub_Chess,
        {zscore_chess_sql("Player_Elo_Raw", "elo_mean", "elo_sd")} AS Player_Elo_Raw_Chess,
        {_skill_z_sql("pwq")} AS Skill_Z
      FROM player_with_quality pwq CROSS JOIN elo_stats CROSS JOIN skill_stats ss
    )
    SELECT
      CAST(ROW_NUMBER() OVER (ORDER BY Player_Elo_Rank ASC) AS INTEGER) AS Player_Elo_Rank,
      Player_Elo_Pub_Chess AS Player_Elo_Score,
      Player_Elo_Raw_Chess AS Player_Elo_Raw,
      Player_Elo_Pub_Chess AS Player_Elo_Published,
      {ELO_TITLE_SQL_CASE.format(elo_col="Player_Elo_Pub_Chess")} AS Title,
      ROUND(Skill_Z, 3) AS Skill_Z,
      Player_ID, Player_Name, CAST(MasterPoints AS INTEGER) AS MasterPoints,
      MasterPoint_Rank,
      CAST(Sessions_Played AS INTEGER) AS Sessions_Played,
      Quality_Rank,
      ROUND(Par_Suit_Rate * 100, 1) AS Par_Suit_Rate_Pct,
      Par_Suit_Rank,
      ROUND(Par_Contract_Rate * 100, 1) AS Par_Contract_Rate_Pct,
      Par_Contract_Rank,
      ROUND(Sacrifice_Rate * 100, 1) AS Sacrifice_Rate_Pct,
      Sacrifice_Rank,
      ROUND(DD_Tricks_Diff_Avg, 2) AS DD_Tricks_Diff_Avg,
      DD_Tricks_Diff_Rank
    FROM player_scaled
    {_skill_gate_clause(min_skill_z)}
    ORDER BY Player_Elo_Rank ASC
    LIMIT {top_n}
    """.strip()


def generate_top_pairs_sql(
    top_n: int,
    min_sessions: int,
    rating_method: str,
    elo_rating_type: str,
    *,
    prior_anchor: float | None = None,
    prior_sessions: int = 0,
    has_round: bool = True,
    min_skill_z: float = SKILL_GATE_DEFAULT_Z,
) -> str:
    """Build the top-pairs ranking SQL.

    Same Raw / Published / Quality_Rank handling as
    :func:`generate_top_players_sql`. Rows are ordered by ``Pair_Elo_Rank``
    (pure Bayesian-shrunk Elo); ``Quality_Rank`` is emitted as a sidecar
    column after Sessions for users who want to spot weak-field-inflated
    pairs.
    """
    rating_expr = _rating_agg_expr(rating_method, "Elo_R_Pair", has_round=has_round)
    round_col = "Round" if has_round else "NULL AS Round"
    elo_cols = get_elo_column_names(elo_rating_type)
    pair_ns_col = elo_cols.get("pair_ns")
    pair_ew_col = elo_cols.get("pair_ew")
    player_pattern = elo_cols.get("player_pattern")
    player_elo_n = player_pattern.format(pos="N") if player_pattern else None
    player_elo_s = player_pattern.format(pos="S") if player_pattern else None
    player_elo_e = player_pattern.format(pos="E") if player_pattern else None
    player_elo_w = player_pattern.format(pos="W") if player_pattern else None
    avg_elo_ns = (
        f"""CASE
            WHEN {player_elo_n} IS NOT NULL AND {player_elo_s} IS NOT NULL AND NOT isnan({player_elo_n}) AND NOT isnan({player_elo_s})
            THEN ({player_elo_n} + {player_elo_s}) / 2.0
            ELSE NULL
        END"""
        if player_elo_n is not None
        else "NULL"
    )
    avg_elo_ew = (
        f"""CASE
            WHEN {player_elo_e} IS NOT NULL AND {player_elo_w} IS NOT NULL AND NOT isnan({player_elo_e}) AND NOT isnan({player_elo_w})
            THEN ({player_elo_e} + {player_elo_w}) / 2.0
            ELSE NULL
        END"""
        if player_elo_e is not None
        else "NULL"
    )
    return f"""
    WITH pair_partnerships AS (
      SELECT
        Date, session_id, {round_col}, Board,
        CASE WHEN Player_ID_N < Player_ID_S THEN Player_ID_N || '-' || Player_ID_S ELSE Player_ID_S || '-' || Player_ID_N END AS Pair_IDs,
        CASE WHEN Player_ID_N <= Player_ID_S THEN Player_Name_N || ' - ' || Player_Name_S ELSE Player_Name_S || ' - ' || Player_Name_N END AS Pair_Names,
        {pair_ns_col} AS Elo_R_Pair,
        (COALESCE(MasterPoints_N, 0) + COALESCE(MasterPoints_S, 0)) / 2.0 AS Avg_MPs,
        SQRT(COALESCE(MasterPoints_N, 0) * COALESCE(MasterPoints_S, 0)) AS Geo_MPs,
        {avg_elo_ns} AS Avg_Player_Elo,
        Is_Par_Suit, Is_Par_Contract, Is_Sacrifice, DD_Tricks_Diff
      FROM self
      WHERE {pair_ns_col} IS NOT NULL AND NOT isnan({pair_ns_col})
      UNION ALL
      SELECT
        Date, session_id, {round_col}, Board,
        CASE WHEN Player_ID_E < Player_ID_W THEN Player_ID_E || '-' || Player_ID_W ELSE Player_ID_W || '-' || Player_ID_E END AS Pair_IDs,
        CASE WHEN Player_ID_E <= Player_ID_W THEN Player_Name_E || ' - ' || Player_Name_W ELSE Player_Name_W || ' - ' || Player_Name_E END AS Pair_Names,
        {pair_ew_col} AS Elo_R_Pair,
        (COALESCE(MasterPoints_E, 0) + COALESCE(MasterPoints_W, 0)) / 2.0 AS Avg_MPs,
        SQRT(COALESCE(MasterPoints_E, 0) * COALESCE(MasterPoints_W, 0)) AS Geo_MPs,
        {avg_elo_ew} AS Avg_Player_Elo,
        Is_Par_Suit, Is_Par_Contract, Is_Sacrifice, DD_Tricks_Diff
      FROM self
      WHERE {pair_ew_col} IS NOT NULL AND NOT isnan({pair_ew_col})
    ),
    pair_aggregates AS (
      SELECT
        Pair_IDs, LAST(Pair_Names ORDER BY Date, session_id) AS Pair_Names, {rating_expr} AS Pair_Elo_Raw_Float,
        AVG(Avg_MPs) AS Avg_MPs, AVG(Geo_MPs) AS Geo_MPs,
        COUNT(DISTINCT session_id) AS Sessions,
        AVG(Avg_Player_Elo) AS Avg_Player_Elo,
        AVG(CAST(Is_Par_Suit AS INTEGER)) AS Par_Suit_Rate,
        AVG(CAST(Is_Par_Contract AS INTEGER)) AS Par_Contract_Rate,
        AVG(CAST(Is_Sacrifice AS INTEGER)) AS Sacrifice_Rate,
        AVG(DD_Tricks_Diff) AS DD_Tricks_Diff_Avg
      FROM pair_partnerships
      GROUP BY Pair_IDs
      HAVING COUNT(DISTINCT session_id) >= {min_sessions}
    ),
    pair_with_published AS (
      SELECT
        Pair_IDs, Pair_Names,
        CAST(COALESCE(Pair_Elo_Raw_Float, 0) AS INTEGER) AS Pair_Elo_Raw,
        {_published_elo_sql("Pair_Elo_Raw_Float", "Sessions", prior_anchor, prior_sessions)} AS Pair_Elo_Published,
        Avg_MPs, Geo_MPs, Sessions, Avg_Player_Elo,
        Par_Suit_Rate, Par_Contract_Rate, Sacrifice_Rate, DD_Tricks_Diff_Avg
      FROM pair_aggregates
    ),
    pair_with_ranks AS (
      SELECT
        Pair_IDs, Pair_Names, Pair_Elo_Raw, Pair_Elo_Published,
        Avg_MPs, Geo_MPs, Sessions, Avg_Player_Elo,
        Par_Suit_Rate, Par_Contract_Rate, Sacrifice_Rate, DD_Tricks_Diff_Avg,
        CAST(ROW_NUMBER() OVER (ORDER BY Pair_Elo_Published DESC, Avg_MPs DESC, Pair_IDs ASC) AS INTEGER) AS Pair_Elo_Rank,
        CAST(RANK() OVER (ORDER BY Avg_Player_Elo DESC NULLS LAST) AS INTEGER) AS Avg_Elo_Rank,
        CAST(RANK() OVER (ORDER BY Avg_MPs DESC) AS INTEGER) AS Avg_MPs_Rank,
        CAST(RANK() OVER (ORDER BY Geo_MPs DESC) AS INTEGER) AS Geo_MPs_Rank,
        CAST(RANK() OVER (ORDER BY Par_Suit_Rate DESC) AS INTEGER) AS Par_Suit_Rank,
        CAST(RANK() OVER (ORDER BY Par_Contract_Rate DESC) AS INTEGER) AS Par_Contract_Rank,
        CAST(RANK() OVER (ORDER BY Sacrifice_Rate DESC) AS INTEGER) AS Sacrifice_Rank,
        CAST(RANK() OVER (ORDER BY DD_Tricks_Diff_Avg DESC NULLS LAST) AS INTEGER) AS DD_Tricks_Diff_Rank
      FROM pair_with_published
    ),
    pair_with_quality AS (
      SELECT
        *,
        {_QUALITY_SCORE_EXPR_PAIR} AS Quality_Score,
        CAST(RANK() OVER (ORDER BY {_QUALITY_SCORE_EXPR_PAIR} ASC) AS INTEGER) AS Quality_Rank
      FROM pair_with_ranks
    ),
    elo_stats AS (
      SELECT AVG(CAST(Pair_Elo_Published AS DOUBLE)) AS elo_mean,
             STDDEV_POP(CAST(Pair_Elo_Published AS DOUBLE)) AS elo_sd
      FROM pair_with_published
    ),{_SKILL_STATS_CTE_TEMPLATE.format(name="skill_stats", source="pair_with_published")},
    pair_scaled AS (
      SELECT pwq.*,
        {zscore_chess_sql("Pair_Elo_Published", "elo_mean", "elo_sd")} AS Pair_Elo_Pub_Chess,
        {zscore_chess_sql("Pair_Elo_Raw", "elo_mean", "elo_sd")} AS Pair_Elo_Raw_Chess,
        {_skill_z_sql("pwq")} AS Skill_Z
      FROM pair_with_quality pwq CROSS JOIN elo_stats CROSS JOIN skill_stats ss
    )
    SELECT
      CAST(ROW_NUMBER() OVER (ORDER BY Pair_Elo_Rank ASC) AS INTEGER) AS Pair_Elo_Rank,
      Pair_Elo_Pub_Chess AS Pair_Elo_Score,
      Pair_Elo_Raw_Chess AS Pair_Elo_Raw,
      Pair_Elo_Pub_Chess AS Pair_Elo_Published,
      {ELO_TITLE_SQL_CASE.format(elo_col="Pair_Elo_Pub_Chess")} AS Title,
      ROUND(Skill_Z, 3) AS Skill_Z,
      Avg_Elo_Rank, Pair_IDs, Pair_Names,
      CAST(Avg_MPs AS INTEGER) AS Avg_MPs, Avg_MPs_Rank, Geo_MPs_Rank, CAST(Sessions AS INTEGER) AS Sessions,
      Quality_Rank,
      ROUND(Par_Suit_Rate * 100, 1) AS Par_Suit_Rate_Pct, Par_Suit_Rank,
      ROUND(Par_Contract_Rate * 100, 1) AS Par_Contract_Rate_Pct, Par_Contract_Rank,
      ROUND(Sacrifice_Rate * 100, 1) AS Sacrifice_Rate_Pct, Sacrifice_Rank,
      ROUND(DD_Tricks_Diff_Avg, 2) AS DD_Tricks_Diff_Avg, DD_Tricks_Diff_Rank
    FROM pair_scaled
    {_skill_gate_clause(min_skill_z)}
    ORDER BY Pair_Elo_Rank ASC
    LIMIT {top_n}
    """.strip()


def _other_event_type(club_or_tournament: str) -> str:
    return "tournament" if club_or_tournament.lower() == "club" else "club"


def _crossover_session_column(club_or_tournament: str, rating_type: str) -> str:
    """Column name for sessions played in the *other* event type."""
    other = _other_event_type(club_or_tournament)
    if rating_type == "Players":
        return "Tournament_Sessions_Played" if other == "tournament" else "Club_Sessions_Played"
    return "Tournament_Sessions" if other == "tournament" else "Club_Sessions"


def _crossover_elo_column(club_or_tournament: str) -> str:
    """Column name for Elo in the *other* event type (chess-scaled published)."""
    other = _other_event_type(club_or_tournament)
    return "Tournament_Elo" if other == "tournament" else "Club_Elo"


def _insert_column_after(df: pl.DataFrame, col: str, after: str) -> pl.DataFrame:
    if col not in df.columns:
        return df
    cols = [c for c in df.columns if c != col]
    if after in cols:
        i = cols.index(after) + 1
        cols = cols[:i] + [col] + cols[i:]
    else:
        cols = cols + [col]
    return df.select(cols)


def _lazy_crossover_source(other_event: str, extra_cols: list[str] | None = None) -> pl.LazyFrame:
    """Projection of the other event parquet for crossover metrics.

    Uses the in-memory frame cache when present; otherwise scans parquet with
    column pushdown so a Club report does not load/evict the Tournament cache
    (and vice versa).
    """
    source_path, storage_options = _parquet_source_for(other_event)
    id_cols = ["Player_ID_N", "Player_ID_S", "Player_ID_E", "Player_ID_W"]
    want = ["session_id", "Date", "Board", "Round", "is_virtual_game", *id_cols, *(extra_cols or [])]
    with _FRAME_LOCK:
        cached = _FRAME_CACHE.get(source_path)
    if cached is not None:
        cols = [c for c in dict.fromkeys(want) if c in cached.columns]
        return cached.select(cols).lazy()
    schema = load_elo_ratings_schema_map(other_event)
    cols = [c for c in dict.fromkeys(want) if c in schema]
    if "session_id" not in cols or not any(c in cols for c in id_cols):
        raise RuntimeError(f"crossover source missing session/player columns: {other_event}")
    return pl.scan_parquet(source_path, storage_options=storage_options).select(cols)


def _apply_crossover_filters(
    lf: pl.LazyFrame,
    date_from: datetime | None,
    online_filter: str,
) -> pl.LazyFrame:
    names = set(lf.collect_schema().names())
    if date_from is not None and "Date" in names:
        day = pl.col("Date").cast(pl.Utf8).str.slice(0, 10)
        lf = lf.filter(day >= date_from.strftime("%Y-%m-%d"))
    if "is_virtual_game" in names:
        if online_filter == "Local Only":
            lf = lf.filter(pl.col("is_virtual_game") == False)  # noqa: E712
        elif online_filter == "Online Only":
            lf = lf.filter(pl.col("is_virtual_game").is_null())
    return lf


def _pair_ids_expr(id_a: str, id_b: str) -> pl.Expr:
    a = pl.col(id_a).cast(pl.Utf8)
    b = pl.col(id_b).cast(pl.Utf8)
    return (
        pl.when(a.is_not_null() & b.is_not_null())
        .then(pl.when(a < b).then(a + pl.lit("-") + b).otherwise(b + pl.lit("-") + a))
        .otherwise(None)
    )


def _polars_rating_agg(rating_method: str, value_col: str = "Elo") -> pl.Expr:
    """Polars aggregate matching ``_rating_agg_expr`` (Latest / Avg / Max)."""
    if rating_method == "Avg":
        return pl.col(value_col).mean()
    if rating_method == "Max":
        return pl.col(value_col).max()
    # Latest: last board in chronological order (Round optional).
    sort_keys = [pl.col("Date"), pl.col("session_id")]
    return pl.col(value_col).sort_by(sort_keys + [pl.col("Board")]).last()


def _chess_scale_series(values: pl.Series, pop_mean: float | None, pop_sd: float | None) -> pl.Series:
    if pop_mean is None or pop_sd is None or pop_sd <= 0:
        return values.cast(pl.Int64, strict=False)
    scaled = (
        CHESS_DISPLAY_MEAN
        + (values.cast(pl.Float64) - pop_mean) / pop_sd * CHESS_DISPLAY_SD
    ).clip(0.0, 3500.0).round(0)
    return scaled.cast(pl.Int64, strict=False)


def _crossover_metrics(
    *,
    other_event: str,
    rating_type: str,
    entity_ids: list[str],
    count_col: str,
    elo_col: str,
    date_from: datetime | None,
    online_filter: str,
    rating_method: str,
    elo_rating_type: str,
    prior_sessions: int,
) -> pl.DataFrame:
    """Sessions + published chess-scaled Elo on the other event type for given IDs."""
    id_name = "Player_ID" if rating_type == "Players" else "Pair_IDs"
    empty = pl.DataFrame({id_name: [], count_col: [], elo_col: []}).cast(
        {id_name: pl.Utf8, count_col: pl.Int64, elo_col: pl.Int64}
    )
    if not entity_ids:
        return empty

    elo_names = get_elo_column_names(elo_rating_type)
    extra: list[str] = []
    if rating_type == "Players":
        pattern = elo_names.get("player_pattern")
        if not pattern:
            return empty
        extra = [pattern.format(pos=p) for p in "NESW"]
    else:
        for key in ("pair_ns", "pair_ew"):
            col = elo_names.get(key)
            if col:
                extra.append(col)
        if not extra:
            return empty

    lf = _apply_crossover_filters(
        _lazy_crossover_source(other_event, extra_cols=extra),
        date_from,
        online_filter,
    )
    names = set(lf.collect_schema().names())
    id_cols = [c for c in ("Player_ID_N", "Player_ID_S", "Player_ID_E", "Player_ID_W") if c in names]
    if not id_cols or "session_id" not in names:
        raise RuntimeError(f"crossover source incomplete for {other_event}")

    id_set = list(dict.fromkeys(str(x) for x in entity_ids if x is not None and str(x)))
    present_elo = [c for c in extra if c in names]
    if not present_elo:
        return empty

    if rating_type == "Players":
        pattern = elo_names["player_pattern"]
        any_match = pl.any_horizontal([pl.col(c).cast(pl.Utf8).is_in(id_set) for c in id_cols])
        matched = lf.filter(any_match)
        parts: list[pl.LazyFrame] = []
        for pos, id_c in (("N", "Player_ID_N"), ("E", "Player_ID_E"), ("S", "Player_ID_S"), ("W", "Player_ID_W")):
            if id_c not in names:
                continue
            elo_c = pattern.format(pos=pos)
            if elo_c not in names:
                continue
            parts.append(
                matched.select(
                    pl.col(id_c).cast(pl.Utf8).alias("Player_ID"),
                    pl.col("session_id"),
                    pl.col("Date"),
                    pl.col("Board") if "Board" in names else pl.lit(0).alias("Board"),
                    pl.col(elo_c).cast(pl.Float64).alias("Elo"),
                ).filter(
                    pl.col("Player_ID").is_in(id_set)
                    & pl.col("Elo").is_not_null()
                    & pl.col("Elo").is_not_nan()
                )
            )
        if not parts:
            return empty
        long = pl.concat(parts)
    else:
        pair_parts: list[pl.LazyFrame] = []
        for side, id_a, id_b, elo_key in (
            ("NS", "Player_ID_N", "Player_ID_S", "pair_ns"),
            ("EW", "Player_ID_E", "Player_ID_W", "pair_ew"),
        ):
            elo_c = elo_names.get(elo_key)
            if not elo_c or elo_c not in names or id_a not in names or id_b not in names:
                continue
            pair_parts.append(
                lf.select(
                    _pair_ids_expr(id_a, id_b).alias("Pair_IDs"),
                    pl.col("session_id"),
                    pl.col("Date"),
                    pl.col("Board") if "Board" in names else pl.lit(0).alias("Board"),
                    pl.col(elo_c).cast(pl.Float64).alias("Elo"),
                ).filter(
                    pl.col("Pair_IDs").is_in(id_set)
                    & pl.col("Elo").is_not_null()
                    & pl.col("Elo").is_not_nan()
                )
            )
        if not pair_parts:
            return empty
        long = pl.concat(pair_parts)

    # One collect: board-level Elo calibration + per-entity sessions/rating.
    long_df = long.collect(engine="streaming")
    if long_df.is_empty():
        return empty
    elo_vals = long_df.get_column("Elo").drop_nulls().drop_nans()
    pop_mean = float(elo_vals.mean()) if elo_vals.len() else None
    pop_sd = float(elo_vals.std(ddof=0)) if elo_vals.len() >= 2 else None

    agg = long_df.group_by(id_name).agg(
        pl.col("session_id").n_unique().cast(pl.Int64).alias(count_col),
        _polars_rating_agg(rating_method, "Elo").alias("_elo_raw"),
    )

    prior_anchor = _shrinkage_anchor(_load_shrinkage_meta(other_event), "player" if rating_type == "Players" else "pair")
    raw = agg.get_column("_elo_raw").cast(pl.Float64)
    sessions = agg.get_column(count_col).cast(pl.Float64)
    if prior_anchor is not None and prior_sessions > 0:
        published = (
            (sessions * raw + float(prior_sessions) * float(prior_anchor))
            / (sessions + float(prior_sessions))
        )
    else:
        published = raw
    chess = _chess_scale_series(published, pop_mean, pop_sd if pop_sd and pop_sd > 0 else None)
    return agg.select(
        pl.col(id_name).cast(pl.Utf8),
        pl.col(count_col),
        chess.alias(elo_col),
    )


def _attach_crossover_columns(
    result_df: pl.DataFrame,
    *,
    club_or_tournament: str,
    rating_type: str,
    date_from: datetime | None,
    online_filter: str,
    rating_method: str,
    elo_rating_type: str,
    prior_sessions: int,
) -> pl.DataFrame:
    """Add other-event session counts and Elo for Club↔Tournament crossover."""
    if result_df.is_empty():
        return result_df
    count_col = _crossover_session_column(club_or_tournament, rating_type)
    elo_col = _crossover_elo_column(club_or_tournament)
    id_col = "Player_ID" if rating_type == "Players" else "Pair_IDs"
    if id_col not in result_df.columns:
        return result_df
    after_col = "Sessions_Played" if rating_type == "Players" else "Sessions"
    other = _other_event_type(club_or_tournament)
    entity_ids = result_df.get_column(id_col).cast(pl.Utf8).to_list()
    metrics = _crossover_metrics(
        other_event=other,
        rating_type=rating_type,
        entity_ids=entity_ids,
        count_col=count_col,
        elo_col=elo_col,
        date_from=date_from,
        online_filter=online_filter,
        rating_method=rating_method,
        elo_rating_type=elo_rating_type,
        prior_sessions=prior_sessions,
    )
    out = (
        result_df.with_columns(pl.col(id_col).cast(pl.Utf8))
        .join(metrics, on=id_col, how="left")
        .with_columns(pl.col(count_col).fill_null(0).cast(pl.Int64))
    )
    # Elo stays null when the entity has no sessions in the other event.
    out = _insert_column_after(out, count_col, after_col)
    return _insert_column_after(out, elo_col, count_col)


@app.get("/health")
def health() -> dict:
    from streamlitlib.memory_usage import get_memory_usage_dict

    runtime = _server_runtime_info()
    return {
        "status": "ok",
        "service": "acbl-api",
        "memory": get_memory_usage_dict(),
        "server": runtime,
        "frame_cache": runtime.get("frame_cache", {}),
        "dual_frame_cache": runtime.get("dual_frame_cache"),
        "cached_frame_gb": round(_cached_frame_bytes() / (1024 ** 3), 2),
        "build_tag": API_BUILD_TAG,
    }


@app.get("/acbl/report")
def acbl_report(
    club_or_tournament: str = Query(..., pattern="^(club|tournament)$"),
    rating_type: str = Query(..., pattern="^(Players|Pairs)$"),
    top_n: int = Query(100, ge=1, le=5000),
    min_sessions: int = Query(10, ge=1, le=10000),
    rating_method: str = Query("Latest"),
    moving_avg_days: int = Query(10, ge=1, le=3650),
    elo_rating_type: str = Query("Current Rating (End of Session)"),
    date_from: str | None = Query(None),
    online_filter: str = Query("All"),
    prior_sessions: int = Query(
        SHRINKAGE_DEFAULT_PRIOR_SESSIONS, ge=0, le=1000,
        description="Bayesian shrinkage prior weight (in 'sessions equivalent'). "
                    "0 disables shrinkage (Published == Raw).",
    ),
    min_skill_z: float = Query(
        SKILL_GATE_DEFAULT_Z, ge=-100.0, le=5.0,
        description="Elite skill gate: exclude players/pairs whose field-"
                    "independent card-play+bidding z-score (over the qualifying "
                    "pool) is below this. Higher = stricter. Set <= -90 to "
                    "disable (show all qualifying entities).",
    ),
) -> dict:
    with _REPORT_LOCK:
        started_at = datetime.now()
        t0 = time.perf_counter()
        try:
            t_parse_start = time.perf_counter()
            parsed_date_from = None if not date_from else datetime.fromisoformat(date_from)
            t_parse_end = time.perf_counter()

            t_load_start = time.perf_counter()
            source_path, _storage_options = _parquet_source_for(club_or_tournament)
            full_df = _load_full_frame(club_or_tournament)
            t_load_end = time.perf_counter()

            t_filter_start = time.perf_counter()
            con = _get_db_connection()
            input_rows, date_range = _prepare_self_view(
                con, full_df, source_path, parsed_date_from, online_filter,
            )
            t_filter_end = time.perf_counter()

            t_sql_start = time.perf_counter()
            try:
                shrinkage_meta = _load_shrinkage_meta(club_or_tournament)
                anchor_kind = "player" if rating_type == "Players" else "pair"
                prior_anchor = _shrinkage_anchor(shrinkage_meta, anchor_kind)
                has_round = "Round" in full_df.columns
                if rating_type == "Players":
                    generated_sql = generate_top_players_sql(
                        top_n, min_sessions, rating_method, elo_rating_type,
                        prior_anchor=prior_anchor, prior_sessions=prior_sessions,
                        has_round=has_round, min_skill_z=min_skill_z,
                    )
                else:
                    generated_sql = generate_top_pairs_sql(
                        top_n, min_sessions, rating_method, elo_rating_type,
                        prior_anchor=prior_anchor, prior_sessions=prior_sessions,
                        has_round=has_round, min_skill_z=min_skill_z,
                    )
                with _DB_LOCK:
                    result_df = con.execute(generated_sql).pl()
            finally:
                _teardown_self_view(con)
                _reset_duckdb_connection()
            t_sql_end = time.perf_counter()

            t_xover_start = time.perf_counter()
            result_df = _attach_crossover_columns(
                result_df,
                club_or_tournament=club_or_tournament,
                rating_type=rating_type,
                date_from=parsed_date_from,
                online_filter=online_filter,
                rating_method=rating_method,
                elo_rating_type=elo_rating_type,
                prior_sessions=prior_sessions,
            )
            t_xover_end = time.perf_counter()

            t_serialize_start = time.perf_counter()
            result_rows = result_df.to_dicts()
            t_serialize_end = time.perf_counter()

            ended_at = datetime.now()
            elapsed = (ended_at - started_at).total_seconds()
            output_rows = len(result_df)
            perf = {
                "source": "r2" if _r2_enabled() else "local",
                "parse_seconds": round(t_parse_end - t_parse_start, 3),
                "load_seconds": round(t_load_end - t_load_start, 3),
                "filter_seconds": round(t_filter_end - t_filter_start, 3),
                "sql_seconds": round(t_sql_end - t_sql_start, 3),
                "crossover_seconds": round(t_xover_end - t_xover_start, 3),
                "serialize_seconds": round(t_serialize_end - t_serialize_start, 3),
                "input_rows": input_rows,
                "output_rows": output_rows,
            }
            response_payload = {
                "rows": result_rows,
                "generated_sql": generated_sql,
                "date_range": date_range,
                "row_count": output_rows,
                "started_at": started_at.isoformat(),
                "ended_at": ended_at.isoformat(),
                "elapsed_seconds": elapsed,
                "moving_avg_days": moving_avg_days,
                "shrinkage": {
                    "prior_sessions": int(prior_sessions),
                    "prior_anchor": prior_anchor,
                    "applied": prior_anchor is not None and prior_sessions > 0,
                    "kind": anchor_kind,
                },
                "skill_gate": {
                    "min_skill_z": float(min_skill_z),
                    "applied": min_skill_z > SKILL_GATE_DISABLED,
                    "metric": "pool z-score of DD_Tricks_Diff + Par_Suit + Par_Contract",
                },
                "perf": perf,
                "server": _server_runtime_info(),
            }
            del result_df
            gc.collect()
            _malloc_trim()
            print(
                f"[acbl-api] report {club_or_tournament}/{rating_type} done "
                f"({output_rows} rows, mem {_cgroup_mem_summary()})",
                flush=True,
            )
            response_payload["perf"]["total_seconds"] = round(time.perf_counter() - t0, 3)
            return response_payload
        except HTTPException:
            raise
        except Exception as exc:
            print(
                f"[acbl-api] report failed {club_or_tournament}/{rating_type}: "
                f"{exc!r} mem {_cgroup_mem_summary()}",
                flush=True,
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/acbl/detail")
def acbl_detail(
    club_or_tournament: str = Query(..., pattern="^(club|tournament)$"),
    rating_type: str = Query(..., pattern="^(Players|Pairs)$"),
    elo_rating_type: str = Query("Current Rating (End of Session)"),
    date_from: str | None = Query(None),
    online_filter: str = Query("All"),
    player_id: str | None = Query(None),
    pair_ids: str | None = Query(None),
) -> dict:
    with _REPORT_LOCK:
        started_at = datetime.now()
        t0 = time.perf_counter()
        try:
            t_parse_start = time.perf_counter()
            parsed_date_from = None if not date_from else datetime.fromisoformat(date_from)
            t_parse_end = time.perf_counter()

            t_load_start = time.perf_counter()
            required_columns = _required_columns_for_detail(rating_type, elo_rating_type)
            df = load_elo_ratings(club_or_tournament, columns=required_columns, date_from=parsed_date_from)
            df = _filter_valid_percentages_acbl(df)
            t_load_end = time.perf_counter()

            t_filter_start = time.perf_counter()
            if online_filter == "Local Only" and "is_virtual_game" in df.columns:
                df = df.filter(pl.col("is_virtual_game") == False)
            elif online_filter == "Online Only" and "is_virtual_game" in df.columns:
                df = df.filter(pl.col("is_virtual_game").is_null())
            t_filter_end = time.perf_counter()

            t_build_start = time.perf_counter()
            if rating_type == "Players":
                if not player_id:
                    raise HTTPException(status_code=400, detail="player_id is required for Players detail.")
                detail = _build_player_detail(df, player_id=str(player_id), elo_rating_type=elo_rating_type)
            else:
                if not pair_ids:
                    raise HTTPException(status_code=400, detail="pair_ids is required for Pairs detail.")
                detail = _build_pair_detail(df, pair_ids=str(pair_ids), elo_rating_type=elo_rating_type)
            t_build_end = time.perf_counter()

            t_serialize_start = time.perf_counter()
            detail_rows = detail.to_dicts()
            t_serialize_end = time.perf_counter()

            ended_at = datetime.now()
            elapsed = (ended_at - started_at).total_seconds()
            input_rows = len(df)
            output_rows = len(detail)
            perf = {
                "source": "r2" if _r2_enabled() else "local",
                "parse_seconds": round(t_parse_end - t_parse_start, 3),
                "load_seconds": round(t_load_end - t_load_start, 3),
                "filter_seconds": round(t_filter_end - t_filter_start, 3),
                "build_seconds": round(t_build_end - t_build_start, 3),
                "serialize_seconds": round(t_serialize_end - t_serialize_start, 3),
                "input_rows": input_rows,
                "output_rows": output_rows,
            }
            response_payload = {
                "rows": detail_rows,
                "row_count": output_rows,
                "started_at": started_at.isoformat(),
                "ended_at": ended_at.isoformat(),
                "elapsed_seconds": elapsed,
                "perf": perf,
                "server": _server_runtime_info(),
            }
            del detail, df
            _reset_duckdb_connection()
            gc.collect()
            _malloc_trim()
            response_payload["perf"]["total_seconds"] = round(time.perf_counter() - t0, 3)
            return response_payload
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

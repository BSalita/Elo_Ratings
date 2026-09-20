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

from acbl_awards import attach_award_totals, attach_session_awards, load_awards_for_players, pair_member_ids
from acbl_platinum import load_platinum_events, platinum_event_ids, platinum_mp_color_expr
from acbl_strata import STRATA_DEFAULT, strata_label_to_bucket
from elo_filter_common import acbl_date_from_for_range, filter_acbl_leaderboard
from elo_session_common import acbl_results_url_expr, results_url_status
from elo_common import (
    CHESS_DISPLAY_MEAN,
    CHESS_DISPLAY_SD,
    SKILL_GATE_DEFAULT_CLUB_Z,
    SKILL_GATE_DEFAULT_TOURNAMENT_Z,
    SKILL_GATE_DISABLED,
    default_min_skill_z,
    title_from_elo_expr,
)
from elo_favorites import (
    button_prompt_ids,
    flatten_favorites,
    load_favorites,
    run_favorite,
    run_sql,
)
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import re

DATA_ROOT = pathlib.Path(
    os.environ.get("DATA_ROOT") or (pathlib.Path(__file__).resolve().parent / "data")
)
API_SOURCE_PATH = pathlib.Path(__file__).resolve()
API_PROCESS_STARTED_AT = datetime.now(timezone.utc)
# Bump when deploying memory/toggle fixes so /health confirms the running build.
API_BUILD_TAG = "2026-09-20-elo-favorites"

QUALITY_METRIC_DEFINITIONS = {
    "DD_Tricks_Diff_Avg": {
        "label": "T-DD",
        "population": "Declarations only; credited to the declarer and declaring pair.",
        "calculation": "Mean DD_Tricks_Diff (actual tricks minus double-dummy tricks).",
    },
    "Par_Contract_Rate_Pct": {
        "label": "Par Contract success percentage",
        "population": "Every pair direction on every board; credited to both pair members.",
        "calculation": (
            "Directional score is +1 when DD_Score_NS/EW >= Par_NS/EW, otherwise -1; "
            "rows with a missing DD or par score are excluded. Displayed as "
            "(mean signed score + 1) * 50."
        ),
    },
    "Par_Suit_Rate_Pct": {
        "label": "Par Suit percentage",
        "population": "All declarations; credited to the declaring pair and both pair members.",
        "calculation": "Hit when BidSuit occurs in the Strain fields of ParContracts.",
    },
    "Sacrifice_Rate_Pct": {
        "label": "Sacrifice percentage",
        "population": (
            "Declarations where the declaring direction's Par_Declarer is negative; "
            "credited to the declaring pair and both pair members."
        ),
        "calculation": "Hit when DD_Score_Declarer equals negative Par_Declarer.",
    },
}

# Module-level caches to avoid re-reading parquet files on every request.
# Keys are source paths; values are the cached objects.
_SCHEMA_CACHE: dict[str, dict] = {}
_FRAME_CACHE: dict[str, pl.DataFrame] = {}
_FRAME_CACHE_TIMES: dict[str, float] = {}
# Full-dataset Date bounds captured once at frame load (avoids per-request scans).
_FRAME_DATE_BOUNDS: dict[str, tuple[datetime | None, datetime | None]] = {}
_PLATINUM_EVENTS_CACHE: pl.DataFrame | None = None

import threading as _threading
_DB_LOCK = _threading.Lock()
_FRAME_LOCK = _threading.Lock()
# Serialize report/detail handlers so Club↔Tournament toggles cannot overlap two
# full-frame loads (~13 GB Club + reload) and OOM the container.
_REPORT_LOCK = _threading.Lock()
_DB_CON: duckdb.DuckDBPyConnection | None = None

app = FastAPI(title="ACBL Elo API", version="1.2.0")
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
    source_path = _parquet_source_for(club_or_tournament)
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
    source_path = _parquet_source_for(club_or_tournament)
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


# In-memory cache for shrinkage sidecars (one per club|tournament). Loaded
# lazily on first /acbl/report request and reused across calls.
_SHRINKAGE_META_CACHE: dict[str, dict | None] = {}

# Default Bayesian shrinkage prior weight (sessions equivalent). The Streamlit
# /acbl/report client may override via the prior_sessions query parameter.
SHRINKAGE_DEFAULT_PRIOR_SESSIONS = 50

# Favorites SQL disables the skill gate at or below this value.
SKILL_GATE_DEFAULT_Z = SKILL_GATE_DISABLED

_SQL_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|COPY|PRAGMA|ATTACH|DETACH|"
    r"EXPORT|IMPORT|INSTALL|LOAD|CALL|SET|RESET|VACUUM|CHECKPOINT)\b",
    re.IGNORECASE,
)
MAX_ACBL_SQL_ROWS = 10000
_ACBL_FAVORITES: dict | None = None


class AcblSqlBody(BaseModel):
    sql: str = Field(..., min_length=1)


def _acbl_favorites() -> dict:
    global _ACBL_FAVORITES
    if _ACBL_FAVORITES is None:
        _ACBL_FAVORITES = load_favorites("acbl")
    return _ACBL_FAVORITES


def _require_select_sql(sql: str) -> str:
    cleaned = (sql or "").strip().rstrip(";")
    if not cleaned:
        raise ValueError("sql is required")
    if _SQL_FORBIDDEN.search(cleaned):
        raise ValueError("Only SELECT/WITH queries against table self are allowed")
    head = cleaned.split(None, 1)[0].upper()
    if head not in {"SELECT", "WITH"}:
        raise ValueError("Only SELECT/WITH queries against table self are allowed")
    return cleaned


def acbl_favorites_meta(
    *,
    top_n: int,
    min_sessions: int,
    rating_method: str,
    elo_rating_type: str,
    rating_type: str,
    prior_anchor: float | None,
    prior_sessions: int,
    min_skill_z: float,
) -> dict:
    """Sidebar/API widgets become brace-macro values for default.acbl.favorites.json."""
    elo_cols = get_elo_column_names(elo_rating_type)
    player_pattern = elo_cols.get("player_pattern")
    if rating_type == "Players" and not player_pattern:
        raise ValueError(f"Player ratings not available for {elo_rating_type}")

    def player_col(pos: str) -> str:
        if player_pattern:
            return player_pattern.format(pos=pos)
        return "NULL"

    suffix = ""
    if player_pattern and "{pos}" in player_pattern:
        suffix = player_pattern.split("{pos}", 1)[1]
    return {
        "Top_N": int(top_n),
        "Min_Sessions": int(min_sessions),
        "Prior_Sessions": int(prior_sessions),
        "Prior_Anchor": "NULL" if prior_anchor is None else repr(float(prior_anchor)),
        "Min_Skill_Z": float(min_skill_z),
        "Rating_Method": rating_method,
        "Rating_Type": rating_type,
        "Elo_Suffix": suffix,
        "Elo_Col_N": player_col("N"),
        "Elo_Col_S": player_col("S"),
        "Elo_Col_E": player_col("E"),
        "Elo_Col_W": player_col("W"),
        "Elo_Col_NS": elo_cols.get("pair_ns") or "NULL",
        "Elo_Col_EW": elo_cols.get("pair_ew") or "NULL",
    }


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


def _load_shrinkage_meta(club_or_tournament: str) -> dict | None:
    """Load the shrinkage sidecar JSON written by acbl_elo_ratings_create.py.

    Returns the parsed dict, or None when the sidecar is unavailable (e.g.
    first deployment after the math change but before the recompute has
    rebuilt the lookup parquets). In that case the report falls back to
    Published == Raw so the API still works.

    Looks in DATA_ROOT, optional ACBL_SHRINKAGE_DIR, then e:/bridge/data/acbl.
    Cached at module level.
    """
    key = club_or_tournament.lower()
    if key in _SHRINKAGE_META_CACHE:
        return _SHRINKAGE_META_CACHE[key]

    filename = f"acbl_{key}_elo_shrinkage.json"

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


def _parquet_source_for(club_or_tournament: str) -> str:
    filename = f"acbl_{club_or_tournament.lower()}_elo_ratings.parquet"
    file_path = DATA_ROOT.joinpath(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")
    return str(file_path)


def load_elo_ratings_schema_map(club_or_tournament: str) -> dict:
    source_path = _parquet_source_for(club_or_tournament)
    if source_path in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[source_path]
    df0 = pl.read_parquet(source_path, n_rows=0)
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
    source_path = _parquet_source_for(club_or_tournament)
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
        lf = pl.scan_parquet(source_path)

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


def _require_strata_column(df: pl.DataFrame | pl.LazyFrame, *, names: set[str] | None = None) -> None:
    cols = names if names is not None else set(df.columns)
    if "strata_bucket" not in cols:
        raise HTTPException(
            status_code=500,
            detail=(
                "Elo parquet is missing strata_bucket. Rebuild via "
                "acbl_sql_to_board_results_clean.py (club mpLimits→mp_limit) then "
                "acbl_elo_ratings_create.py, and refresh the deployed parquet."
            ),
        )


def _sql_string_list(values: list[str]) -> str:
    escaped = [value.replace("'", "''") for value in values]
    return "(" + ", ".join(f"'{value}'" for value in escaped) + ")"


def _cached_platinum_events() -> pl.DataFrame:
    global _PLATINUM_EVENTS_CACHE
    if _PLATINUM_EVENTS_CACHE is None:
        _PLATINUM_EVENTS_CACHE = load_platinum_events(DATA_ROOT)
    return _PLATINUM_EVENTS_CACHE


def _reject_club_platinum(club_or_tournament: str, platinum_events: bool) -> None:
    if platinum_events and club_or_tournament.lower() != "tournament":
        raise HTTPException(
            status_code=400,
            detail="platinum_events applies only to club_or_tournament=tournament.",
        )


def _require_platinum_event_ids() -> list[str]:
    ids = platinum_event_ids(_cached_platinum_events())
    if not ids:
        raise HTTPException(
            status_code=500,
            detail=(
                "No tournament events with mp_color=Platinum were found. "
                "Provide the tournament board-results parquet "
                "(ACBL_TOURNAMENT_BOARD_RESULTS) or "
                "acbl_tournament_platinum_events.parquet in DATA_ROOT."
            ),
        )
    return ids


def _apply_platinum_event_filter(df: pl.DataFrame) -> pl.DataFrame:
    if "mp_color" in df.columns:
        return df.filter(platinum_mp_color_expr())
    if "event_id" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail="Elo parquet is missing event_id; cannot filter platinum events.",
        )
    ids = _require_platinum_event_ids()
    return df.filter(pl.col("event_id").cast(pl.Utf8).is_in(ids))


def _self_filter_sql_clauses(
    full_df: pl.DataFrame,
    date_from: datetime | None,
    online_filter: str,
    strata: str,
    platinum_events: bool = False,
) -> list[str]:
    clauses: list[str] = []
    if date_from is not None and "Date" in full_df.columns:
        clauses.append(f"Date >= {_duckdb_timestamp_literal(date_from)}")
    if online_filter == "Local Only" and "is_virtual_game" in full_df.columns:
        clauses.append("is_virtual_game = false")
    elif online_filter == "Online Only" and "is_virtual_game" in full_df.columns:
        clauses.append("is_virtual_game IS NULL")
    bucket = strata_label_to_bucket(strata)
    if bucket is not None:
        _require_strata_column(full_df)
        # Bucket ids are controlled constants (no user free-text).
        clauses.append(f"strata_bucket = '{bucket}'")
    if platinum_events:
        if "mp_color" in full_df.columns:
            clauses.append("lower(trim(CAST(mp_color AS VARCHAR))) = 'platinum'")
        elif "event_id" not in full_df.columns:
            raise HTTPException(
                status_code=500,
                detail="Elo parquet is missing event_id; cannot filter platinum events.",
            )
        else:
            clauses.append(f"event_id IN {_sql_string_list(_require_platinum_event_ids())}")
    if "Pct_NS" in full_df.columns:
        clauses.append("(Pct_NS IS NULL OR (Pct_NS >= 0 AND Pct_NS <= 1))")
    return clauses


def _prepare_self_view(
    con: duckdb.DuckDBPyConnection,
    full_df: pl.DataFrame,
    source_path: str,
    date_from: datetime | None,
    online_filter: str,
    strata: str = STRATA_DEFAULT,
    platinum_events: bool = False,
) -> tuple[int | None, str]:
    """Register the cached full frame and expose filtered rows as temp view ``self``.

    Avoids per-request Polars ``select``/``filter`` copies of the ~19 GB club
    frame; DuckDB reads columns lazily from the registered Arrow buffer.
    Row counts and date ranges come from load-time metadata (no COUNT scans).
    """
    where_clauses = _self_filter_sql_clauses(
        full_df, date_from, online_filter, strata, platinum_events=platinum_events,
    )
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
        select_sql = "*" if "Round" in full_df.columns else "*, CAST(NULL AS INTEGER) AS Round"
        con.execute(f"CREATE TEMP VIEW self AS SELECT {select_sql} FROM _full WHERE {where_sql}")
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
        "Date", "session_id", "is_virtual_game", "strata_bucket", "Pct_NS", "Round", "Board",
        "DD_Tricks_Diff", "Declarer_Direction", "Declarer_Pair_Direction",
        "BidSuit", "ParContracts", "DD_Score_NS", "DD_Score_EW",
        "Par_NS", "Par_EW", "DD_Score_Declarer", "Par_Declarer",
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
    # is_virtual_game / strata_bucket required so filters match the leaderboard report.
    cols = {
        "Date",
        "session_id",
        "Pct_NS",
        "Round",
        "Board",
        "is_virtual_game",
        "strata_bucket",
    }
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
        cols_to_select = [
            pl.col("Date"),
            pl.col("session_id").alias("Session"),
        ]
        if "event_id" in df.columns:
            cols_to_select.append(pl.col("event_id").alias("Event_ID"))
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
        cols_to_select = [
            pl.col("Date"),
            pl.col("session_id").alias("Session"),
        ]
        if "event_id" in df.columns:
            cols_to_select.append(pl.col("event_id").alias("Event_ID"))
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
    source_path = _parquet_source_for(other_event)
    id_cols = ["Player_ID_N", "Player_ID_S", "Player_ID_E", "Player_ID_W"]
    want = [
        "session_id", "Date", "Board", "Round", "is_virtual_game", "strata_bucket",
        *id_cols, *(extra_cols or []),
    ]
    with _FRAME_LOCK:
        cached = _FRAME_CACHE.get(source_path)
    if cached is not None:
        cols = [c for c in dict.fromkeys(want) if c in cached.columns]
        return cached.select(cols).lazy()
    schema = load_elo_ratings_schema_map(other_event)
    cols = [c for c in dict.fromkeys(want) if c in schema]
    if "session_id" not in cols or not any(c in cols for c in id_cols):
        raise RuntimeError(f"crossover source missing session/player columns: {other_event}")
    return pl.scan_parquet(source_path).select(cols)


def _apply_crossover_filters(
    lf: pl.LazyFrame,
    date_from: datetime | None,
    online_filter: str,
    strata: str = STRATA_DEFAULT,
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
    bucket = strata_label_to_bucket(strata)
    if bucket is not None:
        _require_strata_column(lf, names=names)
        lf = lf.filter(pl.col("strata_bucket") == bucket)
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
    """Polars aggregate matching favorites Latest / Avg / Max (crossover only)."""
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
    strata: str = STRATA_DEFAULT,
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
        strata,
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


def _leaderboard_player_ids(result_df: pl.DataFrame, rating_type: str) -> list[str]:
    if rating_type == "Pairs":
        if "Pair_IDs" not in result_df.columns:
            return []
        ids: list[str] = []
        for pair in result_df.get_column("Pair_IDs").cast(pl.Utf8).to_list():
            ids.extend(pair_member_ids(pair))
        return ids
    if "Player_ID" not in result_df.columns:
        return []
    return result_df.get_column("Player_ID").cast(pl.Utf8).to_list()


def _attach_award_columns(
    result_df: pl.DataFrame,
    *,
    club_or_tournament: str,
    rating_type: str,
    session_ids: list[str],
) -> pl.DataFrame:
    if result_df.is_empty():
        return result_df
    awards = load_awards_for_players(
        DATA_ROOT,
        club_or_tournament,
        _leaderboard_player_ids(result_df, rating_type),
    )
    return attach_award_totals(
        result_df,
        awards,
        rating_type=rating_type,
        session_ids=session_ids,
    )


def _attach_detail_awards(
    detail: pl.DataFrame,
    *,
    club_or_tournament: str,
    rating_type: str,
    player_id: str | None,
    pair_ids: str | None,
) -> pl.DataFrame:
    if detail.is_empty():
        return detail
    if rating_type == "Pairs":
        ids = pair_member_ids(str(pair_ids or ""))
    else:
        ids = [str(player_id)] if player_id else []
    awards = load_awards_for_players(DATA_ROOT, club_or_tournament, ids)
    return attach_session_awards(
        detail,
        awards,
        player_id=None if rating_type == "Pairs" else player_id,
        pair_ids=pair_ids if rating_type == "Pairs" else None,
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
    strata: str = STRATA_DEFAULT,
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
        strata=strata,
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
        "api_version": app.version,
        "memory": get_memory_usage_dict(),
        "server": runtime,
        "frame_cache": runtime.get("frame_cache", {}),
        "dual_frame_cache": runtime.get("dual_frame_cache"),
        "cached_frame_gb": round(_cached_frame_bytes() / (1024 ** 3), 2),
        "build_tag": API_BUILD_TAG,
        "skill_gate": {
            "disabled_at_or_below": SKILL_GATE_DISABLED,
            "defaults": {
                "club": SKILL_GATE_DEFAULT_CLUB_Z,
                "tournament": SKILL_GATE_DEFAULT_TOURNAMENT_Z,
            },
            "enabled_by_default": True,
        },
        "quality_metric_definitions": QUALITY_METRIC_DEFINITIONS,
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
    date_range: str | None = Query(None),
    online_filter: str = Query("All"),
    strata: str = Query(
        STRATA_DEFAULT,
        description="Event MP-limit strata filter (Open / restricted buckets / All).",
    ),
    prior_sessions: int = Query(
        SHRINKAGE_DEFAULT_PRIOR_SESSIONS, ge=0, le=1000,
        description="Bayesian shrinkage prior weight (in 'sessions equivalent'). "
                    "0 disables shrinkage (Published == Raw).",
    ),
    min_skill_z: float | None = Query(
        None, ge=-100.0, le=5.0,
        description="Elite skill gate on field-independent Skill_Z (card play + "
                    "par bidding over the qualifying pool). Omitted: 0.0 for "
                    "tournament (drops Thomas-class spikes), 0.7 for club "
                    "(drops Zubatch-class local inflation). Higher = stricter. "
                    "Set <= -90 to disable.",
    ),
    player_name: str | None = Query(None),
    player_number: str | None = Query(None, pattern=r"^\d*$"),
    masterpoints_range: str = Query("All"),
    platinum_events: bool = Query(
        False,
        description=(
            "Tournament only: rank using only events whose mp_color is Platinum. "
            "Excludes every other tournament event."
        ),
    ),
) -> dict:
    with _REPORT_LOCK:
        started_at = datetime.now()
        t0 = time.perf_counter()
        try:
            t_parse_start = time.perf_counter()
            _reject_club_platinum(club_or_tournament, platinum_events)
            if min_skill_z is None:
                min_skill_z = default_min_skill_z(club_or_tournament)
            effective_date_from = date_from or (
                acbl_date_from_for_range(date_range) if date_range else None
            )
            parsed_date_from = (
                None if not effective_date_from else datetime.fromisoformat(effective_date_from)
            )
            t_parse_end = time.perf_counter()

            t_load_start = time.perf_counter()
            source_path = _parquet_source_for(club_or_tournament)
            full_df = _load_full_frame(club_or_tournament)
            t_load_end = time.perf_counter()

            t_filter_start = time.perf_counter()
            con = _get_db_connection()
            input_rows, date_range = _prepare_self_view(
                con, full_df, source_path, parsed_date_from, online_filter, strata,
                platinum_events=platinum_events,
            )
            t_filter_end = time.perf_counter()

            t_sql_start = time.perf_counter()
            try:
                shrinkage_meta = _load_shrinkage_meta(club_or_tournament)
                anchor_kind = "player" if rating_type == "Players" else "pair"
                prior_anchor = _shrinkage_anchor(shrinkage_meta, anchor_kind)
                favorites = _acbl_favorites()
                meta = acbl_favorites_meta(
                    top_n=top_n,
                    min_sessions=min_sessions,
                    rating_method=rating_method,
                    elo_rating_type=elo_rating_type,
                    rating_type=rating_type,
                    prior_anchor=prior_anchor,
                    prior_sessions=prior_sessions,
                    min_skill_z=min_skill_z,
                )
                prompt_ids = button_prompt_ids(favorites, "Leaderboard", meta)
                with _DB_LOCK:
                    result_df, generated_sql = run_favorite(
                        con, favorites, prompt_ids[0], meta
                    )
                    session_ids_df = con.execute(
                        "SELECT DISTINCT CAST(session_id AS VARCHAR) AS session_id FROM self"
                    ).pl()
            finally:
                _teardown_self_view(con)
                _reset_duckdb_connection()
            t_sql_end = time.perf_counter()

            t_xover_start = time.perf_counter()
            result_df = _attach_award_columns(
                result_df,
                club_or_tournament=club_or_tournament,
                rating_type=rating_type,
                session_ids=session_ids_df.get_column("session_id").to_list(),
            )
            result_df = _attach_crossover_columns(
                result_df,
                club_or_tournament=club_or_tournament,
                rating_type=rating_type,
                date_from=parsed_date_from,
                online_filter=online_filter,
                rating_method=rating_method,
                elo_rating_type=elo_rating_type,
                prior_sessions=prior_sessions,
                strata=strata,
            )
            result_df = filter_acbl_leaderboard(
                result_df,
                rating_type=rating_type,
                player_name=player_name,
                player_number=player_number,
                masterpoints_range=masterpoints_range,
            )
            t_xover_end = time.perf_counter()

            t_serialize_start = time.perf_counter()
            result_rows = result_df.to_dicts()
            t_serialize_end = time.perf_counter()

            ended_at = datetime.now()
            elapsed = (ended_at - started_at).total_seconds()
            output_rows = len(result_df)
            perf = {
                "source": "local",
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
                    "default_min_skill_z": default_min_skill_z(club_or_tournament),
                    "applied": min_skill_z > SKILL_GATE_DISABLED,
                    "metric": "pool z-score of DD_Tricks_Diff + Par_Suit + Par_Contract",
                },
                "quality_metric_definitions": QUALITY_METRIC_DEFINITIONS,
                "platinum_events": {
                    "applied": bool(platinum_events),
                    "event_count": (
                        len(platinum_event_ids(_cached_platinum_events()))
                        if platinum_events else 0
                    ),
                    "events": (
                        _cached_platinum_events().to_dicts() if platinum_events else []
                    ),
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
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            print(
                f"[acbl-api] report failed {club_or_tournament}/{rating_type}: "
                f"{exc!r} mem {_cgroup_mem_summary()}",
                flush=True,
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/acbl/favorites")
def acbl_favorites_catalog(favorite_id: str | None = Query(None)) -> dict:
    payload = _acbl_favorites()
    items = flatten_favorites(payload)
    wanted = (favorite_id or "").strip()
    if wanted:
        items = [item for item in items if item["id"] == wanted]
        if not items:
            raise HTTPException(status_code=404, detail=f"Unknown favorite id {wanted!r}")
    return {"organization": "acbl", "count": len(items), "favorites": items}


@app.post("/acbl/sql")
def acbl_sql(
    body: AcblSqlBody,
    club_or_tournament: str = Query(..., pattern="^(club|tournament)$"),
    rating_type: str = Query("Players", pattern="^(Players|Pairs)$"),
    top_n: int = Query(100, ge=1, le=5000),
    min_sessions: int = Query(10, ge=1, le=10000),
    rating_method: str = Query("Latest"),
    elo_rating_type: str = Query("Current Rating (End of Session)"),
    date_from: str | None = Query(None),
    date_range: str | None = Query(None),
    online_filter: str = Query("All"),
    strata: str = Query(STRATA_DEFAULT),
    prior_sessions: int = Query(SHRINKAGE_DEFAULT_PRIOR_SESSIONS, ge=0, le=1000),
    min_skill_z: float | None = Query(None, ge=-100.0, le=5.0),
    platinum_events: bool = Query(False),
) -> dict:
    """Run SELECT/WITH SQL against the filtered board-level DuckDB table ``self``."""
    with _REPORT_LOCK:
        try:
            _reject_club_platinum(club_or_tournament, platinum_events)
            if min_skill_z is None:
                min_skill_z = default_min_skill_z(club_or_tournament)
            effective_date_from = date_from or (
                acbl_date_from_for_range(date_range) if date_range else None
            )
            parsed_date_from = (
                None if not effective_date_from else datetime.fromisoformat(effective_date_from)
            )
            sql = _require_select_sql(body.sql)
            source_path = _parquet_source_for(club_or_tournament)
            full_df = _load_full_frame(club_or_tournament)
            con = _get_db_connection()
            _prepare_self_view(
                con, full_df, source_path, parsed_date_from, online_filter, strata,
                platinum_events=platinum_events,
            )
            try:
                shrinkage_meta = _load_shrinkage_meta(club_or_tournament)
                anchor_kind = "player" if rating_type == "Players" else "pair"
                prior_anchor = _shrinkage_anchor(shrinkage_meta, anchor_kind)
                meta = acbl_favorites_meta(
                    top_n=top_n,
                    min_sessions=min_sessions,
                    rating_method=rating_method,
                    elo_rating_type=elo_rating_type,
                    rating_type=rating_type,
                    prior_anchor=prior_anchor,
                    prior_sessions=prior_sessions,
                    min_skill_z=min_skill_z,
                )
                with _DB_LOCK:
                    result_df, generated_sql = run_sql(con, sql, meta)
            finally:
                _teardown_self_view(con)
                _reset_duckdb_connection()
            if result_df.height > MAX_ACBL_SQL_ROWS:
                result_df = result_df.head(MAX_ACBL_SQL_ROWS)
            return {
                "rows": result_df.to_dicts() if not result_df.is_empty() else [],
                "generated_sql": generated_sql,
                "row_count": result_df.height,
            }
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/acbl/detail")
def acbl_detail(
    club_or_tournament: str = Query(..., pattern="^(club|tournament)$"),
    rating_type: str = Query(..., pattern="^(Players|Pairs)$"),
    elo_rating_type: str = Query("Current Rating (End of Session)"),
    date_from: str | None = Query(None),
    online_filter: str = Query("All"),
    strata: str = Query(STRATA_DEFAULT),
    player_id: str | None = Query(None),
    pair_ids: str | None = Query(None),
    platinum_events: bool = Query(
        False,
        description="Tournament only: restrict session history to platinum-awarding events.",
    ),
) -> dict:
    """Return board detail with the published session Results_URL."""
    with _REPORT_LOCK:
        started_at = datetime.now()
        t0 = time.perf_counter()
        try:
            t_parse_start = time.perf_counter()
            _reject_club_platinum(club_or_tournament, platinum_events)
            parsed_date_from = None if not date_from else datetime.fromisoformat(date_from)
            t_parse_end = time.perf_counter()

            t_load_start = time.perf_counter()
            required_columns = _required_columns_for_detail(rating_type, elo_rating_type)
            if "event_id" in load_elo_ratings_schema_map(club_or_tournament):
                required_columns.append("event_id")
            df = load_elo_ratings(club_or_tournament, columns=required_columns, date_from=parsed_date_from)
            df = _filter_valid_percentages_acbl(df)
            t_load_end = time.perf_counter()

            t_filter_start = time.perf_counter()
            if online_filter == "Local Only" and "is_virtual_game" in df.columns:
                df = df.filter(pl.col("is_virtual_game") == False)
            elif online_filter == "Online Only" and "is_virtual_game" in df.columns:
                df = df.filter(pl.col("is_virtual_game").is_null())
            bucket = strata_label_to_bucket(strata)
            if bucket is not None:
                _require_strata_column(df)
                df = df.filter(pl.col("strata_bucket") == bucket)
            if platinum_events:
                df = _apply_platinum_event_filter(df)
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
            detail = _attach_detail_awards(
                detail,
                club_or_tournament=club_or_tournament,
                rating_type=rating_type,
                player_id=player_id,
                pair_ids=pair_ids,
            )
            if club_or_tournament == "club":
                if "Event_ID" in detail.columns:
                    detail = detail.with_columns(acbl_results_url_expr("club"))
                else:
                    detail = detail.with_columns(
                        pl.lit(None, dtype=pl.Utf8).alias("Results_URL")
                    )
            else:
                detail = detail.with_columns(acbl_results_url_expr("tournament"))
            detail = detail.select(
                *[column for column in detail.columns if column != "Results_URL"],
                "Results_URL",
            )
            t_build_end = time.perf_counter()

            t_serialize_start = time.perf_counter()
            link_status = results_url_status(detail)
            detail_rows = detail.to_dicts()
            t_serialize_end = time.perf_counter()

            ended_at = datetime.now()
            elapsed = (ended_at - started_at).total_seconds()
            input_rows = len(df)
            output_rows = len(detail)
            perf = {
                "source": "local",
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
                "results_url_status": link_status,
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

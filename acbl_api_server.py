from __future__ import annotations

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

# Module-level caches to avoid re-reading parquet files on every request.
# Keys are source paths; values are the cached objects.
_SCHEMA_CACHE: dict[str, dict] = {}
_FRAME_CACHE: dict[str, pl.DataFrame] = {}
_FRAME_CACHE_TIMES: dict[str, float] = {}

import threading as _threading
_DB_LOCK = _threading.Lock()
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
    memory_limit_bytes: int | None = None
    cpu_limit_cores: float | None = None

    # cgroup v2
    mem_max_v2 = _read_text(pathlib.Path("/sys/fs/cgroup/memory.max"))
    if mem_max_v2 and mem_max_v2 != "max":
        try:
            memory_limit_bytes = int(mem_max_v2)
        except ValueError:
            memory_limit_bytes = None

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

    # cgroup v1 fallbacks
    if memory_limit_bytes is None:
        mem_v1 = _read_text(pathlib.Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
        if mem_v1:
            try:
                value = int(mem_v1)
                # Ignore "effectively unlimited" sentinel values.
                if value < (1 << 60):
                    memory_limit_bytes = value
            except ValueError:
                pass

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


def _container_memory_bytes() -> int:
    """Total memory budget: cgroup limit if present, else host RAM."""
    limits = _detect_cgroup_limits()
    if limits["memory_limit_bytes"]:
        return int(limits["memory_limit_bytes"])
    return int(psutil.virtual_memory().total)


def _duckdb_memory_limit_bytes() -> int:
    """Hard cap for DuckDB's buffer manager.

    DuckDB defaults its ``memory_limit`` to ~80% of detected RAM. With the
    resident Polars frame (~13 GB for Club) cached alongside, that default
    overcommits the container and triggers OOM when the user pages through
    filters or flips Club<->Tournament (DuckDB retains buffer-pool memory
    between queries). We instead reserve headroom for the cached frame plus
    per-request/runtime overhead and give DuckDB the remainder, capped at a
    fraction of total so we never approach the limit. Override at deploy time
    with ``DUCKDB_MEMORY_LIMIT_GB`` (e.g. after profiling on a bigger plan).
    """
    override = os.getenv("DUCKDB_MEMORY_LIMIT_GB", "").strip()
    if override:
        try:
            return max(int(0.5 * 1024 ** 3), int(float(override) * 1024 ** 3))
        except ValueError:
            pass
    total = _container_memory_bytes()
    # Reserve for: worst-case resident frame (~13 GB Club) + per-request
    # column copies + Python/Arrow/R2 client overhead.
    reserve = 18 * 1024 ** 3
    budget = min(total - reserve, int(total * 0.45))
    floor = 2 * 1024 ** 3
    return max(floor, budget)


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
       ``R2_BUCKET`` is set. This is the canonical source on Railway
       deployments where parquets also live in R2.
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


def _evict_other_frames(keep_source_path: str) -> None:
    """Drop any cached frames other than the one we're about to use.

    The ACBL UI only ever queries one dataset at a time (Club OR Tournament),
    yet the old code kept both frames resident permanently, doubling the
    cache footprint (~15.7 GB + ~4.4 GB on real data). Evicting the other
    one on dataset switch keeps the live cache to a single frame.
    """
    for old_path in list(_FRAME_CACHE.keys()):
        if old_path == keep_source_path:
            continue
        _FRAME_CACHE.pop(old_path, None)
        _FRAME_CACHE_TIMES.pop(old_path, None)
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


def _malloc_trim() -> None:
    """Prod glibc to return freed pages to the OS.

    On Linux/Railway this releases per-request allocation slack that the
    allocator otherwise keeps in its arenas forever. No-ops on Windows
    (no libc.so.6) and on platforms whose malloc lacks malloc_trim.
    """
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _load_full_frame(club_or_tournament: str) -> pl.DataFrame:
    """Load the full parquet once and cache it at module level.

    On first access for a given dataset (Club or Tournament), evicts any
    other cached frame (single-frame LRU) and down-casts wasteful String /
    Int64 columns to Categorical / Int32 to keep RAM usage bounded. The
    dtype casts are spliced into the lazy plan so the streaming collect
    materializes the already-shrunk frame and peak load memory stays close
    to the post-cast size, not the raw String size.
    """
    source_path, storage_options = _parquet_source_for(club_or_tournament)
    if source_path in _FRAME_CACHE:
        return _FRAME_CACHE[source_path]

    # Single-frame LRU: drop the other dataset before loading the new one so
    # we never hold both ~15 GB Club + ~4 GB Tournament frames at once.
    _evict_other_frames(source_path)

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
    _malloc_trim()
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
    cols = {"Date", "session_id", "Pct_NS", "Round", "Board"}
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
        # Container-aware metrics (preferred for Railway limits)
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
        frames.append(df.filter(pl.col(f"Player_ID_{pos}") == player_id).select(cols_to_select))

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
        side_df = df.filter(
            ((pl.col(id1_col) == player_a) & (pl.col(id2_col) == player_b))
            | ((pl.col(id1_col) == player_b) & (pl.col(id2_col) == player_a))
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


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "acbl-api"}


@app.get("/acbl/report")
def acbl_report(
    club_or_tournament: str = Query(..., pattern="^(club|tournament)$"),
    rating_type: str = Query(..., pattern="^(Players|Pairs)$"),
    top_n: int = Query(100, ge=1, le=1000),
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
    started_at = datetime.now()
    t0 = time.perf_counter()
    try:
        t_parse_start = time.perf_counter()
        parsed_date_from = None if not date_from else datetime.fromisoformat(date_from)
        t_parse_end = time.perf_counter()

        t_load_start = time.perf_counter()
        required_columns = _required_columns_for_mode(rating_type, elo_rating_type)
        df = load_elo_ratings(club_or_tournament, columns=required_columns, date_from=parsed_date_from)
        df = _filter_valid_percentages_acbl(df)
        t_load_end = time.perf_counter()

        t_filter_start = time.perf_counter()
        if online_filter == "Local Only" and "is_virtual_game" in df.columns:
            df = df.filter(pl.col("is_virtual_game") == False)
        elif online_filter == "Online Only" and "is_virtual_game" in df.columns:
            df = df.filter(pl.col("is_virtual_game").is_null())
        t_filter_end = time.perf_counter()

        t_sql_start = time.perf_counter()
        con = _get_db_connection()
        with _DB_LOCK:
            try:
                con.unregister("self")
            except Exception:
                pass
            con.register("self", df)
            shrinkage_meta = _load_shrinkage_meta(club_or_tournament)
            anchor_kind = "player" if rating_type == "Players" else "pair"
            prior_anchor = _shrinkage_anchor(shrinkage_meta, anchor_kind)
            has_round = "Round" in df.columns
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
            result_df = con.execute(generated_sql).pl()
        t_sql_end = time.perf_counter()

        t_serialize_start = time.perf_counter()
        result_rows = result_df.to_dicts()
        t_serialize_end = time.perf_counter()

        date_range = ""
        if "Date" in df.columns and not df.is_empty():
            date_min, date_max = df.select([pl.col("Date").min().alias("min"), pl.col("Date").max().alias("max")]).row(0)
            date_range = f"{str(date_min)[:10]} to {str(date_max)[:10]}"

        ended_at = datetime.now()
        elapsed = (ended_at - started_at).total_seconds()
        input_rows = len(df)
        output_rows = len(result_df)
        perf = {
            "source": "r2" if _r2_enabled() else "local",
            "parse_seconds": round(t_parse_end - t_parse_start, 3),
            "load_seconds": round(t_load_end - t_load_start, 3),
            "filter_seconds": round(t_filter_end - t_filter_start, 3),
            "sql_seconds": round(t_sql_end - t_sql_start, 3),
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
        del result_df, df
        # Per-request memory hygiene: return per-query slack to the OS so
        # process RSS doesn't climb forever as the user pages through
        # different filters. No-op on Windows; real win on Railway/Linux.
        _malloc_trim()
        response_payload["perf"]["total_seconds"] = round(time.perf_counter() - t0, 3)
        return response_payload
    except HTTPException:
        raise
    except Exception as exc:
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
        # Per-request memory hygiene: return per-query slack to the OS.
        _malloc_trim()
        response_payload["perf"]["total_seconds"] = round(time.perf_counter() - t0, 3)
        return response_payload
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

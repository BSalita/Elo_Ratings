from __future__ import annotations

import os
import pathlib
import time
from datetime import datetime, timezone

import duckdb
import polars as pl
import psutil
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


def _load_full_frame(club_or_tournament: str) -> pl.DataFrame:
    """Load the full parquet once and cache it at module level."""
    source_path, storage_options = _parquet_source_for(club_or_tournament)
    if source_path in _FRAME_CACHE:
        return _FRAME_CACHE[source_path]

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

    full_df = lf.collect(engine="streaming")
    _FRAME_CACHE[source_path] = full_df
    _FRAME_CACHE_TIMES[source_path] = time.time()
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
    else:
        pair_ns = elo_cols.get("pair_ns")
        pair_ew = elo_cols.get("pair_ew")
        if pair_ns:
            cols.add(pair_ns)
        if pair_ew:
            cols.add(pair_ew)
        cols.update({"Elo_R_NS_Before", "Elo_R_EW_Before"})
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
    }



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
        side_df = df.filter(
            (pl.min_horizontal(pl.col(id1_col), pl.col(id2_col)) == player_a)
            & (pl.max_horizontal(pl.col(id1_col), pl.col(id2_col)) == player_b)
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
    return detail


def generate_top_players_sql(top_n: int, min_sessions: int, rating_method: str, elo_rating_type: str) -> str:
    rating_agg = {"Avg": "AVG", "Max": "MAX", "Latest": "LAST"}.get(rating_method)
    if rating_agg is None:
        raise ValueError("Invalid rating_method")
    elo_cols = get_elo_column_names(elo_rating_type)
    player_pattern = elo_cols.get("player_pattern")
    if not player_pattern:
        raise ValueError(f"Player ratings not available for {elo_rating_type}")
    union_parts = []
    for pos in "NESW":
        elo_col = player_pattern.format(pos=pos)
        union_parts.append(
            f"""
            SELECT Date, session_id, Player_ID_{pos} AS Player_ID, Player_Name_{pos} AS Player_Name,
                   MasterPoints_{pos} AS MasterPoints, {elo_col} AS Elo_R_Player,
                   Is_Par_Suit, Is_Par_Contract, Is_Sacrifice, DD_Tricks_Diff
            FROM self
            WHERE Player_ID_{pos} IS NOT NULL AND {elo_col} IS NOT NULL AND NOT isnan({elo_col})
            """
        )
    return f"""
    WITH player_positions AS (
      {' UNION ALL '.join(union_parts)}
    ),
    player_aggregates AS (
      SELECT
        Player_ID,
        LAST(Player_Name) AS Player_Name,
        MAX(MasterPoints) AS MasterPoints,
        {rating_agg}(Elo_R_Player) AS Player_Elo_Score,
        COUNT(DISTINCT session_id) AS Sessions_Played,
        AVG(CAST(Is_Par_Suit AS INTEGER)) AS Par_Suit_Rate,
        AVG(CAST(Is_Par_Contract AS INTEGER)) AS Par_Contract_Rate,
        AVG(CAST(Is_Sacrifice AS INTEGER)) AS Sacrifice_Rate,
        AVG(DD_Tricks_Diff) AS DD_Tricks_Diff_Avg
      FROM player_positions
      GROUP BY Player_ID
      HAVING COUNT(DISTINCT session_id) >= {min_sessions}
    )
    SELECT
      CAST(ROW_NUMBER() OVER (ORDER BY CAST(COALESCE(Player_Elo_Score, 0) AS INTEGER) DESC, MasterPoints DESC, Player_ID ASC) AS INTEGER) AS Player_Elo_Rank,
      CAST(COALESCE(Player_Elo_Score, 0) AS INTEGER) AS Player_Elo_Score,
      Player_ID, Player_Name, CAST(MasterPoints AS INTEGER) AS MasterPoints,
      CAST(RANK() OVER (ORDER BY MasterPoints DESC) AS INTEGER) AS MasterPoint_Rank,
      CAST(Sessions_Played AS INTEGER) AS Sessions_Played,
      ROUND(Par_Suit_Rate * 100, 1) AS Par_Suit_Rate_Pct,
      CAST(RANK() OVER (ORDER BY Par_Suit_Rate DESC) AS INTEGER) AS Par_Suit_Rank,
      ROUND(Par_Contract_Rate * 100, 1) AS Par_Contract_Rate_Pct,
      CAST(RANK() OVER (ORDER BY Par_Contract_Rate DESC) AS INTEGER) AS Par_Contract_Rank,
      ROUND(Sacrifice_Rate * 100, 1) AS Sacrifice_Rate_Pct,
      CAST(RANK() OVER (ORDER BY Sacrifice_Rate DESC) AS INTEGER) AS Sacrifice_Rank,
      ROUND(DD_Tricks_Diff_Avg, 2) AS DD_Tricks_Diff_Avg,
      CAST(RANK() OVER (ORDER BY DD_Tricks_Diff_Avg DESC) AS INTEGER) AS DD_Tricks_Diff_Rank
    FROM player_aggregates
    ORDER BY Player_Elo_Score DESC, MasterPoints DESC, Player_ID ASC
    LIMIT {top_n}
    """.strip()


def generate_top_pairs_sql(top_n: int, min_sessions: int, rating_method: str, elo_rating_type: str) -> str:
    rating_agg = {"Avg": "AVG", "Max": "MAX", "Latest": "LAST"}.get(rating_method)
    if rating_agg is None:
        raise ValueError("Invalid rating_method")
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
        Date, session_id,
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
        Date, session_id,
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
        Pair_IDs, LAST(Pair_Names) AS Pair_Names, {rating_agg}(Elo_R_Pair) AS Pair_Elo_Score,
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
    pair_aggregates_with_ranks AS (
      SELECT
        Pair_IDs, Pair_Names, Pair_Elo_Score, Avg_MPs, Geo_MPs, Sessions, Avg_Player_Elo,
        CAST(RANK() OVER (ORDER BY Avg_Player_Elo DESC NULLS LAST) AS INTEGER) AS Avg_Elo_Rank,
        CAST(RANK() OVER (ORDER BY Avg_MPs DESC) AS INTEGER) AS Avg_MPs_Rank,
        CAST(RANK() OVER (ORDER BY Geo_MPs DESC) AS INTEGER) AS Geo_MPs_Rank,
        Par_Suit_Rate, CAST(RANK() OVER (ORDER BY Par_Suit_Rate DESC) AS INTEGER) AS Par_Suit_Rank,
        Par_Contract_Rate, CAST(RANK() OVER (ORDER BY Par_Contract_Rate DESC) AS INTEGER) AS Par_Contract_Rank,
        Sacrifice_Rate, CAST(RANK() OVER (ORDER BY Sacrifice_Rate DESC) AS INTEGER) AS Sacrifice_Rank,
        DD_Tricks_Diff_Avg, CAST(RANK() OVER (ORDER BY DD_Tricks_Diff_Avg DESC) AS INTEGER) AS DD_Tricks_Diff_Rank
      FROM pair_aggregates
    )
    SELECT
      CAST(ROW_NUMBER() OVER (ORDER BY CAST(Pair_Elo_Score AS INTEGER) DESC, Avg_MPs DESC) AS INTEGER) AS Pair_Elo_Rank,
      CAST(Pair_Elo_Score AS INTEGER) AS Pair_Elo_Score,
      Avg_Elo_Rank, Pair_IDs, Pair_Names,
      CAST(Avg_MPs AS INTEGER) AS Avg_MPs, Avg_MPs_Rank, Geo_MPs_Rank, CAST(Sessions AS INTEGER) AS Sessions,
      ROUND(Par_Suit_Rate * 100, 1) AS Par_Suit_Rate_Pct, Par_Suit_Rank,
      ROUND(Par_Contract_Rate * 100, 1) AS Par_Contract_Rate_Pct, Par_Contract_Rank,
      ROUND(Sacrifice_Rate * 100, 1) AS Sacrifice_Rate_Pct, Sacrifice_Rank,
      ROUND(DD_Tricks_Diff_Avg, 2) AS DD_Tricks_Diff_Avg, DD_Tricks_Diff_Rank
    FROM pair_aggregates_with_ranks
    ORDER BY Pair_Elo_Score DESC, Avg_MPs_Rank ASC
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
    rating_method: str = Query("Avg"),
    moving_avg_days: int = Query(10, ge=1, le=3650),
    elo_rating_type: str = Query("Current Rating (End of Session)"),
    date_from: str | None = Query(None),
    online_filter: str = Query("All"),
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
            if rating_type == "Players":
                generated_sql = generate_top_players_sql(top_n, min_sessions, rating_method, elo_rating_type)
            else:
                generated_sql = generate_top_pairs_sql(top_n, min_sessions, rating_method, elo_rating_type)
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
            "perf": perf,
            "server": _server_runtime_info(),
        }
        del result_df
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
        del detail
        response_payload["perf"]["total_seconds"] = round(time.perf_counter() - t0, 3)
        return response_payload
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

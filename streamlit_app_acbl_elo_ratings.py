# streamlit_app_elo_ratings.py

# Previous steps:
# acbl/acbl_elo_ratings_create.py

import os

# Prevent Intel Fortran runtime (libifcoremd.dll / MKL) from installing its own
# Ctrl+C handler that crashes with "forrtl: error (200)".
# Must be set before any numpy/scipy/MKL imports.
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

import logging
import pathlib
import sys
import time
from datetime import datetime, timedelta, timezone

# On Windows, install a process-level console control handler via ctypes
# for immediate clean exit on Ctrl+C. Works from any thread (unlike signal.signal).
if sys.platform == "win32":
    try:
        import ctypes
        _kernel32 = ctypes.windll.kernel32
        _CTRL_C_EVENT = 0
        _CTRL_BREAK_EVENT = 1
        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)
        def _console_ctrl_handler(event):
            if event in (_CTRL_C_EVENT, _CTRL_BREAK_EVENT):
                os._exit(0)
            return False
        _kernel32.SetConsoleCtrlHandler(_console_ctrl_handler, True)
    except (AttributeError, OSError) as exc:
        sys.stderr.write(f"[acbl] console ctrl handler setup skipped: {exc}\n")

import pandas as pd
import polars as pl
import streamlit as st
import duckdb
from streamlit_extras.bottom_container import bottom

try:
    import endplay
    ENDPLAY_VERSION = endplay.__version__
except (ImportError, AttributeError):
    ENDPLAY_VERSION = "N/A"

from streamlitlib.streamlitlib import (
    ShowDataFrameTable,
    create_pdf,
    stick_it_good,
    widen_scrollbars,
)

# Import common Elo utilities (shared with FFBridge app)
from elo_common import (
    ASSISTANT_LOGO_URL,
    apply_app_theme,
    post_process_elo_table,
    render_app_footer,
)

logger = logging.getLogger(__name__)

# -------------------------------
# Masterpoints Range - Single Source of Truth
# -------------------------------
# Define ranges once and derive labels from bounds
MASTERPOINT_RANGES = [
    (0, 5),
    (5, 20),
    (20, 50),
    (50, 100),
    (100, 200),
    (200, 300),
    (300, 500),
    (500, 750),
    (750, 1000),
    (1000, 1500),
    (1500, 2500),
    (2500, 3500),
    (3500, 5000),
    (5000, 7500),
    (7500, 10000),
    (10000, None),
]

def format_masterpoints_label(lower: float | int, upper: float | int | None) -> str:
    """Build a human-readable label from bounds."""
    if upper is None:
        return f"{int(lower)}+"
    return f"{int(lower)}-{int(upper)}"

def get_masterpoints_bounds(range_label: str) -> tuple[float | None, float | None]:
    """Return (lower, upper) bounds for the given range label. 'All' -> (None, None)."""
    if not range_label or range_label == "All":
        return (None, None)
    for lo, hi in MASTERPOINT_RANGES:
        if format_masterpoints_label(lo, hi) == range_label:
            return (lo, hi)
    return (None, None)

def apply_masterpoints_filter_polars(df: pl.DataFrame, range_label: str) -> pl.DataFrame:
    """Filter Polars DataFrame by MasterPoints according to range_label."""
    lower, upper = get_masterpoints_bounds(range_label)
    if ('MasterPoints' not in df.columns) or (lower is None and upper is None):
        return df
    mp_col = pl.col('MasterPoints').cast(pl.Float64)
    if upper is None:
        return df.filter(mp_col >= lower)
    return df.filter((mp_col >= lower) & (mp_col < upper))

 # No pandas fallback: filtering is standardized on Polars only

# -------------------------------
# Config / Paths
# -------------------------------
DATA_ROOT = pathlib.Path(__file__).resolve().parent / "data"


def _r2_enabled() -> bool:
    return bool(os.getenv("R2_BUCKET", "").strip())


def _r2_storage_options() -> dict:
    """Build Polars S3 storage options for Cloudflare R2."""
    bucket = os.getenv("R2_BUCKET", "").strip()
    endpoint = os.getenv("R2_ENDPOINT", "").strip()
    access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
    region = os.getenv("R2_REGION", "auto").strip() or "auto"

    if not bucket:
        raise ValueError("R2_BUCKET is required when using R2 data source.")
    if not endpoint:
        raise ValueError("R2_ENDPOINT is required when using R2 data source.")
    if not access_key:
        raise ValueError("R2_ACCESS_KEY_ID is required when using R2 data source.")
    if not secret_key:
        raise ValueError("R2_SECRET_ACCESS_KEY is required when using R2 data source.")

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


def _table_cache_keys() -> list[str]:
    return [k for k in st.session_state.keys() if isinstance(k, str) and k.startswith("cached_table_")]


def _clear_table_cache() -> None:
    for k in _table_cache_keys():
        del st.session_state[k]


def _prune_table_cache(current_key: str, max_entries: int = 2) -> None:
    """Keep at most max_entries cached table DataFrames in session_state."""
    keys = _table_cache_keys()
    # Keep current key and the newest keys by insertion order (dict order is insertion-ordered).
    keep = [k for k in keys if k == current_key]
    for k in reversed(keys):
        if k == current_key:
            continue
        if len(keep) >= max_entries:
            break
        keep.append(k)
    keep_set = set(keep)
    for k in keys:
        if k not in keep_set:
            del st.session_state[k]

# -------------------------------
# SQL Query Support
# -------------------------------

def get_db_connection():
    """Get or create a session-specific database connection.
    
    This ensures each Streamlit session has its own database connection,
    preventing concurrency issues when multiple users access the app.
    
    Returns:
        duckdb.DuckDBPyConnection: Session-specific database connection
    """
    if 'db_connection' not in st.session_state:
        # Create a new connection for this session
        st.session_state.db_connection = duckdb.connect()
    # Configure DuckDB pragmas on every access (idempotent), so existing sessions pick up settings
    try:
        import tempfile
        tmp_dir = tempfile.gettempdir().replace('\\', '/')
        st.session_state.db_connection.execute(f"PRAGMA temp_directory='{tmp_dir}';")
        # Use a conservative memory limit and enable parallelism sensibly
        st.session_state.db_connection.execute("PRAGMA memory_limit='6GB';")
        threads = max(4, (os.cpu_count() or 4) // 2)
        st.session_state.db_connection.execute(f"PRAGMA threads={threads};")
        # Reduce overhead in certain aggregations
        st.session_state.db_connection.execute("PRAGMA preserve_insertion_order=false;")
    except (duckdb.Error, OSError, RuntimeError) as exc:
        logger.warning("DuckDB PRAGMA setup failed; continuing with defaults: %s", exc)
    return st.session_state.db_connection


def _db_register(con, name: str, df) -> None:
    """Unregister any existing view then register the new frame.

    DuckDB holds a reference to the registered Arrow/Polars buffer, so
    re-registering without unregistering first leaks the old frame.
    """
    try:
        con.unregister(name)
    except Exception:
        pass
    con.register(name, df)


def _get_or_build_report_table_df(
    df: pl.DataFrame,
    generated_sql: str,
    rating_type: str,
    top_n: int,
    min_sessions: int,
    rating_method: str,
    moving_avg_days: int,
    elo_rating_type: str,
    cache_key: str,
    use_sql_engine: bool,
) -> tuple[pl.DataFrame, bool, str]:
    """Build report dataframe once and share between table/PDF flows."""
    engine_name = "SQL" if use_sql_engine else "Polars"
    if cache_key in st.session_state:
        return st.session_state[cache_key], True, engine_name

    if use_sql_engine:
        con = get_db_connection()
        _db_register(con, 'self', df)
        table_df = con.execute(generated_sql).pl()
    else:
        if rating_type == "Players":
            table_df = show_top_players(
                df, int(top_n), int(min_sessions), rating_method, moving_avg_days, elo_rating_type
            )
        elif rating_type == "Pairs":
            table_df = show_top_pairs(
                df, int(top_n), int(min_sessions), rating_method, moving_avg_days, elo_rating_type
            )
        else:
            raise ValueError(f"Invalid rating type: {rating_type}")

    # Post-process: scale Elo to chess range (~2800 top) and add Title column.
    # ACBL uses 1.6x scaling (FFBridge uses 2.0x).
    elo_col = 'Player_Elo_Score' if rating_type == "Players" else 'Pair_Elo_Score'
    table_df = post_process_elo_table(table_df, elo_col, scale_factor=1.6)
    st.session_state[cache_key] = table_df
    _prune_table_cache(current_key=cache_key, max_entries=2)
    return table_df, False, engine_name

def execute_sql_query(df, query, key):
    """Execute SQL query on dataframe using DuckDB."""
    try:
        # Show SQL query if enabled
        if st.session_state.get('show_sql_query', False):
            st.text(f"SQL Query: {query}")

        # If query doesn't contain 'FROM', add 'FROM self' to the beginning
        if 'from ' not in query.lower():
            query = 'FROM self ' + query

        con = get_db_connection()
        result_df = con.execute(query).pl()
        
        if st.session_state.get('show_sql_query', False):
            st.text(f"Result is a dataframe of {len(result_df)} rows.")
        
        # Use the existing ShowDataFrameTable function
        ShowDataFrameTable(result_df, key)
        return result_df
        
    except Exception as e:
        st.error(f"SQL error: {e}")
        st.text(f"Query: {query}")
        return None

def sql_input_callback():
    """Handle SQL query input submission."""
    query = st.session_state.get('sql_query_input', '').strip()
    if query:
        # Get the current dataset
        dataset_type = st.session_state.get('current_dataset_type', 'club')
        if dataset_type in st.session_state.get('all_data', {}):
            df = st.session_state.all_data[dataset_type]
            # Register the dataframe with DuckDB
            con = get_db_connection()
            _db_register(con, 'self', df)
            # Execute the query
            execute_sql_query(df, query, f'sql_query_result_{len(st.session_state.get("sql_queries", []))}')
            # Store query in history
            if 'sql_queries' not in st.session_state:
                st.session_state.sql_queries = []
            st.session_state.sql_queries.append(query)


# -------------------------------
# Data Loading
# -------------------------------
def load_elo_ratings_schema(club_or_tournament: str) -> list[str]:
    source_path, storage_options = _parquet_source_for(club_or_tournament)
    df0 = pl.read_parquet(source_path, n_rows=0, storage_options=storage_options)
    return df0.columns


def load_elo_ratings_schema_map(club_or_tournament: str) -> dict:
    source_path, storage_options = _parquet_source_for(club_or_tournament)
    df0 = pl.read_parquet(source_path, n_rows=0, storage_options=storage_options)
    return df0.schema  # dict[str, pl.DataType]


def load_elo_ratings(
    club_or_tournament: str,
    columns: list[str] | None = None,
    date_from: datetime | None = None,
) -> pl.DataFrame:
    source_path, storage_options = _parquet_source_for(club_or_tournament)

    # Inspect schema once
    schema_map = load_elo_ratings_schema_map(club_or_tournament)

    # Lazy scan for performance
    lf = pl.scan_parquet(source_path, storage_options=storage_options)

    # Reduce columns early if given
    if columns:
        columns = list(dict.fromkeys(columns))
        # Ensure 'Date' is included if filtering
        if date_from is not None and 'Date' in schema_map and 'Date' not in columns:
            columns = ['Date', *columns]
        lf = lf.select([c for c in columns if c in schema_map])

    # Normalize Date column to Datetime for safe comparisons
    if 'Date' in schema_map:
        if schema_map['Date'] == pl.Utf8:
            # Try common formats explicitly; fall back to Date-only
            parsed_dt = pl.coalesce([
                pl.col('Date').str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S%.f', strict=False),
                pl.col('Date').str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S', strict=False),
                pl.col('Date').str.strptime(pl.Datetime, format='%Y-%m-%dT%H:%M:%S%.f', strict=False),
                pl.col('Date').str.strptime(pl.Datetime, format='%Y-%m-%dT%H:%M:%S', strict=False),
                pl.col('Date').str.strptime(pl.Date, format='%Y-%m-%d', strict=False).cast(pl.Datetime, strict=False),
            ])
            lf = lf.with_columns(parsed_dt.alias('Date'))
        else:
            # Cast Date/Datetime to Datetime uniformly (Date -> midnight)
            lf = lf.with_columns(pl.col('Date').cast(pl.Datetime, strict=False).alias('Date'))

    # Optional date filter
    if date_from is not None and 'Date' in schema_map:
        lf = lf.filter(pl.col('Date') >= pl.lit(date_from))

    # Collect with streaming engine for memory efficiency
    df = lf.collect(engine="streaming")
    
    # Clean up Player_Name columns - remove '(swap names)' suffix
    # todo: do this in previous step. maybe in sql_to_board_results_clean?
    #df = df.with_columns(
    #    pl.col('^Player_Name_[NESW]$').str.replace_all(r'\s*\(swap names\)\s*$', '', literal=False)
    #)
    
    return df


# -------------------------------
# Elo Rating Type Mapping
# -------------------------------
def get_elo_column_names(elo_rating_type: str) -> dict:
    """Map Elo rating type to column name patterns."""
    if elo_rating_type == "Current Rating (End of Session)":
        return {
            "player_pattern": "Elo_R_{pos}",  # Elo_R_N, Elo_R_E, etc.
            "pair_ns": "Elo_R_NS",
            "pair_ew": "Elo_R_EW"
        }
    elif elo_rating_type == "Rating at Start of Session":
        return {
            "player_pattern": "Elo_R_{pos}_Before",  # Elo_R_N_Before, etc.
            "pair_ns": "Elo_R_NS_Before", 
            "pair_ew": "Elo_R_EW_Before"
        }
    elif elo_rating_type == "Rating at Event Start":
        return {
            "player_pattern": "Elo_R_{pos}_EventStart",  # Elo_R_N_EventStart, etc.
            "pair_ns": "Elo_R_NS_EventStart",
            "pair_ew": "Elo_R_EW_EventStart"
        }
    elif elo_rating_type == "Rating at Event End":
        return {
            "player_pattern": "Elo_R_{pos}_EventEnd",  # Elo_R_N_EventEnd, etc.
            "pair_ns": "Elo_R_NS_EventEnd",
            "pair_ew": "Elo_R_EW_EventEnd"
        }
    elif elo_rating_type == "Expected Rating":
        return {
            "player_pattern": None,  # No individual player expected ratings
            "pair_ns": "Elo_E_Pair_NS",
            "pair_ew": "Elo_E_Pair_EW"
        }
    else:
        # Default to current rating
        return {
            "player_pattern": "Elo_R_{pos}",
            "pair_ns": "Elo_R_NS",
            "pair_ew": "Elo_R_EW"
        }


def _required_columns_for_mode(rating_type: str, elo_rating_type: str) -> list[str]:
    """Return minimal parquet columns required for the selected report mode."""
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
                cols.add(f"Elo_R_{p}_Before")  # used by detail view for before/after
    else:  # Pairs
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
        cols.update({"Elo_R_NS_Before", "Elo_R_EW_Before"})  # detail before/after

    return sorted(cols)


# -------------------------------
# Computation Helpers
# -------------------------------
def show_top_players(df: pl.DataFrame, top_n: int, min_elo_count: int = 30, rating_method: str = 'Avg', moving_avg_days: int = 10, elo_rating_type: str = "Current Rating (End of Session)") -> pl.DataFrame:
    
    # if 'MasterPoints_N' not in df.columns:
    #     df = df.with_columns([pl.lit(None).alias(f"MasterPoints_{d}") for d in "NESW"])

    # Get the appropriate Elo column names for the selected rating type
    elo_columns = get_elo_column_names(elo_rating_type)
    
    # Check if player ratings are available for this rating type
    if elo_columns["player_pattern"] is None:
        raise ValueError(f"Player ratings not available for '{elo_rating_type}'. Please select a different rating type.")

    position_frames: list[pl.DataFrame] = []
    for d in "NESW":
        # Get the specific Elo column name for this position
        elo_col = elo_columns["player_pattern"].format(pos=d)
        
        # Check if the column exists in the dataframe
        if elo_col not in df.columns:
            continue  # Skip positions where the column doesn't exist
            
        # Only process positions that have valid data
        position_frame = df.select(
            pl.col("Date"),
            pl.col("session_id"),
            pl.col(f"Player_ID_{d}").alias("Player_ID"),
            pl.col(f"Player_Name_{d}").alias("Player_Name"),
            pl.col(f"MasterPoints_{d}").alias("MasterPoints"),
            pl.when(pl.col(elo_col).is_nan()).then(None).otherwise(pl.col(elo_col)).alias("Elo_R_Player"),
            pl.lit(d).alias("Position"),
            pl.col("Is_Par_Suit"),
            pl.col("Is_Par_Contract"),
            pl.col("Is_Sacrifice"),
            pl.col("DD_Tricks_Diff"),
        )
        
        position_frames.append(position_frame)

    # Concat and clean data
    players_stacked = pl.concat(position_frames, how="vertical").drop_nulls(subset=["Player_ID", "Elo_R_Player"])
    
    # Only sort if needed for Latest method (sorting 447M rows takes 90+ seconds!)
    if rating_method == 'Latest':
        # Sort to match SQL ORDER BY behavior exactly - this affects which record is "last"
        players_stacked = players_stacked.sort(['Player_ID', 'Date', 'Position', 'session_id', 'Elo_R_Player'])
    
    # Keep data in memory - don't delete or garbage collect
    del position_frames  # Only delete local references

    if rating_method == 'Avg':
        rating_agg = pl.col('Elo_R_Player').mean().alias('Elo_R_Player')
    elif rating_method == 'Max':
        rating_agg = pl.col('Elo_R_Player').max().alias('Elo_R_Player')
    elif rating_method == 'Latest':
        rating_agg = pl.col('Elo_R_Player').last().alias('Elo_R_Player')
    else:
        raise ValueError(f"Invalid rating method: {rating_method}. Supported methods: Avg, Max, Latest")

    # Memory optimization: Process aggregation in stages to reduce peak memory
    # Stage 1: Group and aggregate core metrics
    player_aggregates = (
        players_stacked
        .group_by('Player_ID')
        .agg([
            rating_agg,
            pl.col('Player_Name').last().alias('Player_Name'),  # Use last (most recent) for consistency
            pl.col('MasterPoints').max().alias('MasterPoints'),  # Use max for most recent/highest value
            pl.col('session_id').n_unique().alias('Elo_Count'),
            pl.col('Position').n_unique().alias('Positions_Played'),
            # Add ability aggregations (average rates)
            pl.col('Is_Par_Suit').mean().alias('Par_Suit_Rate'),
            pl.col('Is_Par_Contract').mean().alias('Par_Contract_Rate'),
            pl.col('Is_Sacrifice').mean().alias('Sacrifice_Rate'),
            pl.col('DD_Tricks_Diff').mean().alias('DD_Tricks_Diff_Avg'),
        ])
        .filter(pl.col('Elo_Count') >= min_elo_count)  # Filter early to reduce data
    )
    
    # Keep data in memory - only delete local reference
    del players_stacked
    
    # Stage 2: Add rankings on full dataset first (to match SQL behavior)
    # SQL ranks on raw MasterPoints values, not CAST values
    player_aggregates_with_ranks = (
        player_aggregates
        .with_columns([
            pl.col('MasterPoints').rank(method='min', descending=True).alias('MasterPoint_Rank'),
            # Add ability rankings (higher rates = better rank)
            pl.col('Par_Suit_Rate').rank(method='min', descending=True).alias('Par_Suit_Rank'),
            pl.col('Par_Contract_Rate').rank(method='min', descending=True).alias('Par_Contract_Rank'),
            pl.col('Sacrifice_Rate').rank(method='min', descending=True).alias('Sacrifice_Rank'),
            pl.col('DD_Tricks_Diff_Avg').rank(method='min', descending=True).alias('DD_Tricks_Diff_Rank'),
        ])
    )
    
    # Stage 3: Sort by CAST(Elo_R_Player) like SQL ROW_NUMBER() OVER clause
    top_players = (
        player_aggregates_with_ranks
        .with_columns([
            pl.col('Elo_R_Player').round(0).cast(pl.Int32, strict=False).alias('Elo_R_Int')
        ])
        .sort(['Elo_R_Int', 'MasterPoints', 'Player_ID'], descending=[True, True, False])  # Match SQL ROW_NUMBER() ORDER BY
        .head(top_n)  # Limit after ranking to preserve correct ranks
        .with_row_index(name='Rank', offset=1)
        .select(['Rank', 'Elo_R_Player', 'Player_ID', 'Player_Name', 'MasterPoints', 'MasterPoint_Rank', 'Elo_Count', 
                 'Par_Suit_Rate', 'Par_Suit_Rank', 'Par_Contract_Rate', 'Par_Contract_Rank', 'Sacrifice_Rate', 'Sacrifice_Rank',
                 'DD_Tricks_Diff_Avg', 'DD_Tricks_Diff_Rank'])
    )
    
    # Keep data in memory - only delete local reference
    del player_aggregates, player_aggregates_with_ranks

    top_players = top_players.with_columns([
        pl.col('Elo_R_Player').round(0).cast(pl.Int32, strict=False),  # Use round() to match SQL behavior
        pl.col('MasterPoints').round(0).cast(pl.Int32, strict=False),  # Apply same rounding to MasterPoints
        pl.col('MasterPoint_Rank').cast(pl.Int32, strict=False),
        pl.col('Elo_Count').cast(pl.Int32, strict=False).alias('Sessions_Played'),
        # Format ability rates as percentages and cast ranks to integers
        (pl.col('Par_Suit_Rate') * 100).round(1).alias('Par_Suit_Rate_Pct'),
        (pl.col('Par_Contract_Rate') * 100).round(1).alias('Par_Contract_Rate_Pct'),
        (pl.col('Sacrifice_Rate') * 100).round(1).alias('Sacrifice_Rate_Pct'),
        pl.col('DD_Tricks_Diff_Avg').round(2).alias('DD_Tricks_Diff_Avg'),
        pl.col('Par_Suit_Rank').cast(pl.Int32, strict=False),
        pl.col('Par_Contract_Rank').cast(pl.Int32, strict=False),
        pl.col('Sacrifice_Rank').cast(pl.Int32, strict=False),
        pl.col('DD_Tricks_Diff_Rank').cast(pl.Int32, strict=False),
    ]).drop(['Elo_Count', 'Par_Suit_Rate', 'Par_Contract_Rate', 'Sacrifice_Rate']).rename({'Elo_R_Player': 'Player_Elo_Score', 'Rank': 'Player_Elo_Rank'})

    # Force garbage collection to free intermediate DataFrames
    import gc
    gc.collect()
    
    return top_players


def show_top_pairs(df: pl.DataFrame, top_n: int, min_elo_count: int = 30, rating_method: str = 'Avg', moving_avg_days: int = 10, elo_rating_type: str = "Current Rating (End of Session)") -> pl.DataFrame:
    
    # if 'MasterPoints_N' not in df.columns:
    #     df = df.with_columns(
    #         pl.lit(None).alias('MasterPoints_N'),
    #         pl.lit(None).alias('MasterPoints_S'),
    #         pl.lit(None).alias('MasterPoints_E'),
    #         pl.lit(None).alias('MasterPoints_W')
    #     )
    
    # Get the appropriate Elo column names for the selected rating type
    elo_columns = get_elo_column_names(elo_rating_type)
    
    # Memory optimization: Pre-filter to only rows with valid pair data
    df_filtered = df.filter(
        (pl.col(elo_columns["pair_ns"]).is_not_null()) | (pl.col(elo_columns["pair_ew"]).is_not_null())
    )

    # Get individual player Elo columns (if available)
    if elo_columns["player_pattern"] is not None:
        player_elo_n = elo_columns["player_pattern"].format(pos="N")
        player_elo_s = elo_columns["player_pattern"].format(pos="S")
        player_elo_e = elo_columns["player_pattern"].format(pos="E")
        player_elo_w = elo_columns["player_pattern"].format(pos="W")
        
        # Check if these columns actually exist in the dataframe
        available_columns = df_filtered.columns
        if (player_elo_n not in available_columns or player_elo_s not in available_columns or 
            player_elo_e not in available_columns or player_elo_w not in available_columns):
            # Individual player columns don't exist, set to None
            player_elo_n = player_elo_s = player_elo_e = player_elo_w = None
    else:
        # For Expected Rating, no individual player ratings available
        player_elo_n = player_elo_s = player_elo_e = player_elo_w = None

    # Process NS partnerships from filtered data
    ns_partnerships = df_filtered.select(
        pl.col("Date"),
        pl.col("session_id"),
        pl.concat_str([
            pl.min_horizontal(pl.col('Player_ID_N'), pl.col('Player_ID_S')),
            pl.max_horizontal(pl.col('Player_ID_N'), pl.col('Player_ID_S')),
        ], separator='-').alias("Pair_IDs"),
        pl.min_horizontal(pl.col('Player_ID_N'), pl.col('Player_ID_S')).alias("Player_ID_A"),
        pl.max_horizontal(pl.col('Player_ID_N'), pl.col('Player_ID_S')).alias("Player_ID_B"),
        pl.when(pl.col(elo_columns["pair_ns"]).is_nan()).then(None).otherwise(pl.col(elo_columns["pair_ns"])).alias("Elo_R_Pair"),
        (
            pl.when(pl.col('Player_ID_N') <= pl.col('Player_ID_S'))
              .then(pl.col('Player_Name_N') + " - " + pl.col('Player_Name_S'))
              .otherwise(pl.col('Player_Name_S') + " - " + pl.col('Player_Name_N'))
        ),#.str.replace_all("(swap names)", "", literal=True).alias("Pair_Names"),
        ((pl.col("MasterPoints_N") + pl.col("MasterPoints_S")) / 2).alias("Avg_MPs"),
        (pl.col("MasterPoints_N") * pl.col("MasterPoints_S")).sqrt().alias("Geo_MPs"),
        # Calculate average player Elo if individual ratings are available
        (pl.when((player_elo_n is not None) & (player_elo_s is not None))
         .then(pl.when((pl.col(player_elo_n).is_not_null()) & (pl.col(player_elo_s).is_not_null()) & 
                       (~pl.col(player_elo_n).is_nan()) & (~pl.col(player_elo_s).is_nan()))
               .then(pl.mean_horizontal([pl.col(player_elo_n), pl.col(player_elo_s)]))
               .otherwise(None))
         .otherwise(None))
         .alias("Avg_Player_Elo_Row"),
        pl.col("DD_Tricks_Diff"),
    )

    # Process EW partnerships from filtered data  
    ew_partnerships = df_filtered.select(
        pl.col("Date"),
        pl.col("session_id"),
        pl.concat_str([
            pl.min_horizontal(pl.col('Player_ID_E'), pl.col('Player_ID_W')),
            pl.max_horizontal(pl.col('Player_ID_E'), pl.col('Player_ID_W')),
        ], separator='-').alias("Pair_IDs"),
        pl.min_horizontal(pl.col('Player_ID_E'), pl.col('Player_ID_W')).alias("Player_ID_A"),
        pl.max_horizontal(pl.col('Player_ID_E'), pl.col('Player_ID_W')).alias("Player_ID_B"),
        pl.when(pl.col(elo_columns["pair_ew"]).is_nan()).then(None).otherwise(pl.col(elo_columns["pair_ew"])).alias("Elo_R_Pair"),
        (
            pl.when(pl.col('Player_ID_E') <= pl.col('Player_ID_W'))
              .then(pl.col('Player_Name_E') + " - " + pl.col('Player_Name_W'))
              .otherwise(pl.col('Player_Name_W') + " - " + pl.col('Player_Name_E'))
        ),#.str.replace_all("(swap names)", "", literal=True).alias("Pair_Names"),
        ((pl.col("MasterPoints_E") + pl.col("MasterPoints_W")) / 2).alias("Avg_MPs"),
        (pl.col("MasterPoints_E") * pl.col("MasterPoints_W")).sqrt().alias("Geo_MPs"),
        # Calculate average player Elo if individual ratings are available
        (pl.when((player_elo_e is not None) & (player_elo_w is not None))
         .then(pl.when((pl.col(player_elo_e).is_not_null()) & (pl.col(player_elo_w).is_not_null()) & 
                       (~pl.col(player_elo_e).is_nan()) & (~pl.col(player_elo_w).is_nan()))
               .then(pl.mean_horizontal([pl.col(player_elo_e), pl.col(player_elo_w)]))
               .otherwise(None))
         .otherwise(None))
         .alias("Avg_Player_Elo_Row"),
        pl.col("DD_Tricks_Diff"),
    )

    # Apply filtering to each partnership DataFrame before concatenation
    ns_partnerships = ns_partnerships.drop_nulls(subset=["Pair_IDs", "Elo_R_Pair"])
    ew_partnerships = ew_partnerships.drop_nulls(subset=["Pair_IDs", "Elo_R_Pair"])
    
    partnerships_stacked = pl.concat([ns_partnerships, ew_partnerships], how="vertical")
    
    # Keep data in memory - only delete local references
    del df_filtered, ns_partnerships, ew_partnerships

    if rating_method == 'Avg':
        rating_agg = pl.col('Elo_R_Pair').mean().alias('Elo_Score')
    elif rating_method == 'Max':
        rating_agg = pl.col('Elo_R_Pair').max().alias('Elo_Score')
    elif rating_method == 'Latest':
        # To match SQL LAST() behavior for pairs
        rating_agg = (
            pl.struct(['Date', 'Elo_R_Pair'])
            .sort_by('Date')
            .last()
            .struct.field('Elo_R_Pair')
            .alias('Elo_Score')
        )
    else:
        raise ValueError(f"Invalid rating method: {rating_method}. Supported methods: Avg, Max, Latest")

    # Memory optimization: Process pairs aggregation in stages
    # Stage 1: Group and aggregate core metrics
    pair_aggregates = (
        partnerships_stacked
        .group_by('Pair_IDs')
        .agg([
            rating_agg,
            pl.col('Pair_Names').last().alias('Pair_Names'),  # Already using last - good
            pl.col('Avg_MPs').mean().alias('Avg_MPs'),
            pl.col('Geo_MPs').mean().alias('Geo_MPs'),
            pl.col('session_id').n_unique().alias('Sessions'),
            pl.col('Player_ID_A').first().alias('Player_ID_A'),
            pl.col('Player_ID_B').first().alias('Player_ID_B'),
            pl.col('Avg_Player_Elo_Row').mean().alias('Avg_Player_Elo'),
            pl.col('DD_Tricks_Diff').mean().alias('DD_Tricks_Diff_Avg'),
        ])
        .filter(pl.col('Sessions') >= min_elo_count)  # Filter early to reduce data
    )
    
    # Keep data in memory - only delete local reference
    del partnerships_stacked
    
    # Stage 2: Add rankings on full dataset first (to match SQL behavior)
    # SQL ranks on raw values, not CAST values
    pair_aggregates_with_ranks = (
        pair_aggregates
        .with_columns([
            pl.col('Avg_Player_Elo').rank(method='min', descending=True).alias('Avg_Elo_Rank'),
            pl.col('Avg_MPs').rank(method='min', descending=True).alias('Avg_MPs_Rank'),
            pl.col('Geo_MPs').rank(method='min', descending=True).alias('Geo_MPs_Rank'),
            pl.col('DD_Tricks_Diff_Avg').rank(method='min', descending=True).alias('DD_Tricks_Diff_Rank'),
        ])
    )
    
    # Stage 3: Sort by CAST(Elo_Score) like SQL ROW_NUMBER() OVER clause
    top_partnerships = (
        pair_aggregates_with_ranks
        .with_columns([
            pl.col('Elo_Score').round(0).cast(pl.Int32, strict=False).alias('Elo_Score_Int')
        ])
        .sort(['Elo_Score_Int', 'Avg_MPs'], descending=[True, True])  # Match SQL ROW_NUMBER() ORDER BY
        .head(top_n)  # Limit after ranking to preserve correct ranks
        .with_row_index(name='Pair_Elo_Rank', offset=1)
        .select(['Pair_Elo_Rank', 'Elo_Score', 'Avg_Elo_Rank', 'Pair_IDs', 'Pair_Names', 'Avg_MPs', 'Avg_MPs_Rank', 'Geo_MPs_Rank', 'Sessions', 'DD_Tricks_Diff_Avg', 'DD_Tricks_Diff_Rank'])
    )
    
    # Keep data in memory - only delete local reference
    del pair_aggregates, pair_aggregates_with_ranks

    top_partnerships = top_partnerships.with_columns([
        pl.col('Elo_Score').round(0).cast(pl.Int32, strict=False),  # Use round() to match SQL behavior
        pl.col('Avg_MPs').round(0).cast(pl.Int32, strict=False),  # Apply same rounding to Avg_MPs
        pl.col('Avg_Elo_Rank').cast(pl.Int32, strict=False),
        pl.col('Geo_MPs_Rank').cast(pl.Int32, strict=False),
        pl.col('Sessions').cast(pl.Int32, strict=False),
        pl.col('DD_Tricks_Diff_Avg').round(2).alias('DD_Tricks_Diff_Avg'),
        pl.col('DD_Tricks_Diff_Rank').cast(pl.Int32, strict=False),
    ]).rename({'Elo_Score': 'Pair_Elo_Score'})

    # Force garbage collection to free intermediate DataFrames
    import gc
    gc.collect()
    
    return top_partnerships


# -------------------------------
# SQL Query Generation
# -------------------------------
def generate_top_players_sql(top_n: int, min_sessions: int, rating_method: str, moving_avg_days: int = 10, elo_rating_type: str = "Current Rating (End of Session)", available_columns: set = None) -> str:
    """Generate SQL query for top players report."""
    
    # Determine aggregation function based on rating method
    if rating_method == 'Avg':
        rating_agg = 'AVG'
    elif rating_method == 'Max':
        rating_agg = 'MAX'
    elif rating_method == 'Latest':
        # Don't use LAST() as it doesn't respect ORDER BY in CTE
        # We'll use a different approach with window functions
        rating_agg = 'LAST'  # Will be replaced below
    else:
        raise ValueError(f"Invalid rating method: {rating_method}. Supported methods: Avg, Max, Latest")
    
    # Get the appropriate Elo column names for the selected rating type
    elo_columns = get_elo_column_names(elo_rating_type)
    
    # Check if player ratings are available for this rating type
    if elo_columns["player_pattern"] is None:
        raise ValueError(f"Player ratings not available for '{elo_rating_type}'. Please select a different rating type.")
    
    # Get individual player Elo column names and check if they exist
    player_elo_n = elo_columns["player_pattern"].format(pos="N")
    player_elo_s = elo_columns["player_pattern"].format(pos="S")
    player_elo_e = elo_columns["player_pattern"].format(pos="E")
    player_elo_w = elo_columns["player_pattern"].format(pos="W")
    
    # If available_columns is provided, check which columns actually exist
    if available_columns is not None:
        existing_positions = []
        for pos, col in [("N", player_elo_n), ("E", player_elo_e), ("S", player_elo_s), ("W", player_elo_w)]:
            if col in available_columns:
                existing_positions.append((pos, col))
        
        if not existing_positions:
            raise ValueError(f"No Elo columns found for rating type '{elo_rating_type}'")
    else:
        # Assume all positions exist (original behavior)
        existing_positions = [("N", player_elo_n), ("E", player_elo_e), ("S", player_elo_s), ("W", player_elo_w)]
    
    # Build UNION ALL clauses only for existing positions
    union_clauses = []
    for pos, elo_col in existing_positions:
        union_clauses.append(f"""
        SELECT Date, session_id, Player_ID_{pos} as Player_ID, Player_Name_{pos} as Player_Name, 
               MasterPoints_{pos} as MasterPoints, {elo_col} as Elo_R_Player, '{pos}' as Position,
               Is_Par_Suit, Is_Par_Contract, Is_Sacrifice, DD_Tricks_Diff
        FROM self WHERE Player_ID_{pos} IS NOT NULL AND {elo_col} IS NOT NULL AND NOT isnan({elo_col})""")
    
    # For Latest method, use simple LAST() aggregation like pairs do
    if rating_method == 'Latest':
        sql_query = f"""
        WITH player_positions AS (
            -- Extract all player positions from NESW columns (only existing ones)
            {' UNION ALL '.join(union_clauses)}
        ),
        player_aggregates AS (
            SELECT 
                Player_ID,
                LAST(Player_Name) as Player_Name,
                MAX(MasterPoints) as MasterPoints,
                LAST(Elo_R_Player) as Player_Elo_Score,
                COUNT(DISTINCT session_id) as Sessions_Played,
                COUNT(DISTINCT Position) as Positions_Played,
                AVG(CAST(Is_Par_Suit AS INTEGER)) as Par_Suit_Rate,
                AVG(CAST(Is_Par_Contract AS INTEGER)) as Par_Contract_Rate,
                AVG(CAST(Is_Sacrifice AS INTEGER)) as Sacrifice_Rate,
                AVG(DD_Tricks_Diff) as DD_Tricks_Diff_Avg
            FROM player_positions
            GROUP BY Player_ID
            HAVING COUNT(DISTINCT session_id) >= {min_sessions}
        )
        SELECT 
            CAST(ROW_NUMBER() OVER (ORDER BY CAST(COALESCE(Player_Elo_Score, 0) AS INTEGER) DESC, MasterPoints DESC, Player_ID ASC) AS INTEGER) as Player_Elo_Rank,
            CAST(COALESCE(Player_Elo_Score, 0) AS INTEGER) as Player_Elo_Score,
            Player_ID,
            Player_Name,
            CAST(MasterPoints AS INTEGER) as MasterPoints,
            CAST(RANK() OVER (ORDER BY MasterPoints DESC) AS INTEGER) as MasterPoint_Rank,
            CAST(Sessions_Played AS INTEGER) as Sessions_Played,
            ROUND(Par_Suit_Rate * 100, 1) as Par_Suit_Rate_Pct,
            CAST(RANK() OVER (ORDER BY Par_Suit_Rate DESC) AS INTEGER) as Par_Suit_Rank,
            ROUND(Par_Contract_Rate * 100, 1) as Par_Contract_Rate_Pct,
            CAST(RANK() OVER (ORDER BY Par_Contract_Rate DESC) AS INTEGER) as Par_Contract_Rank,
            ROUND(Sacrifice_Rate * 100, 1) as Sacrifice_Rate_Pct,
            CAST(RANK() OVER (ORDER BY Sacrifice_Rate DESC) AS INTEGER) as Sacrifice_Rank,
            ROUND(DD_Tricks_Diff_Avg, 2) as DD_Tricks_Diff_Avg,
            CAST(RANK() OVER (ORDER BY DD_Tricks_Diff_Avg DESC) AS INTEGER) as DD_Tricks_Diff_Rank
        FROM player_aggregates
        ORDER BY Player_Elo_Score DESC, MasterPoints DESC, Player_ID ASC
        LIMIT {top_n}
        """
    else:
        # Use simplified approach for Avg/Max methods like Latest method
        sql_query = f"""
        WITH player_positions AS (
            -- Extract all player positions from NESW columns (only existing ones)
            {' UNION ALL '.join(union_clauses)}
        ),
        player_aggregates AS (
            SELECT 
                Player_ID,
                LAST(Player_Name) as Player_Name,
                MAX(MasterPoints) as MasterPoints,
                {f'{rating_agg}(Elo_R_Player)'} as Player_Elo_Score,
                COUNT(DISTINCT session_id) as Sessions_Played,
                COUNT(DISTINCT Position) as Positions_Played,
                AVG(CAST(Is_Par_Suit AS INTEGER)) as Par_Suit_Rate,
                AVG(CAST(Is_Par_Contract AS INTEGER)) as Par_Contract_Rate,
                AVG(CAST(Is_Sacrifice AS INTEGER)) as Sacrifice_Rate,
                AVG(DD_Tricks_Diff) as DD_Tricks_Diff_Avg
            FROM player_positions
            GROUP BY Player_ID
            HAVING COUNT(DISTINCT session_id) >= {min_sessions}
        )
        SELECT 
            CAST(ROW_NUMBER() OVER (ORDER BY CAST(COALESCE(Player_Elo_Score, 0) AS INTEGER) DESC, MasterPoints DESC, Player_ID ASC) AS INTEGER) as Player_Elo_Rank,
            CAST(COALESCE(Player_Elo_Score, 0) AS INTEGER) as Player_Elo_Score,
            Player_ID,
            Player_Name,
            CAST(MasterPoints AS INTEGER) as MasterPoints,
            CAST(RANK() OVER (ORDER BY MasterPoints DESC) AS INTEGER) as MasterPoint_Rank,
            CAST(Sessions_Played AS INTEGER) as Sessions_Played,
            ROUND(Par_Suit_Rate * 100, 1) as Par_Suit_Rate_Pct,
            CAST(RANK() OVER (ORDER BY Par_Suit_Rate DESC) AS INTEGER) as Par_Suit_Rank,
            ROUND(Par_Contract_Rate * 100, 1) as Par_Contract_Rate_Pct,
            CAST(RANK() OVER (ORDER BY Par_Contract_Rate DESC) AS INTEGER) as Par_Contract_Rank,
            ROUND(Sacrifice_Rate * 100, 1) as Sacrifice_Rate_Pct,
            CAST(RANK() OVER (ORDER BY Sacrifice_Rate DESC) AS INTEGER) as Sacrifice_Rank,
            ROUND(DD_Tricks_Diff_Avg, 2) as DD_Tricks_Diff_Avg,
            CAST(RANK() OVER (ORDER BY DD_Tricks_Diff_Avg DESC) AS INTEGER) as DD_Tricks_Diff_Rank
        FROM player_aggregates
        ORDER BY Player_Elo_Score DESC, MasterPoints DESC, Player_ID ASC
        LIMIT {top_n}
        """
    
    return sql_query.strip()


def generate_top_pairs_sql(top_n: int, min_sessions: int, rating_method: str, moving_avg_days: int = 10, elo_rating_type: str = "Current Rating (End of Session)", available_columns: set | None = None) -> str:
    """Generate SQL query for top pairs report."""
    
    # Determine aggregation function based on rating method
    if rating_method == 'Avg':
        rating_agg = 'AVG'
    elif rating_method == 'Max':
        rating_agg = 'MAX'
    elif rating_method == 'Latest':
        rating_agg = 'LAST'
    else:
        raise ValueError(f"Invalid rating method: {rating_method}. Supported methods: Avg, Max, Latest")
    
    # Get the appropriate Elo column names for the selected rating type
    elo_columns = get_elo_column_names(elo_rating_type)
    
    # Determine individual player Elo column names
    if elo_columns["player_pattern"] is not None:
        player_elo_n = elo_columns["player_pattern"].format(pos="N")
        player_elo_s = elo_columns["player_pattern"].format(pos="S")
        player_elo_e = elo_columns["player_pattern"].format(pos="E")
        player_elo_w = elo_columns["player_pattern"].format(pos="W")
    else:
        player_elo_n = player_elo_s = player_elo_e = player_elo_w = None
    
    # Adjust for datasets with alternative pair column naming (e.g., Elo_R_NS vs Elo_R_NS)
    if available_columns is not None:
        def pick_col(candidates: list[str]) -> str | None:
            for c in candidates:
                if c in available_columns:
                    return c
            return None
        if elo_rating_type == "Current Rating (End of Session)":
            pair_ns_col = pick_col(["Elo_R_NS", "Elo_R_NS"])  # prefer standard, fallback pair_
            pair_ew_col = pick_col(["Elo_R_EW", "Elo_R_EW"])  # prefer standard, fallback pair_
        elif elo_rating_type == "Rating at Start of Session":
            pair_ns_col = pick_col(["Elo_R_NS_Before", "Elo_R_NS_Before"])  # if exists
            pair_ew_col = pick_col(["Elo_R_EW_Before", "Elo_R_EW_Before"])  # if exists
        elif elo_rating_type == "Rating at Event Start":
            pair_ns_col = pick_col(["Elo_R_NS_EventStart", "Elo_R_NS_EventStart"])  # prefer explicit pair naming
            pair_ew_col = pick_col(["Elo_R_EW_EventStart", "Elo_R_EW_EventStart"])  # prefer explicit pair naming
        elif elo_rating_type == "Rating at Event End":
            pair_ns_col = pick_col(["Elo_R_NS_EventEnd", "Elo_R_NS_EventEnd"])  # prefer explicit pair naming
            pair_ew_col = pick_col(["Elo_R_EW_EventEnd", "Elo_R_EW_EventEnd"])  # prefer explicit pair naming
        else:
            pair_ns_col = pick_col([elo_columns["pair_ns"]])
            pair_ew_col = pick_col([elo_columns["pair_ew"]])
    else:
        pair_ns_col = elo_columns["pair_ns"]
        pair_ew_col = elo_columns["pair_ew"]
    
    sql_query = f"""
    WITH pair_partnerships AS (
        -- NS partnerships
        SELECT 
            Date, session_id,
            CASE WHEN Player_ID_N < Player_ID_S 
                 THEN Player_ID_N || '-' || Player_ID_S 
                 ELSE Player_ID_S || '-' || Player_ID_N 
            END as Pair_IDs,
            CASE WHEN Player_ID_N <= Player_ID_S
                 THEN Player_Name_N || ' - ' || Player_Name_S
                 ELSE Player_Name_S || ' - ' || Player_Name_N
            END as Pair_Names,
            {('NULL' if available_columns is not None and pair_ns_col is None else pair_ns_col)} as Elo_R_Pair,
            (COALESCE(MasterPoints_N, 0) + COALESCE(MasterPoints_S, 0)) / 2.0 as Avg_MPs,
            SQRT(COALESCE(MasterPoints_N, 0) * COALESCE(MasterPoints_S, 0)) as Geo_MPs,
            {f'''CASE 
                WHEN {player_elo_n} IS NOT NULL AND {player_elo_s} IS NOT NULL AND NOT isnan({player_elo_n}) AND NOT isnan({player_elo_s})
                THEN ({player_elo_n} + {player_elo_s}) / 2.0
                ELSE NULL
            END''' if player_elo_n is not None else 'NULL'} as Avg_Player_Elo,
            Is_Par_Suit,
            Is_Par_Contract,
            Is_Sacrifice,
            DD_Tricks_Diff
        FROM self 
        WHERE {('1=0' if available_columns is not None and pair_ns_col is None else f"{pair_ns_col} IS NOT NULL AND NOT isnan({pair_ns_col})")}
        
        UNION ALL
        
        -- EW partnerships  
        SELECT 
            Date, session_id,
            CASE WHEN Player_ID_E < Player_ID_W 
                 THEN Player_ID_E || '-' || Player_ID_W 
                 ELSE Player_ID_W || '-' || Player_ID_E 
            END as Pair_IDs,
            CASE WHEN Player_ID_E <= Player_ID_W
                 THEN Player_Name_E || ' - ' || Player_Name_W
                 ELSE Player_Name_W || ' - ' || Player_Name_E
            END as Pair_Names,
            {('NULL' if available_columns is not None and pair_ew_col is None else pair_ew_col)} as Elo_R_Pair,
            (COALESCE(MasterPoints_E, 0) + COALESCE(MasterPoints_W, 0)) / 2.0 as Avg_MPs,
            SQRT(COALESCE(MasterPoints_E, 0) * COALESCE(MasterPoints_W, 0)) as Geo_MPs,
            {f'''CASE 
                WHEN {player_elo_e} IS NOT NULL AND {player_elo_w} IS NOT NULL AND NOT isnan({player_elo_e}) AND NOT isnan({player_elo_w})
                THEN ({player_elo_e} + {player_elo_w}) / 2.0
                ELSE NULL
            END''' if player_elo_e is not None else 'NULL'} as Avg_Player_Elo,
            Is_Par_Suit,
            Is_Par_Contract,
            Is_Sacrifice,
            DD_Tricks_Diff
        FROM self 
        WHERE {('1=0' if available_columns is not None and pair_ew_col is None else f"{pair_ew_col} IS NOT NULL AND NOT isnan({pair_ew_col})")}
    ),
    pair_aggregates AS (
        SELECT 
            Pair_IDs,
            LAST(Pair_Names) as Pair_Names,
            {f'{rating_agg}(Elo_R_Pair)'} as Pair_Elo_Score,
            AVG(Avg_MPs) as Avg_MPs,
            AVG(Geo_MPs) as Geo_MPs,
            COUNT(DISTINCT session_id) as Sessions,
            AVG(Avg_Player_Elo) as Avg_Player_Elo,
            AVG(CAST(Is_Par_Suit AS INTEGER)) as Par_Suit_Rate,
            AVG(CAST(Is_Par_Contract AS INTEGER)) as Par_Contract_Rate,
            AVG(CAST(Is_Sacrifice AS INTEGER)) as Sacrifice_Rate,
            AVG(DD_Tricks_Diff) as DD_Tricks_Diff_Avg
        FROM pair_partnerships
        GROUP BY Pair_IDs
        HAVING COUNT(DISTINCT session_id) >= {min_sessions}
    ),
    pair_aggregates_with_ranks AS (
        SELECT 
            Pair_IDs,
            Pair_Names,
            Pair_Elo_Score,
            Avg_MPs,
            Geo_MPs,
            Sessions,
            Avg_Player_Elo,
            CAST(RANK() OVER (ORDER BY Avg_Player_Elo DESC NULLS LAST) AS INTEGER) as Avg_Elo_Rank,
            CAST(RANK() OVER (ORDER BY Avg_MPs DESC) AS INTEGER) as Avg_MPs_Rank,
            CAST(RANK() OVER (ORDER BY Geo_MPs DESC) AS INTEGER) as Geo_MPs_Rank,
            Par_Suit_Rate,
            CAST(RANK() OVER (ORDER BY Par_Suit_Rate DESC) AS INTEGER) as Par_Suit_Rank,
            Par_Contract_Rate,
            CAST(RANK() OVER (ORDER BY Par_Contract_Rate DESC) AS INTEGER) as Par_Contract_Rank,
            Sacrifice_Rate,
            CAST(RANK() OVER (ORDER BY Sacrifice_Rate DESC) AS INTEGER) as Sacrifice_Rank,
            DD_Tricks_Diff_Avg,
            CAST(RANK() OVER (ORDER BY DD_Tricks_Diff_Avg DESC) AS INTEGER) as DD_Tricks_Diff_Rank
        FROM pair_aggregates
    )
    SELECT 
        CAST(ROW_NUMBER() OVER (ORDER BY CAST(Pair_Elo_Score AS INTEGER) DESC, Avg_MPs DESC) AS INTEGER) as Pair_Elo_Rank,
        CAST(Pair_Elo_Score AS INTEGER) as Pair_Elo_Score,
        Avg_Elo_Rank,
        Pair_IDs,
        Pair_Names,
        CAST(Avg_MPs AS INTEGER) as Avg_MPs,
        Avg_MPs_Rank,
        Geo_MPs_Rank,
        CAST(Sessions AS INTEGER) as Sessions,
        ROUND(Par_Suit_Rate * 100, 1) as Par_Suit_Rate_Pct,
        Par_Suit_Rank,
        ROUND(Par_Contract_Rate * 100, 1) as Par_Contract_Rate_Pct,
        Par_Contract_Rank,
        ROUND(Sacrifice_Rate * 100, 1) as Sacrifice_Rate_Pct,
        Sacrifice_Rank,
        ROUND(DD_Tricks_Diff_Avg, 2) as DD_Tricks_Diff_Avg,
        DD_Tricks_Diff_Rank
    FROM pair_aggregates_with_ranks
    ORDER BY Pair_Elo_Score DESC, Avg_MPs_Rank ASC
    LIMIT {top_n}
    """
    
    return sql_query.strip()


# -------------------------------
# Result Comparison Functions
# -------------------------------
def compare_polars_sql_results(polars_df: pl.DataFrame, sql_df: pl.DataFrame, comparison_type: str) -> dict:
    """Compare Polars and SQL results to ensure consistency.
    
    Args:
        polars_df: Result from Polars implementation
        sql_df: Result from SQL implementation  
        comparison_type: Type of comparison ("players" or "pairs")
        
    Returns:
        dict: Comparison results with status and details
    """
    try:
        # Convert both to pandas for easier comparison
        polars_pd = polars_df.to_pandas() if hasattr(polars_df, 'to_pandas') else polars_df
        sql_pd = sql_df.to_pandas() if hasattr(sql_df, 'to_pandas') else sql_df
        
        # Basic shape comparison
        shape_match = polars_pd.shape == sql_pd.shape
        
        # Column comparison
        polars_cols = set(polars_pd.columns)
        sql_cols = set(sql_pd.columns)
        columns_match = polars_cols == sql_cols
        
        # Sort both dataframes by the primary ranking column for comparison
        if comparison_type == "players":
            sort_col = "Player_Elo_Rank" if "Player_Elo_Rank" in polars_cols else polars_pd.columns[0]
        else:  # pairs
            sort_col = "Pair_Elo_Rank" if "Pair_Elo_Rank" in polars_cols else polars_pd.columns[0]
            
        polars_sorted = polars_pd.sort_values(sort_col).reset_index(drop=True)
        sql_sorted = sql_pd.sort_values(sort_col).reset_index(drop=True)
        
        # Detailed comparison
        differences = []
        name_mismatches = []
        numeric_tolerance = 1e-6  # Tolerance for floating point comparisons
        
        if shape_match and columns_match:
            # Determine relaxed/strict mode from session state (default relaxed)
            try:
                strict_comparison = bool(st.session_state.get('strict_comparison', False))
            except Exception:
                strict_comparison = False
            # ID/name mismatch detection
            try:
                id_cols = []
                name_cols = []
                if comparison_type == "players":
                    id_cols = [c for c in ["Player_ID", "Player_ID_A", "Player_ID_B"] if c in polars_cols and c in sql_cols]
                    name_cols = [c for c in ["Player_Name", "Pair_Names"] if c in polars_cols and c in sql_cols]
                else:
                    id_cols = [c for c in ["Pair_IDs", "Player_ID_A", "Player_ID_B"] if c in polars_cols and c in sql_cols]
                    name_cols = [c for c in ["Pair_Names", "Player_Name"] if c in polars_cols and c in sql_cols]
                # Choose a stable join key preference order
                join_key = None
                for c in ["Player_ID", "Pair_IDs", "Player_ID_A", "Player_ID_B"]:
                    if c in id_cols:
                        join_key = c
                        break
                if join_key is not None and name_cols:
                    l = polars_sorted[[join_key] + name_cols].copy()
                    r = sql_sorted[[join_key] + name_cols].copy()
                    l.columns = [join_key] + [f"polars__{c}" for c in name_cols]
                    r.columns = [join_key] + [f"sql__{c}" for c in name_cols]
                    merged = l.merge(r, on=join_key, how="inner")
                    for c in name_cols:
                        pol = f"polars__{c}"
                        sq = f"sql__{c}"
                        mismatch_mask = merged[pol] != merged[sq]
                        if mismatch_mask.any():
                            sample = merged.loc[mismatch_mask, [join_key, pol, sq]].head(5)
                            name_mismatches.append({
                                'column': c,
                                'type': 'name_mismatch',
                                'ids': sample[join_key].tolist(),
                                'polars_values': sample[pol].tolist(),
                                'sql_values': sample[sq].tolist(),
                            })
            except Exception:
                pass
            # Compare each column
            for col in polars_sorted.columns:
                # In relaxed mode, ignore name columns (IDs are source of truth)
                if not strict_comparison and col in ["Player_Name", "Pair_Names"]:
                    continue
                if col in sql_sorted.columns:
                    polars_col = polars_sorted[col]
                    sql_col = sql_sorted[col]
                    
                    # Handle numeric columns with tolerance
                    if polars_col.dtype in ['int64', 'float64'] and sql_col.dtype in ['int64', 'float64']:
                        # Convert to numeric, handling NaN values
                        polars_numeric = pd.to_numeric(polars_col, errors='coerce')
                        sql_numeric = pd.to_numeric(sql_col, errors='coerce')
                        
                        # Compare with tolerance
                        if not polars_numeric.equals(sql_numeric):
                            # Check if differences are within tolerance
                            diff_mask = ~(
                                (polars_numeric.isna() & sql_numeric.isna()) |
                                (abs(polars_numeric - sql_numeric) <= numeric_tolerance)
                            )
                            
                            if diff_mask.any():
                                diff_indices = diff_mask[diff_mask].index.tolist()
                                differences.append({
                                    'column': col,
                                    'type': 'numeric_difference',
                                    'indices': diff_indices[:5],  # Show first 5 differences
                                    'polars_values': polars_numeric[diff_indices[:5]].tolist(),
                                    'sql_values': sql_numeric[diff_indices[:5]].tolist()
                                })
                    else:
                        # String/categorical comparison
                        if not polars_col.equals(sql_col):
                            diff_mask = polars_col != sql_col
                            if diff_mask.any():
                                diff_indices = diff_mask[diff_mask].index.tolist()
                                differences.append({
                                    'column': col,
                                    'type': 'value_difference',
                                    'indices': diff_indices[:5],  # Show first 5 differences
                                    'polars_values': polars_col[diff_indices[:5]].tolist(),
                                    'sql_values': sql_col[diff_indices[:5]].tolist()
                                })
        
        # Overall assessment
        is_identical = shape_match and columns_match and len(differences) == 0
        
        return {
            'identical': is_identical,
            'shape_match': shape_match,
            'columns_match': columns_match,
            'polars_shape': polars_pd.shape,
            'sql_shape': sql_pd.shape,
            'polars_columns': sorted(polars_cols),
            'sql_columns': sorted(sql_cols),
            'differences': differences,
            'name_mismatches': name_mismatches,
            'summary': f"{'✅ IDENTICAL' if is_identical else '❌ DIFFERENCES FOUND'}"
        }
        
    except Exception as e:
        return {
            'identical': False,
            'error': str(e),
            'summary': f"❌ COMPARISON FAILED: {str(e)}"
        }


def display_comparison_results(comparison_result: dict, comparison_type: str):
    """Display comparison results in Streamlit UI."""
    
    if comparison_result['identical']:
        st.success(f"✅ **Polars and SQL implementations produce IDENTICAL results!**")
        st.info(f"📊 Both returned {comparison_result['polars_shape'][0]} rows × {comparison_result['polars_shape'][1]} columns")
    else:
        st.error(f"❌ **Differences found between Polars and SQL implementations!**")
        
        # Show shape differences
        if not comparison_result['shape_match']:
            st.error(f"**Shape mismatch:**")
            st.write(f"- Polars: {comparison_result['polars_shape']}")
            st.write(f"- SQL: {comparison_result['sql_shape']}")
        
        # Show column differences  
        if not comparison_result['columns_match']:
            st.error(f"**Column mismatch:**")
            polars_only = set(comparison_result['polars_columns']) - set(comparison_result['sql_columns'])
            sql_only = set(comparison_result['sql_columns']) - set(comparison_result['polars_columns'])
            if polars_only:
                st.write(f"- Polars only: {sorted(polars_only)}")
            if sql_only:
                st.write(f"- SQL only: {sorted(sql_only)}")
        
        # Show value differences
        if comparison_result.get('differences'):
            st.error(f"**Value differences found in {len(comparison_result['differences'])} columns:**")
            for diff in comparison_result['differences'][:3]:  # Show first 3 column differences
                with st.expander(f"Column: {diff['column']} ({diff['type']})"):
                    st.write(f"**Indices with differences:** {diff['indices']}")
                    st.write(f"**Polars values:** {diff['polars_values']}")
                    st.write(f"**SQL values:** {diff['sql_values']}")

        # Show name mismatches (IDs match but names differ) - informational only
        if comparison_result.get('name_mismatches'):
            st.warning(f"**Name mismatches detected (IDs match, names differ): {len(comparison_result['name_mismatches'])} columns**")
            for nm in comparison_result['name_mismatches'][:2]:  # show up to 2 columns
                with st.expander(f"Name column: {nm.get('column','')} (sample)"):
                    ids = nm.get('ids', [])
                    pol = nm.get('polars_values', [])
                    sql = nm.get('sql_values', [])
                    rows = min(5, len(ids))
                    for i in range(rows):
                        st.write(f"ID: {ids[i]} | Polars: {pol[i]} | SQL: {sql[i]}")
        
        # Show error if comparison failed
        if 'error' in comparison_result:
            st.error(f"**Comparison error:** {comparison_result['error']}")


def debug_single_comparison(df: pl.DataFrame, top_n: int = 10, min_sessions: int = 5, rating_method: str = 'Avg', elo_rating_type: str = "Current Rating (End of Session)") -> dict:
    """Debug a single comparison to understand differences."""
    
    # Run Polars implementation
    polars_df = show_top_players(df, top_n, min_sessions, rating_method, 10, elo_rating_type)
    
    # Run SQL implementation
    con = get_db_connection()
    _db_register(con, 'self', df)
    sql_query = generate_top_players_sql(top_n, min_sessions, rating_method, 10, elo_rating_type, set(df.columns))
    sql_df = con.execute(sql_query).pl()
    
    # Compare intermediate steps - let's look at the raw aggregated data before ranking
    # Get the player_aggregates step from Polars
    elo_columns = get_elo_column_names(elo_rating_type)
    position_frames = []
    for d in "NESW":
        elo_col = elo_columns["player_pattern"].format(pos=d)
        if elo_col not in df.columns:
            continue
        position_frame = df.select(
            pl.col("Date"),
            pl.col("session_id"),
            pl.col(f"Player_ID_{d}").alias("Player_ID"),
            pl.col(f"Player_Name_{d}").alias("Player_Name"),
            pl.col(f"MasterPoints_{d}").alias("MasterPoints"),
            pl.when(pl.col(elo_col).is_nan()).then(None).otherwise(pl.col(elo_col)).alias("Elo_R_Player"),
            pl.lit(d).alias("Position"),
            pl.col("Is_Par_Suit"),
            pl.col("Is_Par_Contract"),
            pl.col("Is_Sacrifice"),
            pl.col("DD_Tricks_Diff"),
        )
        position_frames.append(position_frame)
    
    players_stacked = pl.concat(position_frames, how="vertical").drop_nulls(subset=["Player_ID", "Elo_R_Player"]).sort(['Player_ID', 'Date', 'Position'])
    
    if rating_method == 'Avg':
        rating_agg = pl.col('Elo_R_Player').mean().alias('Elo_R_Player')
    else:
        rating_agg = pl.col('Elo_R_Player').mean().alias('Elo_R_Player')  # Default to avg for debug
    
    polars_aggregates = (
        players_stacked
        .group_by('Player_ID')
        .agg([
            rating_agg,
            pl.col('Player_Name').first().alias('Player_Name'),
            pl.col('MasterPoints').max().alias('MasterPoints'),
            pl.col('session_id').n_unique().alias('Elo_Count'),
        ])
        .filter(pl.col('Elo_Count') >= min_sessions)
        .sort('Player_ID')
    )
    
    return {
        'polars_final': polars_df,
        'sql_final': sql_df,
        'polars_aggregates': polars_aggregates,
        'sql_query': sql_query,
        'comparison': compare_polars_sql_results(polars_df, sql_df, "players")
    }


def run_comprehensive_comparison(all_data: dict, top_n: int = 100, min_sessions: int = 10, datasets_filter: list[str] | None = None) -> dict:
    """Run comprehensive comparison across all option variations.
    
    Args:
        all_data: Dictionary with 'club' and 'tournament' dataframes
        top_n: Number of top results to compare (smaller for faster testing)
        min_sessions: Minimum sessions for comparison
        
    Returns:
        dict: Comprehensive comparison results
    """
    
    # Define all variations to test
    datasets = ['club', 'tournament']
    if datasets_filter:
        datasets = [d for d in datasets if d in datasets_filter]
    rating_types = ['Players', 'Pairs']
    rating_methods = ['Avg', 'Max', 'Latest']  # Moving Avg disabled due to memory issues
    elo_rating_types = [
        "Current Rating (End of Session)",
        "Rating at Start of Session",
        "Rating at Event Start", 
        "Rating at Event End"
        # Skip "Expected Rating" as it doesn't have individual player ratings
    ]
    
    results = {
        'total_tests': 0,
        'passed_tests': 0,
        'failed_tests': 0,
        'error_tests': 0,
        'test_results': [],
        'summary': {}
    }
    
    total_combinations = len(datasets) * len(rating_types) * len(rating_methods) * len(elo_rating_types)
    
    # Create progress container
    progress_container = st.container()
    with progress_container:
        st.info(f"🔄 **Running comprehensive comparison across {total_combinations} combinations...**")
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    test_count = 0
    
    for dataset in datasets:
        df = all_data[dataset]
        
        for rating_type in rating_types:
            for rating_method in rating_methods:
                for elo_rating_type in elo_rating_types:
                    test_count += 1
                    results['total_tests'] += 1
                    
                    # Update progress
                    progress = test_count / total_combinations
                    progress_bar.progress(progress)
                    status_text.text(f"Testing: {dataset} | {rating_type} | {rating_method} | {elo_rating_type[:20]}...")
                    
                    test_name = f"{dataset}_{rating_type}_{rating_method}_{elo_rating_type.replace(' ', '_')}"
                    
                    try:
                        import time
                        
                        # Apply fast comparison filter for club dataset if enabled
                        df_use = df
                        try:
                            if st.session_state.get('fast_comparison', True) and dataset == 'club' and 'Date' in df.columns:
                                days = int(st.session_state.get('fast_comparison_days', 180))
                                cutoff = datetime.now() - timedelta(days=days)
                                df_use = df.filter(pl.col('Date') >= pl.lit(cutoff))
                        except Exception:
                            df_use = df
                        
                        # Apply online filter if it exists in session state
                        online_filter = st.session_state.get('online_filter', 'All')
                        if online_filter == "Local Only":
                            if "is_virtual_game" in df_use.columns:
                                df_use = df_use.filter(pl.col("is_virtual_game") == False)
                        elif online_filter == "Online Only":
                            if "is_virtual_game" in df_use.columns:
                                df_use = df_use.filter(pl.col("is_virtual_game").is_null())
                        
                        # Run Polars implementation with timing
                        polars_start = time.time()
                        if rating_type == "Players":
                            polars_df = show_top_players(df_use, top_n, min_sessions, rating_method, 10, elo_rating_type)
                        else:  # Pairs
                            polars_df = show_top_pairs(df_use, top_n, min_sessions, rating_method, 10, elo_rating_type)
                        polars_time = time.time() - polars_start
                        
                        # Run SQL implementation with timing
                        sql_start = time.time()
                        con = get_db_connection()
                        _db_register(con, 'self', df_use)
                        # Ensure PRAGMAs are applied for potentially large club queries
                        try:
                            con.execute("PRAGMA preserve_insertion_order=false;")
                        except Exception:
                            pass
                        if rating_type == "Players":
                            sql_query = generate_top_players_sql(top_n, min_sessions, rating_method, 10, elo_rating_type, set(df_use.columns))
                        else:  # Pairs
                            sql_query = generate_top_pairs_sql(top_n, min_sessions, rating_method, 10, elo_rating_type, set(df_use.columns))
                        
                        sql_df = con.execute(sql_query).pl()
                        sql_time = time.time() - sql_start
                        
                        # Compare results
                        comparison = compare_polars_sql_results(polars_df, sql_df, rating_type.lower())
                        
                        # Determine status (relaxed by default):
                        # - strict: PASS only if identical (0 diffs and columns/shapes match)
                        # - relaxed: PASS if 0 diffs; otherwise mark DIFF (not FAIL)
                        try:
                            strict_mode = bool(st.session_state.get('strict_comparison', False))
                        except Exception:
                            strict_mode = False
                        diffs_count = len(comparison.get('differences', []))
                        if strict_mode:
                            status_value = 'PASS' if comparison.get('identical') else 'FAIL'
                        else:
                            status_value = 'PASS' if diffs_count == 0 else 'DIFF'

                        # Store result with timing data
                        test_result = {
                            'test_name': test_name,
                            'dataset': dataset,
                            'rating_type': rating_type,
                            'rating_method': rating_method,
                            'elo_rating_type': elo_rating_type,
                            'status': status_value,
                            'identical': comparison['identical'],
                            'polars_rows': comparison['polars_shape'][0] if 'polars_shape' in comparison else 0,
                            'sql_rows': comparison['sql_shape'][0] if 'sql_shape' in comparison else 0,
                            'polars_time_ms': round(polars_time * 1000, 1),
                            'sql_time_ms': round(sql_time * 1000, 1),
                            'time_diff_ms': round((sql_time - polars_time) * 1000, 1),
                            'speed_ratio': round(sql_time / polars_time, 2) if polars_time > 0 else 0,
                            'differences': len(comparison.get('differences', [])),
                            'error': comparison.get('error', None)
                        }
                        
                        results['test_results'].append(test_result)
                        
                        if status_value == 'PASS':
                            results['passed_tests'] += 1
                        elif status_value == 'DIFF':
                            results['failed_tests'] += 0  # do not count DIFF as failed
                        else:  # FAIL
                            results['failed_tests'] += 1
                            
                    except Exception as e:
                        results['error_tests'] += 1
                        test_result = {
                            'test_name': test_name,
                            'dataset': dataset,
                            'rating_type': rating_type,
                            'rating_method': rating_method,
                            'elo_rating_type': elo_rating_type,
                            'status': 'ERROR',
                            'identical': False,
                            'polars_rows': 0,
                            'sql_rows': 0,
                            'polars_time_ms': 0,
                            'sql_time_ms': 0,
                            'time_diff_ms': 0,
                            'speed_ratio': 0,
                            'differences': 0,
                            'error': str(e)
                        }
                        results['test_results'].append(test_result)
    
    # Update final progress
    progress_bar.progress(1.0)
    status_text.text("Comprehensive comparison completed.")
    
    # Generate summary
    results['summary'] = {
        'pass_rate': (results['passed_tests'] / results['total_tests']) * 100 if results['total_tests'] > 0 else 0,
        'datasets_tested': len(datasets),
        'variations_tested': len(rating_types) * len(rating_methods) * len(elo_rating_types)
    }
    
    return results


def display_comprehensive_results(results: dict):
    """Display comprehensive comparison results."""
    
    # Overall summary
    pass_rate = results['summary']['pass_rate']
    if pass_rate == 100:
        st.success(f"🎉 **ALL TESTS PASSED!** ({results['passed_tests']}/{results['total_tests']})")
        st.balloons()
    elif pass_rate >= 90:
        st.warning(f"⚠️ **Most tests passed** ({results['passed_tests']}/{results['total_tests']}) - {pass_rate:.1f}% success rate")
    else:
        st.error(f"❌ **Multiple failures detected** ({results['passed_tests']}/{results['total_tests']}) - {pass_rate:.1f}% success rate")
    
    # Detailed breakdown
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("✅ Passed", results['passed_tests'])
    with col2:
        st.metric("❌ Failed", results['failed_tests'])
    with col3:
        st.metric("🚫 Errors", results['error_tests'])
    with col4:
        st.metric("📊 Total", results['total_tests'])
    
    # Convert results to DataFrame for detailed analysis
    if results['test_results']:
        import pandas as pd
        
        # Create DataFrame from test results
        df_results = pd.DataFrame(results['test_results'])
        
        # Reorder and select columns for display
        display_columns = [
            'dataset', 'rating_type', 'rating_method', 'elo_rating_type', 'status',
            'polars_rows', 'sql_rows', 'polars_time_ms', 'sql_time_ms', 
            'time_diff_ms', 'speed_ratio', 'differences', 'error'
        ]
        
        # Filter to only existing columns
        available_columns = [col for col in display_columns if col in df_results.columns]
        df_display = df_results[available_columns].copy()
        
        # Format columns for better display
        if 'elo_rating_type' in df_display.columns:
            df_display['elo_rating_type'] = df_display['elo_rating_type'].str.replace('Rating (End of Session)', 'Current', regex=False)
            df_display['elo_rating_type'] = df_display['elo_rating_type'].str.replace('Rating at ', '', regex=False)
            df_display['elo_rating_type'] = df_display['elo_rating_type'].str.replace('Start of Session', 'Session Start', regex=False)
        
        # Rename columns for better readability
        column_renames = {
            'dataset': 'Dataset',
            'rating_type': 'Type',
            'rating_method': 'Method',
            'elo_rating_type': 'Elo Type',
            'status': 'Status',
            'polars_rows': 'Polars Rows',
            'sql_rows': 'SQL Rows',
            'polars_time_ms': 'Polars (ms)',
            'sql_time_ms': 'SQL (ms)',
            'time_diff_ms': 'Diff (ms)',
            'speed_ratio': 'SQL/Polars Ratio',
            'differences': 'Diffs',
            'error': 'Error'
        }
        
        df_display = df_display.rename(columns=column_renames)
        
        # Display the detailed results table
        st.markdown("### 📊 **Detailed Test Results**")
        
        # Add filtering options
        col1, col2, col3 = st.columns(3)
        with col1:
            status_filter = st.selectbox("Filter by Status", ["All", "PASS", "FAIL", "ERROR"])
        with col2:
            dataset_filter = st.selectbox("Filter by Dataset", ["All", "club", "tournament"])
        with col3:
            type_filter = st.selectbox("Filter by Type", ["All", "Players", "Pairs"])
        
        # Apply filters
        filtered_df = df_display.copy()
        if status_filter != "All":
            filtered_df = filtered_df[filtered_df['Status'] == status_filter]
        if dataset_filter != "All":
            filtered_df = filtered_df[filtered_df['Dataset'] == dataset_filter]
        if type_filter != "All":
            filtered_df = filtered_df[filtered_df['Type'] == type_filter]
        
        # Display the filtered dataframe
        st.dataframe(
            filtered_df,
            width='stretch',
            height=400,
            column_config={
                "Status": st.column_config.TextColumn(
                    width="small",
                ),
                "Polars (ms)": st.column_config.NumberColumn(
                    format="%.1f",
                    width="small"
                ),
                "SQL (ms)": st.column_config.NumberColumn(
                    format="%.1f", 
                    width="small"
                ),
                "Diff (ms)": st.column_config.NumberColumn(
                    format="%.1f",
                    width="small"
                ),
                "SQL/Polars Ratio": st.column_config.NumberColumn(
                    format="%.2f",
                    width="small"
                ),
                "Error": st.column_config.TextColumn(
                    width="large"
                )
            }
        )
        
        # Performance summary
        if len(filtered_df) > 0:
            st.markdown("### ⚡ **Performance Summary**")
            
            # Calculate performance statistics
            passed_df = filtered_df[filtered_df['Status'] == 'PASS']
            if len(passed_df) > 0:
                avg_polars_time = passed_df['Polars (ms)'].mean()
                avg_sql_time = passed_df['SQL (ms)'].mean()
                avg_speed_ratio = passed_df['SQL/Polars Ratio'].mean()
                
                perf_col1, perf_col2, perf_col3 = st.columns(3)
                with perf_col1:
                    st.metric("Avg Polars Time", f"{avg_polars_time:.1f} ms")
                with perf_col2:
                    st.metric("Avg SQL Time", f"{avg_sql_time:.1f} ms")
                with perf_col3:
                    st.metric("Avg Speed Ratio", f"{avg_speed_ratio:.2f}x")
                
                # Show which is faster overall
                if avg_speed_ratio < 1:
                    st.success(f"🚀 **SQL is {1/avg_speed_ratio:.2f}x faster** than Polars on average")
                elif avg_speed_ratio > 1:
                    st.info(f"🚀 **Polars is {avg_speed_ratio:.2f}x faster** than SQL on average")
                else:
                    st.info("⚖️ **Performance is roughly equivalent**")
    
    # Show error details for failed tests
    if results['failed_tests'] > 0 or results['error_tests'] > 0:
        with st.expander("🔍 **Error Details**", expanded=False):
            failed_tests = [t for t in results['test_results'] if not t['identical']]
            
            for test in failed_tests[:5]:  # Show first 5 failures
                st.markdown(f"**❌ {test['test_name']}**")
                if test.get('error'):
                    st.error(f"Error: {test['error']}")
                else:
                    st.write(f"Differences in {test.get('differences', 0)} columns")
                st.markdown("---")


# -------------------------------
# First-time Initialization
# -------------------------------
def initialize_session_state():
    """Initialize session state variables on first run."""
    if 'first_time' not in st.session_state:
        st.session_state.first_time = True
        
        # First-time only logic
        st.session_state.app_datetime = datetime.fromtimestamp(
            pathlib.Path(__file__).stat().st_mtime, 
            tz=timezone.utc
        ).strftime('%Y-%m-%d %H:%M:%S %Z')
        
        # Initialize other session state variables
        if 'show_sql_query' not in st.session_state:
            st.session_state.show_sql_query = False
        if 'sql_queries' not in st.session_state:
            st.session_state.sql_queries = []
        if 'use_sql_engine' not in st.session_state:
            st.session_state.use_sql_engine = True  # Default to SQL (faster)
        # Auto-display table on first page load
        st.session_state.show_main_content = True
        st.session_state.content_mode = 'table'
    else:
        st.session_state.first_time = False


# -------------------------------
# UI
# -------------------------------

def app_info() -> None:
    """Display app information"""
    st.caption(f"Project lead is Robert Salita research@AiPolice.org. Code written in Python. UI written in streamlit. Data engine is polars. Query engine is duckdb. Bridge lib is endplay. Self hosted using Cloudflare Tunnel. Repo:https://github.com/BSalita")
    st.caption(f"App:{st.session_state.app_datetime} Streamlit:{st.__version__} Query Params:{st.query_params.to_dict()} Environment:{os.getenv('STREAMLIT_ENV','')}")
    st.caption(f"Python:{'.'.join(map(str, sys.version_info[:3]))} pandas:{pd.__version__} polars:{pl.__version__} endplay:{ENDPLAY_VERSION}")
    try:
        import psutil

        def _gb(v: int) -> float:
            return v / (1024 ** 3)

        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        st.caption(
            f"Memory: RAM {_gb(vm.used):.2f}/{_gb(vm.total):.2f} GB ({vm.percent:.1f}%) | "
            f"Virtual/Pagefile {_gb(sm.used):.2f}/{_gb(sm.total):.2f} GB ({sm.percent:.1f}%)"
        )
    except Exception:
        st.caption("Memory: RAM/Virtual usage unavailable")
    return


def _render_detail_aggrid(detail_df: pl.DataFrame, key: str, selectable: bool = False):
    """Render a detail DataFrame as an AgGrid table.

    Returns the grid response dict when *selectable* is True, else None.
    """
    import pandas as pd
    from st_aggrid import GridOptionsBuilder, AgGrid, AgGridTheme

    pdf = detail_df.to_pandas()

    gb = GridOptionsBuilder.from_dataframe(pdf)
    if selectable:
        gb.configure_selection(selection_mode='single', use_checkbox=False, suppressRowClickSelection=False)
    gb.configure_default_column(
        cellStyle={'color': 'black', 'font-size': '12px'},
        suppressMenu=True,
        wrapHeaderText=True,
        autoHeaderHeight=True,
    )
    # Configure numeric columns for proper sorting
    for col in pdf.columns:
        if pd.api.types.is_numeric_dtype(pdf[col]):
            gb.configure_column(col, type=['numericColumn'], filter='agNumberColumnFilter')

    grid_options = gb.build()
    grid_options['rowHeight'] = 28
    grid_options['domLayout'] = 'normal'

    # Dynamic height: up to 25 rows visible
    header_height = 50
    row_height = grid_options['rowHeight']
    max_visible = 25
    n_rows = len(pdf)
    visible = max(1, min(n_rows, max_visible))
    if n_rows <= max_visible:
        grid_options['alwaysShowVerticalScroll'] = False
        height = header_height + visible * row_height + 20
    else:
        grid_options['alwaysShowVerticalScroll'] = True
        height = header_height + max_visible * row_height + 20

    response = AgGrid(
        pdf,
        gridOptions=grid_options,
        height=height,
        theme=AgGridTheme.BALHAM,
        key=key,
    )
    return response if selectable else None


def _show_opponent_aggregation(detail: pl.DataFrame, selected_row) -> None:
    """Given the full board-level detail and a clicked row, show per-opponent aggregation for that session."""
    session_id = selected_row.get('Session')
    if session_id is None:
        return

    session_boards = detail.filter(pl.col("Session") == session_id)
    if session_boards.is_empty():
        return

    date_val = session_boards.select("Date").row(0)[0]
    st.markdown(f"#### Opponent Breakdown — Session {session_id} ({str(date_val)[:10]})")

    agg_cols = [
        pl.col("Opponents").first().alias("Opponents"),
        pl.len().alias("Boards"),
    ]
    if "Pct" in session_boards.columns:
        agg_cols.append(pl.col("Pct").mean().cast(pl.Float64).round(1).alias("Avg_Pct"))
    if "Elo_Before" in session_boards.columns:
        agg_cols.append(pl.col("Elo_Before").first().cast(pl.Int32, strict=False).alias("Elo_Start"))
    if "Elo_After" in session_boards.columns:
        agg_cols.append(pl.col("Elo_After").last().cast(pl.Int32, strict=False).alias("Elo_End"))

    opp_agg = (
        session_boards
        .sort("Round" if "Round" in session_boards.columns else "Board")
        .group_by("Opponents", maintain_order=True)
        .agg(agg_cols[1:])  # skip the first Opponents alias — it's the group key
    )

    if "Elo_Start" in opp_agg.columns and "Elo_End" in opp_agg.columns:
        opp_agg = opp_agg.with_columns(
            (pl.col("Elo_End") - pl.col("Elo_Start")).alias("Elo_Delta")
        )

    st.caption(f"{len(opp_agg)} opponent pairs, {len(session_boards)} boards")
    _render_detail_aggrid(opp_agg, key=f"opp_agg_{session_id}")


def _show_all_opponents_aggregation(detail: pl.DataFrame, key_suffix: str) -> None:
    """Show per-opponent aggregation across ALL sessions in the detail data."""
    if detail.is_empty() or "Opponents" not in detail.columns:
        return

    st.markdown("#### Opponent Summary — All Sessions")

    agg_cols = [
        pl.len().alias("Boards"),
        pl.col("Session").n_unique().alias("Sessions"),
    ]
    if "Pct" in detail.columns:
        agg_cols.append(pl.col("Pct").mean().cast(pl.Float64).round(1).alias("Avg_Pct"))
    if "Elo_Delta" in detail.columns:
        agg_cols.append(pl.col("Elo_Delta").mean().cast(pl.Float64).round(1).alias("Avg_Elo_Delta"))

    opp_all = (
        detail
        .group_by("Opponents")
        .agg(agg_cols)
        .sort("Boards", descending=True)
    )

    st.caption(f"{len(opp_all)} unique opponent pairs")
    _render_detail_aggrid(opp_all, key=f"opp_all_{key_suffix}")


def _show_detail_for_selected_row(
    selected_row,
    raw_df: pl.DataFrame,
    rating_type: str,
    elo_rating_type: str,
) -> None:
    """Show session-level detail table for the clicked row in the summary grid.

    For *Players* we filter by Player_ID across all four seat positions (N/E/S/W).
    For *Pairs* we filter by the canonical Pair_IDs string (min-max of two player IDs).
    """
    elo_columns = get_elo_column_names(elo_rating_type)

    if rating_type == "Players":
        player_id = str(selected_row.get('Player_ID', ''))
        player_name = selected_row.get('Player_Name', 'Unknown')
        if not player_id:
            return

        st.markdown(f"#### Session History: **{player_name}** ({player_id})")

        with st.spinner("Loading session history..."):
            # Collect rows where this player sat in any seat
            frames: list[pl.DataFrame] = []
            for pos in "NESW":
                elo_col = elo_columns["player_pattern"].format(pos=pos) if elo_columns["player_pattern"] else None
                if elo_col is None or elo_col not in raw_df.columns:
                    continue

                # Determine the partner position
                partner_pos_map = {"N": "S", "S": "N", "E": "W", "W": "E"}
                partner = partner_pos_map[pos]
                opp1, opp2 = (("E", "W") if pos in ("N", "S") else ("N", "S"))

                cols_to_select = [
                    pl.col("Date"),
                    pl.col("session_id").alias("Session"),
                ]
                if "Round" in raw_df.columns:
                    cols_to_select.append(pl.col("Round"))
                if "Board" in raw_df.columns:
                    cols_to_select.append(pl.col("Board"))
                # Pct from this player's perspective (EW = 100 - Pct_NS)
                is_ns = pos in ("N", "S")
                if "Pct_NS" in raw_df.columns:
                    pct_expr = (pl.col("Pct_NS").cast(pl.Float64) * 100).round(1) if is_ns else ((1 - pl.col("Pct_NS").cast(pl.Float64)) * 100).round(1)
                else:
                    pct_expr = pl.lit(None, dtype=pl.Float64)

                # Add Elo_Before if available (only for current/start-of-session types)
                before_col = f"Elo_R_{pos}_Before" if elo_rating_type in (
                    "Current Rating (End of Session)", "Rating at Start of Session"
                ) else None

                cols_to_select += [
                    pl.lit(pos).alias("Seat"),
                    pl.col(f"Player_Name_{partner}").alias("Partner"),
                    (pl.col(f"Player_Name_{opp1}") + " - " + pl.col(f"Player_Name_{opp2}")).alias("Opponents"),
                    pct_expr.alias("Pct"),
                ]
                # Elo_Before first, then Elo_After
                if before_col and before_col in raw_df.columns:
                    cols_to_select.append(pl.col(before_col).cast(pl.Float64).round(0).cast(pl.Int32, strict=False).alias("Elo_Before"))
                cols_to_select.append(pl.col(elo_col).cast(pl.Float64).round(0).cast(pl.Int32, strict=False).alias("Elo_After"))

                seat_df = (
                    raw_df
                    .filter(pl.col(f"Player_ID_{pos}") == player_id)
                    .select(cols_to_select)
                )
                frames.append(seat_df)

            if not frames:
                st.info("No session data found for this player.")
                return

            detail = pl.concat(frames, how="diagonal")

            # Sort by Date desc, then Session, Round, Board for logical ordering
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
                detail = detail.with_columns(
                    (pl.col("Elo_After") - pl.col("Elo_Before")).alias("Elo_Delta")
                )

        n_sessions = detail.select("Session").n_unique()
        st.caption(f"{len(detail)} boards across {n_sessions} sessions — click a row to see opponent breakdown")

        # --- All-sessions opponent summary (shown first for visibility) ---
        _show_all_opponents_aggregation(detail, key_suffix=f"player_{player_id}")

        grid_resp = _render_detail_aggrid(detail, key=f"detail_player_{player_id}", selectable=True)

        # --- Opponent aggregation for selected session ---
        if grid_resp is not None:
            sel = grid_resp.get('selected_rows', None)
            if sel is not None and len(sel) > 0:
                sel_row = sel.iloc[0] if hasattr(sel, 'iloc') else sel[0]
                _show_opponent_aggregation(detail, sel_row)

    elif rating_type == "Pairs":
        pair_ids = str(selected_row.get('Pair_IDs', ''))
        pair_names = selected_row.get('Pair_Names', 'Unknown')
        if not pair_ids or '-' not in pair_ids:
            return

        st.markdown(f"#### Session History: **{pair_names}**")

        player_a, player_b = pair_ids.split('-', 1)

        with st.spinner("Loading session history..."):
            # Collect rows where this pair sat NS or EW
            frames: list[pl.DataFrame] = []
            for side, id1_col, id2_col in [
                ("NS", "Player_ID_N", "Player_ID_S"),
                ("EW", "Player_ID_E", "Player_ID_W"),
            ]:
                pair_elo_col = elo_columns[f"pair_{side.lower()}"]
                if pair_elo_col not in raw_df.columns:
                    continue

                # Canonical pair match: min(id1,id2)==player_a and max(id1,id2)==player_b
                side_df = raw_df.filter(
                    (pl.min_horizontal(pl.col(id1_col), pl.col(id2_col)) == player_a)
                    & (pl.max_horizontal(pl.col(id1_col), pl.col(id2_col)) == player_b)
                )

                if side_df.is_empty():
                    continue

                opp_side = "EW" if side == "NS" else "NS"
                opp1, opp2 = (("E", "W") if opp_side == "EW" else ("N", "S"))

                cols_to_select = [
                    pl.col("Date"),
                    pl.col("session_id").alias("Session"),
                ]
                if "Round" in raw_df.columns:
                    cols_to_select.append(pl.col("Round"))
                if "Board" in raw_df.columns:
                    cols_to_select.append(pl.col("Board"))
                # Pct from this pair's perspective (EW = 100 - Pct_NS)
                is_ns = (side == "NS")
                if "Pct_NS" in raw_df.columns:
                    pct_expr = (pl.col("Pct_NS").cast(pl.Float64) * 100).round(1) if is_ns else ((1 - pl.col("Pct_NS").cast(pl.Float64)) * 100).round(1)
                else:
                    pct_expr = pl.lit(None, dtype=pl.Float64)

                # Add Elo_Before if available (only for current/start-of-session types)
                before_col = f"Elo_R_{side}_Before" if elo_rating_type in (
                    "Current Rating (End of Session)", "Rating at Start of Session"
                ) else None

                cols_to_select += [
                    pl.lit(side).alias("Side"),
                    (pl.col(f"Player_Name_{opp1}") + " - " + pl.col(f"Player_Name_{opp2}")).alias("Opponents"),
                    pct_expr.alias("Pct"),
                ]
                # Elo_Before first, then Elo_After
                if before_col and before_col in raw_df.columns:
                    cols_to_select.append(pl.col(before_col).cast(pl.Float64).round(0).cast(pl.Int32, strict=False).alias("Elo_Before"))
                cols_to_select.append(pl.col(pair_elo_col).cast(pl.Float64).round(0).cast(pl.Int32, strict=False).alias("Elo_After"))

                frames.append(side_df.select(cols_to_select))

            if not frames:
                st.info("No session data found for this pair.")
                return

            detail = pl.concat(frames, how="diagonal")

            # Sort by Date desc, then Session, Round, Board for logical ordering
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
                detail = detail.with_columns(
                    (pl.col("Elo_After") - pl.col("Elo_Before")).alias("Elo_Delta")
                )

        n_sessions = detail.select("Session").n_unique()
        st.caption(f"{len(detail)} boards across {n_sessions} sessions — click a row to see opponent breakdown")

        # --- All-sessions opponent summary (shown first for visibility) ---
        _show_all_opponents_aggregation(detail, key_suffix=f"pair_{pair_ids}")

        grid_resp = _render_detail_aggrid(detail, key=f"detail_pair_{pair_ids}", selectable=True)

        # --- Opponent aggregation for selected session ---
        if grid_resp is not None:
            sel = grid_resp.get('selected_rows', None)
            if sel is not None and len(sel) > 0:
                sel_row = sel.iloc[0] if hasattr(sel, 'iloc') else sel[0]
                _show_opponent_aggregation(detail, sel_row)


def _filter_valid_percentages_acbl(df: pl.DataFrame) -> pl.DataFrame:
    """Drop rows with invalid percentage values (<0 or >100 equivalent)."""
    if df.is_empty() or "Pct_NS" not in df.columns:
        return df

    pct_ns = pl.col("Pct_NS").cast(pl.Float64, strict=False)
    # Pct_NS is stored as a fraction (0..1), which corresponds to 0..100%.
    return df.filter(pct_ns.is_null() | ((pct_ns >= 0.0) & (pct_ns <= 1.0)))


def main():
    """Main application function."""
    # UI Configuration - must be first Streamlit command
    st.set_page_config(
        page_title="Unofficial ACBL Elo Ratings Playground",
        page_icon=ASSISTANT_LOGO_URL,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Apply common dark green/gold theme (shared with FFBridge app)
    apply_app_theme(st)
    widen_scrollbars()
    
    # Styled header
    st.markdown(f"""
        <div style="text-align: center; padding: 0 0 1rem 0; margin-top: -2rem;">
            <h1 style="font-size: 2.8rem; margin-bottom: 0.2rem;">
                <img src="{ASSISTANT_LOGO_URL}" style="height: 2.5rem; vertical-align: middle; margin-right: 0.5rem;">
                Morty's Unofficial ACBL Elo Ratings Playground
            </h1>
            <p style="color: #ffc107; font-size: 1.2rem; font-weight: 500; opacity: 0.9;">
                An interactive playground for ACBL Elo ratings
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    stick_it_good()

    # Sidebar will be created after data loading is complete
    # Set default values for data loading
    date_from = None  # Default to all time for initial loading

    # Load all columns initially (sidebar controls will be available after loading)
    needed_cols = None  # Load all columns

    # -------------------------------
    # Data Loading
    # -------------------------------

    def load_dataset(dataset_name: str, date_from_str: str, columns: list[str] | None = None):
        """Load one dataset lazily to reduce memory pressure."""
        date_from = None if date_from_str == "None" else datetime.fromisoformat(date_from_str)
        df = load_elo_ratings(dataset_name, columns=columns, date_from=date_from)
        return _filter_valid_percentages_acbl(df)

    # Convert date_from to string for caching key
    date_from_str = "None" if date_from is None else date_from.isoformat()

    # Load only the currently selected event dataset (lazy loading).
    selected_event_for_load = st.session_state.get("event_type", "Club").lower()
    selected_event_for_load = "club" if selected_event_for_load not in ("club", "tournament") else selected_event_for_load
    selected_rating_for_load = st.session_state.get("rating_type", "Players")
    selected_elo_type_for_load = st.session_state.get("elo_rating_type", "Current Rating (End of Session)")
    required_columns = _required_columns_for_mode(selected_rating_for_load, selected_elo_type_for_load)
    required_cols_set = set(required_columns)
    stored_cols_set = set(st.session_state.get('loaded_columns', {}).get(selected_event_for_load, []))
    must_reload = (
        'all_data' not in st.session_state
        or 'data_date_from' not in st.session_state
        or st.session_state.data_date_from != date_from_str
        or st.session_state.get('loaded_dataset') != selected_event_for_load
        or selected_event_for_load not in st.session_state.get('all_data', {})
        or not required_cols_set.issubset(stored_cols_set)  # missing columns
        or stored_cols_set - required_cols_set  # excess columns — trim to save memory
    )
    if must_reload:
        try:
            import gc

            # Hard reset of switch-sensitive state to prevent retained references.
            _clear_table_cache()
            if 'sql_query_history' in st.session_state:
                st.session_state.sql_query_history = []

            # Drop all loaded dataframes immediately.
            if 'all_data' in st.session_state:
                st.session_state.all_data = {}

            # Recreate DuckDB connection on dataset switch.
            # This guarantees no registered/relation state can pin prior frames.
            if 'db_connection' in st.session_state:
                try:
                    st.session_state.db_connection.close()
                except Exception:
                    pass
                del st.session_state.db_connection

            # Keep metadata only for the active dataset.
            st.session_state.loaded_columns = {}

            gc.collect()
            with st.spinner(f"Loading {selected_event_for_load} dataset..."):
                all_data = {
                    selected_event_for_load: load_dataset(
                        selected_event_for_load, date_from_str, columns=required_columns
                    )
                }

            st.success("Datasets loaded successfully")

            # Store data and column set in session state for reuse
            st.session_state.all_data = all_data
            st.session_state.data_date_from = date_from_str
            st.session_state.loaded_dataset = selected_event_for_load
            st.session_state.loaded_columns[selected_event_for_load] = required_cols_set

        except Exception as e:
            st.error(f"Failed to load datasets: {e}")
            st.stop()
    else:
        # Use already loaded data
        all_data = st.session_state.all_data

    def ensure_dataset_loaded(dataset_name: str, columns: list[str] | None = None):
        """Load missing dataset on-demand; reload only if required columns are missing."""
        nonlocal all_data
        if dataset_name in all_data:
            if columns is None:
                return
            existing_cols = set(all_data[dataset_name].columns)
            needed_cols = set(columns)
            if needed_cols.issubset(existing_cols):
                return
            # Drop old frame before reloading to free memory promptly
            import gc
            all_data[dataset_name] = None
            gc.collect()
        with st.spinner(f"Loading {dataset_name} dataset..."):
            all_data[dataset_name] = load_dataset(dataset_name, date_from_str, columns=columns)
        if 'loaded_columns' not in st.session_state:
            st.session_state.loaded_columns = {}
        st.session_state.loaded_columns[dataset_name] = set(columns) if columns else None
        st.session_state.all_data = all_data
    
    # Initialize SQL query settings
    if 'show_sql_query' not in st.session_state:
        st.session_state.show_sql_query = False
    if 'sql_queries' not in st.session_state:
        st.session_state.sql_queries = []

    # -------------------------------
    # Create Sidebar Controls (After Data Loading)
    # -------------------------------

    with st.sidebar:
        st.sidebar.caption(f"Build:{st.session_state.app_datetime}")
        st.sidebar.markdown("🔗 [What is Elo Rating?](https://en.wikipedia.org/wiki/Elo_rating_system)")
        club_or_tournament = st.radio("Event type", options=["Club", "Tournament"], index=0, horizontal=True, key="event_type")
        rating_type = st.radio("Rating type", options=["Players", "Pairs"], index=0, horizontal=True, key="rating_type")
        top_n = st.number_input("Top N players or pairs", min_value=50, max_value=5000, value=1000, step=50)
        min_sessions = st.number_input("Minimum sessions played", min_value=1, max_value=200, value=30, step=1)
        rating_method = st.selectbox("Elo Rating statistic", options=["Avg", "Max", "Latest"], index=0)
        
        # Moving average days - not used anymore (Moving Avg disabled due to memory issues)
        moving_avg_days = 10  # Default value
        
        # Elo rating type selector - filter options based on rating type
        if rating_type == "Players":
            # Players don't have Expected Rating (only individual player ratings)
            elo_options = [
                "Current Rating (End of Session)",
                "Rating at Start of Session", 
                "Rating at Event Start",
                "Rating at Event End"
            ]
        else:  # Pairs
            # Pairs have all rating types including Expected Rating
            elo_options = [
                "Current Rating (End of Session)",
                "Rating at Start of Session", 
                "Rating at Event Start",
                "Rating at Event End",
                "Expected Rating"
            ]
        
        elo_rating_type = st.selectbox(
            "Elo rating moment",
            options=elo_options,
            index=0,
            key="elo_rating_type",
            help="Choose timing of Elo analysis"
        )
        
        # Player name filter (use a stable session key to avoid rerun inconsistencies)
        def _on_player_name_enter():
            # Ensure table is shown when ENTER is pressed in the name box
            st.session_state.show_main_content = True
            st.session_state.content_mode = 'table'

        player_name_filter = st.text_input(
            "Filter by Player Name",
            value=st.session_state.get("player_name_filter", ""),
            placeholder="Enter name to search...",
            help="Filter results to show only players whose names contain this text (case-insensitive)",
            key="player_name_filter",
            on_change=_on_player_name_enter
        )
        
        # Date range quick filter (default All time)
        date_range_choice = st.selectbox(
            "Date range",
            options=["All time", "Last 3 months", "Last 6 months", "Last 1 year", "Last 2 years", "Last 3 years", "Last 4 years", "Last 5 years"],
            index=0,
        )
        
        # Game type filter (Local / Online / All)
        online_filter = st.selectbox(
            "Game type",
            options=["All", "Local Only", "Online Only"],
            index=1,
            help="Filter by game type: Local (in-person), Online (virtual), or All games"
        )
        
        # Masterpoints range filter (Players only)
        masterpoints_filter = "All"
        if rating_type == "Players":
            masterpoints_filter = st.selectbox(
                "Masterpoints Range",
                options=["All"] + [format_masterpoints_label(lo, hi) for (lo, hi) in MASTERPOINT_RANGES],
                index=0,
                help="Filter players by ACBL MasterPoints range"
            )
        st.session_state.masterpoints_filter = masterpoints_filter
        
        generate_pdf = st.button("Generate PDF", type="primary")
        
        
        # Automated Postmortem Apps
        st.sidebar.markdown('<p style="color: #ffc107; font-weight: 600;">Morty\'s Automated Postmortem Apps</p>', unsafe_allow_html=True)
        st.sidebar.markdown("🔗 [ACBL Postmortem](https://acbl.postmortem.chat)")
        st.sidebar.markdown("🔗 [French ffbridge Postmortem](https://ffbridge.postmortem.chat)")
        #st.sidebar.markdown("🔗 [BridgeWebs Postmortem](https://bridgewebs.postmortem.chat)")
        
        # Developer Options
        with st.expander("🔧 **Developer Options**"):
            show_sql = st.checkbox('Show SQL Query', value=st.session_state.show_sql_query, help='Show SQL used to query dataframes.')
            st.session_state.show_sql_query = show_sql
            
            # Query engine selection (SQL is 2-3x faster)
            use_sql_engine = st.checkbox('Use SQL Engine (2-3x faster)', value=st.session_state.get('use_sql_engine', True), 
                                        help='SQL engine is 2-3x faster than Polars. Uncheck to use Polars engine for comparison.')
            st.session_state.use_sql_engine = use_sql_engine
            
            # Comparison mode toggle (default relaxed)
            strict = st.checkbox('Strict comparison (include name columns)', value=st.session_state.get('strict_comparison', False))
            st.session_state.strict_comparison = strict
            # Fast comparison options (limit club dataset by recent days)
            fast_comp = st.checkbox('Fast comparison (limit club to recent days)', value=st.session_state.get('fast_comparison', True))
            st.session_state.fast_comparison = fast_comp
            fast_days = st.number_input('Fast comparison club days', min_value=30, max_value=3650, value=int(st.session_state.get('fast_comparison_days', 180)) if isinstance(st.session_state.get('fast_comparison_days', 180), int) else 180, step=30, help='When enabled, comparisons filter club data to the most recent N days for speed.')
            st.session_state.fast_comparison_days = int(fast_days)
            
            run_comprehensive = st.button("Run Polars vs SQL Comparison", help="Test all combinations: datasets × rating types × methods × elo types")
            if run_comprehensive:
                st.session_state.run_comprehensive_comparison = True
            # New: dataset-specific comparison buttons
            run_club_only = st.button("Run Club-only Comparison", help="Compare only club dataset across all variations")
            if run_club_only:
                st.session_state.run_comprehensive_comparison_club = True
            run_tournament_only = st.button("Run Tournament-only Comparison", help="Compare only tournament dataset across all variations")
            if run_tournament_only:
                st.session_state.run_comprehensive_comparison_tournament = True
            
            # Debug single test button
            debug_single = st.button("Run Polars vs SQL Single Test", help="Run a single test to debug issues")
            if debug_single:
                st.session_state.debug_single_test = True

    # Determine date_from based on selection
    now = datetime.now()
    if date_range_choice == "All time":
        date_from = None
    elif date_range_choice == "Last 3 months":
        date_from = now - timedelta(days=90)
    elif date_range_choice == "Last 6 months":
        date_from = now - timedelta(days=182)
    elif date_range_choice == "Last 1 year":
        date_from = now - timedelta(days=365)
    elif date_range_choice == "Last 2 years":
        date_from = now - timedelta(days=365*2)
    elif date_range_choice == "Last 3 years":
        date_from = now - timedelta(days=365*3)
    elif date_range_choice == "Last 4 years":
        date_from = now - timedelta(days=365*4)
    elif date_range_choice == "Last 5 years":
        date_from = now - timedelta(days=365*5)
    else:
        date_from = None # Default to all time

    # -------------------------------
    # Report Generation
    # -------------------------------
    # Track current settings to detect changes
    current_settings = {
        "club_or_tournament": club_or_tournament,
        "rating_type": rating_type,
        "top_n": int(top_n),
        "min_sessions": int(min_sessions),
        "rating_method": rating_method,
        "moving_avg_days": moving_avg_days,
        "elo_rating_type": elo_rating_type,
        "date_range_choice": date_range_choice,
        "online_filter": online_filter,
    }
    
    # Check if settings have changed since last report
    if 'last_settings' not in st.session_state:
        st.session_state.last_settings = current_settings
        settings_changed = False
    else:
        settings_changed = st.session_state.last_settings != current_settings
    
    # Clear table_displayed flag if settings changed
    if settings_changed:
        st.session_state.table_displayed = False
        st.session_state.last_settings = current_settings
        
        # Clear previous results immediately when settings change
        import gc
        _clear_table_cache()
        if 'sql_query_history' in st.session_state:
            st.session_state.sql_query_history = []
        gc.collect()
    
    # Check if comprehensive comparison was requested
    if st.session_state.get('run_comprehensive_comparison', False):
        st.markdown("## 🔍 **Comprehensive Implementation Comparison**")
        st.markdown("Testing all combinations of datasets, rating types, methods, and Elo rating types...")
        ensure_dataset_loaded("club", columns=None)
        ensure_dataset_loaded("tournament", columns=None)
        
        # Run comprehensive comparison
        comprehensive_results = run_comprehensive_comparison(all_data, top_n=50, min_sessions=5)
        
        # Display results
        display_comprehensive_results(comprehensive_results)
        
        # Clear the flag
        st.session_state.run_comprehensive_comparison = False
        
        # Stop here - don't show regular content
        return

    # Check if club-only comparison was requested
    if st.session_state.get('run_comprehensive_comparison_club', False):
        st.markdown("## 🔍 **Club-only Polars vs SQL Comparison**")
        st.markdown("Testing all variations on the club dataset only...")
        ensure_dataset_loaded("club", columns=None)
        comprehensive_results = run_comprehensive_comparison(all_data, top_n=50, min_sessions=5, datasets_filter=['club'])
        display_comprehensive_results(comprehensive_results)
        st.session_state.run_comprehensive_comparison_club = False
        return

    # Check if tournament-only comparison was requested
    if st.session_state.get('run_comprehensive_comparison_tournament', False):
        st.markdown("## 🔍 **Tournament-only Polars vs SQL Comparison**")
        st.markdown("Testing all variations on the tournament dataset only...")
        ensure_dataset_loaded("tournament", columns=None)
        comprehensive_results = run_comprehensive_comparison(all_data, top_n=50, min_sessions=5, datasets_filter=['tournament'])
        display_comprehensive_results(comprehensive_results)
        st.session_state.run_comprehensive_comparison_tournament = False
        return
    
    # Check if debug single test was requested
    if st.session_state.get('debug_single_test', False):
        st.markdown("## 🐛 **Debug Single Test**")
        
        # Run a single test to debug
        try:
            ensure_dataset_loaded("club", columns=None)
            df = all_data['club']
            rating_type = "Players"
            rating_method = "Avg"
            elo_rating_type = "Current Rating (End of Session)"
            
            st.write(f"**Testing:** {rating_type} | {rating_method} | {elo_rating_type}")
            
            # Run Polars implementation
            st.write("Running Polars implementation...")
            polars_df = show_top_players(df, 10, 5, rating_method, 10, elo_rating_type)
            st.write(f"Polars result shape: {polars_df.shape}")
            st.write("Polars columns:", list(polars_df.columns))
            st.dataframe(polars_df.head(3))
            
            # Run SQL implementation
            st.write("Running SQL implementation...")
            con = get_db_connection()
            _db_register(con, 'self', df)
            sql_query = generate_top_players_sql(10, 5, rating_method, 10, elo_rating_type, set(df.columns))
            st.code(sql_query, language='sql')
            
            sql_df = con.execute(sql_query).pl()
            st.write(f"SQL result shape: {sql_df.shape}")
            st.write("SQL columns:", list(sql_df.columns))
            st.dataframe(sql_df.head(3))
            
            # Compare results
            st.write("Comparing results...")
            comparison = compare_polars_sql_results(polars_df, sql_df, "players")
            st.json(comparison)
            
        except Exception as e:
            st.error(f"Debug test failed: {e}")
            import traceback
            st.code(traceback.format_exc())
        
        # Clear the flag
        st.session_state.debug_single_test = False
        
        # Stop here - don't show regular content
        return
    
    # Persist the display state in session_state
    if generate_pdf:
        st.session_state.show_main_content = True
        st.session_state.content_mode = 'pdf'
    
    # Any control change keeps content visible (auto-refresh); just ensure mode is set
    if st.session_state.get('show_main_content') and not st.session_state.get('content_mode'):
        st.session_state.content_mode = 'table'
    
    st.session_state.previous_settings = current_settings.copy()
    
    # Always show main content (auto-started on first load)
    show_main_content = st.session_state.get('show_main_content', False)
    
    if not show_main_content:
        st.info("Select left sidebar options or click 'Generate PDF' button.")
    else:
        # Memory cleanup - clear previous results and force garbage collection
        import gc
        
        # Keep only bounded cached tables; avoid duplicate heavy references
        _prune_table_cache(current_key="")
        if 'sql_query_history' in st.session_state:
            # Keep only the last 5 queries to prevent memory buildup
            st.session_state.sql_query_history = st.session_state.sql_query_history[-5:]
        
        # Force garbage collection
        gc.collect()
        
        # Build options signature for reuse
        opts = {
            "club_or_tournament": club_or_tournament,
            "rating_type": rating_type,
            "top_n": int(top_n),
            "min_sessions": int(min_sessions),
            "rating_method": rating_method,
            "date_from": None if date_from is None else date_from.isoformat(),
        }

        # Get data
        dataset_type = club_or_tournament.lower()
        ensure_dataset_loaded(dataset_type, columns=_required_columns_for_mode(rating_type, elo_rating_type))
        df = all_data[dataset_type]
        
        # Apply date range filter
        if date_from is not None and 'Date' in df.columns:
            df = df.filter(pl.col('Date') >= pl.lit(date_from))
        
        # Apply online filter if specified
        if online_filter == "Local Only":
            # Local games: rows explicitly marked as not virtual (False)
            if "is_virtual_game" in df.columns:
                df = df.filter(pl.col("is_virtual_game") == False)
        elif online_filter == "Online Only":
            # Online games: rows where is_virtual_game is null (virtual/online)
            if "is_virtual_game" in df.columns:
                df = df.filter(pl.col("is_virtual_game").is_null())
        # For "All", no filtering is applied
        
        # Store online filter in session state for comprehensive comparison
        st.session_state.online_filter = online_filter
        
        # Avoid computing len(df) which can be expensive on large datasets
        st.info(f"✅ Using {dataset_type} dataset ({online_filter.lower()} games)")
        
        # Store current dataset type
        st.session_state.current_dataset_type = dataset_type

        # Compute date range for captions
        try:
            date_min, date_max = df.select([pl.col("Date").min().alias("min"), pl.col("Date").max().alias("max")]).row(0)
            date_range = f"{str(date_min)[:10]} to {str(date_max)[:10]}"
        except Exception:
            date_range = ""

        # Prepare column names before any external registration to avoid borrow issues
        try:
            df_columns = set(df.schema.keys())
        except Exception:
            df_columns = set(df.columns)

        # Generate SQL query based on report type
        if rating_type == "Players":
            generated_sql = generate_top_players_sql(int(top_n), int(min_sessions), rating_method, moving_avg_days, elo_rating_type, df_columns)
            method_desc = f"{rating_method} method"
            title = f"Top {top_n} ACBL {club_or_tournament} Players by {elo_rating_type} ({method_desc})"
        elif rating_type == "Pairs":
            generated_sql = generate_top_pairs_sql(int(top_n), int(min_sessions), rating_method, moving_avg_days, elo_rating_type, df_columns)
            method_desc = f"{rating_method} method"
            title = f"Top {top_n} ACBL {club_or_tournament} Pairs by {elo_rating_type} ({method_desc})"
        else:
            raise ValueError(f"Invalid rating type: {rating_type}")
        
        # Store the generated SQL for the display table functionality
        st.session_state.generated_sql = generated_sql
        st.session_state.report_title = title

        # Show SQL-based interface when in table mode
        show_table = st.session_state.get('content_mode') == 'table'
        if show_table:
            if date_range:
                st.subheader(f"{title} From {date_range}")
            else:
                st.subheader(title)
            
            # 1. Show the SQL query used in a compact scrollable container (only if enabled)
            if st.session_state.get('show_sql_query', False):
                st.markdown("### 📝 SQL Query Used")
                st.text_area("SQL Query", value=generated_sql, height=200, disabled=True, label_visibility="collapsed")
            
            # 2. Execute and show results - Use selected engine (SQL is 2-3x faster)
            use_sql = st.session_state.get('use_sql_engine', True)
            # Cache the table_df to avoid regenerating on every rerun (e.g., when Execute Query button is clicked)
            cache_key = f"cached_table_{club_or_tournament}_{rating_type}_{top_n}_{min_sessions}_{rating_method}_{moving_avg_days}_{elo_rating_type}_{date_range}_{online_filter}_{st.session_state.get('masterpoints_filter','All')}"
            try:
                if cache_key in st.session_state:
                    table_df, _used_cache, engine_name = _get_or_build_report_table_df(
                        df=df,
                        generated_sql=generated_sql,
                        rating_type=rating_type,
                        top_n=int(top_n),
                        min_sessions=int(min_sessions),
                        rating_method=rating_method,
                        moving_avg_days=moving_avg_days,
                        elo_rating_type=elo_rating_type,
                        cache_key=cache_key,
                        use_sql_engine=use_sql,
                    )
                    st.info(f"✅ Using cached {rating_type} report ({len(table_df)} rows)")
                else:
                    engine_name = "SQL" if use_sql else "Polars"
                    with st.spinner(f"Building {rating_type} report using {engine_name} engine..."):
                        table_df, _used_cache, engine_name = _get_or_build_report_table_df(
                            df=df,
                            generated_sql=generated_sql,
                            rating_type=rating_type,
                            top_n=int(top_n),
                            min_sessions=int(min_sessions),
                            rating_method=rating_method,
                            moving_avg_days=moving_avg_days,
                            elo_rating_type=elo_rating_type,
                            cache_key=cache_key,
                            use_sql_engine=use_sql,
                        )
                    st.success(f"✅ Report generated successfully using {engine_name} engine. Returned {len(table_df)} rows.")
            except Exception as e:
                st.error(f"❌ Report generation failed: {e}")
                st.error("Please try again or contact support if the problem persists.")
                return
            
            # Display results with exactly 25 viewable rows (common for both paths)
            if 'table_df' in locals():
                st.markdown("### 📊 Query Results")
                # Standardize on Polars for filtering; convert to pandas only right before AgGrid
                work_df = table_df
                try:
                    if not hasattr(work_df, 'select'):
                        # Convert pandas -> Polars if needed
                        work_df = pl.from_pandas(work_df)
                except Exception:
                    pass
                # Apply player name filter if provided (read from session to handle button click reruns)
                player_name_filter_value = st.session_state.get('player_name_filter', '').strip()
                if player_name_filter_value:
                    try:
                        if hasattr(work_df, 'select'):  # Polars DataFrame
                            import re
                            pattern = '(?i)' + re.escape(player_name_filter_value)
                            original_count = len(work_df)
                            if 'Player_Name' in work_df.columns:
                                work_df = work_df.filter(pl.col('Player_Name').cast(pl.Utf8).str.contains(pattern, literal=False))
                                filtered_count = len(work_df)
                                st.info(f"🔍 Filtered to {filtered_count} of {original_count} rows matching '{player_name_filter_value}'")
                            elif 'Pair_Names' in work_df.columns:
                                work_df = work_df.filter(pl.col('Pair_Names').cast(pl.Utf8).str.contains(pattern, literal=False))
                                filtered_count = len(work_df)
                                st.info(f"🔍 Filtered to {filtered_count} of {original_count} rows matching '{player_name_filter_value}'")
                            else:
                                st.warning("⚠️ No Player_Name or Pair_Names column found to filter on")
                    except Exception:
                        pass
                
                # Apply Masterpoints range filter for Players view
                if rating_type == "Players":
                    mp_filter_label = st.session_state.get('masterpoints_filter', 'All')
                    if mp_filter_label != "All":
                        try:
                            original_count = len(work_df)
                            work_df = apply_masterpoints_filter_polars(work_df, mp_filter_label)
                            filtered_count = len(work_df)
                            st.info(f"🎯 Masterpoints {mp_filter_label}: {filtered_count} of {original_count} players")
                        except Exception:
                            pass
                
                # Convert to pandas for AgGrid
                # Convert to pandas for AgGrid rendering
                display_df = work_df.to_pandas()
                
                # Use AgGrid directly with precise height control for exactly 25 rows
                from st_aggrid import GridOptionsBuilder, AgGrid, AgGridTheme
                
                gb = GridOptionsBuilder.from_dataframe(display_df)
                gb.configure_selection(selection_mode='single', use_checkbox=False)  # Enable single row selection
                gb.configure_default_column(cellStyle={'color': 'black', 'font-size': '12px'}, suppressMenu=True, wrapHeaderText=True, autoHeaderHeight=True)
                
                # Configure numeric columns for proper sorting
                for col in display_df.columns:
                    if pd.api.types.is_numeric_dtype(display_df[col]):
                        gb.configure_column(col, type=['numericColumn'], filter='agNumberColumnFilter')
                
                # Don't configure pagination - we want scrolling instead
                gb.configure_side_bar()
                gridOptions = gb.build()
                
                # Configure for scrolling with adjustable height
                gridOptions['rowHeight'] = 28
                gridOptions['suppressPaginationPanel'] = True
                gridOptions['suppressHorizontalScroll'] = False  # Allow horizontal scrollbar if needed
                gridOptions['domLayout'] = 'normal'  # Use normal layout (not autoHeight)

                # Dynamically size height: show up to 25 rows; if fewer, shrink to fit
                header_height = 50
                row_height = gridOptions['rowHeight']
                max_rows_visible = 25
                total_rows = len(display_df)
                visible_rows = max(1, min(total_rows, max_rows_visible))
                if total_rows <= max_rows_visible:
                    # No vertical scrollbar needed; fit exactly to number of rows
                    gridOptions['alwaysShowVerticalScroll'] = False
                    exact_height = header_height + visible_rows * row_height + 20
                else:
                    # Use fixed height with vertical scrollbar
                    gridOptions['alwaysShowVerticalScroll'] = True
                    exact_height = header_height + max_rows_visible * row_height + 20
                
                # Custom CSS to ensure scrollbars are visible
                custom_css = {
                    ".ag-theme-balham .ag-body-viewport": {
                        "overflow-y": "auto !important",
                        "overflow-x": "auto !important"
                    },
                    ".ag-theme-balham .ag-body-horizontal-scroll": {
                        "display": "block !important"
                    },
                    ".ag-theme-balham .ag-body-vertical-scroll": {
                        "display": "block !important"
                    }
                }
                
                st.caption("Click a row to view session history details")
                
                # Create dynamic key that resets selection when data/filters change
                dynamic_key = f"table-{rating_type}-{club_or_tournament}-{top_n}-{min_sessions}-{rating_method}-{elo_rating_type}-{date_range}-{online_filter}-{st.session_state.get('masterpoints_filter','All')}-{st.session_state.get('player_name_filter','')}"
                
                grid_response = AgGrid(
                    display_df,
                    gridOptions=gridOptions,
                    height=exact_height,
                    theme=AgGridTheme.BALHAM,
                    custom_css=custom_css,
                    key=dynamic_key
                )
                
                # --- Row-click detail view ---
                selected_rows = grid_response.get('selected_rows', None)
                if selected_rows is not None and len(selected_rows) > 0:
                    selected_row = selected_rows.iloc[0] if hasattr(selected_rows, 'iloc') else selected_rows[0]
                    _show_detail_for_selected_row(selected_row, df, rating_type, elo_rating_type)
                
                # Mark that table is displayed (lightweight state only)
                st.session_state.table_displayed = True
            
            # 3. SQL Query Interface for additional queries (only if enabled)
            if st.session_state.get('show_sql_query', False):
                st.markdown("---")
                st.markdown("### 🔍 Run Additional SQL Queries")
                st.caption("Query the results above. Only the displayed columns are available. The results table is available as 'self'.")
                
                
                # Initialize SQL query history if not exists
                if 'sql_query_history' not in st.session_state:
                    st.session_state.sql_query_history = []
                
                # Add the generated query to history automatically (only once)
                content_is_table = st.session_state.get('content_mode') == 'table'
                if content_is_table and st.session_state.get('generated_sql'):
                    if not any(h['query'] == st.session_state.generated_sql for h in st.session_state.sql_query_history):
                        st.session_state.sql_query_history.append({
                            'query': st.session_state.generated_sql,
                            'result': table_df,
                            'timestamp': datetime.now().strftime('%H:%M:%S'),
                            'auto_generated': True
                        })
                
                # Display additional query results (excluding the auto-generated one already shown above)
                additional_queries = [q for q in st.session_state.sql_query_history if not q.get('auto_generated', False)]
                if additional_queries:
                    st.markdown("### 📋 Additional Query Results")
                    
                    # Show results in reverse order (most recent first)
                    for i, query_result in enumerate(reversed(additional_queries)):
                        query_label = f"Query {len(additional_queries) - i} ({query_result['timestamp']})"
                        
                        with st.expander(query_label, expanded=(i == 0)):
                            st.code(query_result['query'], language='sql')
                            ShowDataFrameTable(
                                query_result['result'], 
                                key=f"sql_result_{len(additional_queries) - i}",
                                output_method='aggrid',
                                height_rows=25
                            )
                
                # SQL Query input - fixed at bottom using streamlit-extras bottom container
                with bottom():
                    query = st.text_input(
                        "💬 SQL Query (press Enter to execute):",
                        value='',
                        placeholder="SELECT * FROM self WHERE Player_Elo_Score > 1500 ORDER BY Player_Elo_Rank LIMIT 10",
                        key="sql_query_text_input",
                        on_change=lambda: st.session_state.update({"execute_query_now": True})
                    )

                # Execute query when Enter pressed
                if st.session_state.get('execute_query_now') and st.session_state.get('sql_query_text_input', '').strip():
                    with st.spinner('⏳ Executing query...'):
                        try:
                            # Process query
                            processed_query = st.session_state.get('sql_query_text_input', '').strip()
                            if 'from ' not in processed_query.lower():
                                processed_query = 'FROM self ' + processed_query
                            
                            # Show the query being executed
                            if st.session_state.get('show_sql_query', False):
                                st.code(processed_query, language='sql')
                            
                            # Execute query on the query results table, not the raw dataset
                            con = get_db_connection()
                            _db_register(con, 'self', table_df)
                            result_df = con.execute(processed_query).pl()
                            
                            # Store in history
                            st.session_state.sql_query_history.append({
                                'query': st.session_state.get('sql_query_text_input', ''),
                                'result': result_df,
                                'timestamp': datetime.now().strftime('%H:%M:%S')
                            })
                            
                            st.success(f"✅ Query executed successfully! Returned {len(result_df)} rows.")
                            
                            # Display the result immediately
                            st.markdown("### 📊 Query Result")
                            st.code(st.session_state.get('sql_query_text_input', ''), language='sql')
                            ShowDataFrameTable(
                                result_df, 
                                key=f"sql_result_current_{datetime.now().strftime('%H%M%S')}",
                                output_method='aggrid',
                                height_rows=25
                            )
                            
                        except Exception as e:
                            st.error(f"❌ SQL Error: {e}")
                            if 'processed_query' in locals():
                                st.info(f"📝 Your query was transformed to:")
                                st.code(processed_query, language='sql')
                        finally:
                            # Reset flag so we don't re-run on each rerun
                            st.session_state.execute_query_now = False

        # PDF generation when in PDF mode
        if st.session_state.get('content_mode') == 'pdf':
            # Use the same caching mechanism as Display Table
            cache_key = f"cached_table_{club_or_tournament}_{rating_type}_{top_n}_{min_sessions}_{rating_method}_{moving_avg_days}_{elo_rating_type}_{date_range}_{online_filter}_{st.session_state.get('masterpoints_filter','All')}"
            use_sql = st.session_state.get('use_sql_engine', True)
            try:
                if cache_key in st.session_state:
                    table_df, _used_cache, engine_name = _get_or_build_report_table_df(
                        df=df,
                        generated_sql=generated_sql,
                        rating_type=rating_type,
                        top_n=int(top_n),
                        min_sessions=int(min_sessions),
                        rating_method=rating_method,
                        moving_avg_days=moving_avg_days,
                        elo_rating_type=elo_rating_type,
                        cache_key=cache_key,
                        use_sql_engine=use_sql,
                    )
                    st.info(f"✅ Using cached {rating_type} report for PDF generation ({len(table_df)} rows)")
                else:
                    engine_name = "SQL" if use_sql else "Polars"
                    with st.spinner(f"Generating {rating_type} data for PDF using {engine_name} engine..."):
                        table_df, _used_cache, engine_name = _get_or_build_report_table_df(
                            df=df,
                            generated_sql=generated_sql,
                            rating_type=rating_type,
                            top_n=int(top_n),
                            min_sessions=int(min_sessions),
                            rating_method=rating_method,
                            moving_avg_days=moving_avg_days,
                            elo_rating_type=elo_rating_type,
                            cache_key=cache_key,
                            use_sql_engine=use_sql,
                        )
                    st.info(f"✅ Using {engine_name} engine for PDF generation")
            except Exception as e:
                st.error(f"❌ PDF Generation Failed: {e}")
                st.error("Unable to generate PDF. Please try again or contact support if the problem persists.")
                return
            
            created_on = time.strftime("%Y-%m-%d")
            #pdf_title = f"{title} From {date_range}"
            pdf_filename = f"Unofficial Elo Scores for ACBL {club_or_tournament} MatchPoint Games - Top {top_n} {rating_type} {created_on}.pdf"
            # Generate PDF
            try:
                # Apply Masterpoints range filter to table_df for Players PDF if requested
                if rating_type == "Players":
                    mp_filter_label = st.session_state.get('masterpoints_filter', 'All')
                    if mp_filter_label != "All":
                        try:
                            if not hasattr(table_df, 'select'):
                                # Ensure Polars for filtering
                                table_df = pl.from_pandas(table_df)
                            table_df = apply_masterpoints_filter_polars(table_df, mp_filter_label)
                        except Exception:
                            pass
                # Enable shrink_to_fit for both Player and Pair reports to prevent truncation
                # really want title, from date to be centered with reduced line spacing between them.
                pdf_bytes = create_pdf([f"## {title}", f"### From {date_range}", "### Created by https://elo.7nt.info", table_df], title, max_rows=int(top_n), max_cols=None, rows_per_page=(21, 28), shrink_to_fit=True)
            except Exception as e:
                st.error(f"❌ PDF Creation Failed: {e}")
                st.error("Unable to create PDF file. Please try again or contact support if the problem persists.")
                return

            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name=pdf_filename,
                mime="application/pdf",
            )

    render_app_footer(
        st,
        ENDPLAY_VERSION,
        dependency_versions={
            "pandas": pd.__version__,
            "polars": pl.__version__,
            "duckdb": duckdb.__version__,
        },
    )


if __name__ == "__main__":
    main()


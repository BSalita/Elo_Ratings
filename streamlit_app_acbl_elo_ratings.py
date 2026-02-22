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
import requests
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
# Config / API
# -------------------------------
def _acbl_api_base_url() -> str | None:
    """Return configured ACBL API base URL, if enabled."""
    raw = os.getenv("ACBL_API_BASE_URL", "").strip()
    if not raw:
        return None
    return raw.rstrip("/")


def _fetch_remote_report_table(
    club_or_tournament: str,
    rating_type: str,
    top_n: int,
    min_sessions: int,
    rating_method: str,
    moving_avg_days: int,
    elo_rating_type: str,
    date_from: datetime | None,
    online_filter: str,
) -> tuple[pl.DataFrame, dict]:
    """Fetch pre-aggregated report rows from the ACBL API service."""
    base_url = _acbl_api_base_url()
    if base_url is None:
        raise ValueError("ACBL_API_BASE_URL is not configured")

    params = {
        "club_or_tournament": club_or_tournament.lower(),
        "rating_type": rating_type,
        "top_n": int(top_n),
        "min_sessions": int(min_sessions),
        "rating_method": rating_method,
        "moving_avg_days": int(moving_avg_days),
        "elo_rating_type": elo_rating_type,
        "date_from": None if date_from is None else date_from.isoformat(),
        "online_filter": online_filter,
    }

    timeout_seconds = int(os.getenv("ACBL_API_TIMEOUT_SECONDS", "180"))
    request_url = f"{base_url}/acbl/report"
    try:
        response = requests.get(
            request_url,
            params=params,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.exceptions.RequestException as exc:
        hint = (
            "If Streamlit and API are in different Railway projects, use the API public URL "
            "(https://...) for ACBL_API_BASE_URL instead of *.railway.internal."
        )
        raise RuntimeError(
            f"ACBL API request failed for {request_url}: {exc}. {hint}"
        ) from exc
    rows = payload.get("rows", [])
    return pl.DataFrame(rows), payload


def _fetch_remote_detail_table(
    club_or_tournament: str,
    rating_type: str,
    elo_rating_type: str,
    date_from: datetime | None,
    online_filter: str,
    player_id: str | None = None,
    pair_ids: str | None = None,
) -> pl.DataFrame:
    """Fetch session-level detail rows from the ACBL API service."""
    base_url = _acbl_api_base_url()
    if base_url is None:
        raise ValueError("ACBL_API_BASE_URL is not configured")

    params = {
        "club_or_tournament": club_or_tournament.lower(),
        "rating_type": rating_type,
        "elo_rating_type": elo_rating_type,
        "date_from": None if date_from is None else date_from.isoformat(),
        "online_filter": online_filter,
        "player_id": player_id,
        "pair_ids": pair_ids,
    }

    timeout_seconds = int(os.getenv("ACBL_API_TIMEOUT_SECONDS", "180"))
    request_url = f"{base_url}/acbl/detail"
    try:
        response = requests.get(request_url, params=params, timeout=timeout_seconds)
        response.raise_for_status()
        payload = response.json()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"ACBL API detail request failed for {request_url}: {exc}") from exc

    return pl.DataFrame(payload.get("rows", []))


def _table_cache_keys() -> list[str]:
    return [k for k in st.session_state.keys() if isinstance(k, str) and k.startswith("cached_table_")]


def _clear_table_cache() -> None:
    for k in _table_cache_keys():
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


def _get_cached_report_table_df(
    cache_key: str,
) -> tuple[pl.DataFrame, bool]:
    """Return cached report DataFrame. Raises KeyError if not cached."""
    if cache_key in st.session_state:
        return st.session_state[cache_key], True
    raise KeyError(f"Report not cached for key: {cache_key}")


# (Data is fetched exclusively via the remote ACBL API.)


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
        if 'enable_custom_queries' not in st.session_state:
            st.session_state.enable_custom_queries = False
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
    info_line_1 = f"Query Params:{st.query_params.to_dict()} Environment:{os.getenv('STREAMLIT_ENV','')}"
    info_line_2 = (
        f"Streamlit:{st.__version__} Python:{'.'.join(map(str, sys.version_info[:3]))} "
        f"pandas:{pd.__version__} polars:{pl.__version__} duckdb:{duckdb.__version__} endplay:{ENDPLAY_VERSION}"
    )
    try:
        import psutil

        def _gb(v: int) -> float:
            return v / (1024 ** 3)

        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        cpu_count = os.cpu_count() or 0
        threads = max(4, cpu_count // 2) if cpu_count else 0
        if sm.total > 0:
            swap_str = f"{_gb(sm.used):.2f}/{_gb(sm.total):.2f} GB ({sm.percent:.1f}%)"
        else:
            swap_str = "N/A (swap disabled)"
        info_line_3 = (
            f"Memory: RAM {_gb(vm.used):.2f}/{_gb(vm.total):.2f} GB ({vm.percent:.1f}%) "
            f"• Virtual/Pagefile {swap_str} • CPU/Threads {cpu_count}/{threads}"
        )
    except Exception:
        info_line_3 = "Memory: RAM/Virtual usage unavailable"
    info_line_4 = f"System Current Date: {datetime.now().strftime('%Y-%m-%d')}"

    st.markdown(
        f"""
        <div style="text-align: center;">
            <p>{info_line_1}</p>
            <p>{info_line_2}</p>
            <p>{info_line_3}</p>
            <p>{info_line_4}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
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


def _show_sql_query_block(sql_text: str) -> None:
    """Render SQL in a collapsed expander when SQL visibility is enabled."""
    if not st.session_state.get("show_sql_query", False):
        return
    with st.expander("SQL Query", expanded=False):
        st.code(sql_text, language="sql")


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
    _show_sql_query_block(
        f"""SELECT
  Opponents,
  COUNT(*) AS Boards,
  AVG(Pct) AS Avg_Pct,
  FIRST(Elo_Before) AS Elo_Start,
  LAST(Elo_After) AS Elo_End,
  LAST(Elo_After) - FIRST(Elo_Before) AS Elo_Delta
FROM detail
WHERE Session = '{session_id}'
GROUP BY Opponents
ORDER BY Opponents;"""
    )

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
    _show_sql_query_block(
        """SELECT
  Opponents,
  COUNT(*) AS Boards,
  COUNT(DISTINCT Session) AS Sessions,
  AVG(Pct) AS Avg_Pct,
  AVG(Elo_Delta) AS Avg_Elo_Delta
FROM detail
GROUP BY Opponents
ORDER BY Boards DESC;"""
    )

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

    # API-only mode: require remote API base URL.
    api_base_url = _acbl_api_base_url()
    if not api_base_url:
        st.error(
            "ACBL_API_BASE_URL is required. This Streamlit app now runs in API-only mode. "
            "Set ACBL_API_BASE_URL to your deployed ACBL API service URL."
        )
        st.stop()
    
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
    # Initialize SQL query settings
    if 'show_sql_query' not in st.session_state:
        st.session_state.show_sql_query = False
    if 'enable_custom_queries' not in st.session_state:
        st.session_state.enable_custom_queries = False
    if 'sql_queries' not in st.session_state:
        st.session_state.sql_queries = []

    # -------------------------------
    # Create Sidebar Controls (After Data Loading)
    # -------------------------------

    with st.sidebar:
        # Improve contrast for sidebar expander titles (e.g., Developer Settings)
        st.markdown(
            """
            <style>
            section[data-testid="stSidebar"] details summary p {
                color: #ffc107 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
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
        
        # Developer Settings (kept in sidebar for SQL-gating control)
        with st.sidebar.expander("🔧 **Developer Settings**"):
            show_sql = st.checkbox('Show SQL Query', value=st.session_state.show_sql_query, help='Show SQL used to query dataframes.')
            st.session_state.show_sql_query = show_sql
            enable_custom_queries = st.checkbox(
                "Enable custom queries",
                value=st.session_state.get("enable_custom_queries", False),
                help="Show the 'Run Additional SQL Queries' panel below the results table."
            )
            st.session_state.enable_custom_queries = enable_custom_queries
            
            st.session_state.use_sql_engine = True

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
        if 'sql_query_history' in st.session_state:
            st.session_state.sql_query_history = st.session_state.sql_query_history[-5:]

        # Get data from remote API
        dataset_type = club_or_tournament.lower()
        remote_payload: dict = {}
        remote_table_df: pl.DataFrame | None = None

        try:
            with st.spinner("Fetching pre-aggregated report from ACBL API..."):
                remote_table_df, remote_payload = _fetch_remote_report_table(
                    club_or_tournament=club_or_tournament,
                    rating_type=rating_type,
                    top_n=int(top_n),
                    min_sessions=int(min_sessions),
                    rating_method=rating_method,
                    moving_avg_days=int(moving_avg_days),
                    elo_rating_type=elo_rating_type,
                    date_from=date_from,
                    online_filter=online_filter,
                )
        except Exception as exc:
            st.error(str(exc))
            st.stop()
        date_range = str(remote_payload.get("date_range", "") or "")
        generated_sql = str(remote_payload.get("generated_sql", "") or "")
        st.info(f"✅ Using remote ACBL API ({dataset_type}, {online_filter.lower()} games)")
        perf = remote_payload.get("perf", {}) if isinstance(remote_payload, dict) else {}
        server = remote_payload.get("server", {}) if isinstance(remote_payload, dict) else {}
        if isinstance(perf, dict) and perf:
            if isinstance(server, dict) and server:
                source_mtime = server.get("api_source_mtime", "n/a")
                uptime_seconds = float(server.get("api_uptime_seconds", 0) or 0)
                uptime_minutes = uptime_seconds / 60.0
                if server.get("swap_enabled", False):
                    swap_str = (
                        f"{server.get('swap_used_gb', 0)}/{server.get('swap_total_gb', 0)} GB "
                        f"({server.get('swap_percent', 0)}%)"
                    )
                else:
                    swap_str = "N/A (swap disabled)"
                api_meta_str = (
                    f"api_source_datetime:{source_mtime} | "
                    f"api_uptime:{uptime_minutes:.1f}m ({uptime_seconds:.1f}s)"
                )
                server_resources_str = (
                    f"Memory: RAM {server.get('ram_used_gb', 0)}/{server.get('ram_total_gb', 0)} GB "
                    f"({server.get('ram_percent', 0)}%) • "
                    f"Virtual/Pagefile {swap_str} • "
                    f"CPU/Threads {server.get('cpu_count', 0)}/{server.get('threads', 0)}"
                )
            else:
                api_meta_str = "api_source_datetime:n/a | api_uptime:n/a"
                server_resources_str = "Memory/CPU unavailable"
            st.caption(
                "API performance — "
                f"{api_meta_str} • "
                f"source:{perf.get('source', 'unknown')} • "
                f"parse:{perf.get('parse_seconds', 0)}s • "
                f"load:{perf.get('load_seconds', 0)}s • "
                f"filter:{perf.get('filter_seconds', 0)}s • "
                f"sql:{perf.get('sql_seconds', 0)}s • "
                f"serialize:{perf.get('serialize_seconds', 0)}s • "
                f"total:{perf.get('total_seconds', 0)}s • "
                f"rows in/out:{perf.get('input_rows', '?')}/{perf.get('output_rows', '?')}"
            )
            st.caption(f"Server resources — {server_resources_str}")

        # Store online filter and current dataset type for downstream controls
        st.session_state.online_filter = online_filter
        st.session_state.current_dataset_type = dataset_type

        method_desc = f"{rating_method} method"
        if rating_type == "Players":
            title = f"Top {top_n} ACBL {club_or_tournament} Players by {elo_rating_type} ({method_desc})"
        elif rating_type == "Pairs":
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
                with st.expander("SQL Query", expanded=False):
                    st.code(generated_sql, language='sql')
            
            # 2. Show results from remote API
            cache_key = f"cached_table_{club_or_tournament}_{rating_type}_{top_n}_{min_sessions}_{rating_method}_{moving_avg_days}_{elo_rating_type}_{date_range}_{online_filter}_{st.session_state.get('masterpoints_filter','All')}"
            if remote_table_df is not None:
                st.session_state[cache_key] = remote_table_df
            try:
                table_df, _used_cache = _get_cached_report_table_df(cache_key)
            except KeyError:
                st.error("Report data not available. Please refresh.")
                return
            
            # Display results with exactly 25 viewable rows (common for both paths)
            if 'table_df' in locals():
                st.markdown(f"### 📊 Query Results ({len(table_df)} rows)")
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
                    try:
                        with st.spinner("Loading session history from ACBL API..."):
                            if rating_type == "Players":
                                player_id = str(selected_row.get("Player_ID", ""))
                                player_name = selected_row.get("Player_Name", "Unknown")
                                if not player_id:
                                    st.warning("Missing Player_ID in selected row.")
                                else:
                                    st.markdown(f"#### Session History: **{player_name}** ({player_id})")
                                    detail = _fetch_remote_detail_table(
                                        club_or_tournament=club_or_tournament,
                                        rating_type=rating_type,
                                        elo_rating_type=elo_rating_type,
                                        date_from=date_from,
                                        online_filter=online_filter,
                                        player_id=player_id,
                                    )
                                    if detail.is_empty():
                                        st.info("No session data found for this player.")
                                    else:
                                        n_sessions = detail.select("Session").n_unique()
                                        st.caption(f"{len(detail)} boards across {n_sessions} sessions — click a row to see opponent breakdown")
                                        _show_all_opponents_aggregation(detail, key_suffix=f"player_{player_id}")
                                        _show_sql_query_block(
                                            f"""SELECT *
FROM acbl_detail_api
WHERE rating_type = 'Players'
  AND player_id = '{player_id}'
ORDER BY Date DESC, Session DESC, Round ASC, Board ASC;"""
                                        )
                                        st.markdown("#### Board-by-Board Detail")
                                        detail_grid = _render_detail_aggrid(detail, key=f"detail_player_remote_{player_id}", selectable=True)
                                        if detail_grid is not None:
                                            sel = detail_grid.get("selected_rows", None)
                                            if sel is not None and len(sel) > 0:
                                                sel_row = sel.iloc[0] if hasattr(sel, "iloc") else sel[0]
                                                _show_opponent_aggregation(detail, sel_row)
                            else:
                                pair_ids = str(selected_row.get("Pair_IDs", ""))
                                pair_names = selected_row.get("Pair_Names", "Unknown")
                                if not pair_ids:
                                    st.warning("Missing Pair_IDs in selected row.")
                                else:
                                    st.markdown(f"#### Session History: **{pair_names}**")
                                    detail = _fetch_remote_detail_table(
                                        club_or_tournament=club_or_tournament,
                                        rating_type=rating_type,
                                        elo_rating_type=elo_rating_type,
                                        date_from=date_from,
                                        online_filter=online_filter,
                                        pair_ids=pair_ids,
                                    )
                                    if detail.is_empty():
                                        st.info("No session data found for this pair.")
                                    else:
                                        n_sessions = detail.select("Session").n_unique()
                                        st.caption(f"{len(detail)} boards across {n_sessions} sessions — click a row to see opponent breakdown")
                                        _show_all_opponents_aggregation(detail, key_suffix=f"pair_{pair_ids}")
                                        _show_sql_query_block(
                                            f"""SELECT *
FROM acbl_detail_api
WHERE rating_type = 'Pairs'
  AND pair_ids = '{pair_ids}'
ORDER BY Date DESC, Session DESC, Round ASC, Board ASC;"""
                                        )
                                        st.markdown("#### Board-by-Board Detail")
                                        detail_grid = _render_detail_aggrid(detail, key=f"detail_pair_remote_{pair_ids}", selectable=True)
                                        if detail_grid is not None:
                                            sel = detail_grid.get("selected_rows", None)
                                            if sel is not None and len(sel) > 0:
                                                sel_row = sel.iloc[0] if hasattr(sel, "iloc") else sel[0]
                                                _show_opponent_aggregation(detail, sel_row)
                    except Exception as exc:
                        st.error(f"Session detail API request failed: {exc}")
                
                # Mark that table is displayed (lightweight state only)
                st.session_state.table_displayed = True
            
            # 3. SQL Query Interface for additional queries (only if enabled)
            if st.session_state.get('show_sql_query', False) and st.session_state.get('enable_custom_queries', False):
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
            cache_key = f"cached_table_{club_or_tournament}_{rating_type}_{top_n}_{min_sessions}_{rating_method}_{moving_avg_days}_{elo_rating_type}_{date_range}_{online_filter}_{st.session_state.get('masterpoints_filter','All')}"
            if remote_table_df is not None:
                st.session_state[cache_key] = remote_table_df
            try:
                table_df, _used_cache = _get_cached_report_table_df(cache_key)
                st.info(f"✅ Using cached {rating_type} report for PDF generation ({len(table_df)} rows)")
            except (KeyError, Exception) as e:
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


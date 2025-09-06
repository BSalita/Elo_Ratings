import pathlib
import time
from datetime import datetime, timedelta

import polars as pl
import streamlit as st
import duckdb
from streamlit_extras.bottom_container import bottom

from streamlitlib.streamlitlib import (
    ShowDataFrameTable,
    create_pdf,
    stick_it_good,
    widen_scrollbars,
)

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
    return st.session_state.db_connection

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
            con.register('self', df)
            # Execute the query
            execute_sql_query(df, query, f'sql_query_result_{len(st.session_state.get("sql_queries", []))}')
            # Store query in history
            if 'sql_queries' not in st.session_state:
                st.session_state.sql_queries = []
            st.session_state.sql_queries.append(query)




# -------------------------------
# Config / Paths
# -------------------------------
DATA_ROOT = pathlib.Path('data')


# -------------------------------
# Data Loading
# -------------------------------
def load_elo_ratings_schema(club_or_tournament: str) -> list[str]:
    filename = f'acbl_{club_or_tournament.lower()}_elo_ratings.parquet'
    file_path = DATA_ROOT.joinpath(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")
    df0 = pl.read_parquet(file_path, n_rows=0)
    return df0.columns


def load_elo_ratings_schema_map(club_or_tournament: str) -> dict:
    filename = f'acbl_{club_or_tournament.lower()}_elo_ratings.parquet'
    file_path = DATA_ROOT.joinpath(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")
    df0 = pl.read_parquet(file_path, n_rows=0)
    return df0.schema  # dict[str, pl.DataType]


def load_elo_ratings(
    club_or_tournament: str,
    columns: list[str] | None = None,
    date_from: datetime | None = None,
) -> pl.DataFrame:
    filename = f'acbl_{club_or_tournament.lower()}_elo_ratings.parquet'
    file_path = DATA_ROOT.joinpath(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

    # Inspect schema once
    schema_map = load_elo_ratings_schema_map(club_or_tournament)

    # Lazy scan for performance
    lf = pl.scan_parquet(file_path)

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
    return lf.collect(engine="streaming")


# -------------------------------
# Data Enrichment
# -------------------------------
def enrich_tournament_with_club_masterpoints_direct(tournament_df: pl.DataFrame, club_df: pl.DataFrame) -> pl.DataFrame:
    """
    Enrich tournament data with MasterPoints from club data by joining on Player_IDs.
    Takes both DataFrames directly (used during startup enrichment).
    """
    try:
        # Extract unique player MasterPoints from club data
        # Get the most recent MasterPoints for each player
        club_players = []
        for pos in "NESW":
            if f"Player_ID_{pos}" in club_df.columns and f"MasterPoints_{pos}" in club_df.columns:
                club_players.append(
                    club_df.select([
                        pl.col(f"Player_ID_{pos}").alias("Player_ID"),
                        pl.col(f"MasterPoints_{pos}").alias("MasterPoints"),
                        pl.col("Date")
                    ]).filter(pl.col("Player_ID").is_not_null())
                )
        
        if not club_players:
            return tournament_df
            
        # Combine all club player data and get latest MasterPoints per player
        club_players_combined = pl.concat(club_players, how="vertical")
        latest_masterpoints = (
            club_players_combined
            .sort("Date", descending=True)
            .group_by("Player_ID")
            .agg([
                pl.col("MasterPoints").first().alias("MasterPoints")
            ])
        )
        
        # Join MasterPoints to tournament data for each position
        enriched_df = tournament_df
        for pos in "NESW":
            if f"Player_ID_{pos}" in tournament_df.columns:
                # Join MasterPoints for this position
                enriched_df = enriched_df.join(
                    latest_masterpoints.select([
                        pl.col("Player_ID"),
                        pl.col("MasterPoints").alias(f"MasterPoints_{pos}")
                    ]),
                    left_on=f"Player_ID_{pos}",
                    right_on="Player_ID",
                    how="left"
                )
        
        return enriched_df
        
    except Exception as e:
        st.warning(f"⚠️ Could not enrich tournament data with club MasterPoints: {e}")
        return tournament_df


# -------------------------------
# Computation Helpers
# -------------------------------
def show_top_players(df: pl.DataFrame, top_n: int, min_elo_count: int = 30, rating_method: str = 'Avg') -> pl.DataFrame:
    
    if 'MasterPoints_N' not in df.columns:
        df = df.with_columns([pl.lit(None).alias(f"MasterPoints_{d}") for d in "NESW"])

    # Don't filter out null Player_IDs - process all rows

    position_frames: list[pl.DataFrame] = []
    for d in "NESW":
        # Only process positions that have valid data
        position_frame = df.select(
            pl.col("Date"),
            pl.col("session_id"),
            pl.col(f"Player_ID_{d}").alias("Player_ID"),
            pl.col(f"Player_Name_{d}").alias("Player_Name"),
            pl.col(f"MasterPoints_{d}").alias("MasterPoints"),
            pl.when(pl.col(f"Elo_R_{d}").is_nan()).then(None).otherwise(pl.col(f"Elo_R_{d}")).alias("Elo_R_Player"),
            pl.lit(d).alias("Position"),
        )
        
        position_frames.append(position_frame)

    players_stacked = pl.concat(position_frames, how="vertical").drop_nulls(subset=["Player_ID", "Elo_R_Player"])
    
    # Keep data in memory - don't delete or garbage collect
    del position_frames  # Only delete local references

    if rating_method == 'Avg':
        rating_agg = pl.col('Elo_R_Player').mean().alias('Elo_R_Player')
    elif rating_method == 'Max':
        rating_agg = pl.col('Elo_R_Player').max().alias('Elo_R_Player')
    elif rating_method == 'Latest':
        rating_agg = pl.col('Elo_R_Player').last().alias('Elo_R_Player')
    else:
        raise ValueError(f"Invalid rating method: {rating_method}")

    # Memory optimization: Process aggregation in stages to reduce peak memory
    # Stage 1: Group and aggregate core metrics
    player_aggregates = (
        players_stacked
        .group_by('Player_ID')
        .agg([
            rating_agg,
            pl.col('Player_Name').last().alias('Player_Name'),
            pl.col('MasterPoints').last().alias('MasterPoints'),
            pl.col('session_id').n_unique().alias('Elo_Count'),
            pl.col('Position').n_unique().alias('Positions_Played'),
        ])
        .filter(pl.col('Elo_Count') >= min_elo_count)  # Filter early to reduce data
    )
    
    # Keep data in memory - only delete local reference
    del players_stacked
    
    # Stage 2: Add rankings and final processing on smaller dataset
    top_players = (
        player_aggregates
        .with_columns([
            pl.col('MasterPoints').rank(method='ordinal', descending=True).alias('MasterPoint_Rank')
        ])
        .sort('Elo_R_Player', descending=True, nulls_last=True)
        .select(['Elo_R_Player', 'Player_ID', 'Player_Name', 'MasterPoints', 'MasterPoint_Rank', 'Elo_Count'])
        .head(top_n)  # Limit early to reduce final processing
        .with_row_index(name='Rank', offset=1)
        .select(['Rank', 'Elo_R_Player', 'Player_ID', 'Player_Name', 'MasterPoints', 'MasterPoint_Rank', 'Elo_Count'])
    )
    
    # Keep data in memory - only delete local reference
    del player_aggregates

    top_players = top_players.with_columns([
        pl.col('Elo_R_Player').cast(pl.Int32, strict=False),
        pl.col('MasterPoints').cast(pl.Int32, strict=False),
        pl.col('MasterPoint_Rank').cast(pl.Int32, strict=False),
        pl.col('Elo_Count').alias('Sessions_Played'),
    ]).drop(['Elo_Count']).rename({'Elo_R_Player': 'Player_Elo_Score', 'Rank': 'Player_Elo_Rank'})

    return top_players


def show_top_pairs(df: pl.DataFrame, top_n: int, min_elo_count: int = 30, rating_method: str = 'Avg') -> pl.DataFrame:
    
    if 'MasterPoints_N' not in df.columns:
        df = df.with_columns(
            pl.lit(None).alias('MasterPoints_N'),
            pl.lit(None).alias('MasterPoints_S'),
            pl.lit(None).alias('MasterPoints_E'),
            pl.lit(None).alias('MasterPoints_W')
        )
    
    # Memory optimization: Pre-filter to only rows with valid pair data
    df_filtered = df.filter(
        (pl.col("Elo_R_NS").is_not_null()) | (pl.col("Elo_R_EW").is_not_null())
    )

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
        pl.when(pl.col("Elo_R_NS").is_nan()).then(None).otherwise(pl.col("Elo_R_NS")).alias("Elo_R_Pair"),
        (pl.col("Player_Name_N") + " - " + pl.col("Player_Name_S")).str.replace_all("(swap names)", "", literal=True).alias("Pair_Names"),
        ((pl.col("MasterPoints_N") + pl.col("MasterPoints_S")) / 2).alias("Avg_MPs"),
        (pl.col("MasterPoints_N") * pl.col("MasterPoints_S")).sqrt().alias("Geo_MPs"),
        pl.when(pl.mean_horizontal([pl.col("Elo_R_N"), pl.col("Elo_R_S")]).is_nan())
         .then(None)
         .otherwise(pl.mean_horizontal([pl.col("Elo_R_N"), pl.col("Elo_R_S")]))
         .alias("Avg_Player_Elo_Row"),
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
        pl.when(pl.col("Elo_R_EW").is_nan()).then(None).otherwise(pl.col("Elo_R_EW")).alias("Elo_R_Pair"),
        (pl.col("Player_Name_E") + " - " + pl.col("Player_Name_W")).str.replace_all("(swap names)", "", literal=True).alias("Pair_Names"),
        ((pl.col("MasterPoints_E") + pl.col("MasterPoints_W")) / 2).alias("Avg_MPs"),
        (pl.col("MasterPoints_E") * pl.col("MasterPoints_W")).sqrt().alias("Geo_MPs"),
        pl.when(pl.mean_horizontal([pl.col("Elo_R_E"), pl.col("Elo_R_W")]).is_nan())
         .then(None)
         .otherwise(pl.mean_horizontal([pl.col("Elo_R_E"), pl.col("Elo_R_W")]))
         .alias("Avg_Player_Elo_Row"),
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
        rating_agg = pl.col('Elo_R_Pair').last().alias('Elo_Score')
    else:
        raise ValueError(f"Invalid rating method: {rating_method}")

    # Memory optimization: Process pairs aggregation in stages
    # Stage 1: Group and aggregate core metrics
    pair_aggregates = (
        partnerships_stacked
        .group_by('Pair_IDs')
        .agg([
            rating_agg,
            pl.col('Pair_Names').last().alias('Pair_Names'),
            pl.col('Avg_MPs').mean().alias('Avg_MPs'),
            pl.col('Geo_MPs').mean().alias('Geo_MPs'),
            pl.col('session_id').n_unique().alias('Sessions'),
            pl.col('Player_ID_A').first().alias('Player_ID_A'),
            pl.col('Player_ID_B').first().alias('Player_ID_B'),
            pl.col('Avg_Player_Elo_Row').mean().alias('Avg_Player_Elo'),
        ])
        .filter(pl.col('Sessions') >= min_elo_count)  # Filter early to reduce data
    )
    
    # Keep data in memory - only delete local reference
    del partnerships_stacked
    
    # Stage 2: Add rankings and final processing on smaller dataset
    top_partnerships = (
        pair_aggregates
        .with_columns([
            pl.col('Avg_Player_Elo').rank(method='ordinal', descending=True).alias('Avg_Elo_Rank'),
            pl.col('Avg_MPs').rank(method='ordinal', descending=True).alias('Avg_MPs_Rank'),
            pl.col('Geo_MPs').rank(method='ordinal', descending=True).alias('Geo_MPs_Rank'),
        ])
        .sort('Elo_Score', descending=True, nulls_last=True)
        .select(['Elo_Score', 'Avg_Elo_Rank', 'Pair_IDs', 'Pair_Names', 'Avg_MPs', 'Avg_MPs_Rank', 'Geo_MPs_Rank', 'Sessions'])
        .head(top_n)  # Limit early to reduce final processing
        .with_row_index(name='Pair_Elo_Rank', offset=1)
        .select(['Pair_Elo_Rank', 'Elo_Score', 'Avg_Elo_Rank', 'Pair_IDs', 'Pair_Names', 'Avg_MPs', 'Avg_MPs_Rank', 'Geo_MPs_Rank', 'Sessions'])
    )
    
    # Keep data in memory - only delete local reference
    del pair_aggregates

    top_partnerships = top_partnerships.with_columns([
        pl.col('Elo_Score').cast(pl.Int32, strict=False),
        pl.col('Avg_MPs').cast(pl.Int32, strict=False),
        pl.col('Avg_Elo_Rank').cast(pl.Int32, strict=False),
        pl.col('Geo_MPs_Rank').cast(pl.Int32, strict=False),
    ]).rename({'Elo_Score': 'Pair_Elo_Score'})

    return top_partnerships


# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Unofficial ACBL Elo Ratings", layout="wide")

# -------------------------------
# Main App
# -------------------------------
widen_scrollbars()
st.title("Unofficial ACBL Elo Ratings Playground")
st.caption("An interactive playground for fiddling with ACBL Elo ratings")

stick_it_good()

st.markdown(
    """
    <style>
      .stButton > button {
        background-color: #2E7D32 !important; /* green */
        color: white !important;
        border-color: #2E7D32 !important;
      }
      .stButton > button:hover {
        background-color: #1B5E20 !important; /* darker green */
        border-color: #1B5E20 !important;
        color: white !important;
      }
      .stDownloadButton > button {
        background-color: #2E7D32 !important; /* green */
        color: white !important;
        border-color: #2E7D32 !important;
      }
      .stDownloadButton > button:hover {
        background-color: #1B5E20 !important; /* darker green */
        border-color: #1B5E20 !important;
        color: white !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar will be created after data loading is complete
# Set default values for data loading
date_from = None  # Default to all time for initial loading

# Load all columns initially (sidebar controls will be available after loading)
needed_cols = None  # Load all columns

# -------------------------------
# Data Loading
# -------------------------------

#@st.cache_data(ttl=3600)
def load_and_enrich_datasets(date_from_str: str):
    """Load and enrich datasets."""
    date_from = None if date_from_str == "None" else datetime.fromisoformat(date_from_str)
    
    # Load both datasets
    club_df = load_elo_ratings("club", columns=None, date_from=date_from)
    tournament_df = load_elo_ratings("tournament", columns=None, date_from=date_from)
    
    # Enrich tournament data with club MasterPoints
    enriched_tournament_df = enrich_tournament_with_club_masterpoints_direct(tournament_df, club_df)
    
    return {
        "club": club_df,
        "tournament": enriched_tournament_df
    }

# Convert date_from to string for caching key
date_from_str = "None" if date_from is None else date_from.isoformat()

# Load data
try:
    with st.spinner("Loading datasets and augmenting tournament data..."):
        all_data = load_and_enrich_datasets(date_from_str)
    st.success("✅ Datasets loaded successfully.")
    
    # Store data in session state for SQL queries
    st.session_state.all_data = all_data
    
    # Initialize SQL query settings
    if 'show_sql_query' not in st.session_state:
        st.session_state.show_sql_query = False
    if 'sql_queries' not in st.session_state:
        st.session_state.sql_queries = []
        
except Exception as e:
    st.error(f"❌ Failed to load datasets: {e}")
    st.stop()

# -------------------------------
# Create Sidebar Controls (After Data Loading)
# -------------------------------

with st.sidebar:
    st.header("Controls")
    club_or_tournament = st.selectbox("Dataset", options=["Club", "Tournament"], index=0)
    rating_type = st.radio("Rating type", options=["Players", "Pairs"], index=0, horizontal=False)
    top_n = st.number_input("Top N", min_value=50, max_value=5000, value=1000, step=50)
    min_sessions = st.number_input("Minimum sessions played", min_value=1, max_value=200, value=30, step=1)
    rating_method = st.selectbox("Rating method", options=["Avg", "Max", "Latest"], index=0)
    # Date range quick filter (default All time)
    date_range_choice = st.selectbox(
        "Date range",
        options=["All time", "Last 3 months", "Last 6 months", "Last 1 year", "Last 2 years", "Last 3 years", "Last 4 years", "Last 5 years"],
        index=0,
    )
    
    display_table = st.button("Display Table", type="primary")
    generate_pdf = st.button("Generate PDF", type="primary")
    
    # SQL Query Controls
    st.markdown("---")
    st.markdown("**SQL Query Options**")
    show_sql = st.checkbox('Show SQL Query', value=st.session_state.show_sql_query, help='Show SQL used to query dataframes.')
    st.session_state.show_sql_query = show_sql
    
    # Automated Postmortem Apps
    st.markdown("---")
    st.markdown("**Automated Postmortem Apps**")
    st.markdown("🔗 [ACBL Postmortem](https://acbl.postmortem.chat)")
    st.markdown("🔗 [French ffbridge Postmortem](https://ffbridge.postmortem.chat)")
    #st.markdown("🔗 [BridgeWebs Postmortem](https://bridgewebs.postmortem.chat)")

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
if not (display_table or generate_pdf):
    st.info("Select left sidebar options then click 'Display Table' or 'Generate PDF' button.")
else:
    
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
    df = all_data[dataset_type]
    st.info(f"✅ Using {dataset_type} dataset with {df.height:,} rows")
    
    # Store current dataset type and register with DuckDB for SQL queries
    st.session_state.current_dataset_type = dataset_type
    con = get_db_connection()
    con.register('self', df)

    # Tournament data is already enriched during startup if both datasets were loaded

    # Compute date range for captions
    try:
        date_min, date_max = df.select([pl.col("Date").min().alias("min"), pl.col("Date").max().alias("max")]).row(0)
        date_range = f"{str(date_min)[:10]} to {str(date_max)[:10]}"
    except Exception:
        date_range = ""

    # Compute table and title
    with st.spinner(f"Building {rating_type} report for {dataset_type} data..."):
        if rating_type == "Players":
            table_df = show_top_players(df, int(top_n), int(min_sessions), rating_method)
            title = f"Top {top_n} ACBL {club_or_tournament} Players by Elo Rating ({rating_method} method)"
        elif rating_type == "Pairs":
            table_df = show_top_pairs(df, int(top_n), int(min_sessions), rating_method)
            title = f"Top {top_n} ACBL {club_or_tournament} Pairs by Elo Rating ({rating_method} method)"
        else:
            raise ValueError(f"Invalid rating type: {rating_type}")
    st.success(f"✅ Report built with {table_df.height:,} rows")

    # Show table only when requested
    if display_table:
        if date_range:
            st.subheader(f"{title} From {date_range}")
        else:
            st.subheader(title)
        # Prefer large grid height equal to top_n; fallback if not supported
        try:
            ShowDataFrameTable(table_df, key=f"table-{rating_type}", output_method='aggrid', height_rows=int(top_n))
        except TypeError:
            ShowDataFrameTable(table_df, key=f"table-{rating_type}", output_method='aggrid')

    # PDF generation regardless of whether table is shown
    if generate_pdf:
        
        created_on = time.strftime("%Y-%m-%d")
        pdf_title = f"{title} From {date_range}"
        pdf_filename = f"Unofficial Elo Scores for ACBL {club_or_tournament} MatchPoint Games - Top {top_n} {rating_type} {created_on}.pdf"
        # Try new API with max_rows; fallback to chunking when older API is loaded
        def build_pdf():
            # Enable shrink_to_fit for Pair reports to better fit the wider tables
            shrink_pairs = (rating_type == "Pairs")
            return create_pdf([f"## {pdf_title}", "### Created by https://elo.7nt.info", table_df], pdf_title, max_rows=int(top_n), rows_per_page=(20, 24), shrink_to_fit=shrink_pairs)
        try:
            pdf_bytes = build_pdf()
        except TypeError:
            def build_pdf_fallback():
                assets = [f"## {pdf_title}", "### Created by https://elo.7nt.info"]
                try:
                    if isinstance(table_df, pl.DataFrame):
                        total = min(int(top_n), table_df.height)
                        first = min(20, total)
                        if first > 0:
                            assets.append(table_df.slice(0, first))
                        idx = first
                        while idx < total:
                            take = min(24, total - idx)
                            assets.append(table_df.slice(idx, take))
                            idx += take
                    else:
                        import pandas as pd  # type: ignore
                        total = min(int(top_n), len(table_df))
                        first = min(20, total)
                        if first > 0:
                            assets.append(table_df.iloc[0:first, :])
                        idx = first
                        while idx < total:
                            take = min(24, total - idx)
                            assets.append(table_df.iloc[idx:idx+take, :])
                            idx += take
                except Exception:
                    assets.append(table_df)
                # Try to pass shrink_to_fit for pairs, fallback if not supported
                try:
                    shrink_pairs = (rating_type == "Pairs")
                    return create_pdf(assets, pdf_title, shrink_to_fit=shrink_pairs)
                except TypeError:
                    return create_pdf(assets, pdf_title)

            pdf_bytes = build_pdf_fallback()

        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name=pdf_filename,
            mime="application/pdf",
        )

# -------------------------------
# SQL Query Input (Bottom of Page)
# -------------------------------
with bottom():
    st.chat_input(
        'Enter a SQL query (e.g., SELECT * FROM self ORDER BY Elo_R_Player DESC LIMIT 100)',
        key='sql_query_input',
        on_submit=sql_input_callback
    )

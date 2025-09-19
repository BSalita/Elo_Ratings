import os
import pathlib
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import polars as pl
import streamlit as st
import duckdb
import endplay
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
            "player_pattern": "Elo_R_Player_{pos}_EventStart",  # Elo_R_Player_N_EventStart, etc.
            "pair_ns": "Elo_R_Pair_NS_EventStart",
            "pair_ew": "Elo_R_Pair_EW_EventStart"
        }
    elif elo_rating_type == "Rating at Event End":
        return {
            "player_pattern": "Elo_R_Player_{pos}_EventEnd",  # Elo_R_Player_N_EventEnd, etc.
            "pair_ns": "Elo_R_Pair_NS_EventEnd",
            "pair_ew": "Elo_R_Pair_EW_EventEnd"
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


# -------------------------------
# Computation Helpers
# -------------------------------
def show_top_players(df: pl.DataFrame, top_n: int, min_elo_count: int = 30, rating_method: str = 'Avg', moving_avg_days: int = 10, elo_rating_type: str = "Current Rating (End of Session)") -> pl.DataFrame:
    
    if 'MasterPoints_N' not in df.columns:
        df = df.with_columns([pl.lit(None).alias(f"MasterPoints_{d}") for d in "NESW"])

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
    elif rating_method == 'Moving Avg':
        # Use a rolling average of the last N sessions for moving average
        rating_agg = pl.col('Elo_R_Player').tail(moving_avg_days).mean().alias('Elo_R_Player')
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

    # Force garbage collection to free intermediate DataFrames
    import gc
    gc.collect()
    
    return top_players


def show_top_pairs(df: pl.DataFrame, top_n: int, min_elo_count: int = 30, rating_method: str = 'Avg', moving_avg_days: int = 10, elo_rating_type: str = "Current Rating (End of Session)") -> pl.DataFrame:
    
    if 'MasterPoints_N' not in df.columns:
        df = df.with_columns(
            pl.lit(None).alias('MasterPoints_N'),
            pl.lit(None).alias('MasterPoints_S'),
            pl.lit(None).alias('MasterPoints_E'),
            pl.lit(None).alias('MasterPoints_W')
        )
    
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
        (pl.col("Player_Name_N") + " - " + pl.col("Player_Name_S")).str.replace_all("(swap names)", "", literal=True).alias("Pair_Names"),
        ((pl.col("MasterPoints_N") + pl.col("MasterPoints_S")) / 2).alias("Avg_MPs"),
        (pl.col("MasterPoints_N") * pl.col("MasterPoints_S")).sqrt().alias("Geo_MPs"),
        # Calculate average player Elo if individual ratings are available
        (pl.when(player_elo_n is not None and player_elo_s is not None)
         .then(pl.when(pl.mean_horizontal([pl.col(player_elo_n), pl.col(player_elo_s)]).is_nan())
               .then(None)
               .otherwise(pl.mean_horizontal([pl.col(player_elo_n), pl.col(player_elo_s)])))
         .otherwise(None))
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
        pl.when(pl.col(elo_columns["pair_ew"]).is_nan()).then(None).otherwise(pl.col(elo_columns["pair_ew"])).alias("Elo_R_Pair"),
        (pl.col("Player_Name_E") + " - " + pl.col("Player_Name_W")).str.replace_all("(swap names)", "", literal=True).alias("Pair_Names"),
        ((pl.col("MasterPoints_E") + pl.col("MasterPoints_W")) / 2).alias("Avg_MPs"),
        (pl.col("MasterPoints_E") * pl.col("MasterPoints_W")).sqrt().alias("Geo_MPs"),
        # Calculate average player Elo if individual ratings are available
        (pl.when(player_elo_e is not None and player_elo_w is not None)
         .then(pl.when(pl.mean_horizontal([pl.col(player_elo_e), pl.col(player_elo_w)]).is_nan())
               .then(None)
               .otherwise(pl.mean_horizontal([pl.col(player_elo_e), pl.col(player_elo_w)])))
         .otherwise(None))
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
    elif rating_method == 'Moving Avg':
        # Use a rolling average of the last N sessions for moving average
        rating_agg = pl.col('Elo_R_Pair').tail(moving_avg_days).mean().alias('Elo_Score')
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

    # Force garbage collection to free intermediate DataFrames
    import gc
    gc.collect()
    
    return top_partnerships


# -------------------------------
# SQL Query Generation
# -------------------------------
def generate_top_players_sql(top_n: int, min_sessions: int, rating_method: str, moving_avg_days: int = 10) -> str:
    """Generate SQL query for top players report."""
    
    # Determine aggregation function based on rating method
    if rating_method == 'Avg':
        rating_agg = 'AVG'
    elif rating_method == 'Max':
        rating_agg = 'MAX'
    elif rating_method == 'Latest':
        rating_agg = 'LAST'  # DuckDB supports LAST aggregation
    elif rating_method == 'Moving Avg':
        # Use a window function to get moving average of last 10 sessions
        rating_agg = 'AVG'  # Will be modified in the query to use window function
    else:
        rating_agg = 'AVG'
    
    sql_query = f"""
    WITH player_positions AS (
        -- Extract all player positions from NESW columns
        SELECT Date, session_id, Player_ID_N as Player_ID, Player_Name_N as Player_Name, 
               MasterPoints_N as MasterPoints, Elo_R_N as Elo_R_Player, 'N' as Position
        FROM self WHERE Player_ID_N IS NOT NULL AND Elo_R_N IS NOT NULL AND NOT isnan(Elo_R_N)
        
        UNION ALL
        
        SELECT Date, session_id, Player_ID_E as Player_ID, Player_Name_E as Player_Name, 
               MasterPoints_E as MasterPoints, Elo_R_E as Elo_R_Player, 'E' as Position
        FROM self WHERE Player_ID_E IS NOT NULL AND Elo_R_E IS NOT NULL AND NOT isnan(Elo_R_E)
        
        UNION ALL
        
        SELECT Date, session_id, Player_ID_S as Player_ID, Player_Name_S as Player_Name, 
               MasterPoints_S as MasterPoints, Elo_R_S as Elo_R_Player, 'S' as Position
        FROM self WHERE Player_ID_S IS NOT NULL AND Elo_R_S IS NOT NULL AND NOT isnan(Elo_R_S)
        
        UNION ALL
        
        SELECT Date, session_id, Player_ID_W as Player_ID, Player_Name_W as Player_Name, 
               MasterPoints_W as MasterPoints, Elo_R_W as Elo_R_Player, 'W' as Position
        FROM self WHERE Player_ID_W IS NOT NULL AND Elo_R_W IS NOT NULL AND NOT isnan(Elo_R_W)
    ),
    {('player_recent_sessions AS (' +
      '    SELECT Player_ID, Player_Name, MasterPoints, Elo_R_Player, Position, session_id, Date,' +
      '           ROW_NUMBER() OVER (PARTITION BY Player_ID ORDER BY Date DESC) as rn' +
      '    FROM player_positions' +
      '    QUALIFY rn <= ' + str(moving_avg_days) +
      '),' if rating_method == 'Moving Avg' else '')}
    player_aggregates AS (
        SELECT 
            Player_ID,
            LAST(Player_Name) as Player_Name,
            LAST(MasterPoints) as MasterPoints,
            {f'{rating_agg}(Elo_R_Player)' if rating_method != 'Moving Avg' else 'AVG(Elo_R_Player)'} as Player_Elo_Score,
            COUNT(DISTINCT session_id) as Sessions_Played,
            COUNT(DISTINCT Position) as Positions_Played
        FROM {'player_recent_sessions' if rating_method == 'Moving Avg' else 'player_positions'}
        GROUP BY Player_ID
        HAVING COUNT(DISTINCT session_id) >= {min_sessions}
    )
    SELECT 
        ROW_NUMBER() OVER (ORDER BY CAST(Player_Elo_Score AS INTEGER) DESC, MasterPoints DESC) as Player_Elo_Rank,
        CAST(Player_Elo_Score AS INTEGER) as Player_Elo_Score,
        Player_ID,
        Player_Name,
        CAST(MasterPoints AS INTEGER) as MasterPoints,
        RANK() OVER (ORDER BY MasterPoints DESC) as MasterPoint_Rank,
        Sessions_Played
    FROM player_aggregates
    ORDER BY Player_Elo_Rank ASC
    LIMIT {top_n}
    """
    
    return sql_query.strip()


def generate_top_pairs_sql(top_n: int, min_sessions: int, rating_method: str, moving_avg_days: int = 10) -> str:
    """Generate SQL query for top pairs report."""
    
    # Determine aggregation function based on rating method
    if rating_method == 'Avg':
        rating_agg = 'AVG'
    elif rating_method == 'Max':
        rating_agg = 'MAX'
    elif rating_method == 'Latest':
        rating_agg = 'LAST'
    elif rating_method == 'Moving Avg':
        # Use a window function to get moving average of last 10 sessions
        rating_agg = 'AVG'  # Will be modified in the query to use window function
    else:
        rating_agg = 'AVG'
    
    sql_query = f"""
    WITH pair_partnerships AS (
        -- NS partnerships
        SELECT 
            Date, session_id,
            CASE WHEN Player_ID_N < Player_ID_S 
                 THEN Player_ID_N || '-' || Player_ID_S 
                 ELSE Player_ID_S || '-' || Player_ID_N 
            END as Pair_IDs,
            Player_Name_N || ' - ' || Player_Name_S as Pair_Names,
            Elo_R_NS as Elo_R_Pair,
            (COALESCE(MasterPoints_N, 0) + COALESCE(MasterPoints_S, 0)) / 2.0 as Avg_MPs,
            SQRT(COALESCE(MasterPoints_N, 0) * COALESCE(MasterPoints_S, 0)) as Geo_MPs,
            (COALESCE(Elo_R_N, 0) + COALESCE(Elo_R_S, 0)) / 2.0 as Avg_Player_Elo
        FROM self 
        WHERE Elo_R_NS IS NOT NULL AND NOT isnan(Elo_R_NS)
        
        UNION ALL
        
        -- EW partnerships  
        SELECT 
            Date, session_id,
            CASE WHEN Player_ID_E < Player_ID_W 
                 THEN Player_ID_E || '-' || Player_ID_W 
                 ELSE Player_ID_W || '-' || Player_ID_E 
            END as Pair_IDs,
            Player_Name_E || ' - ' || Player_Name_W as Pair_Names,
            Elo_R_EW as Elo_R_Pair,
            (COALESCE(MasterPoints_E, 0) + COALESCE(MasterPoints_W, 0)) / 2.0 as Avg_MPs,
            SQRT(COALESCE(MasterPoints_E, 0) * COALESCE(MasterPoints_W, 0)) as Geo_MPs,
            (COALESCE(Elo_R_E, 0) + COALESCE(Elo_R_W, 0)) / 2.0 as Avg_Player_Elo
        FROM self 
        WHERE Elo_R_EW IS NOT NULL AND NOT isnan(Elo_R_EW)
    ),
    {('pair_recent_sessions AS (' +
      '    SELECT Pair_IDs, Pair_Names, Elo_R_Pair, Avg_MPs, Geo_MPs, Avg_Player_Elo, session_id, Date,' +
      '           ROW_NUMBER() OVER (PARTITION BY Pair_IDs ORDER BY Date DESC) as rn' +
      '    FROM pair_partnerships' +
      '    QUALIFY rn <= ' + str(moving_avg_days) +
      '),' if rating_method == 'Moving Avg' else '')}
    pair_aggregates AS (
        SELECT 
            Pair_IDs,
            LAST(Pair_Names) as Pair_Names,
            {f'{rating_agg}(Elo_R_Pair)' if rating_method != 'Moving Avg' else 'AVG(Elo_R_Pair)'} as Pair_Elo_Score,
            AVG(Avg_MPs) as Avg_MPs,
            AVG(Geo_MPs) as Geo_MPs,
            COUNT(DISTINCT session_id) as Sessions,
            AVG(Avg_Player_Elo) as Avg_Player_Elo
        FROM {'pair_recent_sessions' if rating_method == 'Moving Avg' else 'pair_partnerships'}
        GROUP BY Pair_IDs
        HAVING COUNT(DISTINCT session_id) >= {min_sessions}
    )
    SELECT 
        ROW_NUMBER() OVER (ORDER BY CAST(Pair_Elo_Score AS INTEGER) DESC, Avg_MPs DESC) as Pair_Elo_Rank,
        CAST(Pair_Elo_Score AS INTEGER) as Pair_Elo_Score,
        RANK() OVER (ORDER BY Avg_Player_Elo DESC) as Avg_Elo_Rank,
        Pair_IDs,
        Pair_Names,
        CAST(Avg_MPs AS INTEGER) as Avg_MPs,
        RANK() OVER (ORDER BY Avg_MPs DESC) as Avg_MPs_Rank,
        RANK() OVER (ORDER BY Geo_MPs DESC) as Geo_MPs_Rank,
        Sessions
    FROM pair_aggregates
    ORDER BY Pair_Elo_Score DESC, Avg_MPs_Rank ASC
    LIMIT {top_n}
    """
    
    return sql_query.strip()


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
    else:
        st.session_state.first_time = False


# -------------------------------
# UI
# -------------------------------

def app_info() -> None:
    """Display app information"""
    st.caption(f"Project lead is Robert Salita research@AiPolice.org. Code written in Python. UI written in streamlit. Data engine is polars. Query engine is duckdb. Bridge lib is endplay. Self hosted using Cloudflare Tunnel. Repo:https://github.com/BSalita")
    st.caption(f"App:{st.session_state.app_datetime} Streamlit:{st.__version__} Query Params:{st.query_params.to_dict()} Environment:{os.getenv('STREAMLIT_ENV','')}")
    st.caption(f"Python:{'.'.join(map(str, sys.version_info[:3]))} pandas:{pd.__version__} polars:{pl.__version__} endplay:{endplay.__version__}")
    return


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # UI Configuration
    st.set_page_config(page_title="Unofficial ACBL Elo Ratings", layout="wide")
    
    # -------------------------------
    # Main App
    # -------------------------------
    widen_scrollbars()
    st.title("Unofficial ACBL Elo Ratings Playground")
    st.caption("An interactive playground for fiddling with Unofficial ACBL Elo ratings")
    app_info()
    
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
        
        return {
            "club": club_df,
            "tournament": tournament_df
        }

    # Convert date_from to string for caching key
    date_from_str = "None" if date_from is None else date_from.isoformat()

    # Load data only if not already loaded in session state
    if 'all_data' not in st.session_state or 'data_date_from' not in st.session_state or st.session_state.data_date_from != date_from_str:
        try:
            with st.spinner("Loading club and tournament datasets..."):
                all_data = load_and_enrich_datasets(date_from_str)
            st.success("✅ Datasets loaded successfully.")
            
            # Store data in session state for SQL queries
            st.session_state.all_data = all_data
            st.session_state.data_date_from = date_from_str
            
        except Exception as e:
            st.error(f"❌ Failed to load datasets: {e}")
            st.stop()
    else:
        # Data already loaded, use cached version
        all_data = st.session_state.all_data
    
    # Initialize SQL query settings
    if 'show_sql_query' not in st.session_state:
        st.session_state.show_sql_query = False
    if 'sql_queries' not in st.session_state:
        st.session_state.sql_queries = []

    # -------------------------------
    # Create Sidebar Controls (After Data Loading)
    # -------------------------------

    with st.sidebar:
        st.header("Controls")
        club_or_tournament = st.selectbox("Dataset", options=["Club", "Tournament"], index=0)
        rating_type = st.radio("Rating type", options=["Players", "Pairs"], index=0, horizontal=False)
        top_n = st.number_input("Top N", min_value=50, max_value=5000, value=1000, step=50)
        min_sessions = st.number_input("Minimum sessions played", min_value=1, max_value=200, value=30, step=1)
        rating_method = st.selectbox("Rating method", options=["Avg", "Max", "Latest", "Moving Avg"], index=0)
        
        # Moving average days control (only show when Moving Avg is selected)
        if rating_method == "Moving Avg":
            moving_avg_days = st.number_input("Moving average sessions", min_value=1, max_value=50, value=10, step=1, 
                                            help="Number of most recent sessions to include in the moving average")
        else:
            moving_avg_days = 10  # Default value when not using Moving Avg
        
        # Elo rating type selector
        elo_rating_type = st.selectbox("Elo rating type", options=[
            "Current Rating (End of Session)",
            "Rating at Start of Session", 
            "Rating at Event Start",
            "Rating at Event End",
            "Expected Rating"
        ], index=0, help="Choose which Elo rating statistic to analyze")
        
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
        if 'previous_table_df' in st.session_state:
            del st.session_state.previous_table_df
        if 'cached_table_df' in st.session_state:
            del st.session_state.cached_table_df
        if 'cached_table_settings' in st.session_state:
            del st.session_state.cached_table_settings
        if 'sql_query_history' in st.session_state:
            st.session_state.sql_query_history = []
        gc.collect()
    
    # Only show main content when buttons are explicitly clicked
    show_main_content = display_table or generate_pdf
    
    if not show_main_content:
        st.info("Select left sidebar options then click 'Display Table' or 'Generate PDF' button.")
    else:
        # Memory cleanup - clear previous results and force garbage collection
        import gc
        
        # Clear any previous table data from session state (but preserve cached table if settings haven't changed)
        if 'previous_table_df' in st.session_state:
            del st.session_state.previous_table_df
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
        df = all_data[dataset_type]
        st.info(f"✅ Using {dataset_type} dataset with {len(df):,} rows")
        
        # Store current dataset type and register with DuckDB for SQL queries
        st.session_state.current_dataset_type = dataset_type
        con = get_db_connection()
        con.register('self', df)

        # Compute date range for captions
        try:
            date_min, date_max = df.select([pl.col("Date").min().alias("min"), pl.col("Date").max().alias("max")]).row(0)
            date_range = f"{str(date_min)[:10]} to {str(date_max)[:10]}"
        except Exception:
            date_range = ""

        # Generate SQL query based on report type
        if rating_type == "Players":
            generated_sql = generate_top_players_sql(int(top_n), int(min_sessions), rating_method, moving_avg_days)
            method_desc = f"{rating_method} method" if rating_method != "Moving Avg" else f"Moving Avg ({moving_avg_days} sessions) method"
            title = f"Top {top_n} ACBL {club_or_tournament} Players by {elo_rating_type} ({method_desc})"
        elif rating_type == "Pairs":
            generated_sql = generate_top_pairs_sql(int(top_n), int(min_sessions), rating_method, moving_avg_days)
            method_desc = f"{rating_method} method" if rating_method != "Moving Avg" else f"Moving Avg ({moving_avg_days} sessions) method"
            title = f"Top {top_n} ACBL {club_or_tournament} Pairs by {elo_rating_type} ({method_desc})"
        else:
            raise ValueError(f"Invalid rating type: {rating_type}")
        
        # Store the generated SQL for the display table functionality
        st.session_state.generated_sql = generated_sql
        st.session_state.report_title = title

        # Show SQL-based interface when requested or if previously displayed
        show_table = display_table or (st.session_state.get('table_displayed', False) and not generate_pdf)
        if show_table:
            if date_range:
                st.subheader(f"{title} From {date_range}")
            else:
                st.subheader(title)
            
            # 1. Show the SQL query used in a compact scrollable container (only if enabled)
            if st.session_state.get('show_sql_query', False):
                st.markdown("### 📝 SQL Query Used")
                st.text_area("", value=generated_sql, height=200, disabled=True, label_visibility="collapsed")
            
            # 2. Execute and show results in a 25-row scrollable table
            # Use Polars implementation for Moving Avg (much faster than SQL)
            if rating_method == "Moving Avg":
                with st.spinner(f"Building {rating_type} report using optimized method."):
                    if rating_type == "Players":
                        table_df = show_top_players(df, int(top_n), int(min_sessions), rating_method, moving_avg_days, elo_rating_type)
                    elif rating_type == "Pairs":
                        table_df = show_top_pairs(df, int(top_n), int(min_sessions), rating_method, moving_avg_days, elo_rating_type)
                    st.success(f"✅ Report generated successfully! Returned {len(table_df)} rows.")
            else:
                with st.spinner(f"Executing SQL query for {rating_type} report."):
                    try:
                        con = get_db_connection()
                        con.register('self', df)
                        table_df = con.execute(generated_sql).pl()
                        st.success(f"✅ Query executed successfully! Returned {len(table_df)} rows.")
                    except Exception as e:
                        st.error(f"❌ SQL Query Failed: {e}")
                        st.error("Please try again or contact support if the problem persists.")
                        return
            
            # Display results with exactly 25 viewable rows (common for both paths)
            if 'table_df' in locals():
                st.markdown("### 📊 Query Results")
                
                # Convert to pandas for AgGrid
                if hasattr(table_df, 'to_pandas'):
                    display_df = table_df.to_pandas()
                else:
                    display_df = table_df
                
                # Use AgGrid directly with precise height control for exactly 25 rows
                from st_aggrid import GridOptionsBuilder, AgGrid, AgGridTheme
                
                gb = GridOptionsBuilder.from_dataframe(display_df)
                gb.configure_default_column(cellStyle={'color': 'black', 'font-size': '12px'}, suppressMenu=True, wrapHeaderText=True, autoHeaderHeight=True)
                # Don't configure pagination - we want scrolling instead
                gb.configure_side_bar()
                gridOptions = gb.build()
                
                # Configure for scrolling with fixed height
                gridOptions['rowHeight'] = 28
                gridOptions['suppressPaginationPanel'] = True
                gridOptions['alwaysShowVerticalScroll'] = True  # Force vertical scrollbar
                gridOptions['suppressHorizontalScroll'] = False  # Allow horizontal scrollbar if needed
                gridOptions['domLayout'] = 'normal'  # Use normal layout (not autoHeight)
                
                # Calculate height for exactly 25 rows: header(50) + 25*row_height(28) + scrollbar(20)
                exact_height = 50 + 25 * 28 + 20  # = 770px
                
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
                
                AgGrid(
                    display_df,
                    gridOptions=gridOptions,
                    height=exact_height,
                    theme=AgGridTheme.BALHAM,
                    custom_css=custom_css,
                    key=f"table-{rating_type}"
                )
                
                # Store table_df in session state for cleanup on next run
                st.session_state.previous_table_df = table_df
                
                # Cache table_df and settings for reuse in PDF generation
                st.session_state.cached_table_df = table_df
                st.session_state.cached_table_settings = current_settings.copy()
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
                if display_table and st.session_state.get('generated_sql'):
                    if not any(h['query'] == st.session_state.generated_sql for h in st.session_state.sql_query_history):
                        st.session_state.sql_query_history.append({
                            'query': st.session_state.generated_sql,
                            'result': table_df,
                            'timestamp': datetime.now().strftime('%H:%M:%S'),
                            'auto_generated': True
                        })
                
                # SQL Query input - pre-populate with generated query
                default_query = st.session_state.get('generated_sql', '') if display_table else ''
                query = st.text_area(
                    "Enter SQL Query:",
                    value=default_query,
                    placeholder="SELECT * FROM self WHERE Player_Elo_Score > 1500 ORDER BY Player_Elo_Rank LIMIT 10",
                    height=150,
                    key="sql_query_text_area"
                )
                
                col1, col2, col3 = st.columns([1, 1, 4])
                with col1:
                    execute_query = st.button("Execute Query", type="primary")
                with col2:
                    clear_results = st.button("Clear Results")
                
                # Execute query when button is pressed
                if execute_query and query.strip():
                    try:
                        # Show the query being executed
                        if st.session_state.get('show_sql_query', False):
                            st.code(query, language='sql')
                        
                        # Process query
                        processed_query = query.strip()
                        if 'from ' not in processed_query.lower():
                            processed_query = 'FROM self ' + processed_query
                        
                        # Execute query on the query results table, not the raw dataset
                        con = get_db_connection()
                        con.register('self', table_df)  # Register the query results dataframe
                        result_df = con.execute(processed_query).pl()
                        
                        # Store in history
                        st.session_state.sql_query_history.append({
                            'query': query,
                            'result': result_df,
                            'timestamp': datetime.now().strftime('%H:%M:%S')
                        })
                        
                        st.success(f"✅ Query executed successfully! Returned {len(result_df)} rows.")
                        
                    except Exception as e:
                        st.error(f"❌ SQL Error: {e}")
                        st.code(query, language='sql')
                
                # Clear results when button is pressed
                if clear_results:
                    st.session_state.sql_query_history = []
                    st.success("🗑️ Query results cleared!")
                
                # Display additional query results (excluding the auto-generated one already shown above)
                additional_queries = [q for q in st.session_state.sql_query_history if not q.get('auto_generated', False)]
                if additional_queries:
                    st.markdown("### 📋 Additional Query Results")
                    
                    # Show results in reverse order (most recent first)
                    for i, query_result in enumerate(reversed(additional_queries)):
                        query_label = f"Query {len(additional_queries) - i} ({query_result['timestamp']})"
                        
                        with st.expander(query_label, expanded=(i == 0)):
                            st.code(query_result['query'], language='sql')
                            st.caption(f"Returned {len(query_result['result'])} rows")
                            ShowDataFrameTable(
                                query_result['result'], 
                                key=f"sql_result_{len(additional_queries) - i}",
                                output_method='aggrid',
                                height_rows=25
                            )
            

        # PDF generation regardless of whether table is shown
        if generate_pdf:
            # Check if we can reuse cached table data from Display Table
            cached_df = st.session_state.get('cached_table_df')
            cached_settings = st.session_state.get('cached_table_settings')
            
            if (cached_df is not None and 
                cached_settings is not None and 
                cached_settings == current_settings):
                # Reuse cached dataframe
                table_df = cached_df
                st.info("✅ Using cached table data from Display Table for PDF generation.")
            else:
                # Generate table data for PDF - use same logic as display table
                # Use Polars implementation for Moving Avg (much faster than SQL)
                if rating_method == "Moving Avg":
                    with st.spinner(f"Generating {rating_type} data for PDF using optimized method..."):
                        if rating_type == "Players":
                            table_df = show_top_players(df, int(top_n), int(min_sessions), rating_method, moving_avg_days, elo_rating_type)
                        elif rating_type == "Pairs":
                            table_df = show_top_pairs(df, int(top_n), int(min_sessions), rating_method, moving_avg_days, elo_rating_type)
                else:
                    with st.spinner(f"Generating {rating_type} data for PDF..."):
                        try:
                            con = get_db_connection()
                            con.register('self', df)
                            table_df = con.execute(generated_sql).pl()
                        except Exception as e:
                            st.error(f"❌ PDF Generation Failed: {e}")
                            st.error("Unable to generate PDF. Please try again or contact support if the problem persists.")
                            return
            
            created_on = time.strftime("%Y-%m-%d")
            #pdf_title = f"{title} From {date_range}"
            pdf_filename = f"Unofficial Elo Scores for ACBL {club_or_tournament} MatchPoint Games - Top {top_n} {rating_type} {created_on}.pdf"
            # Generate PDF
            try:
                # Enable shrink_to_fit for Pair reports to better fit the wider tables
                shrink_pairs = (rating_type == "Pairs")
                # really want title, from date to be centered with reduced line spacing between them.
                pdf_bytes = create_pdf([f"## {title}", f"### From {date_range}", "### Created by https://elo.7nt.info", table_df], title, max_rows=int(top_n), rows_per_page=(19, 24), shrink_to_fit=shrink_pairs)
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


if __name__ == "__main__":
    main()

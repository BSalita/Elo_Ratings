import pathlib
import time
import uuid
import json
from datetime import datetime, timedelta

import polars as pl
import streamlit as st
import threading

from streamlitlib.streamlitlib import (
    ShowDataFrameTable,
    create_pdf,
    stick_it_good,
    widen_scrollbars,
)

# -------------------------------
# Singleton Queue Management
# -------------------------------

QUEUE_FILE = pathlib.Path('.queue_state.json')
SESSION_TIMEOUT = 600  # 10 minutes in seconds

def get_user_id():
    """Get or create a unique user ID for this session."""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    return st.session_state.user_id

def load_queue_state():
    """Load the current queue state from file."""
    if not QUEUE_FILE.exists():
        return {
            'active_user': None,
            'active_since': None,
            'queue': []
        }
    
    try:
        with open(QUEUE_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {
            'active_user': None,
            'active_since': None,
            'queue': []
        }

def save_queue_state(state):
    """Save the queue state to file."""
    try:
        with open(QUEUE_FILE, 'w') as f:
            json.dump(state, f)
    except Exception:
        pass

def cleanup_expired_session(state):
    """Remove active user if their session has expired."""
    if state['active_user'] and state['active_since']:
        try:
            active_since = datetime.fromisoformat(state['active_since'])
            if (datetime.now() - active_since).total_seconds() > SESSION_TIMEOUT:
                # Session expired, kick out the user
                state['active_user'] = None
                state['active_since'] = None
                return True
        except Exception:
            # Invalid timestamp, clear it
            state['active_user'] = None
            state['active_since'] = None
            return True
    return False

def get_queue_position(user_id, state):
    """Get the position of user in queue (0-based, -1 if not in queue)."""
    try:
        return state['queue'].index(user_id)
    except ValueError:
        return -1

def manage_user_access():
    """Manage user access to the singleton app."""
    user_id = get_user_id()
    
    # Load current state
    state = load_queue_state()
    
    # Clean up expired sessions
    session_expired = cleanup_expired_session(state)
    
    # Check if this user is the active user
    if state['active_user'] == user_id:
        # Update last activity time
        state['active_since'] = datetime.now().isoformat()
        save_queue_state(state)
        return True, 0, state
    
    # Check if no one is active or session expired
    if not state['active_user'] or session_expired:
        # Make this user active
        state['active_user'] = user_id
        state['active_since'] = datetime.now().isoformat()
        # Remove from queue if they were waiting
        if user_id in state['queue']:
            state['queue'].remove(user_id)
        save_queue_state(state)
        return True, 0, state
    
    # User needs to wait in queue
    if user_id not in state['queue']:
        state['queue'].append(user_id)
        save_queue_state(state)
    
    position = get_queue_position(user_id, state)
    return False, position + 1, state

def promote_next_user():
    """Promote the next user in queue to active status."""
    state = load_queue_state()
    
    if state['queue']:
        # Promote first user in queue
        next_user = state['queue'].pop(0)
        state['active_user'] = next_user
        state['active_since'] = datetime.now().isoformat()
        save_queue_state(state)

def leave_queue():
    """Remove current user from queue and promote next user if they were active."""
    user_id = get_user_id()
    state = load_queue_state()
    
    was_active = state['active_user'] == user_id
    
    # Remove from active or queue
    if state['active_user'] == user_id:
        state['active_user'] = None
        state['active_since'] = None
    
    if user_id in state['queue']:
        state['queue'].remove(user_id)
    
    # If the active user left, promote next in queue
    if was_active and state['queue']:
        next_user = state['queue'].pop(0)
        state['active_user'] = next_user
        state['active_since'] = datetime.now().isoformat()
    
    save_queue_state(state)


# -------------------------------
# Progress helper
# -------------------------------
def run_with_progress(label: str, seconds_hint: int, work_fn):
    container = st.container()
    msg = container.info(label)
    bar = container.progress(0)
    done = threading.Event()
    holder = {}

    def worker():
        try:
            holder['result'] = work_fn()
        except Exception as ex:
            holder['error'] = ex
        finally:
            done.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    start_t = time.time()
    while not done.is_set():
        elapsed = time.time() - start_t
        pct = 100 if seconds_hint <= 0 else min(100, int((elapsed / seconds_hint) * 100))
        bar.progress(pct)
        time.sleep(1)

    # Update to 100% when finished (even if earlier than hint)
    bar.progress(100)
    time.sleep(0.2)  # Brief pause to show 100%
    
    # Clear progress bar and message when finished
    try:
        msg.empty()  # Clear the info message
        bar.empty()  # Clear the progress bar
    except:
        pass
    container.empty()  # Clear the entire container
    
    if 'error' in holder:
        raise holder['error']
    return holder.get('result')

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
st.set_page_config(page_title="UnofficialACBL Elo Ratings", layout="wide")

# -------------------------------
# Queue Management Check
# -------------------------------
is_active, queue_position, queue_state = manage_user_access()

if not is_active:
    # User is in queue, show waiting screen
    st.title("🕐 Please Wait - App in Use")
    st.warning(f"**You are #{queue_position} in the queue**")
    
    # Calculate estimated wait time
    estimated_wait_minutes = (queue_position - 1) * 10  # 10 minutes per person ahead
    
    if queue_position == 1:
        st.info("You're next! The app will be available when the current user finishes or after their 10-minute session expires.")
        
        # Show remaining time for current user if available
        if queue_state['active_since']:
            try:
                active_since = datetime.fromisoformat(queue_state['active_since'])
                current_user_remaining = SESSION_TIMEOUT - (datetime.now() - active_since).total_seconds()
                current_user_remaining_seconds = max(0, current_user_remaining)
                current_user_remaining_minutes = current_user_remaining_seconds / 60
                
                # Progress bar for current user's remaining time
                progress_value = max(0, min(1.0, current_user_remaining_seconds / SESSION_TIMEOUT))
                st.progress(progress_value)
                
                if current_user_remaining_seconds > 60:
                    st.markdown(f"⏱️ **Current user: {current_user_remaining_minutes:.1f} minutes ({int(current_user_remaining_seconds)} seconds) remaining**")
                else:
                    st.markdown(f"⏱️ **Current user: {int(current_user_remaining_seconds)} seconds remaining**")
            except:
                pass
    else:
        st.info(f"There are {queue_position - 1} people ahead of you. Each user gets up to 10 minutes.")
        st.markdown(f"⏰ **Estimated wait time: {estimated_wait_minutes} minutes** (assuming each person uses their full 10 minutes)")
    
    # Show current queue status
    total_in_queue = len(queue_state['queue'])
    if total_in_queue > 0:
        st.markdown(f"**Total users waiting:** {total_in_queue}")
    
    # Auto-refresh every 5 seconds to check queue status
    time.sleep(5)
    st.rerun()

# User is active, show the app
widen_scrollbars()
st.title("Unofficial ACBL Elo Ratings Playground")
st.caption("An interactive playground for fiddling with ACBL Elo ratings")

# Welcome message for new sessions
if 'welcomed' not in st.session_state:
    st.balloons()
    st.info("🎯 **Welcome to your 10-minute playground session!** Explore player rankings, pair statistics, generate PDFs, and discover insights in the ACBL Elo ratings data. Make the most of your time!")
    st.session_state.welcomed = True

# Check if session has expired
remaining_time = SESSION_TIMEOUT - (datetime.now() - datetime.fromisoformat(queue_state['active_since'])).total_seconds()
remaining_seconds = max(0, remaining_time)
remaining_minutes = remaining_seconds / 60

if remaining_time <= 0:
    # Session expired, kick out user
    leave_queue()
    # Clear session state but preserve user_id
    user_id = st.session_state.get('user_id')
    for key in list(st.session_state.keys()):
        if key != 'user_id':  # Keep user_id to maintain identity
            del st.session_state[key]
    st.error("⏰ Your 10-minute session has expired. Redirecting to queue...")
    time.sleep(3)
    st.rerun()

# Progress bar for remaining session time
progress_value = max(0, min(1.0, remaining_seconds / SESSION_TIMEOUT))
st.progress(progress_value)

# Show session info with seconds
if remaining_minutes <= 1:
    st.warning(f"⚠️ **Session ending soon!** {int(remaining_seconds)} seconds remaining to explore the playground")
elif remaining_seconds > 60:
    st.success(f"🎉 **Welcome to the playground!** You have {remaining_minutes:.1f} minutes ({int(remaining_seconds)} seconds) to explore all the Elo ratings features")
else:
    st.success(f"🎉 **Welcome to the playground!** You have {int(remaining_seconds)} seconds to explore all the Elo ratings features")

# Add leave button
if st.button("🚪 Leave App (Let Next Person In)", type="secondary"):
    leave_queue()
    # Clear session state but preserve user_id
    user_id = st.session_state.get('user_id')
    for key in list(st.session_state.keys()):
        if key != 'user_id':  # Keep user_id to maintain identity
            del st.session_state[key]
    st.success("👋 Thanks for using the app! Next user should be promoted automatically!")
    time.sleep(3)
    st.rerun()

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
# Preload Data Files on App Start (Multi-User Safe)
# -------------------------------

@st.cache_data(ttl=3600, show_spinner=False)  # Disable built-in spinner, we'll use our progress bar
def load_and_enrich_datasets(date_from_str: str):
    """Load and enrich datasets with multi-user safe caching."""
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

# Check if data needs loading (use cache key to avoid multi-user conflicts)
cache_key = f"datasets_{date_from_str}"

# Try to load data first (this handles concurrency properly)
try:
    # Check if we need to show progress (data not in cache)
    if cache_key not in st.session_state:
        # Show progress bar for loading
        all_data = run_with_progress(
            "Loading datasets and augmenting tournament data... This may take up to 2 minutes.",
            120,  # 2 minutes
            lambda: load_and_enrich_datasets(date_from_str)
        )
        
        # Store in session state for this user
        st.session_state[cache_key] = all_data
        st.session_state["cached_df_club"] = all_data["club"]
        st.session_state["cached_df_tournament"] = all_data["tournament"]
        st.session_state["cached_date_club"] = date_from_str
        st.session_state["cached_date_tournament"] = date_from_str
        
        # Show completion message only for users who just loaded
        st.success("✅ Datasets loading and augmentation completed.")
    else:
        # Data already loaded, just get it from session state
        all_data = st.session_state[cache_key]
    
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
    
    # Automated Postmortem Apps
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
    st.info("Select options and hit the 'Display Table' or 'Generate PDF' button.")
else:
    # PDF generation will show progress through run_with_progress function
    
    # Build options signature for reuse
    opts = {
        "club_or_tournament": club_or_tournament,
        "rating_type": rating_type,
        "top_n": int(top_n),
        "min_sessions": int(min_sessions),
        "rating_method": rating_method,
        "date_from": None if date_from is None else date_from.isoformat(),
    }

    # Get preloaded data
    dataset_type = club_or_tournament.lower()
    dataset_cache_key = f"cached_df_{dataset_type}"
    
    # Generate report (no caching needed since data is pre-loaded)
    # Use preloaded data
    if dataset_cache_key in st.session_state:
        df = st.session_state[dataset_cache_key]
        st.info(f"✅ Using preloaded {dataset_type} dataset with {df.height:,} rows")
    else:
        st.error(f"❌ {dataset_type.title()} dataset not preloaded. Please refresh the page.")
        st.stop()

    # Tournament data is already enriched during startup if both datasets were loaded

    # Compute date range for captions
    try:
        date_min, date_max = df.select([pl.col("Date").min().alias("min"), pl.col("Date").max().alias("max")]).row(0)
        date_range = f"{str(date_min)[:10]} to {str(date_max)[:10]}"
    except Exception:
        date_range = ""

    # Compute table and title with progress (60s hint)
    def build_table():
        st.info(f"🔄 Building {rating_type} report for {dataset_type} data...")
        if rating_type == "Players":
            tbl = show_top_players(df, int(top_n), int(min_sessions), rating_method)
            ttl = f"Top {top_n} ACBL {club_or_tournament} Players by Elo Rating ({rating_method} method)"
        elif rating_type == "Pairs":
            tbl = show_top_pairs(df, int(top_n), int(min_sessions), rating_method)
            ttl = f"Top {top_n} ACBL {club_or_tournament} Pairs by Elo Rating ({rating_method} method)"
        else:
            raise ValueError(f"Invalid rating type: {rating_type}")
        st.info(f"✅ Report built with {tbl.height:,} rows")
        return tbl, ttl

    seconds_hint = 60 if club_or_tournament == 'Club' else 10
    table_df, title = run_with_progress(
        f"Creating Report Table. Takes up to {seconds_hint} seconds.",
        seconds_hint,
        build_table,
    )

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

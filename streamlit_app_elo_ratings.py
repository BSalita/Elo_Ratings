import pathlib
import time
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
    
    # Clear message when finished
    container.empty()
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
    filename = f'acbl_{club_or_tournament}_elo_ratings.parquet'
    file_path = DATA_ROOT.joinpath(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")
    df0 = pl.read_parquet(file_path, n_rows=0)
    return df0.columns


def load_elo_ratings_schema_map(club_or_tournament: str) -> dict:
    filename = f'acbl_{club_or_tournament}_elo_ratings.parquet'
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
    filename = f'acbl_{club_or_tournament}_elo_ratings.parquet'
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

    # Collect with streaming enabled
    return lf.collect(streaming=True)


# -------------------------------
# Computation Helpers
# -------------------------------
def show_top_players(df: pl.DataFrame, top_n: int, min_elo_count: int = 30, rating_method: str = 'mean') -> pl.DataFrame:
    if 'MasterPoints_N' not in df.columns:
        df = df.with_columns([pl.lit(None).alias(f"MasterPoints_{d}") for d in "NESW"])

    position_frames: list[pl.DataFrame] = []
    for d in "NESW":
        position_frames.append(
            df.select(
                pl.col("Date"),
                pl.col("session_id"),
                pl.col(f"Player_ID_{d}").alias("Player_ID"),
                pl.col(f"Player_Name_{d}").alias("Player_Name"),
                pl.col(f"MasterPoints_{d}").alias("MasterPoints"),
                pl.when(pl.col(f"Elo_R_{d}").is_nan()).then(None).otherwise(pl.col(f"Elo_R_{d}")).alias("Elo_R_Player"),
                pl.lit(d).alias("Position"),
            )
        )
    players_stacked = pl.concat(position_frames, how="vertical").drop_nulls(subset=["Player_ID", "Elo_R_Player"])  # drop rows with missing ID or rating

    if rating_method == 'max':
        rating_agg = pl.col('Elo_R_Player').max().alias('Elo_R_Player')
    elif rating_method == 'mean':
        rating_agg = pl.col('Elo_R_Player').mean().alias('Elo_R_Player')
    elif rating_method == 'last':
        rating_agg = pl.col('Elo_R_Player').last().alias('Elo_R_Player')
    else:
        raise ValueError(f"Invalid rating method: {rating_method}")

    top_players = (
        players_stacked
        .group_by('Player_ID')
        .agg([
            rating_agg,
            pl.col('Player_Name').last().alias('Player_Name'),
            pl.col('MasterPoints').last().alias('MasterPoints'),
            pl.col('session_id').n_unique().alias('Elo_Count'),
            pl.col('Position').n_unique().alias('Positions_Played'),
        ])
        .with_columns([
            pl.col('MasterPoints').rank(method='ordinal', descending=True).alias('MasterPoint_Rank')
        ])
        .filter(pl.col('Elo_Count') >= min_elo_count)
        .sort('Elo_R_Player', descending=True, nulls_last=True)
        .select(['Elo_R_Player', 'Player_ID', 'Player_Name', 'MasterPoints', 'MasterPoint_Rank', 'Elo_Count'])
        .head(top_n)
        .with_row_index(name='Rank', offset=1)
        .select(['Rank', 'Elo_R_Player', 'Player_ID', 'Player_Name', 'MasterPoints', 'MasterPoint_Rank', 'Elo_Count'])
    )

    top_players = top_players.with_columns([
        pl.col('Elo_R_Player').cast(pl.Int32, strict=False),
        pl.col('MasterPoints').cast(pl.Int32, strict=False),
        pl.col('MasterPoint_Rank').cast(pl.Int32, strict=False),
        pl.col('Elo_Count').alias('Sessions_Played'),
    ]).drop(['Elo_Count']).rename({'Elo_R_Player': 'Player_Elo_Score', 'Rank': 'Player_Elo_Rank'})

    return top_players


def show_top_pairs(df: pl.DataFrame, top_n: int, min_elo_count: int = 30, rating_method: str = 'mean') -> pl.DataFrame:
    if 'MasterPoints_N' not in df.columns:
        df = df.with_columns(
            pl.lit(None).alias('MasterPoints_N'),
            pl.lit(None).alias('MasterPoints_S'),
            pl.lit(None).alias('MasterPoints_E'),
            pl.lit(None).alias('MasterPoints_W')
        )

    ns_partnerships = df.select(
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
        ((pl.col("MasterPoints_N").fill_null(0) + pl.col("MasterPoints_S").fill_null(0)) / 2).alias("Avg_MPs"),
        (pl.col("MasterPoints_N").fill_null(0) * pl.col("MasterPoints_S").fill_null(0)).sqrt().alias("Geo_MPs"),
        pl.when(pl.mean_horizontal([pl.col("Elo_R_N"), pl.col("Elo_R_S")]).is_nan())
         .then(None)
         .otherwise(pl.mean_horizontal([pl.col("Elo_R_N"), pl.col("Elo_R_S")]))
         .alias("Avg_Player_Elo_Row"),
    )

    ew_partnerships = df.select(
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
        ((pl.col("MasterPoints_E").fill_null(0) + pl.col("MasterPoints_W").fill_null(0)) / 2).alias("Avg_MPs"),
        (pl.col("MasterPoints_E").fill_null(0) * pl.col("MasterPoints_W").fill_null(0)).sqrt().alias("Geo_MPs"),
        pl.when(pl.mean_horizontal([pl.col("Elo_R_E"), pl.col("Elo_R_W")]).is_nan())
         .then(None)
         .otherwise(pl.mean_horizontal([pl.col("Elo_R_E"), pl.col("Elo_R_W")]))
         .alias("Avg_Player_Elo_Row"),
    )

    partnerships_stacked = pl.concat([ns_partnerships, ew_partnerships], how="vertical").drop_nulls(subset=["Pair_IDs", "Elo_R_Pair"])

    if rating_method == 'max':
        rating_agg = pl.col('Elo_R_Pair').max().alias('Elo_Score')
    elif rating_method == 'mean':
        rating_agg = pl.col('Elo_R_Pair').mean().alias('Elo_Score')
    elif rating_method == 'last':
        rating_agg = pl.col('Elo_R_Pair').last().alias('Elo_Score')
    else:
        raise ValueError(f"Invalid rating method: {rating_method}")

    top_partnerships = (
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
        .with_columns([
            pl.col('Avg_Player_Elo').rank(method='ordinal', descending=True).alias('Avg_Elo_Rank'),
            pl.col('Avg_MPs').rank(method='ordinal', descending=True).alias('Avg_MPs_Rank'),
            pl.col('Geo_MPs').rank(method='ordinal', descending=True).alias('Geo_MPs_Rank'),
        ])
        .filter(pl.col('Sessions') >= min_elo_count)
        .sort('Elo_Score', descending=True, nulls_last=True)
        .select(['Elo_Score', 'Avg_Elo_Rank', 'Pair_IDs', 'Pair_Names', 'Avg_MPs', 'Avg_MPs_Rank', 'Geo_MPs_Rank', 'Sessions'])
        .head(top_n)
        .with_row_index(name='Pair_Elo_Rank', offset=1)
        .select(['Pair_Elo_Rank', 'Elo_Score', 'Avg_Elo_Rank', 'Pair_IDs', 'Pair_Names', 'Avg_MPs', 'Avg_MPs_Rank', 'Geo_MPs_Rank', 'Sessions'])
    )

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
widen_scrollbars()
st.title("Unofficial ACBL Elo Ratings")
st.caption("Interactive viewer for unofficial ACBL Elo ratings")
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

with st.sidebar:
    st.header("Controls")
    club_or_tournament = st.selectbox("Dataset", options=["club", "tournament"], index=0)
    rating_type = st.radio("Rating type", options=["Players", "Pairs"], index=0, horizontal=False)
    top_n = st.number_input("Top N", min_value=50, max_value=5000, value=1000, step=50)
    min_sessions = st.number_input("Minimum sessions", min_value=1, max_value=200, value=30, step=1)
    rating_method = st.selectbox("Rating method", options=["mean", "max", "last"], index=0)
    # Date range quick filter (default All time)
    date_range_choice = st.selectbox(
        "Date range",
        options=["All time", "Last 3 months", "Last 6 months", "Last 12 months"],
        index=0,
    )
    display_table = st.button("Display Table", type="primary")
    generate_pdf = st.button("Generate PDF", type="primary")

# Determine date_from based on selection
now = datetime.now()
if date_range_choice == "All time":
    date_from = None
elif date_range_choice == "Last 3 months":
    date_from = now - timedelta(days=90)
elif date_range_choice == "Last 6 months":
    date_from = now - timedelta(days=182)
else:  # Last 12 months
    date_from = now - timedelta(days=365)

# Determine needed columns to accelerate load
common_cols = ["Date", "session_id"]
if rating_type == "Players":
    needed = []
    for d in "NESW":
        needed += [
            f"Player_ID_{d}", f"Player_Name_{d}", f"MasterPoints_{d}", f"Elo_R_{d}",
        ]
    needed_cols = common_cols + needed
else:
    needed_cols = common_cols + [
        # IDs and Names
        "Player_ID_N", "Player_ID_S", "Player_ID_E", "Player_ID_W",
        "Player_Name_N", "Player_Name_S", "Player_Name_E", "Player_Name_W",
        # MasterPoints and Elo components
        "MasterPoints_N", "MasterPoints_S", "MasterPoints_E", "MasterPoints_W",
        "Elo_R_N", "Elo_R_S", "Elo_R_E", "Elo_R_W",
        # Pair Elo
        "Elo_R_NS", "Elo_R_EW",
    ]

if not (display_table or generate_pdf):
    st.info("Select options and hit the 'Display Table' or 'Generate PDF' button.")
else:
    # Show immediate progress indicator when PDF generation is requested
    if generate_pdf:
        progress_container = st.container()
        with progress_container:
            progress_status = st.status("🔄 Preparing PDF generation...", expanded=True)
            with progress_status:
                st.write("Initializing PDF report generation process. Takes 60 seconds.")
                progress_bar = st.progress(0, text="Starting...")
    
    # Build options signature for reuse
    opts = {
        "club_or_tournament": club_or_tournament,
        "rating_type": rating_type,
        "top_n": int(top_n),
        "min_sessions": int(min_sessions),
        "rating_method": rating_method,
        "date_from": None if date_from is None else date_from.isoformat(),
    }

    # Disable caching for now to avoid memory issues
    use_cache = False

    if not use_cache:
        # Intersect needed columns with available to avoid errors
        try:
            available_cols = set(load_elo_ratings_schema(club_or_tournament))
            columns_to_read = [c for c in needed_cols if c in available_cols]
            if "Date" not in columns_to_read and "Date" in available_cols:
                columns_to_read.insert(0, "Date")
        except Exception as e:
            st.error(f"Failed to read schema: {e}")
            st.stop()

        # Helper: run work with a seconds-based progress bar
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

            # Clear message when finished (even if earlier than hint)
            container.empty()
            if 'error' in holder:
                raise holder['error']
            return holder.get('result')

        # Update progress for PDF generation
        if generate_pdf:
            progress_bar.progress(10, text="Loading dataset...")
            progress_status.update(label="📊 Loading dataset. Takes 30 seconds.", state="running")
        
        # Load data with progress (60s hint)
        df = run_with_progress(
            "Loading dataset. Takes 30 seconds.",
            60,
            lambda: load_elo_ratings(club_or_tournament, columns=columns_to_read, date_from=date_from),
        )

        # Update progress after data loading
        if generate_pdf:
            progress_bar.progress(50, text="Dataset loaded. Creating report table...")
            progress_status.update(label="📋 Creating report table...", state="running")
        
        # Compute date range for captions
        try:
            date_min, date_max = df.select([pl.col("Date").min().alias("min"), pl.col("Date").max().alias("max")]).row(0)
            date_range = f"{str(date_min)[:10]} to {str(date_max)[:10]}"
        except Exception:
            date_range = ""

        # Compute table and title with progress (60s hint)
        def build_table():
            if rating_type == "Players":
                tbl = show_top_players(df, int(top_n), int(min_sessions), rating_method)
                ttl = f"Top {top_n} ACBL {club_or_tournament} Players by Elo ({rating_method})"
            else:
                tbl = show_top_pairs(df, int(top_n), int(min_sessions), rating_method)
                ttl = f"Top {top_n} ACBL {club_or_tournament} Pairs by Elo ({rating_method})"
            return tbl, ttl

        table_df, title = run_with_progress(
            "Creating Report Table. Takes 30 seconds.",
            60,
            build_table,
        )

        # Update progress after table creation
        if generate_pdf:
            progress_bar.progress(75, text="Report table created. Caching results...")
            progress_status.update(label="💾 Caching results...", state="running")

        # Cache results
        st.session_state["report_opts"] = opts
        st.session_state["report_table_df"] = table_df
        st.session_state["report_title"] = title
        st.session_state["report_date_range"] = date_range
    else:
        # Using cached data
        if generate_pdf:
            progress_bar.progress(80, text="Using cached data...")
            progress_status.update(label="⚡ Using cached data...", state="running")
        
        table_df = st.session_state["report_table_df"]
        title = st.session_state["report_title"]
        date_range = st.session_state["report_date_range"]

    # Show table only when requested
    if display_table:
        if date_range:
            st.subheader(f"{title} ({date_range})")
        else:
            st.subheader(title)
        # Prefer large grid height equal to top_n; fallback if not supported
        try:
            ShowDataFrameTable(table_df, key=f"table-{rating_type}", output_method='aggrid', height_rows=int(top_n))
        except TypeError:
            ShowDataFrameTable(table_df, key=f"table-{rating_type}", output_method='aggrid')

    # PDF generation regardless of whether table is shown
    if generate_pdf:
        # Update progress before PDF generation
        progress_bar.progress(85, text="Preparing PDF generation...")
        progress_status.update(label="📄 Generating PDF...", state="running")
        
        created_on = time.strftime("%Y-%m-%d")
        pdf_title = f"{title} ({date_range})"
        pdf_filename = f"Unofficial Elo Scores for ACBL {club_or_tournament} MatchPoint Games - Top {top_n} {rating_type} {created_on}.pdf"
        # Try new API with max_rows; fallback to chunking when older API is loaded
        def build_pdf():
            # Enable shrink_to_fit for Pair reports to better fit the wider tables
            shrink_pairs = (rating_type == "Pairs")
            return create_pdf([pdf_title, table_df], pdf_title, max_rows=int(top_n), rows_per_page=(20, 24), shrink_to_fit=shrink_pairs)
        try:
            pdf_bytes = build_pdf()
        except TypeError:
            def build_pdf_fallback():
                assets = [pdf_title]
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

        # Complete the progress indicator
        progress_bar.progress(100, text="PDF generation complete!")
        progress_status.update(label="✅ PDF generation complete!", state="complete")
        
        # Clear progress indicators after a brief moment
        time.sleep(0.5)
        progress_container.empty()

        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name=pdf_filename,
            mime="application/pdf",
        )

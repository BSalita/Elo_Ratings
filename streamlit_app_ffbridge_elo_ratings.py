# streamlit_app_ffbridge_elo.py
"""
FFBridge Elo Ratings - Unified Streamlit Application

This app fetches duplicate bridge tournament results from FFBridge and calculates
Elo ratings based on percentage scores.

Supports both:
- Classic API (api.ffbridge.fr) - requires authentication
- Lancelot API (api-lancelot.ffbridge.fr) - public access
"""

import os
import pathlib
import sys
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Dict, Any

import pandas as pd
import polars as pl
import streamlit as st
import duckdb

from streamlitlib.streamlitlib import (
    ShowDataFrameTable,
    create_pdf,
    widen_scrollbars,
)
from st_aggrid import GridOptionsBuilder, AgGrid, ColumnsAutoSizeMode, AgGridTheme

# Import shared utilities
from elo_ffbridge_common import (
    DEFAULT_ELO,
    K_FACTOR,
    AGGRID_ROW_HEIGHT,
    AGGRID_HEADER_HEIGHT,
    AGGRID_FOOTER_HEIGHT,
    AGGRID_MAX_DISPLAY_ROWS,
    normalize_series_id,
    calculate_expected_score,
    calculate_elo_from_percentage,
)

# Import API adapters
import elo_ffbridge_classic as classic_api
import elo_ffbridge_lancelot as lancelot_api

# Import for version display only
try:
    import endplay
    ENDPLAY_VERSION = endplay.__version__
except (ImportError, AttributeError):
    ENDPLAY_VERSION = "N/A"

# Available API backends
API_BACKENDS = {
    "FFBridge (Classic)": classic_api,
    "FFBridge (Lancelot)": lancelot_api,
}


# -------------------------------
# Helper Functions
# -------------------------------
def calculate_aggrid_height(row_count: int) -> int:
    """Calculate AgGrid height based on row count."""
    display_rows = min(AGGRID_MAX_DISPLAY_ROWS, row_count)
    return AGGRID_HEADER_HEIGHT + (AGGRID_ROW_HEIGHT * display_rows) + AGGRID_FOOTER_HEIGHT


def build_selectable_aggrid(df: pd.DataFrame, key: str) -> Dict[str, Any]:
    """Build an AgGrid with single-click row selection."""
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_selection(selection_mode='single', use_checkbox=False, suppressRowClickSelection=False)
    gb.configure_default_column(cellStyle={'color': 'black', 'font-size': '12px'}, suppressMenu=True)
    grid_options = gb.build()
    
    return AgGrid(
        df,
        gridOptions=grid_options,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
        theme=AgGridTheme.BALHAM,
        height=calculate_aggrid_height(len(df)),
        key=key,
        allow_unsafe_jscode=True
    )


# -------------------------------
# Data Processing (common for both APIs)
# -------------------------------
def process_tournaments_to_elo(
    tournaments: List[Dict[str, Any]],
    api_module,
    initial_players: Optional[Dict[str, Dict]] = None,
    use_handicap: bool = False,
) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, float]]:
    """
    Process tournament list and calculate Elo ratings.
    Works with both Classic and Lancelot API data.
    """
    all_results = []
    player_ratings: Dict[str, float] = {}
    player_names: Dict[str, str] = {}
    player_games: Dict[str, int] = {}
    player_pct_n: Dict[str, int] = {}
    player_pct_mean: Dict[str, float] = {}
    player_pct_m2: Dict[str, float] = {}
    
    if initial_players:
        for pid, pinfo in initial_players.items():
            player_ratings[pid] = pinfo.get('elo', DEFAULT_ELO)
            player_names[pid] = pinfo.get('name', pid)
            player_games[pid] = pinfo.get('games_played', 0)
    
    # Sort tournaments chronologically
    sorted_tournaments = sorted(tournaments, key=lambda x: x.get('date', ''))
    
    # Filter out future tournaments
    today = datetime.now().strftime('%Y-%m-%d')
    sorted_tournaments = [t for t in sorted_tournaments if t.get('date', '')[:10] <= today]
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    cache_stats = {"cached": 0, "fetched": 0}
    
    total_t = len(sorted_tournaments)
    for i, tournament in enumerate(sorted_tournaments):
        t_id = str(tournament.get('id', ''))
        t_series = normalize_series_id(tournament.get('series_id'))
        t_name = tournament.get('name') or tournament.get('label') or tournament.get('moment_label') or f"Tournament {t_id}"
        t_date = tournament.get('date', '')
        
        cache_info = f"[Cached: {cache_stats['cached']}, Fetched: {cache_stats['fetched']}]"
        status_text.markdown(f"<span style='color: white;'>Processing {i+1}/{total_t}: {t_name[:35]}... {cache_info}</span>", unsafe_allow_html=True)
        progress_bar.progress((i + 1) / total_t)
        
        # Fetch results using the appropriate API module
        results, was_cached = api_module.fetch_tournament_results(t_id, tournament_date=t_date, series_id=t_series)
        
        if was_cached:
            cache_stats["cached"] += 1
        else:
            cache_stats["fetched"] += 1
        
        if not results:
            continue
        
        # Calculate field average rating
        field_ratings = []
        for result in results:
            p1_id = result.get('player1_id')
            p2_id = result.get('player2_id')
            if p1_id and p2_id:
                r1 = player_ratings.get(p1_id, DEFAULT_ELO)
                r2 = player_ratings.get(p2_id, DEFAULT_ELO)
                field_ratings.append((r1 + r2) / 2)
        
        field_avg = sum(field_ratings) / len(field_ratings) if field_ratings else DEFAULT_ELO
        
        # Update ratings for each result
        for result in results:
            p1_id = result.get('player1_id')
            p2_id = result.get('player2_id')
            p1_name = result.get('player1_name', '')
            p2_name = result.get('player2_name', '')
            
            # Get percentages
            raw_pct = float(result.get('percentage', 50.0) or 50.0)
            handicap_pct_raw = result.get('handicap_percentage')
            try:
                handicap_pct = float(handicap_pct_raw) if handicap_pct_raw is not None else raw_pct
            except (ValueError, TypeError):
                handicap_pct = raw_pct
            
            pe_bonus_raw = result.get('pe_bonus', 0)
            try:
                pe_bonus = float(pe_bonus_raw or 0)
            except (ValueError, TypeError):
                pe_bonus = 0.0
            
            club_pct_raw = result.get('club_percentage')
            try:
                club_pct = float(club_pct_raw) if club_pct_raw is not None else (handicap_pct - pe_bonus / 10.0)
            except (ValueError, TypeError):
                club_pct = handicap_pct - pe_bonus / 10.0
            
            percentage = handicap_pct if use_handicap else club_pct
            
            rank_club = result.get('rank', 0)
            rank_handicap = result.get('theoretical_rank', 0)
            rank = rank_handicap if use_handicap and rank_handicap is not None else rank_club
            
            team_id = result.get('team_id', '')
            pe = result.get('pe', 0)
            club_id = result.get('club_id', '')
            club_name = result.get('club_name', '')
            club_code = result.get('club_code', '')
            
            # Create stable pair identification
            if p1_id and p2_id:
                p1_str, p2_str = str(p1_id), str(p2_id)
                if p1_str < p2_str:
                    pair_id = f"{p1_str}_{p2_str}"
                    stable_p1_name, stable_p2_name = p1_name, p2_name
                else:
                    pair_id = f"{p2_str}_{p1_str}"
                    stable_p1_name, stable_p2_name = p2_name, p1_name
                pair_name = f"{stable_p1_name} - {stable_p2_name}"
            else:
                pair_id = result.get('pair_id') or f"{p1_id}-{p2_id}"
                pair_name = f"{p1_name} - {p2_name}"
            
            # Use tournament_id or session_id depending on API
            event_id = str(t_id)
            
            result_record = {
                'tournament_id': event_id,
                'tournament_name': str(t_name),
                'date': str(t_date),
                'series_id': int(t_series) if t_series else 0,
                'team_id': str(team_id),
                'pair_id': str(pair_id),
                'player1_id': str(p1_id),
                'player2_id': str(p2_id),
                'player1_name': str(p1_name),
                'player2_name': str(p2_name),
                'pair_name': str(pair_name),
                'percentage': float(percentage),
                'handicap_percentage': float(handicap_pct),
                'club_percentage': float(club_pct),
                'rank': int(rank) if rank is not None else 0,
                'rank_without_handicap': int(rank_club) if rank_club is not None else 0,
                'theoretical_rank': int(rank_handicap) if rank_handicap is not None else 0,
                'pe': float(pe) if pe is not None else 0.0,
                'pe_bonus': str(pe_bonus) if pe_bonus is not None else '',
                'field_avg_rating': float(field_avg),
                'club_id': str(club_id),
                'club_name': str(club_name),
                'club_code': str(club_code),
            }
            
            # Update player 1 rating
            if p1_id:
                current_r1 = player_ratings.get(p1_id, DEFAULT_ELO)
                new_r1 = calculate_elo_from_percentage(current_r1, percentage, field_avg)
                result_record['player1_elo_before'] = current_r1
                result_record['player1_elo_after'] = new_r1
                player_ratings[p1_id] = new_r1
                player_names[p1_id] = p1_name
                player_games[p1_id] = player_games.get(p1_id, 0) + 1
                
                n = player_pct_n.get(p1_id, 0) + 1
                mean = player_pct_mean.get(p1_id, 0.0)
                m2 = player_pct_m2.get(p1_id, 0.0)
                x = float(percentage)
                delta = x - mean
                mean += delta / n
                delta2 = x - mean
                m2 += delta * delta2
                player_pct_n[p1_id] = n
                player_pct_mean[p1_id] = mean
                player_pct_m2[p1_id] = m2
            
            # Update player 2 rating
            if p2_id:
                current_r2 = player_ratings.get(p2_id, DEFAULT_ELO)
                new_r2 = calculate_elo_from_percentage(current_r2, percentage, field_avg)
                result_record['player2_elo_before'] = current_r2
                result_record['player2_elo_after'] = new_r2
                player_ratings[p2_id] = new_r2
                player_names[p2_id] = p2_name
                player_games[p2_id] = player_games.get(p2_id, 0) + 1
                
                n = player_pct_n.get(p2_id, 0) + 1
                mean = player_pct_mean.get(p2_id, 0.0)
                m2 = player_pct_m2.get(p2_id, 0.0)
                x = float(percentage)
                delta = x - mean
                mean += delta / n
                delta2 = x - mean
                m2 += delta * delta2
                player_pct_n[p2_id] = n
                player_pct_mean[p2_id] = mean
                player_pct_m2[p2_id] = m2
            
            # Calculate pair Elo
            if p1_id and p2_id:
                result_record['pair_elo'] = (player_ratings[p1_id] + player_ratings[p2_id]) / 2
            
            all_results.append(result_record)
    
    progress_bar.empty()
    status_text.empty()
    
    # Convert to DataFrames
    results_df = pl.DataFrame(all_results) if all_results else pl.DataFrame()
    
    # Create player ratings summary
    player_summary = []
    for pid, rating in player_ratings.items():
        n = player_pct_n.get(pid, 0)
        avg_pct = float(player_pct_mean.get(pid, 0.0)) if n > 0 else None
        stdev_pct = float((player_pct_m2.get(pid, 0.0) / (n - 1)) ** 0.5) if n > 1 else None
        player_summary.append({
            'player_id': str(pid),
            'player_name': str(player_names.get(pid, pid)),
            'elo_rating': float(round(rating, 1)),
            'games_played': int(player_games.get(pid, 0)),
            'avg_percentage': avg_pct,
            'stdev_percentage': stdev_pct,
        })
    
    if player_summary:
        players_df = pl.DataFrame(player_summary, schema={
            'player_id': pl.Utf8,
            'player_name': pl.Utf8,
            'elo_rating': pl.Float64,
            'games_played': pl.Int64,
            'avg_percentage': pl.Float64,
            'stdev_percentage': pl.Float64,
        })
    else:
        players_df = pl.DataFrame()
    
    return results_df, players_df, player_ratings


def show_top_players(players_df: pl.DataFrame, top_n: int, min_games: int = 5) -> Tuple[pl.DataFrame, str]:
    """Get top players by Elo rating using SQL."""
    if players_df.is_empty():
        return players_df, ""
    
    query = f"""
        WITH filtered AS (
            SELECT *
            FROM players_df
            WHERE games_played >= {min_games}
        )
        SELECT 
            ROW_NUMBER() OVER (ORDER BY elo_rating DESC, games_played DESC, player_name ASC, player_id ASC) AS Rank,
            CAST(ROUND(elo_rating, 0) AS INTEGER) AS Elo_Rating,
            player_id AS Player_ID,
            player_name AS Player_Name,
            ROUND(avg_percentage, 1) AS Avg_Pct,
            ROUND(stdev_percentage, 1) AS Pct_Stdev,
            games_played AS Games_Played
        FROM filtered
        ORDER BY Rank ASC
        LIMIT {top_n}
    """
    
    result = duckdb.sql(query).pl()
    return result, query


def show_top_pairs(results_df: pl.DataFrame, top_n: int, min_games: int = 5) -> Tuple[pl.DataFrame, str]:
    """Get top pairs by average Elo rating using SQL."""
    if results_df.is_empty():
        return results_df, ""
    
    query = f"""
        WITH pair_stats AS (
            SELECT 
                pair_id,
                ARG_MAX(pair_name, date) AS pair_name,
                ARG_MAX(player1_id, date) AS player1_id,
                ARG_MAX(player2_id, date) AS player2_id,
                AVG(pair_elo) AS avg_pair_elo,
                AVG(percentage) AS avg_percentage,
                STDDEV_SAMP(percentage) AS stdev_percentage,
                COUNT(*) AS games_played
            FROM results_df
            GROUP BY pair_id
        ),
        filtered AS (
            SELECT *
            FROM pair_stats
            WHERE games_played >= {min_games}
        )
        SELECT 
            ROW_NUMBER() OVER (ORDER BY avg_pair_elo DESC, avg_percentage DESC, games_played DESC, pair_name ASC, pair_id ASC) AS Rank,
            CAST(ROUND(avg_pair_elo, 0) AS INTEGER) AS Pair_Elo,
            pair_id AS Pair_ID,
            pair_name AS Pair_Name,
            ROUND(avg_percentage, 1) AS Avg_Pct,
            ROUND(stdev_percentage, 1) AS Pct_Stdev,
            games_played AS Games
        FROM filtered
        ORDER BY Rank ASC
        LIMIT {top_n}
    """
    
    result = duckdb.sql(query).pl()
    return result, query


# -------------------------------
# Session State Initialization
# -------------------------------
def initialize_session_state():
    """Initialize session state variables on first run."""
    if 'first_time' not in st.session_state:
        st.session_state.first_time = True
        st.session_state.app_datetime = datetime.fromtimestamp(
            pathlib.Path(__file__).stat().st_mtime, 
            tz=timezone.utc
        ).strftime('%Y-%m-%d %H:%M:%S %Z')
    else:
        st.session_state.first_time = False


# -------------------------------
# Main UI
# -------------------------------
def main():
    st.set_page_config(
        page_title="Unofficial FFBridge Elo Ratings Playground",
        page_icon="🃏",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    # Apply styling
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        
        .stApp {
            background-color: #004d40;
            color: #f5f5f5;
        }
        
        h1, h2, h3 {
            color: #ffc107 !important;
            font-weight: 700;
            letter-spacing: 1px;
        }
        
        .stSidebar {
            background-color: #00332e !important;
            border-right: 1px solid #00695c;
        }
        
        .stSidebar .stMarkdown, .stSidebar label {
            color: #e0e0e0 !important;
        }
        
        .metric-card {
            background: rgba(0, 105, 92, 0.4);
            border: 1px solid #00796b;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .stDataFrame {
            background-color: white;
            border-radius: 8px;
        }
        
        .stButton > button {
            background-color: #ffc107;
            color: #004d40;
            border: none;
            border-radius: 5px;
            font-weight: 700;
            padding: 0.5rem 1rem;
            transition: all 0.2s ease;
        }
        
        .stButton > button:hover {
            background-color: #ffca28;
            transform: scale(1.02);
            box-shadow: 0 4px 12px rgba(255,193,7,0.3);
        }
        
        .stRadio > label { color: #ffc107 !important; font-weight: 600 !important; }
        .stRadio [data-testid="stMarkdownContainer"] p { color: #ffffff !important; font-size: 1rem !important; font-weight: 500 !important; }
        .stSelectbox > label { color: #ffc107 !important; font-weight: 600 !important; }
        .stSlider > label { color: #ffc107 !important; font-weight: 600 !important; }
        .stCheckbox label span, .stCheckbox label p, .stCheckbox [data-testid="stMarkdownContainer"] p { color: #ffffff !important; font-weight: 500 !important; }
        </style>
    """, unsafe_allow_html=True)
    
    widen_scrollbars()
    
    # -------------------------------
    # Sidebar Controls
    # -------------------------------
    with st.sidebar:
        st.sidebar.caption(f"Build:{st.session_state.app_datetime}")
        
        # API Backend selection
        if "selected_api" not in st.session_state:
            st.session_state.selected_api = "FFBridge (Classic)"
        
        selected_api_name = st.selectbox(
            "Bridge API",
            options=list(API_BACKENDS.keys()),
            key="selected_api",
            help="Select which API to use for data"
        )
        
        # Get the appropriate API module
        api_module = API_BACKENDS[selected_api_name]
        
        # Check authentication if required
        if api_module.REQUIRES_AUTH and not api_module.is_authenticated():
            st.error("❌ **Authentication Error**")
            st.markdown(api_module.get_auth_error_message())
            return
        
        # Tournament selection
        series_names = api_module.SERIES_NAMES
        valid_series_ids = api_module.VALID_SERIES_IDS
        tournament_options_list = ["all"] + valid_series_ids
        tournament_labels = [series_names[k] for k in tournament_options_list]
        
        if "elo_tournament_selectbox" not in st.session_state:
            st.session_state.elo_tournament_selectbox = tournament_labels[0]
        
        if st.session_state.elo_tournament_selectbox not in tournament_labels:
            st.session_state.elo_tournament_selectbox = tournament_labels[0]
        
        selected_tournament_label = st.selectbox(
            "Tournament Names",
            options=tournament_labels,
            key="elo_tournament_selectbox",
            help="Select which simultaneous tournaments to analyze"
        )
        
        simultaneous_type = tournament_options_list[tournament_labels.index(selected_tournament_label)]
        
        # Club filter
        club_options = st.session_state.get('elo_available_clubs', ["All Clubs"])
        
        if "elo_club_selectbox" not in st.session_state:
            st.session_state.elo_club_selectbox = "All Clubs"
        
        if st.session_state.elo_club_selectbox not in club_options:
            st.session_state.elo_club_selectbox = "All Clubs"
        
        selected_club = st.selectbox(
            "Filter by Club",
            options=club_options,
            key="elo_club_selectbox",
            help="Filter results to show only players/pairs from a specific club"
        )
        
        # Ranking type
        rating_type = st.radio(
            "Ranking Type",
            ["Players", "Pairs"],
            index=0,
            key="elo_rating_type",
            horizontal=True,
            help="Switch between individual and partnership rankings"
        )
        
        # Handicap option
        use_handicap = st.checkbox(
            "Use handicap score",
            value=True,
            key="elo_use_handicap",
            help="Uses handicap-adjusted percentage when available"
        )
        
        # Number of results
        top_n = st.slider(
            "Show Top N",
            min_value=50,
            max_value=1000,
            value=250,
            step=50,
            key="elo_top_n"
        )
        
        # Minimum games
        min_games = st.slider(
            "Minimum Games",
            min_value=1,
            max_value=100,
            value=10,
            key="elo_min_games",
            help="Minimum tournaments played to appear in rankings"
        )
        
        # PDF Export
        generate_pdf = st.button("Export Report to PDF File", use_container_width=True)
    
    # Header
    st.markdown(f"""
        <div style="text-align: center; padding: 0 0 1rem 0; margin-top: -2rem;">
            <h1 style="font-size: 2.8rem; margin-bottom: 0.2rem;">
                🃏 Unofficial FFBridge Elo Ratings Playground
            </h1>
            <p style="color: #ffc107; font-size: 1.2rem; font-weight: 500; opacity: 0.9;">
                 {selected_tournament_label} via {selected_api_name}
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # -------------------------------
    # Main Content
    # -------------------------------
    
    # Cache key includes API name to separate caches
    api_key = selected_api_name.replace(" ", "_").replace("(", "").replace(")", "")
    
    # Fetch all tournaments
    with st.spinner("Loading tournament data..."):
        all_tournaments = api_module.fetch_tournament_list(series_id="all", limit=None)
        
        if not all_tournaments:
            st.error("Failed to retrieve tournament data. Please check your connection or authentication.")
            return
    
    # Cache key for full dataset
    cache_key = f"elo_full_v1_{api_key}_{len(all_tournaments)}_handicap_{int(use_handicap)}"
    
    if 'elo_full_cache' not in st.session_state:
        st.session_state.elo_full_cache = {}
    
    full_cache = st.session_state.elo_full_cache
    
    if cache_key in full_cache:
        cached = full_cache[cache_key]
        full_results_df = cached['results_df']
        full_players_df = cached['players_df']
        current_ratings = cached['current_ratings']
    else:
        # Clear old cache entries for different API
        keys_to_remove = [k for k in full_cache.keys() if not k.startswith(f"elo_full_v1_{api_key}_")]
        for k in keys_to_remove:
            del full_cache[k]
        
        # Process all tournaments
        full_results_df, full_players_df, current_ratings = process_tournaments_to_elo(
            all_tournaments, api_module, initial_players=None, use_handicap=use_handicap
        )
        
        # Apply club name mapping for APIs that don't provide club names directly (e.g., Lancelot)
        if not full_results_df.is_empty() and 'club_code' in full_results_df.columns:
            # Check if club_name column is mostly empty
            non_empty_names = full_results_df.filter(pl.col('club_name') != '').height
            if non_empty_names < full_results_df.height * 0.1:  # Less than 10% have names
                # Get unique club codes
                unique_codes = full_results_df.select('club_code').unique().to_series().to_list()
                unique_codes = [str(c) for c in unique_codes if c]
                
                # Build mapping using API's club name mapping function
                if hasattr(api_module, 'build_club_name_mapping'):
                    club_mapping = api_module.build_club_name_mapping(unique_codes, all_tournaments, full_results_df)
                    
                    # Apply mapping to DataFrame
                    if club_mapping:
                        full_results_df = full_results_df.with_columns(
                            pl.col('club_code').map_elements(
                                lambda c: club_mapping.get(str(c).lstrip('0') if c else '', str(c)),
                                return_dtype=pl.Utf8
                            ).alias('club_name')
                        )
        
        full_cache[cache_key] = {
            'results_df': full_results_df,
            'players_df': full_players_df,
            'current_ratings': current_ratings,
        }
    
    # Apply filters
    results_df = full_results_df
    
    if simultaneous_type != "all":
        if 'series_id' in results_df.columns:
            results_df = results_df.filter(pl.col('series_id') == simultaneous_type)
    
    if selected_club != "All Clubs" and not results_df.is_empty():
        if 'club_name' in results_df.columns:
            results_df = results_df.filter(pl.col('club_name') == selected_club)
    
    # Recalculate players_df from filtered results
    if not results_df.is_empty():
        players_df = duckdb.sql("""
            WITH player_results AS (
                SELECT player1_id AS player_id, player1_name AS player_name, player1_elo_after AS elo_rating, percentage, date
                FROM results_df
                UNION ALL
                SELECT player2_id AS player_id, player2_name AS player_name, player2_elo_after AS elo_rating, percentage, date
                FROM results_df
            )
            SELECT 
                player_id,
                ARG_MAX(player_name, date) AS player_name,
                ROUND(ARG_MAX(elo_rating, date), 1) AS elo_rating,
                COUNT(*) AS games_played,
                ROUND(AVG(percentage), 2) AS avg_percentage,
                ROUND(STDDEV_SAMP(percentage), 2) AS stdev_percentage
            FROM player_results
            GROUP BY player_id
        """).pl()
    else:
        players_df = pl.DataFrame()
        st.info("No results found for the selected filters.")
    
    # Populate club options
    if not full_results_df.is_empty() and 'club_name' in full_results_df.columns:
        unique_clubs = sorted(set(full_results_df.select('club_name').to_series().to_list()))
        unique_clubs = [c for c in unique_clubs if c and c.strip()]
        
        old_clubs = st.session_state.get('elo_available_clubs', ["All Clubs"])
        merged_names = {c.strip() for c in old_clubs if c != "All Clubs"}
        merged_names.update(c.strip() for c in unique_clubs if c.strip())
        
        current_selected = st.session_state.get("elo_club_selectbox", "All Clubs")
        if current_selected and current_selected != "All Clubs":
            merged_names.add(current_selected.strip())
        
        new_clubs = ["All Clubs"] + sorted(merged_names)
        
        if len(new_clubs) > len(old_clubs):
            st.session_state.elo_available_clubs = new_clubs
            st.rerun()
        else:
            st.session_state.elo_available_clubs = new_clubs
    
    # Display metrics
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.markdown(f'<div class="metric-card"><small>Tournaments</small><br><span style="font-size:1.4rem; color:#ffc107; font-weight:700;">{len(all_tournaments)}</span></div>', unsafe_allow_html=True)
    
    with m2:
        st.markdown(f'<div class="metric-card"><small>Active Players</small><br><span style="font-size:1.4rem; color:#ffc107; font-weight:700;">{len(players_df)}</span></div>', unsafe_allow_html=True)
    
    with m3:
        if not players_df.is_empty():
            avg_games = players_df.select(pl.col('games_played').mean()).item()
            st.markdown(f'<div class="metric-card"><small>Avg Games</small><br><span style="font-size:1.4rem; color:#ffc107; font-weight:700;">{avg_games:.1f}</span></div>', unsafe_allow_html=True)
    
    with m4:
        if not players_df.is_empty():
            top_rating = players_df.select(pl.col('elo_rating').max()).item()
            st.markdown(f'<div class="metric-card"><small>Highest Elo</small><br><span style="font-size:1.4rem; color:#ffc107; font-weight:700;">{int(top_rating)}</span></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Store for detail views
    st.session_state.full_results_df = results_df
    
    # Display tables
    if rating_type == "Players":
        st.markdown(f"### 🏆 Top {top_n} Players (Min. {min_games} games)")
        if not players_df.is_empty():
            top_players, sql_query = show_top_players(players_df, top_n, min_games)
            
            if sql_query:
                with st.expander("📝 SQL Query", expanded=False):
                    st.code(sql_query, language="sql")
            
            if not top_players.is_empty():
                display_df = top_players.to_pandas()
                grid_response = build_selectable_aggrid(display_df, 'players_table_selectable')
                
                selected_rows = grid_response.get('selected_rows', None)
                if selected_rows is not None and len(selected_rows) > 0:
                    selected_row = selected_rows.iloc[0] if hasattr(selected_rows, 'iloc') else selected_rows[0]
                    player_id = selected_row.get('Player_ID')
                    player_name = selected_row.get('Player_Name', 'Unknown')
                    
                    if player_id and not results_df.is_empty():
                        # Fetch member details if available
                        member_info = api_module.fetch_member_details(str(player_id))
                        
                        st.markdown(f"### 👤 Player Profile: **{player_name}**")
                        
                        if member_info:
                            iv_info = member_info.get('iv', {})
                            iv_rating = iv_info.get('iv', 'N/A')
                            iv_label = iv_info.get('label', '')
                            licence_info = member_info.get('licence', {})
                            home_club = licence_info.get('organization_name', 'Unknown')
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                calc_elo = selected_row.get('Elo_Rating', 'N/A')
                                st.metric("Calculated Elo", f"{calc_elo}")
                            with col2:
                                st.metric("Official IV", f"{iv_rating}", help=iv_label)
                            with col3:
                                games = selected_row.get('Games_Played', 0)
                                st.metric("Games Played", f"{games}")
                            with col4:
                                st.metric("Home Club", home_club[:20] + "..." if len(home_club) > 20 else home_club)
                        
                        # Show tournament history
                        st.markdown("#### 📊 Tournament History")
                        player_results = results_df.filter(
                            (pl.col('player1_id') == str(player_id)) | 
                            (pl.col('player2_id') == str(player_id))
                        ).sort('date', descending=True)
                        
                        if not player_results.is_empty():
                            cols_to_select = [
                                pl.col('date').str.slice(0, 10).alias('Date'),
                                pl.col('tournament_id').alias('Event_ID'),
                                pl.col('tournament_name').alias('Tournament'),
                                pl.col('pair_name').alias('Partner'),
                                (pl.col('club_percentage') if 'club_percentage' in player_results.columns else pl.lit(None, dtype=pl.Float64)).cast(pl.Float64, strict=False).round(2).alias('Club_Score'),
                                (pl.col('handicap_percentage') if 'handicap_percentage' in player_results.columns else pl.lit(None, dtype=pl.Float64)).cast(pl.Float64, strict=False).round(2).alias('Handicap_Score'),
                                pl.col('percentage').cast(pl.Float64, strict=False).round(2).alias('Pct_Used'),
                                pl.col('rank').alias('Rank'),
                            ]
                            if 'theoretical_rank' in player_results.columns and use_handicap:
                                cols_to_select.append(pl.col('theoretical_rank').alias('Hcp_Rank'))
                            cols_to_select.append(pl.col('player1_elo_after').round(0).alias('Elo_After'))
                            
                            detail_df = player_results.select(cols_to_select)
                            ShowDataFrameTable(detail_df, key='player_detail_table', height_rows=10)
                        else:
                            st.info("No results in selected tournaments.")
                
                st.session_state.display_df = top_players
                st.session_state.report_title = f"FFBridge Top Players - {datetime.now().strftime('%Y-%m-%d')}"
            else:
                st.info(f"No players match the minimum requirement of {min_games} games.")
    else:
        st.markdown(f"### 🏆 Top {top_n} Pairs (Min. {min_games} games)")
        if not results_df.is_empty():
            top_pairs, sql_query = show_top_pairs(results_df, top_n, min_games)
            
            if sql_query:
                with st.expander("📝 SQL Query", expanded=False):
                    st.code(sql_query, language="sql")
            
            if not top_pairs.is_empty():
                display_df = top_pairs.to_pandas()
                grid_response = build_selectable_aggrid(display_df, 'pairs_table_selectable')
                
                selected_rows = grid_response.get('selected_rows', None)
                if selected_rows is not None and len(selected_rows) > 0:
                    selected_row = selected_rows.iloc[0] if hasattr(selected_rows, 'iloc') else selected_rows[0]
                    pair_id = selected_row.get('Pair_ID')
                    pair_name = selected_row.get('Pair_Name', 'Unknown')
                    
                    if pair_id and not results_df.is_empty():
                        st.markdown(f"### 📋 Tournament History for **{pair_name}**")
                        
                        pair_results = results_df.filter(
                            pl.col('pair_id') == str(pair_id)
                        ).sort('date', descending=True)
                        
                        if not pair_results.is_empty():
                            cols_to_select = [
                                pl.col('date').str.slice(0, 10).alias('Date'),
                                pl.col('tournament_id').alias('Event_ID'),
                                pl.col('tournament_name').alias('Tournament'),
                                (pl.col('club_percentage') if 'club_percentage' in pair_results.columns else pl.lit(None, dtype=pl.Float64)).cast(pl.Float64, strict=False).round(2).alias('Club_Score'),
                                (pl.col('handicap_percentage') if 'handicap_percentage' in pair_results.columns else pl.lit(None, dtype=pl.Float64)).cast(pl.Float64, strict=False).round(2).alias('Handicap_Score'),
                                pl.col('percentage').cast(pl.Float64, strict=False).round(2).alias('Pct_Used'),
                                pl.col('rank').alias('Rank'),
                            ]
                            if 'theoretical_rank' in pair_results.columns and use_handicap:
                                cols_to_select.append(pl.col('theoretical_rank').alias('Hcp_Rank'))
                            cols_to_select.append(pl.col('pair_elo').round(0).alias('Pair_Elo'))
                            
                            detail_df = pair_results.select(cols_to_select)
                            ShowDataFrameTable(detail_df, key='pair_detail_table', height_rows=10)
                        else:
                            st.info("No detailed results found for this pair.")
                
                st.session_state.display_df = top_pairs
                st.session_state.report_title = f"FFBridge Top Pairs - {datetime.now().strftime('%Y-%m-%d')}"
            else:
                st.info(f"No pairs match the minimum requirement of {min_games} games.")
    
    # PDF Export
    if generate_pdf:
        if 'display_df' in st.session_state and not st.session_state.display_df.is_empty():
            with st.spinner("Preparing PDF export..."):
                title = st.session_state.get('report_title', 'Unofficial FFBridge Elo Ratings')
                pdf_bytes = create_pdf(
                    [f"# {title}\n\nProcessed on {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                     st.session_state.display_df],
                    title=title,
                    shrink_to_fit=True
                )
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"FFBridge_Elo_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )
    
    # Footer
    st.markdown(f"""
        <div style="text-align: center; color: #80cbc4; font-size: 0.8rem; opacity: 0.7;">
            Project lead is Robert Salita research@AiPolice.org. Code written in Python. UI written in streamlit. Data engine is polars. Repo:https://github.com/BSalita<br>
            Query Params:{st.query_params.to_dict()} Environment:{os.getenv('STREAMLIT_ENV','')}<br>
            Streamlit:{st.__version__} Python:{'.'.join(map(str, sys.version_info[:3]))} pandas:{pd.__version__} polars:{pl.__version__} duckdb:{duckdb.__version__} endplay:{ENDPLAY_VERSION}
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div style="text-align: center; padding: 2rem 0; color: #80cbc4; font-size: 0.9rem; opacity: 0.8;">
            Data sourced from {selected_api_name} • {selected_tournament_label}<br>
            System Current Date: {datetime.now().strftime('%Y-%m-%d')}
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

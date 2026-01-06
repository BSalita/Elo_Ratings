# streamlit_app_ffbridge_elo_ratings.py
"""
FFBridge (Fédération Française de Bridge) Elo Ratings Streamlit Application

This app fetches duplicate bridge tournament results from FFBridge and calculates
Elo ratings based on percentage scores.

Data source: https://licencie.ffbridge.fr/#/simultanes/3/tournois
"""

import os
import pathlib
import sys
import json
import hashlib
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, List, Dict, Any

import pandas as pd
import polars as pl
import streamlit as st
import requests
import duckdb
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from streamlitlib.streamlitlib import (
    ShowDataFrameTable,
    create_pdf,
    stick_it_good,
    widen_scrollbars,
)
from st_aggrid import GridOptionsBuilder, AgGrid, ColumnsAutoSizeMode, AgGridTheme, JsCode

# Import for version display only
try:
    import endplay
    ENDPLAY_VERSION = endplay.__version__
except (ImportError, AttributeError):
    ENDPLAY_VERSION = "N/A"

# -------------------------------
# Config / Constants
# -------------------------------
DATA_ROOT = pathlib.Path('data') / 'ffbridge'
CACHE_DIR = DATA_ROOT / 'cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Unified tournament series names (used for cache folders and UI)
SERIES_NAMES = {
    3: "Rondes de France",
    4: "Trophes du Voyage",
    5: "Roy Rene",
    140: "Armour du Bridge",
    384: "Simultanet",
    386: "Simultane Octopus",
    604: "Atout Simultane",
    868: "Festival des Simultanes",
    "all": "All Tournaments"
}

# List of all valid tournament IDs
VALID_SERIES_IDS = [3, 4, 5, 140, 384, 386, 604, 868]

FFBRIDGE_API_BASE = "https://api.ffbridge.fr"

# Default Elo parameters
DEFAULT_ELO = 1500.0
K_FACTOR = 32.0
PERFORMANCE_SCALING = 400  # Standard Elo scaling factor

# UI Constants
REQUEST_TIMEOUT = 15  # seconds
REQUEST_DELAY = 0.1  # seconds between API requests to avoid rate limiting
AGGRID_ROW_HEIGHT = 42  # pixels per row (30 + 12)
AGGRID_HEADER_HEIGHT = 50
AGGRID_FOOTER_HEIGHT = 20
AGGRID_MAX_DISPLAY_ROWS = 10


# -------------------------------
# Persistent Disk Cache Helpers
# -------------------------------
def get_cache_path(identifier: str, params: Optional[Dict] = None, series_id: Optional[Any] = None) -> pathlib.Path:
    """Generate a readable, unique filename in a series-specific subdirectory."""
    import re
    
    # Determine the subdirectory based on series_id (friendly name)
    series_folder = SERIES_NAMES.get(series_id, "General")
    target_dir = CACHE_DIR / series_folder
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize identifier to be a safe filename
    safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', identifier)
    
    # If there are params, append them in a readable format
    if params:
        # Sort items for deterministic filename
        # Replace None with "all" for cleaner filenames
        param_parts = [f"{k}_{'all' if v is None else v}" for k, v in sorted(params.items())]
        safe_name = f"{safe_name}_{'_'.join(param_parts)}"
    
    return target_dir / f"{safe_name}.json"


def save_to_disk_cache(identifier: str, data: Any, params: Optional[Dict] = None, series_id: Optional[Any] = None):
    """Save response data to disk as JSON."""
    cache_path = get_cache_path(identifier, params, series_id)
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'identifier': identifier,
                'series_id': series_id,
                'data': data
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        pass  # Silently fail on cache save errors


def load_from_disk_cache(identifier: str, params: Optional[Dict] = None, max_age_hours: Optional[int] = 72, series_id: Optional[Any] = None) -> Optional[Any]:
    """Load data from disk if it exists and isn't too old."""
    cache_path = get_cache_path(identifier, params, series_id)
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
            
        # Check expiration if max_age_hours is provided
        if max_age_hours is not None:
            cached_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cached_time > timedelta(hours=max_age_hours):
                return None
            
        return cache_data['data']
    except Exception:
        return None


# -------------------------------
# FFBridge Authentication
# -------------------------------
def get_ffbridge_session() -> Optional[requests.Session]:
    """
    Get an authenticated FFBridge session from environment variables.
    Returns None if token is missing.
    """
    # 1. Check if session already exists in Streamlit state
    session = st.session_state.get('ffbridge_session')
    if session:
        return session
    
    # 2. Check for token in environment variable (dotenv)
    env_token = os.getenv('FFBRIDGE_BEARER_TOKEN')
    if env_token:
        # Create a session from the env token
        session = requests.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',
            'Origin': 'https://licencie.ffbridge.fr',
            'Referer': 'https://licencie.ffbridge.fr/',
            'Authorization': f'Bearer {env_token}'
        }
        session.headers.update(headers)
        
        # Store it in session state
        st.session_state.ffbridge_session = session
        st.session_state.ffbridge_user = "Env Token"
        return session
        
    return None


# -------------------------------
# Elo Rating Calculation
# -------------------------------
def calculate_expected_score(rating_a: float, rating_b: float) -> float:
    """Calculate expected score for player A against player B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / PERFORMANCE_SCALING))


def calculate_elo_from_percentage(
    current_rating: float,
    percentage: float,
    field_average_rating: float,
    k_factor: float = K_FACTOR
) -> float:
    """
    Calculate new Elo rating based on percentage score.
    
    Uses percentage as the actual performance and compares to expected
    performance based on rating difference from field average.
    
    Args:
        current_rating: Player's current Elo rating
        percentage: Percentage score achieved (0-100)
        field_average_rating: Average rating of the field
        k_factor: K-factor for rating adjustment
    
    Returns:
        New Elo rating
    """
    # Convert percentage to score (0-1)
    actual_score = percentage / 100.0
    
    # Expected score based on rating difference from field average
    expected_score = calculate_expected_score(current_rating, field_average_rating)
    
    # Calculate rating change
    rating_change = k_factor * (actual_score - expected_score)
    
    return current_rating + rating_change


def update_elo_ratings_from_tournament(
    player_ratings: Dict[str, float],
    tournament_results: List[Dict[str, Any]],
    k_factor: float = K_FACTOR
) -> Dict[str, float]:
    """
    Update Elo ratings for all players based on tournament results.
    
    Args:
        player_ratings: Current ratings {player_id: rating}
        tournament_results: List of {player_id, player_name, percentage, pair_id?}
        k_factor: K-factor for rating adjustment
    
    Returns:
        Updated ratings dictionary
    """
    if not tournament_results:
        return player_ratings
    
    # Calculate field average rating
    field_ratings = []
    for result in tournament_results:
        player_id = result.get('player_id') or result.get('pair_id')
        if player_id:
            rating = player_ratings.get(player_id, DEFAULT_ELO)
            field_ratings.append(rating)
    
    field_average = sum(field_ratings) / len(field_ratings) if field_ratings else DEFAULT_ELO
    
    # Update each player's rating
    updated_ratings = player_ratings.copy()
    for result in tournament_results:
        player_id = result.get('player_id') or result.get('pair_id')
        percentage = result.get('percentage', 50.0)
        
        if player_id:
            current_rating = updated_ratings.get(player_id, DEFAULT_ELO)
            new_rating = calculate_elo_from_percentage(
                current_rating, percentage, field_average, k_factor
            )
            updated_ratings[player_id] = new_rating
    
    return updated_ratings


# -------------------------------
# Helper Functions
# -------------------------------
def get_session_id() -> str:
    """Generate a consistent session ID for cache invalidation."""
    user = st.session_state.get('ffbridge_user', 'env')
    return hashlib.md5(user.encode()).hexdigest()[:8]


def calculate_aggrid_height(row_count: int) -> int:
    """Calculate AgGrid height based on row count."""
    display_rows = min(AGGRID_MAX_DISPLAY_ROWS, row_count)
    return AGGRID_HEADER_HEIGHT + (AGGRID_ROW_HEIGHT * display_rows) + AGGRID_FOOTER_HEIGHT


def build_selectable_aggrid(df: pd.DataFrame, key: str) -> Dict[str, Any]:
    """
    Build an AgGrid with double-click row selection.
    Returns the grid response dictionary.
    """
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_selection(selection_mode='single', use_checkbox=False, suppressRowClickSelection=True)
    gb.configure_default_column(cellStyle={'color': 'black', 'font-size': '12px'}, suppressMenu=True)
    grid_options = gb.build()
    
    # Add double-click handler to select row
    grid_options['onRowDoubleClicked'] = JsCode("""
        function(event) {
            event.node.setSelected(true, true);
        }
    """)
    
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
# FFBridge API Functions
# -------------------------------

@st.cache_data(ttl=86400, show_spinner=False)  # Cache for 24 hours
def fetch_member_details(person_id: str, _session_id: str = "") -> Optional[Dict[str, Any]]:
    """
    Fetch member profile details including official IV rating.
    API: /api/v1/members/{person_id}
    """
    session = get_ffbridge_session()
    if not session:
        return None
    
    url = f"{FFBRIDGE_API_BASE}/api/v1/members/{person_id}"
    
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    
    return None


@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def fetch_team_roadsheets(tournament_id: str, team_id: str, _session_id: str = "") -> Optional[Dict[str, Any]]:
    """
    Fetch complete roadsheet with all deals for a team.
    API: /api/v1/simultaneous-tournaments/{tournament_id}/teams/{team_id}/roadsheets
    """
    session = get_ffbridge_session()
    if not session:
        return None
    
    url = f"{FFBRIDGE_API_BASE}/api/v1/simultaneous-tournaments/{tournament_id}/teams/{team_id}/roadsheets"
    
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    
    return None


@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def fetch_person_results(person_id: str, _session_id: str = "") -> List[Dict[str, Any]]:
    """
    Fetch all tournament results for a person directly from FFBridge.
    API: /api/v1/licensee-results/results/person/{person_id}?date=all&place=0&type=0
    
    This provides complete tournament history with official PE points, ranks, and percentages.
    """
    session = get_ffbridge_session()
    if not session:
        return []
    
    url = f"{FFBRIDGE_API_BASE}/api/v1/licensee-results/results/person/{person_id}?date=all&place=0&type=0"
    
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                return data
    except Exception:
        pass
    
    return []


@st.cache_data(ttl=86400, show_spinner=False)  # Cache for 24 hours
def fetch_all_clubs(_session_id: str = "") -> List[Dict[str, Any]]:
    """
    Fetch the master list of all FFBridge clubs.
    Returns list of club dictionaries with 'id', 'name', 'organization_code', etc.
    """
    session = get_ffbridge_session()
    if not session:
        return []
    
    url = f"{FFBRIDGE_API_BASE}/api/v1/clubs"
    
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                # Filter to only active clubs and sort by name
                active_clubs = [c for c in data if c.get('is_active', False)]
                active_clubs.sort(key=lambda x: x.get('name', ''))
                return active_clubs
    except Exception as e:
        st.warning(f"Error fetching clubs: {e}")
    
    return []


def fetch_tournament_list_with_session(session: Optional[requests.Session], simultaneous_type: int = 3, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Fetch list of simultaneous tournaments from FFBridge using authenticated session.
    Uses persistent disk cache in a series-specific subdirectory.
    """
    # Check disk cache first (never expires as results are static)
    cached_data = load_from_disk_cache(f"tournament_list", {"limit": limit}, max_age_hours=None, series_id=simultaneous_type)
    if cached_data:
        return cached_data

    tournaments = []
    
    # Use provided session or create a new one
    http_client = session or requests.Session()
    
    # The primary API endpoint for the list of tournaments
    url = f"{FFBRIDGE_API_BASE}/api/v1/simultaneous/{simultaneous_type}/tournaments"
    
    try:
        # Use session's headers (includes Authorization) - don't override
        response = http_client.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            try:
                data = response.json()
                if isinstance(data, list):
                    if limit:
                        tournaments = data[:limit]
                    else:
                        tournaments = data
                    
                    # Inject series_id into each tournament for later processing
                    for t in tournaments:
                        t['series_id'] = simultaneous_type
                        
                    # Save to disk cache on success
                    save_to_disk_cache(f"tournament_list", tournaments, {"limit": limit}, series_id=simultaneous_type)
            except json.JSONDecodeError:
                pass
    except Exception as e:
        st.warning(f"Error fetching tournament list: {e}")
    
    return tournaments


@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def fetch_tournament_list(simultaneous_type: Any = 3, limit: Optional[int] = None, _session_id: str = "") -> List[Dict[str, Any]]:
    """
    Cached wrapper for fetching tournament list.
    _session_id is used to invalidate cache when session changes.
    """
    session = get_ffbridge_session()
    
    if simultaneous_type == "all":
        all_tournaments = []
        # Fetch from all known series
        for s_type in VALID_SERIES_IDS:
            all_tournaments.extend(fetch_tournament_list_with_session(session, s_type, limit))
        return all_tournaments
        
    return fetch_tournament_list_with_session(session, simultaneous_type, limit)


def fetch_tournament_results_with_session(session: Optional[requests.Session], tournament_id: str, tournament_date: str = "", series_id: Optional[Any] = None) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Fetch results (ranking) for a specific tournament using authenticated session.
    Uses persistent disk cache in a series-specific subdirectory.
    Uses the detailed tournament endpoint which includes club/organization info.
    
    Returns:
        Tuple of (results dict or None, was_cached bool)
    """
    import re
    
    # Create friendly filename: results_<tournament_id>_<date>
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', tournament_date)
    date_part = date_match.group(1) if date_match else ""
    friendly_name = f"results_{tournament_id}_{date_part}" if date_part else f"results_{tournament_id}"
    
    # Check disk cache first (never expires as results are static)
    cached_data = load_from_disk_cache(friendly_name, max_age_hours=None, series_id=series_id)
    if cached_data:
        return cached_data, True

    # Add delay to avoid rate limiting
    time.sleep(REQUEST_DELAY)
    
    http_client = session or requests.Session()
    
    # Use the detailed tournament endpoint which includes organization/club info
    url = f"{FFBRIDGE_API_BASE}/api/v1/simultaneous-tournaments/{tournament_id}"
    
    try:
        # Use session's headers (includes Authorization) - don't override
        response = http_client.get(url, timeout=REQUEST_TIMEOUT)
        
        # Handle rate limiting
        if response.status_code == 429:
            st.warning("Rate limited by FFBridge API. Waiting 5 seconds...")
            time.sleep(5)
            return fetch_tournament_results_with_session(session, tournament_id, tournament_date, series_id)
        
        if response.status_code == 200:
            try:
                data = response.json()
                if data:
                    # The response contains 'teams' array with detailed info
                    teams = data.get('teams', [])
                    results = []
                    for team in teams:
                        players = team.get('players', [])
                        if len(players) >= 2:
                            p1 = players[0]
                            p2 = players[1]
                            
                            # Get organization/club info
                            org = team.get('organization', {})
                            club_id = str(org.get('id', ''))
                            club_name = org.get('name', '')
                            club_code = org.get('code', '')
                            
                            # Percentage is in 'percent' field
                            try:
                                pct = float(team.get('percent', 0))
                            except (ValueError, TypeError):
                                pct = 0.0
                                
                            results.append({
                                'team_id': str(team.get('id')),  # For roadsheets API
                                'pair_id': str(team.get('id')),
                                'player1_id': str(p1.get('id')),
                                'player2_id': str(p2.get('id')),
                                'player1_name': f"{p1.get('firstname', '')} {p1.get('lastname', '')}".strip(),
                                'player2_name': f"{p2.get('firstname', '')} {p2.get('lastname', '')}".strip(),
                                'percentage': pct,
                                'rank': team.get('ranking', 0),
                                'theoretical_rank': team.get('theoretical_ranking', 0),  # Handicap rank
                                'pe': team.get('PE', 0),  # Performance points
                                'pe_bonus': team.get('PE_bonus', '0'),
                                'club_id': club_id,
                                'club_name': club_name,
                                'club_code': club_code,
                            })
                    
                    processed_data = {'results': results}
                    # Save to disk cache with friendly filename
                    save_to_disk_cache(friendly_name, processed_data, series_id=series_id)
                    return processed_data, False
            except json.JSONDecodeError:
                pass
    except Exception as e:
        st.warning(f"Error fetching tournament results: {e}")
    
    return None, False




def fetch_tournament_results(tournament_id: str, tournament_date: str = "", _session_id: str = "", series_id: Optional[Any] = None) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Wrapper for fetching tournament results with disk caching.
    Returns tuple of (results dict, was_cached bool).
    """
    session = get_ffbridge_session()
    return fetch_tournament_results_with_session(session, tournament_id, tournament_date=tournament_date, series_id=series_id)


# -------------------------------
# Data Processing
# -------------------------------
def process_tournaments_to_elo(
    tournaments: List[Dict[str, Any]],
    initial_players: Optional[Dict[str, Dict]] = None
) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, float]]:
    """
    Process tournament list and calculate Elo ratings.
    
    Args:
        tournaments: List of tournament dicts. If missing 'results', they will be fetched.
        initial_players: Optional dict of initial player info
    
    Returns:
        Tuple of (all_results_df, player_ratings_df, current_ratings_dict)
    """
    all_results = []
    player_ratings: Dict[str, float] = {}
    player_names: Dict[str, str] = {}
    player_games: Dict[str, int] = {}
    
    # Initialize from initial_players if provided
    if initial_players:
        for pid, pinfo in initial_players.items():
            player_ratings[pid] = pinfo.get('elo', DEFAULT_ELO)
            player_names[pid] = pinfo.get('name', pid)
            player_games[pid] = pinfo.get('games_played', 0)
    
    # Sort tournaments chronologically
    sorted_tournaments = sorted(tournaments, key=lambda x: x.get('date', ''))
    
    # Get session for fetching results if needed
    session = get_ffbridge_session()
    session_id = get_session_id()
    
    # Progress tracking for UI
    progress_bar = st.progress(0)
    status_text = st.empty()
    cache_stats = {"cached": 0, "fetched": 0}
    
    total_t = len(sorted_tournaments)
    for i, tournament in enumerate(sorted_tournaments):
        t_id = str(tournament.get('id', ''))
        t_series = tournament.get('series_id')
        # New API uses moment_label, Demo uses name
        t_name = tournament.get('name') or tournament.get('moment_label') or f"Tournament {t_id}"
        t_date = tournament.get('date', '')
        
        # Display progress with cache stats (white text for visibility)
        cache_info = f"[Cached: {cache_stats['cached']}, Fetched: {cache_stats['fetched']}]"
        status_text.markdown(f"<span style='color: white;'>Processing {i+1}/{total_t}: {t_name[:35]}... {cache_info}</span>", unsafe_allow_html=True)
        progress_bar.progress((i + 1) / total_t)
        
        results = tournament.get('results')
        
        # If results are missing (common for live API data), fetch them
        if not results and t_id:
            fetched_data, was_cached = fetch_tournament_results(t_id, tournament_date=t_date, _session_id=session_id, series_id=t_series)
            if was_cached:
                cache_stats["cached"] += 1
            else:
                cache_stats["fetched"] += 1
            if fetched_data:
                results = fetched_data.get('results', [])
        
        if not results:
            continue
        
        # Calculate field average rating before updates
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
            percentage = result.get('percentage', 50.0)
            rank = result.get('rank', 0)
            theoretical_rank = result.get('theoretical_rank', 0)  # Handicap rank
            team_id = result.get('team_id', '')  # For roadsheets
            pe = result.get('pe', 0)
            pe_bonus = result.get('pe_bonus', '0')
            club_id = result.get('club_id', '')
            club_name = result.get('club_name', '')
            
            # Create stable names and IDs
            if p1_id and p2_id:
                # Sort by ID to ensure stable partnership identification
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

            # Store result
            result_record = {
                'tournament_id': t_id,
                'tournament_name': t_name,
                'date': t_date,
                'team_id': team_id,
                'pair_id': pair_id,
                'player1_id': p1_id,
                'player2_id': p2_id,
                'player1_name': p1_name,
                'player2_name': p2_name,
                'pair_name': pair_name,
                'percentage': percentage,
                'rank': rank,
                'theoretical_rank': theoretical_rank,
                'pe': pe,
                'pe_bonus': pe_bonus,
                'field_avg_rating': field_avg,
                'club_id': club_id,
                'club_name': club_name,
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
            
            # Update player 2 rating
            if p2_id:
                current_r2 = player_ratings.get(p2_id, DEFAULT_ELO)
                new_r2 = calculate_elo_from_percentage(current_r2, percentage, field_avg)
                result_record['player2_elo_before'] = current_r2
                result_record['player2_elo_after'] = new_r2
                player_ratings[p2_id] = new_r2
                player_names[p2_id] = p2_name
                player_games[p2_id] = player_games.get(p2_id, 0) + 1
            
            # Calculate pair Elo (average of both players)
            if p1_id and p2_id:
                result_record['pair_elo'] = (player_ratings[p1_id] + player_ratings[p2_id]) / 2
            
            all_results.append(result_record)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Convert to DataFrames
    results_df = pl.DataFrame(all_results) if all_results else pl.DataFrame()
    
    # Create player ratings summary
    player_summary = []
    for pid, rating in player_ratings.items():
        player_summary.append({
            'player_id': pid,
            'player_name': player_names.get(pid, pid),
            'elo_rating': round(rating, 1),
            'games_played': player_games.get(pid, 0)
        })
    
    players_df = pl.DataFrame(player_summary) if player_summary else pl.DataFrame()
    
    return results_df, players_df, player_ratings


def show_top_players(players_df: pl.DataFrame, top_n: int, min_games: int = 5) -> Tuple[pl.DataFrame, str]:
    """Get top players by Elo rating with minimum games requirement using SQL."""
    if players_df.is_empty():
        return players_df, ""
    
    query = f"""
        WITH filtered AS (
            SELECT *
            FROM players_df
            WHERE games_played >= {min_games}
        )
        SELECT 
            ROW_NUMBER() OVER (ORDER BY elo_rating DESC, games_played DESC, player_name ASC) AS Rank,
            CAST(ROUND(elo_rating, 0) AS INTEGER) AS Elo_Rating,
            player_id AS Player_ID,
            player_name AS Player_Name,
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
                LAST(pair_name) AS pair_name,
                FIRST(player1_id) AS player1_id,
                FIRST(player2_id) AS player2_id,
                AVG(pair_elo) AS avg_pair_elo,
                AVG(percentage) AS avg_percentage,
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
            ROW_NUMBER() OVER (ORDER BY avg_pair_elo DESC, avg_percentage DESC, games_played DESC, pair_name ASC) AS Rank,
            CAST(ROUND(avg_pair_elo, 0) AS INTEGER) AS Pair_Elo,
            pair_id AS Pair_ID,
            pair_name AS Pair_Name,
            ROUND(avg_percentage, 1) AS Avg_Pct,
            games_played AS Games
        FROM filtered
        ORDER BY Rank ASC
        LIMIT {top_n}
    """
    
    result = duckdb.sql(query).pl()
    return result, query


# -------------------------------
# First-time Initialization
# -------------------------------
def initialize_session_state():
    """Initialize session state variables on first run."""
    if 'first_time' not in st.session_state:
        st.session_state.first_time = True
        
        # Set app build datetime from file modification time
        st.session_state.app_datetime = datetime.fromtimestamp(
            pathlib.Path(__file__).stat().st_mtime, 
            tz=timezone.utc
        ).strftime('%Y-%m-%d %H:%M:%S %Z')
    else:
        st.session_state.first_time = False


# -------------------------------
# Streamlit UI
# -------------------------------
def main():
    st.set_page_config(
        page_title="Unofficial FFBridge Elo Ratings Playground",
        page_icon="🃏",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Apply modern "Bridge Table" styling
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
        
        /* Radio button styling for better visibility */
        .stRadio > label {
            color: #ffc107 !important;
            font-weight: 600 !important;
        }
        
        .stRadio [data-testid="stMarkdownContainer"] p {
            color: #ffffff !important;
            font-size: 1rem !important;
            font-weight: 500 !important;
        }
        
        /* Selectbox label styling */
        .stSelectbox > label {
            color: #ffc107 !important;
            font-weight: 600 !important;
        }
        
        /* Slider label styling */
        .stSlider > label {
            color: #ffc107 !important;
            font-weight: 600 !important;
        }
        
        /* Checkbox styling */
        .stCheckbox label span,
        .stCheckbox label p,
        .stCheckbox [data-testid="stMarkdownContainer"] p {
            color: #ffffff !important;
            font-weight: 500 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    widen_scrollbars()
    
    # -------------------------------
    # Sidebar Controls (Defined first to get variables)
    # -------------------------------
    with st.sidebar:
        st.sidebar.caption(f"Build:{st.session_state.app_datetime}")
        # Organization selection
        organization = st.selectbox(
            "Bridge Organization",
            options=["FFBridge"],
            index=0,
            help="Select the bridge organization"
        )
        
        # Check authentication status
        session = get_ffbridge_session()
        if not session:
            st.error("❌ **Authentication Error**")
            st.markdown("""
                Please ensure you have a valid Bearer token in your `.env` file:
                `FFBRIDGE_BEARER_TOKEN=your_token_here`
            """)
            return
        

        # Tournament selection - derive from SERIES_NAMES constant
        tournament_options = {SERIES_NAMES[k]: k for k in ["all"] + VALID_SERIES_IDS}
        
        selected_tournament_label = st.selectbox(
            "Tournament Names",
            options=list(tournament_options.keys()),
            index=0,
            help="Select which simultaneous tournaments to analyze"
        )
        simultaneous_type = tournament_options[selected_tournament_label]
        
        # Club filter - uses clubs from previous data load (stored in session state)
        # Will be populated after first data load
        club_options = st.session_state.get('available_clubs', ["All Clubs"])
        selected_club = st.selectbox(
            "Filter by Club",
            options=club_options,
            index=0,
            help="Filter results to show only players/pairs from a specific club"
        )
        
        # Report type
        rating_type = st.radio(
            "Ranking Type",
            ["Players", "Pairs"],
            index=0,
            horizontal=True,
            help="Switch between individual and partnership rankings"
        )
        
        # Handicap score option (placeholder - not yet implemented)
        use_handicap = st.checkbox(
            "Use handicap score",
            value=True,
            help="Use handicap-adjusted scores for Elo calculations (not yet implemented)"
        )
        
        # Number of results
        top_n = st.slider(
            "Show Top N Players or Pairs",
            min_value=50,
            max_value=1000,
            value=250,
            step=50
        )
        
        # Minimum games
        min_games = st.slider(
            "Minimum Games required",
            min_value=1,
            max_value=100,
            value=10,
            help="Minimum tournaments played to appear in rankings"
        )

        # PDF Export button
        generate_pdf = st.button("Export Report to PDF File", use_container_width=True)

    # Header
    st.markdown(f"""
        <div style="text-align: center; padding: 0 0 1rem 0; margin-top: -2rem;">
            <h1 style="font-size: 2.8rem; margin-bottom: 0.2rem;">
                🃏 Unofficial FFBridge Elo Ratings Playground (Proof of Concept)
            </h1>
            <p style="color: #ffc107; font-size: 1.2rem; font-weight: 500; opacity: 0.9;">
                 Elo Ratings for "{selected_tournament_label}"
            </p>
        </div>
    """, unsafe_allow_html=True)

    # -------------------------------
    # Main Content
    # -------------------------------
    
    # Use session ID for cache invalidation
    session_id = hashlib.md5(st.session_state.get('ffbridge_user', 'env').encode()).hexdigest()[:8]
    
    # Automatically fetch data
    with st.spinner(f"Fetching {selected_tournament_label}..."):
        tournaments = fetch_tournament_list(simultaneous_type=simultaneous_type, limit=None, _session_id=session_id)
        
        if not tournaments:
            st.error("Failed to retrieve tournament data from FFBridge. Please verify your token.")
            return

    # Process tournaments and calculate Elo
    # Note: process_tournaments_to_elo handles individual tournament fetching and disk caching
    results_df, players_df, current_ratings = process_tournaments_to_elo(
        tournaments, initial_players=None
    )
    
    # Update available clubs in session state for the filter dropdown
    if not results_df.is_empty() and 'club_name' in results_df.columns:
        unique_clubs = sorted(set(results_df.select('club_name').to_series().to_list()))
        unique_clubs = [c for c in unique_clubs if c and c.strip()]  # Remove empty/whitespace strings
        new_clubs = ["All Clubs"] + unique_clubs
        # If clubs changed, update and rerun to refresh the dropdown
        if st.session_state.get('available_clubs') != new_clubs:
            st.session_state.available_clubs = new_clubs
            st.rerun()
    
    # Apply club filter if selected
    if selected_club != "All Clubs" and not results_df.is_empty():
        # Check if club_name column exists (may not exist in old cached data)
        if 'club_name' not in results_df.columns:
            st.warning("Club filtering requires refreshing cached data. Please delete the cache folder and reload.")
        else:
            # Filter results to only include pairs from the selected club
            results_df = results_df.filter(pl.col('club_name') == selected_club)
            
            # Recalculate players_df based on filtered results
            if not results_df.is_empty():
                # Get unique player IDs from filtered results (ensure string type)
                filtered_player_ids = set(
                    [str(pid) for pid in results_df.select('player1_id').to_series().to_list()] +
                    [str(pid) for pid in results_df.select('player2_id').to_series().to_list()]
                )
                players_df = players_df.filter(pl.col('player_id').cast(pl.Utf8).is_in(list(filtered_player_ids)))
            else:
                st.info(f"No results found for club: {selected_club}")
                players_df = pl.DataFrame()
    
    # Display top metrics in styled cards
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.markdown(f'<div class="metric-card"><small>Tournaments</small><br><span style="font-size:1.4rem; color:#ffc107; font-weight:700;">{len(tournaments)}</span></div>', unsafe_allow_html=True)
    
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
    
    # Store results_df in session state for detail view
    st.session_state.full_results_df = results_df
    
    # Display table logic with selectable rows
    if rating_type == "Players":
        st.markdown(f"### 🏆 Top {top_n} Players (Min. {min_games} games)")
        if not players_df.is_empty():
            top_players, sql_query = show_top_players(players_df, top_n, min_games)
            
            # Show SQL query in expander
            if sql_query:
                with st.expander("📝 SQL Query", expanded=False):
                    st.code(sql_query, language="sql")
            
            if not top_players.is_empty():
                # Convert to pandas for AgGrid
                display_df = top_players.to_pandas() if isinstance(top_players, pl.DataFrame) else top_players
                
                # Build AgGrid with double-click selection
                grid_response = build_selectable_aggrid(display_df, 'players_table_selectable')
                
                # Check if a row was selected
                selected_rows = grid_response.get('selected_rows', None)
                if selected_rows is not None and len(selected_rows) > 0:
                    selected_row = selected_rows.iloc[0] if hasattr(selected_rows, 'iloc') else selected_rows[0]
                    player_id = selected_row.get('Player_ID')
                    player_name = selected_row.get('Player_Name', 'Unknown')
                    
                    if player_id and not results_df.is_empty():
                        # Fetch official player profile with IV rating
                        session_id = get_session_id()
                        member_info = fetch_member_details(str(player_id), _session_id=session_id)
                        
                        # Display player profile card
                        st.markdown(f"### 👤 Player Profile: **{player_name}**")
                        
                        if member_info:
                            iv_info = member_info.get('iv', {})
                            iv_rating = iv_info.get('iv', 'N/A')
                            iv_label = iv_info.get('label', '')
                            licence_info = member_info.get('licence', {})
                            home_club = licence_info.get('organization_name', 'Unknown')
                            license_num = member_info.get('license_number', '')
                            
                            # Show profile in columns
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
                        
                        # Tabs for different views of tournament history
                        tab1, tab2 = st.tabs(["📊 Calculated Elo History", "📋 Official FFBridge Results"])
                        
                        with tab1:
                            # Local data with calculated Elo
                            player_results = results_df.filter(
                                (pl.col('player1_id') == str(player_id)) | 
                                (pl.col('player2_id') == str(player_id))
                            ).sort('date', descending=True)
                            
                            if not player_results.is_empty():
                                cols_to_select = [
                                    pl.col('date').str.slice(0, 10).alias('Date'),
                                    pl.col('tournament_name').alias('Tournament'),
                                    pl.col('pair_name').alias('Partner'),
                                    pl.col('percentage').round(2).alias('Pct'),
                                    pl.col('rank').alias('Rank'),
                                ]
                                if 'theoretical_rank' in player_results.columns and use_handicap:
                                    cols_to_select.append(pl.col('theoretical_rank').alias('Hcp_Rank'))
                                cols_to_select.append(pl.col('player1_elo_after').round(0).alias('Elo_After'))
                                
                                detail_df = player_results.select(cols_to_select)
                                ShowDataFrameTable(detail_df, key='player_detail_table', height_rows=10)
                            else:
                                st.info("No results in selected tournaments.")
                        
                        with tab2:
                            # Fetch complete history from FFBridge Person Results API
                            with st.spinner("Loading official FFBridge results..."):
                                ffb_results = fetch_person_results(str(player_id), _session_id=session_id)
                            
                            if ffb_results:
                                # Convert to DataFrame
                                ffb_data = []
                                for r in ffb_results:
                                    ffb_data.append({
                                        'Date': r.get('date', '')[:10] if r.get('date') else '',
                                        'Tournament': r.get('title', r.get('tournament_name', '')),
                                        'Club': r.get('organization_name', '')[:25],
                                        'Pct': round(float(r.get('result', 0)), 2),
                                        'Rank': r.get('rank', 0),
                                        'PE': r.get('pe', 0),
                                        'PE_Bonus': r.get('pe_bonus', 0),
                                    })
                                
                                ffb_df = pl.DataFrame(ffb_data).sort('Date', descending=True)
                                ShowDataFrameTable(ffb_df, key='ffb_results_table', height_rows=10)
                                
                                # Summary stats
                                total_pe = sum(r.get('pe', 0) or 0 for r in ffb_results)
                                total_bonus = sum(int(r.get('pe_bonus', 0) or 0) for r in ffb_results)
                                avg_pct = sum(float(r.get('result', 0) or 0) for r in ffb_results) / len(ffb_results) if ffb_results else 0
                                st.caption(f"**Total:** {len(ffb_results)} tournaments | Avg: {avg_pct:.1f}% | Total PE: {total_pe} (+{total_bonus} bonus)")
                            else:
                                st.info("No official FFBridge results found for this player.")
                
                st.session_state.display_df = top_players
                st.session_state.report_title = f"FFBridge Top Players - {datetime.now().strftime('%Y-%m-%d')}"
            else:
                st.info(f"No players match the minimum requirement of {min_games} games.")
    else:
        st.markdown(f"### 🏆 Top {top_n} Pairs (Min. {min_games} games)")
        if not results_df.is_empty():
            top_pairs, sql_query = show_top_pairs(results_df, top_n, min_games)
            
            # Show SQL query in expander
            if sql_query:
                with st.expander("📝 SQL Query", expanded=False):
                    st.code(sql_query, language="sql")
            
            if not top_pairs.is_empty():
                # Convert to pandas for AgGrid
                display_df = top_pairs.to_pandas() if isinstance(top_pairs, pl.DataFrame) else top_pairs
                
                # Build AgGrid with double-click selection
                grid_response = build_selectable_aggrid(display_df, 'pairs_table_selectable')
                
                # Check if a row was selected
                selected_rows = grid_response.get('selected_rows', None)
                if selected_rows is not None and len(selected_rows) > 0:
                    selected_row = selected_rows.iloc[0] if hasattr(selected_rows, 'iloc') else selected_rows[0]
                    pair_id = selected_row.get('Pair_ID')
                    pair_name = selected_row.get('Pair_Name', 'Unknown')
                    
                    if pair_id and not results_df.is_empty():
                        st.markdown(f"### 📋 Tournament History for **{pair_name}**")
                        
                        # Filter results for this pair
                        pair_results = results_df.filter(
                            pl.col('pair_id') == str(pair_id)
                        ).sort('date', descending=True)
                        
                        if not pair_results.is_empty():
                            # Select and format relevant columns including handicap rank
                            cols_to_select = [
                                pl.col('date').str.slice(0, 10).alias('Date'),
                                pl.col('tournament_name').alias('Tournament'),
                                pl.col('percentage').round(2).alias('Pct'),
                                pl.col('rank').alias('Rank'),
                            ]
                            # Add theoretical_rank if using handicap
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
    
    # Generate PDF logic
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
    
    # App info
    st.markdown(f"""
        <div style="text-align: center; color: #80cbc4; font-size: 0.8rem; opacity: 0.7;">
            Project lead is Robert Salita research@AiPolice.org. Code written in Python. UI written in streamlit. Data engine is polars. Repo:https://github.com/BSalita<br>
            Query Params:{st.query_params.to_dict()} Environment:{os.getenv('STREAMLIT_ENV','')}<br>
            Streamlit:{st.__version__} Python:{'.'.join(map(str, sys.version_info[:3]))} pandas:{pd.__version__} polars:{pl.__version__} duckdb:{duckdb.__version__} endplay:{ENDPLAY_VERSION}
        </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown(f"""
        <div style="text-align: center; padding: 2rem 0; color: #80cbc4; font-size: 0.9rem; opacity: 0.8;">
            Data sourced from FFBridge API • {selected_tournament_label}<br>
            System Current Date: {datetime.now().strftime('%Y-%m-%d')}
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

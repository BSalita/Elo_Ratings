# streamlit_app_ffbridge_elo_ratings_lancelot_api.py
"""
FFBridge (Fédération Française de Bridge) Elo Ratings Streamlit Application
Using the Lancelot API for more historical data.

This app fetches duplicate bridge tournament results from FFBridge's Lancelot API
and calculates Elo ratings based on percentage scores.

Key benefits of Lancelot API:
- No authentication required (public API)
- More historical data (620+ sessions for Rondes de France alone)
- Full PBN deal data available for board-level analysis
"""

import os
import pathlib
import sys
import json
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, List, Dict, Any

import pandas as pd
import polars as pl
import streamlit as st
import requests
import duckdb

from streamlitlib.streamlitlib import (
    ShowDataFrameTable,
    create_pdf,
    widen_scrollbars,
)
from st_aggrid import GridOptionsBuilder, AgGrid, ColumnsAutoSizeMode, AgGridTheme

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
CACHE_DIR = DATA_ROOT / 'lancelot_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Lancelot API base URL (public, no auth required)
LANCELOT_API_BASE = "https://api-lancelot.ffbridge.fr"

# Lancelot ID to Migration ID (FFBridge series ID) mapping
# Lancelot uses sequential IDs, we need to map to our familiar series IDs
LANCELOT_TO_MIGRATION = {
    1: 3,    # Rondes de France
    2: 4,    # Trophées du Voyage
    3: 5,    # Roy René
    17: 140,  # Amour du Bridge
    25: 384,  # Simultanet
    27: 386,  # Simultané Octopus
    47: 604,  # Atout Simultané
    62: 868,  # Festival des Simultanés
}

# Reverse mapping for lookups
MIGRATION_TO_LANCELOT = {v: k for k, v in LANCELOT_TO_MIGRATION.items()}

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

# List of all valid tournament IDs (migration IDs, same as main API)
VALID_SERIES_IDS = [3, 4, 5, 140, 384, 386, 604, 868]

# Default Elo parameters
DEFAULT_ELO = 1500.0
K_FACTOR = 32.0
PERFORMANCE_SCALING = 400  # Standard Elo scaling factor

# UI Constants
REQUEST_TIMEOUT = 30  # seconds (Lancelot can be slower)
REQUEST_DELAY = 0.1  # seconds between API requests to avoid rate limiting
AGGRID_ROW_HEIGHT = 42
AGGRID_HEADER_HEIGHT = 50
AGGRID_FOOTER_HEIGHT = 20
AGGRID_MAX_DISPLAY_ROWS = 10


# -------------------------------
# Persistent Disk Cache Helpers
# -------------------------------
def _normalize_series_id(series_id: Optional[Any]) -> Optional[Any]:
    """Normalize series_id so cache folders are stable (e.g. '3' and 3 map the same)."""
    if series_id is None:
        return None
    if isinstance(series_id, str):
        s = series_id.strip()
        if s.isdigit():
            try:
                return int(s)
            except Exception:
                return series_id
        return series_id
    try:
        if isinstance(series_id, bool):
            return series_id
        if isinstance(series_id, float) and series_id.is_integer():
            return int(series_id)
    except Exception:
        pass
    return series_id


def get_cache_path(identifier: str, params: Optional[Dict] = None, series_id: Optional[Any] = None) -> pathlib.Path:
    """Generate a readable, unique filename in a series-specific subdirectory."""
    # Determine the subdirectory based on series_id (friendly name)
    norm_series_id = _normalize_series_id(series_id)
    series_folder = SERIES_NAMES.get(norm_series_id, SERIES_NAMES.get(series_id, "General"))
    target_dir = CACHE_DIR / series_folder
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize identifier to be a safe filename
    safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', identifier)
    
    # If there are params, append them in a readable format
    if params:
        param_parts = [f"{k}_{'all' if v is None else v}" for k, v in sorted(params.items())]
        safe_name = f"{safe_name}_{'_'.join(param_parts)}"
    
    return target_dir / f"{safe_name}.json"


def save_to_disk_cache(identifier: str, data: Any, params: Optional[Dict] = None, series_id: Optional[Any] = None):
    """Save response data to disk as JSON."""
    cache_path = get_cache_path(identifier, params, series_id)
    # print(f"[CACHE] Saving to: {cache_path}")
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'identifier': identifier,
                'series_id': series_id,
                'api': 'lancelot',
                'data': data
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # Silently fail on cache save errors


def load_from_disk_cache(identifier: str, params: Optional[Dict] = None, max_age_hours: Optional[int] = None, series_id: Optional[Any] = None) -> Optional[Any]:
    """Load data from disk if it exists and isn't too old."""
    cache_path = get_cache_path(identifier, params, series_id)
    
    # Fallback for older cache filename format (results_ID.json instead of results_ID_DATE.json)
    if not cache_path.exists() and "ranking_" in identifier:
        # Extract just the ID part: ranking_12345_2023-01-01 -> ranking_12345
        id_match = re.search(r'(ranking_\d+)', identifier)
        if id_match:
            legacy_identifier = id_match.group(1)
            legacy_path = get_cache_path(legacy_identifier, params, series_id)
            if legacy_path.exists():
                print(f"[CACHE] Found legacy file: {legacy_path}")
                cache_path = legacy_path

    # Log cache lookup for debugging (disabled - too verbose)
    # print(f"[CACHE] Looking for: {cache_path} (exists: {cache_path.exists()})")
    
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
    """
    actual_score = percentage / 100.0
    expected_score = calculate_expected_score(current_rating, field_average_rating)
    rating_change = k_factor * (actual_score - expected_score)
    return current_rating + rating_change


# -------------------------------
# Helper Functions
# -------------------------------
def calculate_aggrid_height(row_count: int) -> int:
    """Calculate AgGrid height based on row count."""
    display_rows = min(AGGRID_MAX_DISPLAY_ROWS, row_count)
    return AGGRID_HEADER_HEIGHT + (AGGRID_ROW_HEIGHT * display_rows) + AGGRID_FOOTER_HEIGHT


def build_selectable_aggrid(df: pd.DataFrame, key: str) -> Dict[str, Any]:
    """
    Build an AgGrid with single-click row selection.
    Returns the grid response dictionary.
    """
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
# Lancelot API Functions (No Authentication Required)
# -------------------------------

def lancelot_get(endpoint: str, params: Optional[Dict] = None, add_delay: bool = True) -> Optional[Any]:
    """
    Make a GET request to the Lancelot API (public, no auth needed).
    Includes rate limiting to avoid being blacklisted.
    """
    url = f"{LANCELOT_API_BASE}{endpoint}"
    try:
        # Add delay to avoid rate limiting (only for non-cached requests)
        if add_delay:
            time.sleep(REQUEST_DELAY)
        
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            st.warning("Rate limited by Lancelot API. Waiting 5 seconds...")
            time.sleep(5)
            return lancelot_get(endpoint, params, add_delay=False)  # Retry once
    except Exception as e:
        st.warning(f"Lancelot API error: {e}")
    return None


@st.cache_data(ttl=86400, show_spinner=False)  # Cache for 24 hours
def fetch_current_season() -> Optional[Dict[str, Any]]:
    """Fetch current bridge season info."""
    return lancelot_get("/seasons/current")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_series_info(lancelot_id: int) -> Optional[Dict[str, Any]]:
    """Fetch series metadata by Lancelot ID."""
    return lancelot_get(f"/competitions/simultaneous/{lancelot_id}")


def fetch_sessions_for_series(lancelot_id: int, max_pages: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch all sessions (tournament dates) for a simultaneous series.
    Uses pagination to get complete history.
    """
    migration_id = LANCELOT_TO_MIGRATION.get(lancelot_id)
    series_name = SERIES_NAMES.get(migration_id, f"series_{lancelot_id}")
    
    # Friendly cache filename: sessions_list_<series_name>
    friendly_name = f"sessions_list_{series_name.replace(' ', '_')}"
    
    # Check disk cache first (never expires as historical results are static)
    cached_data = load_from_disk_cache(friendly_name, max_age_hours=None, series_id=migration_id)
    if cached_data:
        return cached_data
    
    all_sessions = []
    page = 1
    
    while page <= max_pages:
        data = lancelot_get(
            f"/competitions/simultaneous/{lancelot_id}/sessions",
            params={"currentPage": page, "maxPerPage": 80}
        )
        
        if not data or 'items' not in data:
            break
            
        items = data['items']
        all_sessions.extend(items)
        
        # Check if there are more pages
        pagination = data.get('pagination', {})
        if not pagination.get('has_next_page', False):
            break
            
        page += 1
    
    # Save to disk cache with friendly filename
    if all_sessions:
        save_to_disk_cache(friendly_name, all_sessions, series_id=migration_id)
    
    return all_sessions


def fetch_session_ranking(session_id: int, session_label: str = "", series_id: Optional[int] = None) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Fetch full ranking for a session (all clubs combined).
    Returns tuple of (list of team results, was_cached).
    """
    # Create friendly filename: ranking_<session_id>_<date_label>
    # Extract date from label like "Rondes de France 2026-01-06 Après-midi" -> "2026-01-06"
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', session_label)
    date_part = date_match.group(1) if date_match else ""
    friendly_name = f"ranking_{session_id}_{date_part}" if date_part else f"ranking_{session_id}"
    old_name = f"ranking_{session_id}"
    
    # Check disk cache
    cached_data = load_from_disk_cache(friendly_name, max_age_hours=None, series_id=series_id)
    if cached_data:
        return cached_data, True
    
    # Fetch from API (no pagination needed for ranking - returns full list)
    data = lancelot_get(f"/results/sessions/{session_id}/ranking")
    
    if data and isinstance(data, list):
        # Save to disk cache with friendly filename
        save_to_disk_cache(friendly_name, data, series_id=series_id)
        return data, False  # Return fresh data and flag
    
    return [], False


def fetch_session_clubs(session_id: int) -> List[Dict[str, Any]]:
    """Get all clubs that participated in a session with disk caching."""
    # Check disk cache first
    cache_name = f"clubs_{session_id}"
    cached_data = load_from_disk_cache(cache_name, max_age_hours=None, series_id=None)
    if cached_data:
        return cached_data
    
    # Fetch from API
    data = lancelot_get(f"/results/sessions/{session_id}/simultaneousIds")
    if isinstance(data, list):
        # print(f"[DEBUG] fetch_session_clubs({session_id}) returned {len(data)} clubs")
        save_to_disk_cache(cache_name, data, series_id=None)
        return data
    return []


def build_club_name_mapping(unique_codes: List[str], sessions: List[Dict[str, Any]], results_df=None) -> Dict[str, str]:
    """
    Build a mapping of club codes to club names.
    Uses disk cache and session state to avoid repeated API calls.
    If results_df is provided, uses it to find which sessions contain missing clubs.
    """
    # Check if we already have a cached mapping in session state
    if 'lancelot_club_mapping' not in st.session_state:
        st.session_state.lancelot_club_mapping = {}
    
    mapping = st.session_state.lancelot_club_mapping
    
    # Helper to normalize codes to stripped strings (handle potential leading zeros)
    def normalize_code(c: Any) -> str:
        if c is None: return ""
        s = str(c).strip()
        if not s or s.lower() == 'none': return ""
        # Remove leading zeros but keep at least one digit if it's all zeros
        norm = s.lstrip('0')
        return norm if norm else '0'
    
    # Find which codes we still need to look up
    normalized_unique = [normalize_code(c) for c in unique_codes if c]
    missing_codes = set(normalized_unique) - set(mapping.keys())
    
    if missing_codes:
        # print(f"[DEBUG] Missing club names for normalized codes: {list(missing_codes)[:20]}...")
        # Build a set of session IDs to check - prioritize sessions that have missing clubs
        sessions_to_check = set()
        
        # If we have results_df, find sessions that contain missing clubs
        if results_df is not None and not results_df.is_empty() and 'session_id' in results_df.columns:
            # Check sessions for EACH missing code (up to a limit)
            for code in list(missing_codes)[:100]:
                matches = results_df.filter(pl.col('club_code') == code)
                if not matches.is_empty():
                    # Get up to 3 session IDs for this code to increase chances of finding it
                    s_ids = matches.select('session_id').head(3).to_series().to_list()
                    for sid in s_ids:
                        sessions_to_check.add(str(sid))
        
        # Also add some recent sessions as fallback
        if len(sessions_to_check) < 20:
            sorted_sessions = sorted(sessions, key=lambda x: x.get('date', ''), reverse=True)
            for s in sorted_sessions[:40]:
                sessions_to_check.add(str(s.get('id', '')))
        
        # Fetch clubs from identified sessions
        # print(f"[DEBUG] Checking {len(sessions_to_check)} sessions for club names...")
        found_count = 0
        for session_id in sessions_to_check:
            if not missing_codes:
                break
            if session_id:
                clubs = fetch_session_clubs(int(session_id) if session_id.isdigit() else session_id)
                for club in clubs:
                    # Normalize all possible ID fields
                    l_id = normalize_code(club.get('id'))
                    ffb_code = normalize_code(club.get('ffbCode'))
                    name = club.get('label', '')
                    
                    if name:
                        if l_id and l_id in missing_codes:
                            mapping[l_id] = name
                            missing_codes.discard(l_id)
                            found_count += 1
                        if ffb_code and ffb_code in missing_codes:
                            mapping[ffb_code] = name
                            missing_codes.discard(ffb_code)
                            found_count += 1
        
        print(f"[DEBUG] Club mapping: found {found_count} new names, {len(missing_codes)} still missing")
    
    return mapping


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_available_series() -> List[Dict[str, Any]]:
    """Fetch all available simultaneous series from the Lancelot API."""
    data = lancelot_get(
        "/results/search/",
        params={"searchCompetitionType": "clubSimultaneous", "searchSeason": "current", "currentPage": 1}
    )
    if data and 'items' in data:
        return data['items']
    return []


# -------------------------------
# Data Processing
# -------------------------------
def process_sessions_to_elo(
    sessions: List[Dict[str, Any]],
    series_id: int,
    initial_players: Optional[Dict[str, Dict]] = None,
    use_handicap: bool = False,
) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, float]]:
    """
    Process session list and calculate Elo ratings from Lancelot API data.
    
    Args:
        sessions: List of session dicts from Lancelot API
        series_id: The migration ID (our familiar series ID)
        initial_players: Optional dict of initial player info
    
    Returns:
        Tuple of (all_results_df, player_ratings_df, current_ratings_dict)
    """
    all_results = []
    player_ratings: Dict[str, float] = {}
    player_names: Dict[str, str] = {}
    player_games: Dict[str, int] = {}
    # Welford running stats per player for percentage (mean / sample stdev)
    player_pct_n: Dict[str, int] = {}
    player_pct_mean: Dict[str, float] = {}
    player_pct_m2: Dict[str, float] = {}
    
    # Initialize from initial_players if provided
    if initial_players:
        for pid, pinfo in initial_players.items():
            player_ratings[pid] = pinfo.get('elo', DEFAULT_ELO)
            player_names[pid] = pinfo.get('name', pid)
            player_games[pid] = pinfo.get('games_played', 0)
    
    # Sort sessions chronologically (oldest first for proper Elo progression)
    sorted_sessions = sorted(sessions, key=lambda x: x.get('date', ''))
    
    # Filter out future sessions
    today = datetime.now().strftime('%Y-%m-%d')
    sorted_sessions = [s for s in sorted_sessions if s.get('date', '')[:10] <= today]
    
    # Progress tracking for UI
    progress_bar = st.progress(0)
    status_text = st.empty()
    cache_stats = {"cached": 0, "fetched": 0}
    
    total_s = len(sorted_sessions)
    for i, session in enumerate(sorted_sessions):
        s_id = session.get('id')
        s_label = session.get('label', f"Session {s_id}")
        s_date = session.get('date', '')
        s_series_id = _normalize_series_id(session.get('series_id', series_id))  # Use session's series_id for correct cache folder
        
        # Display progress with cache stats (white text for visibility)
        cache_info = f"[Cached: {cache_stats['cached']}, Fetched: {cache_stats['fetched']}]"
        status_text.markdown(f"<span style='color: white;'>Processing {i+1}/{total_s}: {s_label[:35]}... {cache_info}</span>", unsafe_allow_html=True)
        progress_bar.progress((i + 1) / total_s)
        
        # Fetch ranking for this session (use session's series_id for correct cache folder)
        ranking, was_cached = fetch_session_ranking(s_id, session_label=s_label, series_id=s_series_id)
        
        # Track cache statistics
        if was_cached:
            cache_stats["cached"] += 1
        else:
            cache_stats["fetched"] += 1
        
        if not ranking:
            continue
        
        # Extract results from ranking
        results = []
        for entry in ranking:
            # Skip if entry is not a dict (API sometimes returns unexpected data)
            if not isinstance(entry, dict):
                continue
                
            team = entry.get('team', {})
            if not isinstance(team, dict):
                continue
                
            p1 = team.get('player1')
            p2 = team.get('player2')
            
            # Skip if players are not proper dicts (sometimes they're strings or None)
            if not isinstance(p1, dict) or not isinstance(p2, dict):
                continue
            
            # Use migrationId to get FFBridge-compatible player ID
            p1_id = str(p1.get('migrationId') or p1.get('id', ''))
            p2_id = str(p2.get('migrationId') or p2.get('id', ''))
            p1_name = f"{p1.get('firstName', '')} {p1.get('lastName', '')}".strip()
            p2_name = f"{p2.get('firstName', '')} {p2.get('lastName', '')}".strip()
            
            # Score/percentage from sessionScore or totalScore
            pct = float(entry.get('sessionScore') or entry.get('totalScore') or 0)
            pe_bonus = float(entry.get('peBonus') or 0)
            
            results.append({
                'team_id': str(team.get('id', '')),
                'player1_id': p1_id,
                'player2_id': p2_id,
                'player1_name': p1_name,
                'player2_name': p2_name,
                'percentage': pct,
                # Lancelot `sessionScore/totalScore` behaves like the handicapped score in practice.
                'handicap_percentage': pct,
                # Derive "club score" by subtracting Team_PE_Bonus/10 from handicapped score
                'club_percentage': pct - pe_bonus / 10.0,
                'rank': entry.get('rank', 0),
                'rank_without_handicap': entry.get('rankWithoutHandicap'),
                'pe': entry.get('pe', 0),
                'pe_bonus': pe_bonus,
                'club_code': str(entry.get('simultaneousId', '')),
            })
        
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
            club_pct_raw = result.get('club_percentage')
            handicap_pct_raw = result.get('handicap_percentage')
            club_code = result.get('club_code', '')
            
            try:
                handicap_pct = float(handicap_pct_raw) if handicap_pct_raw is not None else float(result.get('percentage', 50.0) or 50.0)
            except (ValueError, TypeError):
                handicap_pct = float(result.get('percentage', 50.0) or 50.0)

            # Derive club % if missing
            pe_bonus_raw = result.get('pe_bonus', 0)
            try:
                pe_bonus = float(pe_bonus_raw or 0)
            except (ValueError, TypeError):
                pe_bonus = 0.0
            try:
                club_pct = float(club_pct_raw) if club_pct_raw is not None else (handicap_pct - pe_bonus / 10.0)
            except (ValueError, TypeError):
                club_pct = handicap_pct - pe_bonus / 10.0

            percentage = handicap_pct if use_handicap else club_pct

            rank_with_handicap = result.get('rank', 0)
            rank_without_handicap = result.get('rank_without_handicap')
            rank = rank_with_handicap if use_handicap else (rank_without_handicap if rank_without_handicap is not None else rank_with_handicap)
            
            # Create stable pair identification
            if p1_id and p2_id:
                if p1_id < p2_id:
                    pair_id = f"{p1_id}_{p2_id}"
                    stable_p1_name, stable_p2_name = p1_name, p2_name
                else:
                    pair_id = f"{p2_id}_{p1_id}"
                    stable_p1_name, stable_p2_name = p2_name, p1_name
                pair_name = f"{stable_p1_name} - {stable_p2_name}"
            else:
                pair_id = result.get('team_id', f"{p1_id}-{p2_id}")
                pair_name = f"{p1_name} - {p2_name}"
            
            # Store result with explicit type conversion
            # Normalize club_code to string and strip leading zeros for consistent mapping
            cc_raw = str(club_code).strip()
            if not cc_raw or cc_raw.lower() == 'none':
                cc_norm = "0"
            else:
                cc_norm = cc_raw.lstrip('0')
                if not cc_norm: cc_norm = "0"
            
            result_record = {
                'session_id': str(s_id),
                'session_name': str(s_label),
                'date': str(s_date),
                'series_id': int(s_series_id) if s_series_id else 0,
                'team_id': str(result.get('team_id', '')),
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
                'pe': float(result.get('pe', 0) or 0),
                'pe_bonus': float(result.get('pe_bonus', 0) or 0),
                'field_avg_rating': float(field_avg),
                'club_code': cc_norm,
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
                # Update running pct stats for player 1
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
                # Update running pct stats for player 2
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
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Convert to DataFrames
    results_df = pl.DataFrame(all_results) if all_results else pl.DataFrame()
    
    # Create player ratings summary with explicit type conversion
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
    
    # Create DataFrame with explicit schema to avoid type inference issues
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
                LAST(pair_name) AS pair_name,
                FIRST(player1_id) AS player1_id,
                FIRST(player2_id) AS player2_id,
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
            ROW_NUMBER() OVER (ORDER BY avg_pair_elo DESC, avg_percentage DESC, games_played DESC, pair_name ASC) AS Rank,
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
        
        .lancelot-badge {
            background: linear-gradient(135deg, #00796b, #004d40);
            border: 1px solid #4db6ac;
            border-radius: 6px;
            padding: 0.3rem 0.8rem;
            display: inline-block;
            font-size: 0.75rem;
            color: #b2dfdb;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    widen_scrollbars()
    
    # -------------------------------
    # Sidebar Controls
    # -------------------------------
    with st.sidebar:
        st.sidebar.caption(f"Build:{st.session_state.app_datetime}")
        
        # Organization selection
        organization = st.selectbox(
            "Bridge API",
            options=["FFBridge (Lancelot)"],
            index=0,
            help="Using Lancelot API for extended historical data"
        )
        
        # Tournament selection - derive from SERIES_NAMES constant
        tournament_options_list = ["all"] + VALID_SERIES_IDS
        tournament_labels = [SERIES_NAMES[k] for k in tournament_options_list]
        
        # Debug: Track widget state before initialization
        print(f"[DEBUG] Widget state BEFORE init: lancelot_tournament_selectbox = {st.session_state.get('lancelot_tournament_selectbox', 'NOT SET')}")
        
        # Initialize widget state if not present (ensures "All Tournaments" is default)
        if "lancelot_tournament_selectbox" not in st.session_state:
            st.session_state.lancelot_tournament_selectbox = tournament_labels[0]  # "All Tournaments"
            print(f"[DEBUG] Initialized lancelot_tournament_selectbox to: {tournament_labels[0]}")
        
        # Validate the stored value is still valid
        if st.session_state.lancelot_tournament_selectbox not in tournament_labels:
            print(f"[DEBUG] Invalid stored value '{st.session_state.lancelot_tournament_selectbox}', resetting to: {tournament_labels[0]}")
            st.session_state.lancelot_tournament_selectbox = tournament_labels[0]
        
        print(f"[DEBUG] Widget state AFTER init/validation: lancelot_tournament_selectbox = {st.session_state.lancelot_tournament_selectbox}")
        
        selected_tournament_label = st.selectbox(
            "Tournament Names",
            options=tournament_labels,
            key="lancelot_tournament_selectbox",
            help="Select which simultaneous tournaments to analyze"
        )
        
        print(f"[DEBUG] Selectbox RETURNED: {selected_tournament_label}")
        
        # Determine selected series from the label
        selected_series = tournament_options_list[tournament_labels.index(selected_tournament_label)]
        print(f"[DEBUG] selected_series resolved to: {selected_series}")
        
        # Club filter - uses clubs from previous data load (stored in session state)
        club_options = st.session_state.get('lancelot_available_clubs', ["All Clubs"])
        
        # Debug: Track club widget state
        print(f"[DEBUG] Widget state BEFORE init: lancelot_club_selectbox = {st.session_state.get('lancelot_club_selectbox', 'NOT SET')}")
        
        # Initialize widget state if not present
        if "lancelot_club_selectbox" not in st.session_state:
            st.session_state.lancelot_club_selectbox = "All Clubs"
            print(f"[DEBUG] Initialized lancelot_club_selectbox to: All Clubs")
        
        # Validate the stored club is still in the options (list may have changed)
        if st.session_state.lancelot_club_selectbox not in club_options:
            print(f"[DEBUG] Invalid stored club '{st.session_state.lancelot_club_selectbox}', resetting to: All Clubs")
            st.session_state.lancelot_club_selectbox = "All Clubs"
        
        selected_club = st.selectbox(
            "Filter by Club",
            options=club_options,
            key="lancelot_club_selectbox",
            help="Filter results to show only players/pairs from a specific club"
        )
        
        print(f"[DEBUG] Club selectbox RETURNED: {selected_club}")
        
        # Report type
        rating_type = st.radio(
            "Ranking Type",
            ["Players", "Pairs"],
            index=0,
            key="lancelot_rating_type",
            horizontal=True,
            help="Switch between individual and partnership rankings"
        )

        # Handicap score option (controls whether we use handicap-adjusted % when available)
        use_handicap = st.checkbox(
            "Use handicap score",
            value=True,
            key="lancelot_use_handicap",
            help="If available in the API data, uses handicap-adjusted percentage (otherwise uses club percentage)."
        )
        
        # Number of results
        top_n = st.slider(
            "Show Top N Players or Pairs",
            min_value=50,
            max_value=1000,
            value=250,
            step=50,
            key="lancelot_top_n"
        )
        
        # Minimum games
        min_games = st.slider(
            "Minimum Games required",
            min_value=1,
            max_value=100,
            value=10,
            key="lancelot_min_games",
            help="Minimum sessions played to appear in rankings"
        )

        # PDF Export button
        generate_pdf = st.button("Export Report to PDF File", use_container_width=True)

    # Header
    st.markdown(f"""
        <div style="text-align: center; padding: 0 0 1rem 0; margin-top: -2rem;">
            <h1 style="font-size: 2.5rem; margin-bottom: 0.2rem;">
                🃏 Unofficial FFBridge Elo Ratings Playground (Lancelot API) Proof of Concept
            </h1>
            <p style="color: #ffc107; font-size: 1.2rem; font-weight: 500; opacity: 0.9;">
                Extended Historical Data • {selected_tournament_label}
            </p>
        </div>
    """, unsafe_allow_html=True)

    # -------------------------------
    # Main Content - ALWAYS load ALL tournaments once, then filter
    # -------------------------------
    
    # Always fetch ALL sessions regardless of UI selection
    all_sessions = []
    
    with st.spinner("Loading all tournament sessions..."):
        for migration_id in VALID_SERIES_IDS:
            lancelot_id = MIGRATION_TO_LANCELOT.get(migration_id)
            if lancelot_id:
                sessions = fetch_sessions_for_series(lancelot_id, max_pages=15)
                # Add series info to each session
                for s in sessions:
                    s['series_id'] = migration_id
                    s['series_name'] = SERIES_NAMES.get(migration_id, 'Unknown')
                all_sessions.extend(sessions)
        
        if not all_sessions:
            st.error("Failed to retrieve session data from Lancelot API.")
            return
    
    # Sort all sessions chronologically
    all_sessions.sort(key=lambda x: x.get('date', ''))
    
    # Cache key is now ONLY based on handicap setting (we always load everything)
    cache_key = f"lancelot_full_v1_{len(all_sessions)}_handicap_{int(use_handicap)}"
    
    # Use a single cache entry for the full dataset
    if 'lancelot_full_cache' not in st.session_state:
        st.session_state.lancelot_full_cache = {}
    
    full_cache = st.session_state.lancelot_full_cache
    
    # Check if we have cached results for ALL data
    if cache_key in full_cache:
        cached = full_cache[cache_key]
        full_results_df = cached['results_df']
        full_players_df = cached['players_df']
        current_ratings = cached['current_ratings']
        print(f"[DEBUG] Full dataset cache hit: {cache_key}")
    else:
        # Check if this is an unexpected reload (fail fast)
        if full_cache:
            old_keys = list(full_cache.keys())
            st.error(f"⚠️ Unexpected cache miss! Old keys: {old_keys}, New key: {cache_key}. Reload the page to fix.")
            st.stop()
        
        print(f"[DEBUG] Full dataset cache miss (initial load): {cache_key}")
        
        # Process ALL sessions and calculate Elo (this is the one-time expensive operation)
        full_results_df, full_players_df, current_ratings = process_sessions_to_elo(
            all_sessions,
            series_id=3,  # Default series for cache folder
            initial_players=None,
            use_handicap=use_handicap,
        )
        
        # Cache the full dataset
        full_cache[cache_key] = {
            'results_df': full_results_df,
            'players_df': full_players_df,
            'current_ratings': current_ratings,
        }
    
    # Now filter the cached data based on UI selections (instant operation)
    print(f"[DEBUG] Filtering: series={selected_series}, club={selected_club}")
    print(f"[DEBUG] Full results_df has {len(full_results_df)} rows")
    
    # Start with full data, then apply filters
    results_df = full_results_df
    
    # Apply tournament filter if not "all"
    if selected_series != "all":
        if 'series_id' in results_df.columns:
            results_df = results_df.filter(pl.col('series_id') == selected_series)
            print(f"[DEBUG] After series filter: {len(results_df)} rows")
        else:
            st.warning("⚠️ Cached data is missing series_id column. Please reload the page.")
    
    # Apply club filter if not "All Clubs"
    if selected_club != "All Clubs" and not results_df.is_empty():
        if 'club_code' in results_df.columns:
            code_mapping = st.session_state.get('lancelot_club_name_to_code', {})
            filter_code = code_mapping.get(selected_club, selected_club)
            print(f"[DEBUG] Club filter: '{selected_club}' -> code '{filter_code}'")
            results_df = results_df.filter(pl.col('club_code') == filter_code)
            print(f"[DEBUG] After club filter: {len(results_df)} rows")
    
    # Recalculate players_df from filtered results (so stats reflect the filter)
    if not results_df.is_empty():
        # Use DuckDB to aggregate player stats from the filtered results
        players_df = duckdb.sql("""
            WITH player_results AS (
                -- Combine player1 and player2 into a single player view
                SELECT 
                    player1_id AS player_id,
                    player1_name AS player_name,
                    player1_elo_after AS elo_rating,
                    percentage
                FROM results_df
                UNION ALL
                SELECT 
                    player2_id AS player_id,
                    player2_name AS player_name,
                    player2_elo_after AS elo_rating,
                    percentage
                FROM results_df
            )
            SELECT 
                player_id,
                LAST(player_name) AS player_name,
                ROUND(LAST(elo_rating), 1) AS elo_rating,
                COUNT(*) AS games_played,
                ROUND(AVG(percentage), 2) AS avg_percentage,
                ROUND(STDDEV_SAMP(percentage), 2) AS stdev_percentage
            FROM player_results
            GROUP BY player_id
        """).pl()
        print(f"[DEBUG] Recalculated players_df: {len(players_df)} rows")
    else:
        players_df = pl.DataFrame()
        st.info(f"No results found for the selected filters.")
    
    print(f"[DEBUG] Final: results_df={len(results_df)} rows, players_df={len(players_df)} rows")

    # Populate club options with human-readable names (using FULL results, not filtered)
    if not full_results_df.is_empty() and 'club_code' in full_results_df.columns:
        unique_codes = sorted(set(full_results_df.select('club_code').to_series().to_list()))
        unique_codes = [c for c in unique_codes if c and str(c).strip()]
        
        # Build club code -> name mapping (fetches from API as needed)
        club_mapping = build_club_name_mapping(unique_codes, all_sessions, full_results_df)
        
        # Create list of club names (with normalized code as fallback if name not found)
        club_names = []
        code_to_name = {}
        for code in unique_codes:
            name = club_mapping.get(code, f"Club {code}") 
            club_names.append(name)
            code_to_name[name] = code
        
        # Update available clubs in session state
        club_names = sorted(set(club_names))
        current_selected = st.session_state.get("lancelot_club_selectbox", "All Clubs")
        
        old_clubs = st.session_state.get('lancelot_available_clubs', ["All Clubs"])
        merged_names = {c.strip() for c in old_clubs}
        for n in club_names:
            if n.strip():
                merged_names.add(n.strip())
        
        if current_selected and current_selected != "All Clubs":
            merged_names.add(current_selected.strip())
        
        merged_sorted = sorted([c for c in merged_names if c != "All Clubs"])
        new_clubs = ["All Clubs"] + merged_sorted
        
        old_map = st.session_state.get('lancelot_club_name_to_code', {})
        merged_map = dict(old_map)
        merged_map.update(code_to_name)
        
        if len(new_clubs) > len(old_clubs) or merged_map != old_map:
            print(f"[DEBUG] Updating available clubs: {len(old_clubs)} -> {len(new_clubs)}")
            st.session_state.lancelot_available_clubs = new_clubs
            st.session_state.lancelot_club_name_to_code = merged_map
            st.rerun()
        else:
            st.session_state.lancelot_available_clubs = new_clubs
            st.session_state.lancelot_club_name_to_code = merged_map

    
    # Display top metrics in styled cards
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.markdown(f'<div class="metric-card"><small>Sessions</small><br><span style="font-size:1.4rem; color:#ffc107; font-weight:700;">{len(all_sessions)}</span></div>', unsafe_allow_html=True)
    
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
                        st.markdown(f"### 📋 Session History for **{player_name}**")
                        
                        # Filter results for this player
                        player_results = results_df.filter(
                            (pl.col('player1_id') == str(player_id)) | 
                            (pl.col('player2_id') == str(player_id))
                        ).sort('date', descending=True)
                        
                        if not player_results.is_empty():
                            # Select and format relevant columns
                            detail_df = player_results.select([
                                pl.col('date').str.slice(0, 10).alias('Date'),
                                pl.col('session_id').alias('Event_ID'),
                                pl.col('session_name').alias('Session'),
                                pl.col('pair_name').alias('Partner'),
                                (pl.col('club_percentage') if 'club_percentage' in player_results.columns else pl.col('percentage')).cast(pl.Float64, strict=False).round(2).alias('Club_Score'),
                                (pl.col('handicap_percentage') if 'handicap_percentage' in player_results.columns else pl.lit(None, dtype=pl.Float64)).cast(pl.Float64, strict=False).round(2).alias('Handicap_Score'),
                                pl.col('percentage').cast(pl.Float64, strict=False).round(2).alias('Pct_Used'),
                                pl.col('rank').alias('Rank'),
                                pl.col('pe').alias('PE'),
                                pl.col('player1_elo_after').round(0).alias('Elo_After'),
                            ])
                            ShowDataFrameTable(detail_df, key='player_detail_table', height_rows=10)
                        else:
                            st.info("No detailed results found for this player.")
                
                st.session_state.display_df = top_players
                st.session_state.report_title = f"FFBridge Top Players (Lancelot) - {datetime.now().strftime('%Y-%m-%d')}"
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
                        st.markdown(f"### 📋 Session History for **{pair_name}**")
                        
                        # Filter results for this pair
                        pair_results = results_df.filter(
                            pl.col('pair_id') == str(pair_id)
                        ).sort('date', descending=True)
                        
                        if not pair_results.is_empty():
                            # Select and format relevant columns
                            detail_df = pair_results.select([
                                pl.col('date').str.slice(0, 10).alias('Date'),
                                pl.col('session_id').alias('Event_ID'),
                                pl.col('session_name').alias('Session'),
                                (pl.col('club_percentage') if 'club_percentage' in pair_results.columns else pl.col('percentage')).cast(pl.Float64, strict=False).round(2).alias('Club_Score'),
                                (pl.col('handicap_percentage') if 'handicap_percentage' in pair_results.columns else pl.lit(None, dtype=pl.Float64)).cast(pl.Float64, strict=False).round(2).alias('Handicap_Score'),
                                pl.col('percentage').cast(pl.Float64, strict=False).round(2).alias('Pct_Used'),
                                pl.col('rank').alias('Rank'),
                                pl.col('pe').alias('PE'),
                                pl.col('pair_elo').round(0).alias('Pair_Elo'),
                            ])
                            ShowDataFrameTable(detail_df, key='pair_detail_table', height_rows=10)
                        else:
                            st.info("No detailed results found for this pair.")
                
                st.session_state.display_df = top_pairs
                st.session_state.report_title = f"FFBridge Top Pairs (Lancelot) - {datetime.now().strftime('%Y-%m-%d')}"
            else:
                st.info(f"No pairs match the minimum requirement of {min_games} games.")
    
    # Generate PDF logic
    if generate_pdf:
        if 'display_df' in st.session_state and not st.session_state.display_df.is_empty():
            with st.spinner("Preparing PDF export..."):
                title = st.session_state.get('report_title', 'Unofficial FFBridge Elo Ratings (Lancelot API)')
                pdf_bytes = create_pdf(
                    [f"# {title}\n\nProcessed on {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                     st.session_state.display_df],
                    title=title,
                    shrink_to_fit=True
                )
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"FFBridge_Elo_Lancelot_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )
    
    # App info
    st.markdown(f"""
        <div style="text-align: center; color: #80cbc4; font-size: 0.8rem; opacity: 0.7;">
            Project lead is Robert Salita research@AiPolice.org. Code written in Python. UI written in streamlit. Data engine is polars. Repo:https://github.com/BSalita<br>
            <b>API:</b> Lancelot (public, no auth) • Query Params:{st.query_params.to_dict()} Environment:{os.getenv('STREAMLIT_ENV','')}<br>
            Streamlit:{st.__version__} Python:{'.'.join(map(str, sys.version_info[:3]))} pandas:{pd.__version__} polars:{pl.__version__} duckdb:{duckdb.__version__} endplay:{ENDPLAY_VERSION}
        </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown(f"""
        <div style="text-align: center; padding: 2rem 0; color: #80cbc4; font-size: 0.9rem; opacity: 0.8;">
            Data sourced from FFBridge Lancelot API (public) • {selected_tournament_label}<br>
            {len(all_sessions)} sessions processed • System Date: {datetime.now().strftime('%Y-%m-%d')}
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

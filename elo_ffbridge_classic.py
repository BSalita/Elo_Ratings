# elo_ffbridge_classic.py
"""
FFBridge Classic API Adapter

This module provides the API adapter for the authenticated FFBridge API (api.ffbridge.fr).
Requires a Bearer token for authentication.
"""

import os
import re
import time
import pathlib
from typing import Optional, Tuple, List, Dict, Any

import requests
import streamlit as st
from dotenv import load_dotenv

# Import shared utilities
from elo_ffbridge_common import (
    SERIES_NAMES,
    VALID_SERIES_IDS,
    normalize_series_id,
    get_cache_path,
    save_to_disk_cache,
    load_from_disk_cache,
)

# Load environment variables
load_dotenv()

# -------------------------------
# Constants
# -------------------------------
API_NAME = "FFBridge (Classic)"
API_BASE = "https://api.ffbridge.fr"
REQUIRES_AUTH = True

DATA_ROOT = pathlib.Path('data') / 'ffbridge'
CACHE_DIR = DATA_ROOT / 'cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_TIMEOUT = 15  # seconds
REQUEST_DELAY = 0.1  # seconds between API requests


# -------------------------------
# Authentication
# -------------------------------
# Player IV cache (in-memory for session)
_player_iv_cache: Dict[str, Dict] = {}


def fetch_player_iv(person_id: str, session: Optional[requests.Session] = None) -> Optional[Dict]:
    """
    Fetch player IV data from the members endpoint.
    
    Returns dict with:
        - iv: int (e.g., 28)
        - label: str (e.g., "4ème série")
        - code: str (e.g., "4S")
    
    Returns None if not found or error.
    """
    global _player_iv_cache
    
    # Check memory cache
    if person_id in _player_iv_cache:
        return _player_iv_cache[person_id]
    
    # Check disk cache - expires on 15th of each month when FFBridge updates IVs
    cache_key = f"player_iv_{person_id}"
    # Calculate hours until 15th of current/next month
    from datetime import datetime
    now = datetime.now()
    if now.day < 15:
        # Expires on 15th of current month
        next_refresh = now.replace(day=15, hour=0, minute=0, second=0, microsecond=0)
    else:
        # Expires on 15th of next month
        if now.month == 12:
            next_refresh = now.replace(year=now.year + 1, month=1, day=15, hour=0, minute=0, second=0, microsecond=0)
        else:
            next_refresh = now.replace(month=now.month + 1, day=15, hour=0, minute=0, second=0, microsecond=0)
    hours_until_refresh = max(1, int((next_refresh - now).total_seconds() / 3600))
    
    cached = load_from_disk_cache(CACHE_DIR, cache_key, max_age_hours=hours_until_refresh)
    if cached:
        _player_iv_cache[person_id] = cached
        return cached
    
    # Fetch from API
    if session is None:
        session = get_session()
    if not session:
        return None
    
    try:
        time.sleep(0.05)  # Light rate limiting (50ms)
        url = f"{API_BASE}/api/v1/members/{person_id}"
        response = session.get(url, timeout=10)  # Shorter timeout for IV lookups
        
        if response.status_code == 200:
            data = response.json()
            iv_data = data.get('iv', {})
            if iv_data:
                result = {
                    'iv': iv_data.get('iv', 0),
                    'label': iv_data.get('label', ''),
                    'code': iv_data.get('code', ''),
                }
                _player_iv_cache[person_id] = result
                save_to_disk_cache(CACHE_DIR, cache_key, result)
                return result
    except Exception:
        pass
    
    return None


def get_session() -> Optional[requests.Session]:
    """
    Get an authenticated FFBridge session from environment variables.
    Returns None if token is missing.
    """
    session = st.session_state.get('ffbridge_classic_session')
    if session:
        return session
    
    env_token = os.getenv('FFBRIDGE_BEARER_TOKEN')
    if env_token:
        session = requests.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',
            'Origin': 'https://licencie.ffbridge.fr',
            'Referer': 'https://licencie.ffbridge.fr/',
            'Authorization': f'Bearer {env_token}'
        }
        session.headers.update(headers)
        st.session_state.ffbridge_classic_session = session
        st.session_state.ffbridge_classic_user = "Env Token"
        return session
        
    return None


def is_authenticated() -> bool:
    """Check if authentication is available."""
    return get_session() is not None


def get_auth_error_message() -> str:
    """Return the error message to display when authentication fails."""
    return """
Please ensure you have a valid Bearer token in your `.env` file:
`FFBRIDGE_BEARER_TOKEN=your_token_here`
"""


# -------------------------------
# API Functions
# -------------------------------
def fetch_tournament_list(series_id: Any = "all", limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Fetch list of tournaments from FFBridge.
    
    Args:
        series_id: Tournament series ID or "all" for all series
        limit: Maximum number of tournaments to fetch per series
    
    Returns:
        List of tournament dictionaries with normalized structure
    """
    session = get_session()
    if not session:
        return []
    
    if series_id == "all":
        all_tournaments = []
        for s_id in VALID_SERIES_IDS:
            all_tournaments.extend(_fetch_tournament_list_single(session, s_id, limit))
        return all_tournaments
    
    return _fetch_tournament_list_single(session, series_id, limit)


def _fetch_tournament_list_single(session: requests.Session, series_id: int, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fetch tournament list for a single series."""
    cached_data = load_from_disk_cache(CACHE_DIR, "tournament_list", {"limit": limit}, max_age_hours=None, series_id=series_id)
    if cached_data:
        return cached_data
    
    url = f"{API_BASE}/api/v1/simultaneous/{series_id}/tournaments"
    
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                tournaments = data[:limit] if limit else data
                
                # Inject series_id into each tournament
                for t in tournaments:
                    t['series_id'] = series_id
                
                save_to_disk_cache(CACHE_DIR, "tournament_list", tournaments, {"limit": limit}, series_id=series_id)
                return tournaments
    except Exception as e:
        st.warning(f"Error fetching tournament list: {e}")
    
    return []


def fetch_tournament_results(tournament_id: str, tournament_date: str = "", series_id: Optional[Any] = None, fetch_iv: bool = False) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Fetch results for a specific tournament.
    
    Args:
        tournament_id: The tournament ID to fetch
        tournament_date: Optional date string for cache naming
        series_id: Optional series ID for cache organization
        fetch_iv: If True, also fetch IV data for each player (slower)
    
    Returns:
        Tuple of (list of result dicts, was_cached bool)
        Each result dict has normalized keys including:
        - percentage, handicap_percentage, scratch_percentage, iv_bonus
        - player1_iv, player2_iv, pair_iv (if fetch_iv=True)
    """
    session = get_session()
    if not session:
        return [], False
    
    # Create friendly filename
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', tournament_date)
    date_part = date_match.group(1) if date_match else ""
    friendly_name = f"results_{tournament_id}_{date_part}" if date_part else f"results_{tournament_id}"
    
    # Check disk cache
    cached_data = load_from_disk_cache(CACHE_DIR, friendly_name, max_age_hours=None, series_id=series_id)
    if cached_data:
        return cached_data.get('results', []), True
    
    # Rate limiting
    time.sleep(REQUEST_DELAY)
    
    url = f"{API_BASE}/api/v1/simultaneous-tournaments/{tournament_id}"
    
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        
        if response.status_code == 429:
            st.warning("Rate limited by FFBridge API. Waiting 5 seconds...")
            time.sleep(5)
            return fetch_tournament_results(tournament_id, tournament_date, series_id)
        
        if response.status_code == 200:
            data = response.json()
            if data:
                teams = data.get('teams', [])
                results = []
                
                for team in teams:
                    players = team.get('players', [])
                    if len(players) >= 2:
                        p1 = players[0]
                        p2 = players[1]
                        
                        org = team.get('organization', {})
                        club_id = str(org.get('id', ''))
                        club_name = org.get('name', '')
                        club_code = org.get('code', '')
                        
                        try:
                            pct = float(team.get('percent', 0))
                        except (ValueError, TypeError):
                            pct = 0.0
                        
                        # The API's percent is typically the handicap-adjusted score
                        handicap_pct = pct
                        
                        # Try to find explicit club percentage
                        club_pct_raw = None
                        for k in ("percentWithoutHandicap", "clubPercent", "club_percent"):
                            v = team.get(k)
                            if v is not None and v != "":
                                club_pct_raw = v
                                break
                        
                        try:
                            club_pct = float(club_pct_raw) if club_pct_raw is not None else None
                        except (ValueError, TypeError):
                            club_pct = None
                        
                        pe_bonus = team.get('PE_bonus', 0)
                        try:
                            pe_bonus_val = float(pe_bonus or 0)
                        except (ValueError, TypeError):
                            pe_bonus_val = 0.0
                        
                        # Derive IV bonus (PE_bonus is in tenths of a percent)
                        iv_bonus = pe_bonus_val / 10.0
                        
                        # Derive scratch (unhandicapped) score
                        scratch_pct = handicap_pct - iv_bonus
                        
                        # Derive club_pct if not available (same as scratch)
                        if club_pct is None:
                            club_pct = scratch_pct
                        
                        # Base result dict
                        result_dict = {
                            'team_id': str(team.get('id')),
                            'pair_id': str(team.get('id')),
                            'player1_id': str(p1.get('id')),
                            'player2_id': str(p2.get('id')),
                            'player1_name': f"{p1.get('firstname', '')} {p1.get('lastname', '')}".strip(),
                            'player2_name': f"{p2.get('firstname', '')} {p2.get('lastname', '')}".strip(),
                            'percentage': pct,
                            'handicap_percentage': handicap_pct,
                            'scratch_percentage': scratch_pct,  # Derived unhandicapped score
                            'iv_bonus': iv_bonus,  # Derived IV bonus (percentage points)
                            'club_percentage': club_pct,
                            'rank': team.get('ranking', 0),
                            'theoretical_rank': team.get('theoretical_ranking', 0),
                            'pe': team.get('PE', 0),
                            'pe_bonus': str(pe_bonus),
                            'club_id': club_id,
                            'club_name': club_name,
                            'club_code': club_code,
                            # IV fields (populated if fetch_iv=True)
                            'player1_iv': None,
                            'player2_iv': None,
                            'pair_iv': None,
                        }
                        
                        # Optionally fetch IV for each player
                        if fetch_iv:
                            p1_iv_data = fetch_player_iv(str(p1.get('id')), session)
                            p2_iv_data = fetch_player_iv(str(p2.get('id')), session)
                            
                            p1_iv = p1_iv_data.get('iv', 0) if p1_iv_data else 0
                            p2_iv = p2_iv_data.get('iv', 0) if p2_iv_data else 0
                            
                            result_dict['player1_iv'] = p1_iv
                            result_dict['player2_iv'] = p2_iv
                            result_dict['pair_iv'] = p1_iv + p2_iv
                        
                        results.append(result_dict)
                
                processed_data = {'results': results}
                save_to_disk_cache(CACHE_DIR, friendly_name, processed_data, series_id=series_id)
                return results, False
                
    except Exception as e:
        st.warning(f"Error fetching tournament results: {e}")
    
    return [], False


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_member_details(person_id: str) -> Optional[Dict[str, Any]]:
    """Fetch member profile details including official IV rating."""
    session = get_session()
    if not session:
        return None
    
    url = f"{API_BASE}/api/v1/members/{person_id}"
    
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_person_results(person_id: str) -> List[Dict[str, Any]]:
    """Fetch all tournament results for a person from FFBridge."""
    session = get_session()
    if not session:
        return []
    
    url = f"{API_BASE}/api/v1/licensee-results/results/person/{person_id}?date=all&place=0&type=0"
    
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                return data
    except Exception:
        pass
    
    return []


def build_club_name_mapping(results_df) -> Dict[str, str]:
    """
    Build a mapping from club codes to club names.
    For Classic API, club names come directly from the results.
    """
    mapping = {}
    if results_df is not None and not results_df.is_empty():
        if 'club_code' in results_df.columns and 'club_name' in results_df.columns:
            for row in results_df.iter_rows(named=True):
                code = str(row.get('club_code', '')).strip()
                name = str(row.get('club_name', '')).strip()
                if code and name and code not in mapping:
                    mapping[code] = name
    return mapping

# elo_ffbridge_lancelot.py
"""
FFBridge Lancelot API Adapter

This module provides the API adapter for the public FFBridge Lancelot API (api-lancelot.ffbridge.fr).
No authentication required - public access.
"""

import re
import time
import pathlib
from typing import Optional, Tuple, List, Dict, Any

import requests
import streamlit as st
import polars as pl

# Import shared utilities
from elo_ffbridge_common import (
    SERIES_NAMES,
    VALID_SERIES_IDS,
    normalize_series_id,
    normalize_club_code,
    get_cache_path,
    save_to_disk_cache,
    load_from_disk_cache,
)

# -------------------------------
# Constants
# -------------------------------
API_NAME = "FFBridge (Lancelot)"
API_BASE = "https://api-lancelot.ffbridge.fr"
REQUIRES_AUTH = False

DATA_ROOT = pathlib.Path('data') / 'ffbridge'
CACHE_DIR = DATA_ROOT / 'lancelot_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_TIMEOUT = 30  # seconds (Lancelot can be slower)
REQUEST_DELAY = 0.1  # seconds between API requests

# Lancelot ID to Migration ID (FFBridge series ID) mapping
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

MIGRATION_TO_LANCELOT = {v: k for k, v in LANCELOT_TO_MIGRATION.items()}


# -------------------------------
# Authentication (not required for Lancelot)
# -------------------------------
def get_session() -> Optional[requests.Session]:
    """Get a session for Lancelot API (no auth required)."""
    if 'lancelot_session' not in st.session_state:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        })
        st.session_state.lancelot_session = session
    return st.session_state.lancelot_session


def is_authenticated() -> bool:
    """Lancelot API doesn't require authentication."""
    return True


def get_auth_error_message() -> str:
    """No auth error for Lancelot."""
    return ""


# -------------------------------
# API Helpers
# -------------------------------
def lancelot_get(endpoint: str, params: Optional[Dict] = None, add_delay: bool = True) -> Optional[Any]:
    """Make a GET request to Lancelot API with rate limiting."""
    session = get_session()
    url = f"{API_BASE}{endpoint}"
    
    if add_delay:
        time.sleep(REQUEST_DELAY)
    
    try:
        response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            st.warning("Rate limited by Lancelot API. Waiting 5 seconds...")
            time.sleep(5)
            return lancelot_get(endpoint, params, add_delay=False)
    except Exception as e:
        st.warning(f"Lancelot API error: {e}")
    return None


# -------------------------------
# API Functions
# -------------------------------
def fetch_tournament_list(series_id: Any = "all", limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Fetch list of sessions (tournaments) from Lancelot API.
    
    Args:
        series_id: Tournament series ID (migration ID) or "all" for all series
        limit: Maximum number of sessions per series
    
    Returns:
        List of session dictionaries with normalized structure
    """
    if series_id == "all":
        all_sessions = []
        for migration_id in VALID_SERIES_IDS:
            lancelot_id = MIGRATION_TO_LANCELOT.get(migration_id)
            if lancelot_id:
                sessions = _fetch_sessions_for_series(lancelot_id, migration_id, limit)
                all_sessions.extend(sessions)
        return all_sessions
    
    lancelot_id = MIGRATION_TO_LANCELOT.get(series_id)
    if lancelot_id:
        return _fetch_sessions_for_series(lancelot_id, series_id, limit)
    return []


def _fetch_sessions_for_series(lancelot_id: int, migration_id: int, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fetch all sessions for a series from Lancelot API."""
    series_name = SERIES_NAMES.get(migration_id, f"series_{lancelot_id}")
    friendly_name = f"sessions_list_{series_name.replace(' ', '_')}"
    
    # Check disk cache
    cached_data = load_from_disk_cache(CACHE_DIR, friendly_name, max_age_hours=None, series_id=migration_id)
    if cached_data:
        # Ensure series_id is set on each session
        for s in cached_data:
            s['series_id'] = migration_id
        return cached_data[:limit] if limit else cached_data
    
    all_sessions = []
    page = 1
    max_pages = 10
    
    while page <= max_pages:
        data = lancelot_get(
            f"/competitions/simultaneous/{lancelot_id}/sessions",
            params={"currentPage": page, "maxPerPage": 80}
        )
        
        if not data or 'items' not in data:
            break
            
        items = data['items']
        # Inject series_id for each session
        for item in items:
            item['series_id'] = migration_id
        all_sessions.extend(items)
        
        pagination = data.get('pagination', {})
        if not pagination.get('has_next_page', False):
            break
            
        page += 1
    
    if all_sessions:
        save_to_disk_cache(CACHE_DIR, friendly_name, all_sessions, series_id=migration_id)
    
    return all_sessions[:limit] if limit else all_sessions


def fetch_tournament_results(session_id: str, tournament_date: str = "", series_id: Optional[Any] = None) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Fetch results for a specific session from Lancelot.
    
    Returns:
        Tuple of (list of result dicts, was_cached bool)
    """
    # Create friendly filename
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', tournament_date)
    date_part = date_match.group(1) if date_match else ""
    friendly_name = f"ranking_{session_id}_{date_part}" if date_part else f"ranking_{session_id}"
    
    # Check disk cache
    cached_data = load_from_disk_cache(CACHE_DIR, friendly_name, max_age_hours=None, series_id=series_id)
    if cached_data:
        return _normalize_ranking_results(cached_data), True
    
    # Fetch from API
    data = lancelot_get(f"/results/sessions/{session_id}/ranking")
    
    if data and isinstance(data, list):
        save_to_disk_cache(CACHE_DIR, friendly_name, data, series_id=series_id)
        return _normalize_ranking_results(data), False
    
    return [], False


def _normalize_ranking_results(ranking: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize Lancelot ranking data to common result format."""
    results = []
    
    for entry in ranking:
        if not isinstance(entry, dict):
            continue
            
        team = entry.get('team', {})
        if not isinstance(team, dict):
            continue
            
        p1 = team.get('player1')
        p2 = team.get('player2')
        
        if not isinstance(p1, dict) or not isinstance(p2, dict):
            continue
        
        p1_id = str(p1.get('migrationId') or p1.get('id', ''))
        p2_id = str(p2.get('migrationId') or p2.get('id', ''))
        p1_name = f"{p1.get('firstName', '')} {p1.get('lastName', '')}".strip()
        p2_name = f"{p2.get('firstName', '')} {p2.get('lastName', '')}".strip()
        
        pct = float(entry.get('sessionScore') or entry.get('totalScore') or 0)
        pe_bonus = float(entry.get('peBonus') or 0)
        
        # Normalize club code using shared utility
        club_code = normalize_club_code(entry.get('simultaneousId', ''))
        
        results.append({
            'team_id': str(team.get('id', '')),
            'pair_id': str(team.get('id', '')),
            'player1_id': p1_id,
            'player2_id': p2_id,
            'player1_name': p1_name,
            'player2_name': p2_name,
            'percentage': pct,
            'handicap_percentage': pct,
            'club_percentage': pct - pe_bonus / 10.0,
            'rank': entry.get('rank', 0),
            'theoretical_rank': entry.get('rankWithoutHandicap'),
            'pe': entry.get('pe', 0),
            'pe_bonus': str(pe_bonus),
            'club_code': club_code,
            'club_name': '',  # Will be populated by build_club_name_mapping
        })
    
    return results


def fetch_session_clubs(session_id: int) -> List[Dict[str, Any]]:
    """Get all clubs that participated in a session."""
    cache_name = f"clubs_{session_id}"
    cached_data = load_from_disk_cache(CACHE_DIR, cache_name, max_age_hours=None, series_id=None)
    if cached_data:
        return cached_data
    
    data = lancelot_get(f"/results/sessions/{session_id}/simultaneousIds")
    if isinstance(data, list):
        save_to_disk_cache(CACHE_DIR, cache_name, data, series_id=None)
        return data
    return []


def fetch_member_details(person_id: str) -> Optional[Dict[str, Any]]:
    """
    Lancelot doesn't have a member details endpoint like the classic API.
    Return None to indicate no profile data available.
    """
    return None


def fetch_person_results(person_id: str) -> List[Dict[str, Any]]:
    """
    Lancelot doesn't have a person results endpoint.
    Return empty list.
    """
    return []


def build_club_name_mapping(unique_codes: List[str], sessions: List[Dict[str, Any]], results_df=None) -> Dict[str, str]:
    """
    Build a mapping of club codes to club names.
    Uses disk cache and session state to avoid repeated API calls.
    """
    if 'lancelot_club_mapping' not in st.session_state:
        st.session_state.lancelot_club_mapping = {}
    
    mapping = st.session_state.lancelot_club_mapping
    
    normalized_unique = [normalize_club_code(c) for c in unique_codes if c]
    missing_codes = set(normalized_unique) - set(mapping.keys())
    
    if missing_codes:
        sessions_to_check = set()
        
        # If we have results_df, find sessions that contain missing clubs
        # Check for both 'session_id' and 'tournament_id' column names
        id_col = None
        if results_df is not None and not results_df.is_empty():
            if 'session_id' in results_df.columns:
                id_col = 'session_id'
            elif 'tournament_id' in results_df.columns:
                id_col = 'tournament_id'
        
        if id_col:
            for code in list(missing_codes)[:100]:
                matches = results_df.filter(pl.col('club_code') == code)
                if not matches.is_empty():
                    s_ids = matches.select(id_col).head(3).to_series().to_list()
                    for sid in s_ids:
                        sessions_to_check.add(str(sid))
        
        # Add recent sessions as fallback
        if len(sessions_to_check) < 20:
            sorted_sessions = sorted(sessions, key=lambda x: x.get('date', ''), reverse=True)
            for s in sorted_sessions[:40]:
                sessions_to_check.add(str(s.get('id', '')))
        
        for session_id in sessions_to_check:
            if not missing_codes:
                break
            if session_id:
                try:
                    clubs = fetch_session_clubs(int(session_id) if session_id.isdigit() else session_id)
                except Exception:
                    continue
                    
                for club in clubs:
                    l_id = normalize_club_code(club.get('id'))
                    ffb_code = normalize_club_code(club.get('ffbCode'))
                    name = club.get('label', '')
                    
                    if name:
                        if l_id and l_id in missing_codes:
                            mapping[l_id] = name
                            missing_codes.discard(l_id)
                        if ffb_code and ffb_code in missing_codes:
                            mapping[ffb_code] = name
                            missing_codes.discard(ffb_code)
    
    return mapping

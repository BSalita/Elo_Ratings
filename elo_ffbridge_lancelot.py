# elo_ffbridge_lancelot.py
"""
FFBridge Lancelot API Adapter

This module provides the API adapter for the public FFBridge Lancelot API (api-lancelot.ffbridge.fr).
No authentication required - public access.
"""

import os
import re
import pathlib
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

import requests
import streamlit as st
import polars as pl

from ffbridge_provisional_scores import (
    fetch_provisional_pair_percentages,
    national_ranking_is_pending,
)

# Import shared utilities
from elo_ffbridge_common import (
    SERIES_NAMES,
    VALID_SERIES_IDS,
    normalize_series_id,
    normalize_club_code,
    ffbridge_scoring_mode,
    fill_missing_score_ranks,
    get_cache_path,
    save_to_disk_cache,
    load_from_disk_cache,
    mlBridgeFFLib,
)

# -------------------------------
# Constants
# -------------------------------
API_NAME = "FFBridge (Lancelot)"
API_BASE = mlBridgeFFLib.LANCELOT_API_BASE
REQUIRES_AUTH = False

# Cache root can be redirected to a persistent data mount, e.g. /data/ffbridge
DATA_ROOT = pathlib.Path(os.getenv("FFBRIDGE_CACHE_DIR", "data/ffbridge")).resolve()
CACHE_DIR = DATA_ROOT / 'lancelot_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
_cache_file_count = sum(len(files) for _, _, files in os.walk(CACHE_DIR))
print(
    f"[lancelot] FFBRIDGE_CACHE_DIR={os.getenv('FFBRIDGE_CACHE_DIR', '(not set)')!r} "
    f"-> DATA_ROOT={DATA_ROOT} CACHE_DIR={CACHE_DIR} "
    f"exists={CACHE_DIR.exists()} writable={os.access(CACHE_DIR, os.W_OK)} "
    f"cached_files={_cache_file_count}",
    flush=True,
)

REQUEST_TIMEOUT = 10  # seconds (reduced from 30 to fail faster on hung requests)
REQUEST_DELAY = 0.1  # seconds between API requests
PENDING_RESULTS_CACHE_HOURS = 6
RECENT_RESULTS_CACHE_HOURS = 6
RECENT_RESULTS_DAYS = 90
PROVENANCE_API_START = "2026-07-01"
ORGANIZER_SCORE_CACHE_VERSION = "v1"

# Lancelot ID to Migration ID (FFBridge series ID) mapping (shared)
LANCELOT_TO_MIGRATION = mlBridgeFFLib.LANCELOT_TO_MIGRATION
MIGRATION_TO_LANCELOT = mlBridgeFFLib.MIGRATION_TO_LANCELOT


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
def lancelot_get(endpoint: str, params: Optional[Dict] = None, add_delay: bool = True, verbose: bool = True) -> Optional[Any]:
    """Make a GET request to Lancelot API with rate limiting.

    Thin wrapper over the shared mlBridgeFFLib client: keeps this app's
    return-None-on-error contract and Streamlit warnings.
    """
    try:
        return mlBridgeFFLib.lancelot_get(
            endpoint,
            params=params,
            session=get_session(),
            timeout=REQUEST_TIMEOUT,
            rate_limit_delay=REQUEST_DELAY if add_delay else 0.0,
            verbose=verbose,
        )
    except requests.exceptions.Timeout:
        print(f"[Lancelot] TIMEOUT after {REQUEST_TIMEOUT}s: {endpoint}", flush=True)
        st.warning(f"Timeout fetching {endpoint} after {REQUEST_TIMEOUT}s - skipping")
    except requests.exceptions.HTTPError as e:
        if verbose:
            print(f"[Lancelot] HTTP {e.response.status_code}: {endpoint}", flush=True)
    except Exception as e:
        print(f"[Lancelot] ERROR: {e} for {endpoint}", flush=True)
        st.warning(f"Lancelot API error: {e}")
    return None


# -------------------------------
# API Functions
# -------------------------------
def fetch_tournament_list(series_id: Any = "all", limit: Optional[int] = None, force_refresh: bool = False) -> List[Dict[str, Any]]:
    """
    Fetch list of sessions (tournaments) from Lancelot API.
    
    Args:
        series_id: Tournament series ID (migration ID) or "all" for all series
        limit: Maximum number of sessions per series
        force_refresh: If True, bypass the (no-expiry) disk cache and re-fetch
            from the API so newly published sessions are discovered. The fresh
            list is written back to the cache.
    
    Returns:
        List of session dictionaries with normalized structure
    """
    if series_id == "all":
        all_sessions = []
        for migration_id in VALID_SERIES_IDS:
            lancelot_id = MIGRATION_TO_LANCELOT.get(migration_id)
            if lancelot_id:
                sessions = _fetch_sessions_for_series(lancelot_id, migration_id, limit, force_refresh)
                all_sessions.extend(sessions)
        return all_sessions
    
    lancelot_id = MIGRATION_TO_LANCELOT.get(series_id)
    if lancelot_id:
        return _fetch_sessions_for_series(lancelot_id, series_id, limit, force_refresh)
    return []


def _fetch_sessions_for_series(lancelot_id: int, migration_id: int, limit: Optional[int] = None, force_refresh: bool = False) -> List[Dict[str, Any]]:
    """Fetch all sessions for a series from Lancelot API."""
    series_name = SERIES_NAMES.get(migration_id, f"series_{lancelot_id}")
    friendly_name = f"sessions_list_{series_name.replace(' ', '_')}"
    
    # Check disk cache (unless forcing a refresh to discover new sessions)
    if not force_refresh:
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


def _is_recent_result(tournament_date: str) -> bool:
    try:
        session_day = datetime.fromisoformat(tournament_date[:10]).date()
    except (TypeError, ValueError):
        return False
    age_days = (datetime.now().date() - session_day).days
    return 0 <= age_days <= RECENT_RESULTS_DAYS


def _missing_expected_provenance(
    ranking: List[Dict[str, Any]],
    tournament_date: str,
    series_id: Optional[Any],
) -> bool:
    if tournament_date[:10] < PROVENANCE_API_START:
        return False
    rows = [row for row in ranking if isinstance(row, dict)]
    if not rows or national_ranking_is_pending(rows):
        return False
    if not any(row.get("theoreticalRank") is not None for row in rows):
        return True
    return (
        ffbridge_scoring_mode(series_id, tournament_date) == "handicap"
        and not any(row.get("totalBonus") is not None for row in rows)
    )


def _fetch_organizer_scores(
    session_id: str,
    ranking: List[Dict[str, Any]],
    tournament_date: str,
    series_id: Optional[Any],
) -> Dict[str, Dict[str, Any]]:
    if normalize_series_id(series_id) != 386:
        return {}
    cache_name = f"organizer_scores_{ORGANIZER_SCORE_CACHE_VERSION}_{session_id}"
    cached = load_from_disk_cache(
        CACHE_DIR,
        cache_name,
        max_age_hours=None,
        series_id=series_id,
    )
    if isinstance(cached, dict) and "scores" in cached:
        return cached["scores"]
    scores = fetch_provisional_pair_percentages(
        ranking,
        tournament_date,
        series_id,
    )
    has_score = any(
        any(
            row.get(column) is not None
            for column in (
                "national_scratch_percentage",
                "national_handicap_percentage",
                "club_scratch_percentage",
                "club_handicap_percentage",
            )
        )
        for row in scores.values()
    )
    if not has_score:
        return scores
    save_to_disk_cache(
        CACHE_DIR,
        cache_name,
        {"scores": scores},
        series_id=series_id,
    )
    return scores


def fetch_tournament_results(session_id: str, tournament_date: str = "", series_id: Optional[Any] = None, fetch_iv: bool = False) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Fetch results for a specific session from Lancelot.
    
    Args:
        session_id: The session ID to fetch
        tournament_date: Optional date string for cache naming
        series_id: Optional series ID for cache organization
        fetch_iv: Ignored for Lancelot API (requires auth not available)
    
    Returns:
        Tuple of (list of result dicts, was_cached bool)
    """
    # Create friendly filename
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', tournament_date)
    date_part = date_match.group(1) if date_match else ""
    friendly_name = f"ranking_{session_id}_{date_part}" if date_part else f"ranking_{session_id}"
    
    # FFBridge updates row counts, bonuses, and theoretical ranks after a
    # ranking first appears. Revalidate recent finalized sessions as well as
    # pending zero shells; older complete rankings remain immutable.
    cached_data = load_from_disk_cache(
        CACHE_DIR, friendly_name, max_age_hours=None, series_id=series_id
    )
    if cached_data:
        if _missing_expected_provenance(
            cached_data, tournament_date, series_id
        ):
            cached_data = None
        elif national_ranking_is_pending(cached_data):
            cached_data = load_from_disk_cache(
                CACHE_DIR,
                friendly_name,
                max_age_hours=PENDING_RESULTS_CACHE_HOURS,
                series_id=series_id,
            )
        elif _is_recent_result(tournament_date):
            cached_data = load_from_disk_cache(
                CACHE_DIR,
                friendly_name,
                max_age_hours=RECENT_RESULTS_CACHE_HOURS,
                series_id=series_id,
            )
        if cached_data:
            group_ids = fetch_session_group_ids(session_id)
            organizer_scores = _fetch_organizer_scores(
                session_id,
                cached_data,
                tournament_date,
                series_id,
            )
            return _normalize_ranking_results(
                cached_data,
                series_id=series_id,
                tournament_date=tournament_date,
                group_ids=group_ids,
                organizer_scores=organizer_scores,
            ), True
    
    # Fetch from API
    data = lancelot_get(f"/results/sessions/{session_id}/ranking")
    
    if data and isinstance(data, list):
        save_to_disk_cache(CACHE_DIR, friendly_name, data, series_id=series_id)
        group_ids = fetch_session_group_ids(session_id)
        organizer_scores = _fetch_organizer_scores(
            session_id,
            data,
            tournament_date,
            series_id,
        )
        return _normalize_ranking_results(
            data,
            series_id=series_id,
            tournament_date=tournament_date,
            group_ids=group_ids,
            organizer_scores=organizer_scores,
        ), False
    
    return [], False


def _as_number(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _official_scratch_and_handicap(
    entry: Dict[str, Any],
    *,
    session_score: Optional[float],
    scoring_mode: str,
    bonus_is_authoritative: bool,
) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    """Map a published Lancelot ranking row to scratch, handicap, and IV bonus.

    Verified against FFBridge and club tables (2026-08-25):
    - `sessionScore` is scratch for scratch-only series (Rondes de France, etc.).
    - `sessionScore` is the national handicap percentage for registered
      handicap schedules (Octopus, Simultanet, Roy René first Tuesday).
    - `totalBonus` is the Octopus IV handicap in percentage points.
    - `peBonus` is PE ranking points. Never convert it into a percentage.
    """
    explicit_scratch = _as_number(entry.get("totalScoreWithoutHandicap"))
    explicit_handicap = _as_number(
        entry.get("scoreHandicap")
        if entry.get("scoreHandicap") is not None
        else entry.get("handicapPercentage")
    )
    total_bonus_value = _as_number(entry.get("totalBonus"))
    if session_score is None:
        return None, None, None, "unresolved"
    has_explicit_handicap = any(
        value is not None for value in (explicit_scratch, explicit_handicap)
    )
    if (
        scoring_mode != "handicap"
        and not has_explicit_handicap
        and not (total_bonus_value is not None and total_bonus_value > 0)
    ):
        return session_score, None, 0.0, "scratch_only"

    if explicit_scratch is not None and explicit_handicap is not None:
        scratch_pct = explicit_scratch
        handicap_pct = explicit_handicap
    elif explicit_scratch is not None:
        scratch_pct = explicit_scratch
        handicap_pct = session_score
    elif explicit_handicap is not None:
        handicap_pct = explicit_handicap
        scratch_pct = (
            handicap_pct - total_bonus_value
            if total_bonus_value is not None
            and (bonus_is_authoritative or total_bonus_value > 0)
            else None
        )
    elif total_bonus_value is not None and (
        bonus_is_authoritative or total_bonus_value > 0
    ):
        total_bonus = total_bonus_value
        handicap_pct = (
            explicit_handicap if explicit_handicap is not None else session_score
        )
        scratch_pct = handicap_pct - total_bonus
    else:
        handicap_pct = session_score
        scratch_pct = None
    iv_bonus = (
        handicap_pct - scratch_pct
        if handicap_pct is not None and scratch_pct is not None
        else None
    )
    return scratch_pct, handicap_pct, iv_bonus, "official"


def _normalize_ranking_results(
    ranking: List[Dict[str, Any]],
    series_id: Optional[Any] = None,
    tournament_date: str = "",
    group_ids: Optional[Dict[str, str]] = None,
    organizer_scores: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Normalize Lancelot ranking data to common result format."""
    results = []
    group_ids = group_ids or {}

    normalized_series_id = normalize_series_id(series_id)
    scoring_mode = ffbridge_scoring_mode(
        normalized_series_id, tournament_date
    )
    national_pending = national_ranking_is_pending(ranking)
    provisional_scores = organizer_scores or (
        fetch_provisional_pair_percentages(
            ranking, tournament_date, normalized_series_id
        )
        if national_pending
        else {}
    )
    
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
        
        national_pct_raw = (
            entry.get('sessionScore')
            if entry.get('sessionScore') is not None
            else entry.get('totalScore')
        )
        national_pct = _as_number(national_pct_raw)
        pe_bonus_raw = float(entry.get('peBonus') or 0)
        # peBonus is PE ranking points, not a percentage adjustment.
        # Octopus IV handicap lives in totalBonus (percentage points).
        iv_bonus = _as_number(entry.get('totalBonus'))

        team_id = str(team.get('id', ''))
        organizer = provisional_scores.get(team_id, {})
        organizer_national_scratch = organizer.get(
            "national_scratch_percentage"
        )
        organizer_national_handicap = organizer.get(
            "national_handicap_percentage"
        )
        organizer_club_scratch = organizer.get("club_scratch_percentage")
        organizer_club_handicap = organizer.get("club_handicap_percentage")
        if (
            "club_scratch_percentage" not in organizer
            and organizer.get("scratch_percentage") is not None
        ):
            organizer_club_scratch = organizer.get("scratch_percentage")
        if (
            "club_handicap_percentage" not in organizer
            and organizer.get("handicap_percentage") is not None
        ):
            organizer_club_handicap = organizer.get("handicap_percentage")

        if national_pending:
            club_scratch_pct = organizer_club_scratch
            club_handicap_pct = organizer_club_handicap
            national_scratch_pct = organizer_national_scratch
            national_handicap_pct = organizer_national_handicap
            resolved_scratch = (
                national_scratch_pct
                if national_scratch_pct is not None
                else club_scratch_pct
            )
            resolved_handicap = (
                national_handicap_pct
                if national_handicap_pct is not None
                else club_handicap_pct
            )
            if resolved_scratch is not None and resolved_handicap is not None:
                iv_bonus = float(resolved_handicap) - float(resolved_scratch)
            scratch_status = (
                "provisional" if resolved_scratch is not None else "unresolved"
            )
            handicap_status = (
                "provisional" if resolved_handicap is not None else "unresolved"
            )
            has_provisional_score = (
                resolved_scratch is not None or resolved_handicap is not None
            )
            score_source = (
                "club_provisional" if has_provisional_score else "unresolved"
            )
            score_status = (
                "provisional" if has_provisional_score else "unresolved"
            )
            source_url = (
                organizer.get("scratch_url")
                or organizer.get("handicap_url")
            )
        else:
            club_scratch_pct = organizer_club_scratch
            club_handicap_pct = organizer_club_handicap
            national_scratch_pct, national_handicap_pct, iv_bonus, handicap_status = (
                _official_scratch_and_handicap(
                    entry,
                    session_score=national_pct,
                    scoring_mode=scoring_mode,
                    bonus_is_authoritative=normalized_series_id in {384, 386},
                )
            )
            if national_scratch_pct is None:
                national_scratch_pct = organizer_national_scratch
            if national_handicap_pct is None:
                national_handicap_pct = organizer_national_handicap
            if (
                iv_bonus is None
                and national_scratch_pct is not None
                and national_handicap_pct is not None
            ):
                iv_bonus = (
                    float(national_handicap_pct) - float(national_scratch_pct)
                )
            scratch_status = (
                "official" if national_scratch_pct is not None else "unresolved"
            )
            score_source = (
                "national_and_organizer_official"
                if organizer
                else "national_official"
            )
            score_status = "official"
            source_url = (
                organizer.get("scratch_url")
                or organizer.get("handicap_url")
            )

        national_rank = (
            entry.get("rank") if not national_pending else None
        )
        scratch_rank = entry.get("rankWithoutHandicap")
        theoretical_rank = entry.get("theoreticalRank")
        if national_pending:
            national_handicap_rank = None
            national_scratch_rank = None
        elif scoring_mode == "handicap":
            national_handicap_rank = (
                national_rank if national_handicap_pct is not None else None
            )
            national_scratch_rank = (
                scratch_rank if national_scratch_pct is not None else None
            )
        else:
            national_handicap_rank = (
                national_rank if national_handicap_pct is not None else None
            )
            national_scratch_rank = (
                scratch_rank
                if scratch_rank is not None
                else (
                    national_rank if national_scratch_pct is not None else None
                )
            )
        
        # Normalize club code using shared utility
        club_code = normalize_club_code(entry.get('simultaneousId', ''))
        
        results.append({
            'team_id': team_id,
            'pair_id': team_id,
            'player1_id': p1_id,
            'player2_id': p2_id,
            # Preserve every Lancelot identity namespace for the shared
            # player-session index. Elo continues to use migrationId above.
            'player1_lancelot_id': str(p1.get('id') or ''),
            'player2_lancelot_id': str(p2.get('id') or ''),
            'player1_classic_person_id': str(p1.get('migrationId') or ''),
            'player2_classic_person_id': str(p2.get('migrationId') or ''),
            'player1_license_number': str(p1.get('ffbId') or ''),
            'player2_license_number': str(p2.get('ffbId') or ''),
            'player1_name': p1_name,
            'player2_name': p2_name,
            'Club_Scratch_Pct': club_scratch_pct,
            'Club_Handicap_Pct': club_handicap_pct,
            'National_Scratch_Pct': national_scratch_pct,
            'National_Handicap_Pct': national_handicap_pct,
            'Club_Scratch_Rank': None,
            'Club_Handicap_Rank': None,
            'National_Scratch_Rank': national_scratch_rank,
            'National_Handicap_Rank': national_handicap_rank,
            'Theoretical_Rank': theoretical_rank,
            'scoring_mode': scoring_mode,
            'iv_bonus': iv_bonus,  # Derived IV bonus (percentage points)
            'score_source': score_source,
            'score_status': score_status,
            'scratch_score_status': scratch_status,
            'handicap_score_status': handicap_status,
            'score_source_url': source_url,
            'rank': entry.get('rank', 0),
            'theoretical_rank': theoretical_rank,
            'pe': entry.get('pe', 0),
            'pe_bonus': str(pe_bonus_raw),
            'group_id': group_ids.get(club_code),
            # Lancelot exposes the organization route identifier as
            # simultaneousId/ffbCode rather than Classic's organization.id.
            'club_id': club_code,
            'club_code': club_code,
            'club_name': '',  # Will be populated by build_club_name_mapping
            # IV fields (not available from Lancelot API without auth)
            'player1_iv': None,
            'player2_iv': None,
            'pair_iv': None,
        })
    
    fill_missing_score_ranks(results)
    return results


def fetch_session_group_ids(session_id: str) -> Dict[str, str]:
    """Map each participating club's FFB code to its public results group ID."""
    cache_name = f"result_group_ids_{session_id}"
    cached_data = load_from_disk_cache(
        CACHE_DIR,
        cache_name,
        max_age_hours=None,
        series_id=None,
    )
    if isinstance(cached_data, dict):
        return {str(key): str(value) for key, value in cached_data.items()}

    data = None
    for _attempt in range(3):
        data = lancelot_get(f"/competitions/sessions/{session_id}")
        if isinstance(data, dict):
            break
    if not isinstance(data, dict):
        return {}

    mapping: Dict[str, str] = {}
    for group_session in data.get("groupSessions") or []:
        group = group_session.get("group") or {}
        organization = (
            ((group.get("phase") or {}).get("stade") or {}).get("organization")
            or {}
        )
        club_code = normalize_club_code(organization.get("ffbCode"))
        group_id = group.get("id")
        if club_code and group_id not in (None, ""):
            mapping[club_code] = str(group_id)

    if mapping:
        save_to_disk_cache(CACHE_DIR, cache_name, mapping, series_id=None)
    return mapping


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

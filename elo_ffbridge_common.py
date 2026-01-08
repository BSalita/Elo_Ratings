# elo_ffbridge_common.py
"""
FFBridge Elo Ratings - Shared Utilities

This module contains shared constants, cache helpers, and Elo calculation functions
used by both the Classic and Lancelot API adapters.
"""

import json
import re
import pathlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# -------------------------------
# Constants
# -------------------------------
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

# List of all valid tournament series IDs
VALID_SERIES_IDS = [3, 4, 5, 140, 384, 386, 604, 868]

# Default Elo parameters
DEFAULT_ELO = 1500.0
K_FACTOR = 32.0
PERFORMANCE_SCALING = 400  # Standard Elo scaling factor

# UI Constants
AGGRID_ROW_HEIGHT = 42
AGGRID_HEADER_HEIGHT = 50
AGGRID_FOOTER_HEIGHT = 20
AGGRID_MAX_DISPLAY_ROWS = 10


# -------------------------------
# Series ID Normalization
# -------------------------------
def normalize_series_id(series_id: Optional[Any]) -> Optional[Any]:
    """
    Normalize series_id so cache folders are stable.
    Handles string/int/float conversions (e.g., '3' and 3 map the same).
    """
    if series_id is None:
        return None
    # Keep non-numeric strings (e.g., "all") as-is
    if isinstance(series_id, str):
        s = series_id.strip()
        if s.isdigit():
            try:
                return int(s)
            except Exception:
                return series_id
        return series_id
    # Coerce simple numeric types
    try:
        # bool is a subclass of int; don't convert it
        if isinstance(series_id, bool):
            return series_id
        # floats that are integral (e.g., 3.0)
        if isinstance(series_id, float) and series_id.is_integer():
            return int(series_id)
    except Exception:
        pass
    return series_id


# -------------------------------
# Cache Helpers
# -------------------------------
def get_cache_path(
    cache_dir: pathlib.Path,
    identifier: str,
    params: Optional[Dict] = None,
    series_id: Optional[Any] = None
) -> pathlib.Path:
    """
    Generate a readable, unique filename in a series-specific subdirectory.
    
    Args:
        cache_dir: Base cache directory
        identifier: Cache entry identifier (e.g., "tournament_list", "results_12345")
        params: Optional parameters to include in filename
        series_id: Series ID for subdirectory organization
    
    Returns:
        Path to the cache file
    """
    norm_series_id = normalize_series_id(series_id)
    series_folder = SERIES_NAMES.get(norm_series_id, SERIES_NAMES.get(series_id, "General"))
    target_dir = cache_dir / series_folder
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize identifier to be a safe filename
    safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', identifier)
    
    # If there are params, append them in a readable format
    if params:
        param_parts = [f"{k}_{'all' if v is None else v}" for k, v in sorted(params.items())]
        safe_name = f"{safe_name}_{'_'.join(param_parts)}"
    
    return target_dir / f"{safe_name}.json"


def save_to_disk_cache(
    cache_dir: pathlib.Path,
    identifier: str,
    data: Any,
    params: Optional[Dict] = None,
    series_id: Optional[Any] = None
):
    """
    Save response data to disk as JSON.
    
    Args:
        cache_dir: Base cache directory
        identifier: Cache entry identifier
        data: Data to cache
        params: Optional parameters for filename
        series_id: Series ID for subdirectory
    """
    cache_path = get_cache_path(cache_dir, identifier, params, series_id)
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'identifier': identifier,
                'series_id': series_id,
                'data': data
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # Silently fail on cache save errors


def load_from_disk_cache(
    cache_dir: pathlib.Path,
    identifier: str,
    params: Optional[Dict] = None,
    max_age_hours: Optional[int] = 72,
    series_id: Optional[Any] = None
) -> Optional[Any]:
    """
    Load data from disk if it exists and isn't too old.
    
    Args:
        cache_dir: Base cache directory
        identifier: Cache entry identifier
        params: Optional parameters for filename
        max_age_hours: Maximum age in hours (None = never expires)
        series_id: Series ID for subdirectory
    
    Returns:
        Cached data or None if not found/expired
    """
    cache_path = get_cache_path(cache_dir, identifier, params, series_id)
    
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
    """
    Calculate expected score for player A against player B.
    
    Uses the standard Elo formula:
    E_A = 1 / (1 + 10^((R_B - R_A) / 400))
    """
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


# -------------------------------
# Club Code Normalization
# -------------------------------
def normalize_club_code(code: Any) -> str:
    """
    Normalize club codes by stripping leading zeros and handling edge cases.
    
    Args:
        code: Raw club code (string, int, or None)
    
    Returns:
        Normalized code as string
    """
    if code is None:
        return ""
    s = str(code).strip()
    if not s or s.lower() == 'none':
        return ""
    # Remove leading zeros but keep at least one digit if it's all zeros
    norm = s.lstrip('0')
    return norm if norm else '0'

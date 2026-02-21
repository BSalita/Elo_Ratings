# elo_ffbridge_common.py
"""
FFBridge Elo Ratings - Shared Utilities

This module contains FFBridge-specific constants, cache helpers, and re-exports
common Elo functions from elo_common.py.
Used by both the Classic and Lancelot API adapters.
"""

import json
import os
import re
import pathlib
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# Re-export common Elo constants and functions so existing imports keep working.
from elo_common import (  # noqa: F401
    DEFAULT_ELO,
    K_FACTOR,
    PERFORMANCE_SCALING,
    CHESS_SCALING_ENABLED,
    CHESS_TARGET_MIN,
    CHESS_TARGET_MAX,
    CURRENT_REFERENCE_MIN,
    CURRENT_REFERENCE_MAX,
    AGGRID_ROW_HEIGHT,
    AGGRID_HEADER_HEIGHT,
    AGGRID_FOOTER_HEIGHT,
    AGGRID_MAX_DISPLAY_ROWS,
    calculate_expected_score,
    calculate_elo_from_percentage,
    scale_to_chess_range,
    get_elo_title,
)

# -------------------------------
# FFBridge-Specific Constants
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


def _get_lock_path(cache_path: pathlib.Path) -> pathlib.Path:
    return cache_path.with_suffix(cache_path.suffix + ".lock")


def _wait_for_unlock(cache_path: pathlib.Path, max_wait_seconds: float = 3.0, poll_seconds: float = 0.1) -> None:
    """Wait briefly if another process is currently writing this cache key."""
    lock_path = _get_lock_path(cache_path)
    deadline = time.time() + max_wait_seconds
    while lock_path.exists() and time.time() < deadline:
        time.sleep(poll_seconds)


class _CacheFileLock:
    """File lock based on atomic lock-file creation."""

    def __init__(
        self,
        cache_path: pathlib.Path,
        timeout_seconds: float = 30.0,
        stale_after_seconds: float = 300.0,
        poll_seconds: float = 0.1,
    ) -> None:
        self.cache_path = cache_path
        self.lock_path = _get_lock_path(cache_path)
        self.timeout_seconds = timeout_seconds
        self.stale_after_seconds = stale_after_seconds
        self.poll_seconds = poll_seconds
        self._locked = False

    def __enter__(self):
        deadline = time.time() + self.timeout_seconds
        while True:
            try:
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, f"{os.getpid()} {time.time()}".encode("utf-8"))
                finally:
                    os.close(fd)
                self._locked = True
                return self
            except FileExistsError:
                # Best-effort stale lock cleanup.
                try:
                    mtime = self.lock_path.stat().st_mtime
                    if time.time() - mtime > self.stale_after_seconds:
                        self.lock_path.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass

                if time.time() >= deadline:
                    raise TimeoutError(f"Timed out waiting for cache lock: {self.lock_path}")
                time.sleep(self.poll_seconds)

    def __exit__(self, exc_type, exc, tb):
        if self._locked:
            try:
                self.lock_path.unlink(missing_ok=True)
            except OSError:
                pass
        return False


def _write_cache_file(
    cache_path: pathlib.Path,
    identifier: str,
    data: Any,
    series_id: Optional[Any] = None,
) -> None:
    """Write cache payload atomically (temp file + os.replace)."""
    tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "identifier": identifier,
                    "series_id": series_id,
                    "data": data,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, cache_path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


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
        with _CacheFileLock(cache_path):
            _write_cache_file(cache_path, identifier, data, series_id=series_id)
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
        # Another session may currently be writing this cache key.
        _wait_for_unlock(cache_path)
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


def get_or_fetch_with_disk_cache(
    cache_dir: pathlib.Path,
    identifier: str,
    fetch_fn,
    params: Optional[Dict] = None,
    max_age_hours: Optional[int] = None,
    series_id: Optional[Any] = None,
) -> tuple[Optional[Any], bool]:
    """
    Single-flight cache helper:
    - First tries cache.
    - If missing, acquires per-key lock, rechecks cache, then fetches and saves.
    Returns (data, was_cached).
    """
    cached = load_from_disk_cache(cache_dir, identifier, params=params, max_age_hours=max_age_hours, series_id=series_id)
    if cached is not None:
        return cached, True

    cache_path = get_cache_path(cache_dir, identifier, params, series_id)
    with _CacheFileLock(cache_path):
        cached = load_from_disk_cache(cache_dir, identifier, params=params, max_age_hours=max_age_hours, series_id=series_id)
        if cached is not None:
            return cached, True

        data = fetch_fn()
        if data is None:
            return None, False

        _write_cache_file(cache_path, identifier, data, series_id=series_id)
        return data, False


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

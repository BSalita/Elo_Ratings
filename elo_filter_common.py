"""Shared Elo leaderboard filters used by Streamlit, REST, and MCP."""

from __future__ import annotations

from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Optional, Sequence
import pathlib
import re
import sys
import unicodedata

import polars as pl


ACBL_DATE_RANGE_OPTIONS = (
    "All time",
    "Last 3 months",
    "Last 6 months",
    "Last 1 year",
    "Last 2 years",
    "Last 3 years",
    "Last 4 years",
    "Last 5 years",
)

ACBL_MASTERPOINT_RANGES = (
    (0, 5),
    (5, 20),
    (20, 50),
    (50, 100),
    (100, 200),
    (200, 300),
    (300, 500),
    (500, 750),
    (750, 1000),
    (1000, 1500),
    (1500, 2500),
    (2500, 3500),
    (3500, 5000),
    (5000, 7500),
    (7500, 10000),
    (10000, None),
)


def acbl_date_from_for_range(date_range: Optional[str]) -> Optional[str]:
    """Return the sidebar-equivalent inclusive lower date bound."""
    choice = (date_range or "All time").strip()
    days = {
        "All time": None,
        "Last 3 months": 90,
        "Last 6 months": 182,
        "Last 1 year": 365,
        "Last 2 years": 365 * 2,
        "Last 3 years": 365 * 3,
        "Last 4 years": 365 * 4,
        "Last 5 years": 365 * 5,
    }
    if choice not in days:
        raise ValueError(
            f"Unknown ACBL date_range {choice!r}; valid: {list(ACBL_DATE_RANGE_OPTIONS)}"
        )
    if days[choice] is None:
        return None
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    return (today - timedelta(days=days[choice])).strftime("%Y-%m-%d")


def format_masterpoints_label(
    lower: float | int,
    upper: float | int | None,
) -> str:
    if upper is None:
        return f"{int(lower)}+"
    return f"{int(lower)}-{int(upper)}"


def masterpoints_bounds(range_label: Optional[str]) -> tuple[float | None, float | None]:
    label = (range_label or "All").strip()
    if label == "All":
        return None, None
    for lower, upper in ACBL_MASTERPOINT_RANGES:
        if format_masterpoints_label(lower, upper) == label:
            return lower, upper
    valid = ["All"] + [
        format_masterpoints_label(lower, upper)
        for lower, upper in ACBL_MASTERPOINT_RANGES
    ]
    raise ValueError(f"Unknown masterpoints_range {label!r}; valid: {valid}")


def _ffbridge_index_helpers():
    try:
        from mlBridge.mlBridgeFFIndexLib import expand_player_aliases, load_persons

        return expand_player_aliases, load_persons
    except ImportError:
        pass
    root = pathlib.Path(__file__).resolve().parent
    mlbridge = next(
        (path for path in (root / "mlBridge", root.parent / "mlBridge") if path.is_dir()),
        None,
    )
    if mlbridge is None:
        return None
    parent = str(mlbridge.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    try:
        from mlBridge.mlBridgeFFIndexLib import expand_player_aliases, load_persons

        return expand_player_aliases, load_persons
    except ImportError:
        return None


def expand_ffbridge_player_numbers(tokens: Sequence[str]) -> list[str]:
    """Map a license, Lancelot, or Classic id to all known aliases."""
    cleaned = [str(token).strip() for token in tokens if str(token).strip()]
    if not cleaned:
        return []
    helpers = _ffbridge_index_helpers()
    if helpers is None:
        return list(dict.fromkeys(cleaned))
    expand_player_aliases, load_persons = helpers
    try:
        persons = load_persons()
    except (FileNotFoundError, OSError, ValueError):
        return list(dict.fromkeys(cleaned))
    return expand_player_aliases(persons, cleaned)


def _pair_contains_number_expr(column: str, numbers: Sequence[str]) -> pl.Expr:
    parts = (
        pl.col(column)
        .cast(pl.Utf8)
        .str.replace_all("_", "-")
        .str.split("-")
    )
    left = parts.list.get(0, null_on_oob=True)
    right = parts.list.get(1, null_on_oob=True)
    predicate = None
    for number in numbers:
        part = (left == number) | (right == number)
        predicate = part if predicate is None else predicate | part
    if predicate is None:
        return pl.lit(False)
    return predicate


def normalize_fuzzy_text(value: object) -> str:
    """Normalize accents, punctuation, whitespace, and case for fuzzy search."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def fuzzy_text_score(candidate: object, query: object) -> float:
    """Score a free-text candidate, preserving substring matches as exact."""
    haystack = normalize_fuzzy_text(candidate)
    needle = normalize_fuzzy_text(query)
    if not haystack or not needle:
        return 0.0
    if needle in haystack:
        return 1.0
    scores = [SequenceMatcher(None, needle, haystack).ratio()]
    candidate_tokens = haystack.split()
    query_word_count = max(1, len(needle.split()))
    for start in range(len(candidate_tokens)):
        window = " ".join(candidate_tokens[start : start + query_word_count])
        scores.append(SequenceMatcher(None, needle, window).ratio())
    return max(scores)


def filter_fuzzy_text(
    df: pl.DataFrame,
    *,
    column: str,
    query: Optional[str],
    threshold: float = 0.72,
) -> pl.DataFrame:
    """Filter against fuzzy-matched unique values without row-wise Python work."""
    token = (query or "").strip()
    if not token or df.is_empty():
        return df
    if column not in df.columns:
        raise ValueError(f"Missing text filter column {column!r}")
    values = (
        df.select(pl.col(column).cast(pl.Utf8).drop_nulls().unique())
        .to_series()
        .to_list()
    )
    matches = [
        value for value in values if fuzzy_text_score(value, token) >= threshold
    ]
    return df.filter(pl.col(column).cast(pl.Utf8).is_in(matches))


def filter_normalized_substring(
    df: pl.DataFrame,
    *,
    column: str,
    query: Optional[str],
) -> pl.DataFrame:
    """Case- and accent-insensitive literal substring filter."""
    token = normalize_fuzzy_text(query)
    if not token or df.is_empty():
        return df
    if column not in df.columns:
        raise ValueError(f"Missing text filter column {column!r}")
    values = (
        df.select(pl.col(column).cast(pl.Utf8).drop_nulls().unique())
        .to_series()
        .to_list()
    )
    matches = [value for value in values if token in normalize_fuzzy_text(value)]
    return df.filter(pl.col(column).cast(pl.Utf8).is_in(matches))


def filter_identity_table(
    df: pl.DataFrame,
    *,
    rating_type: str,
    player_name: Optional[str] = None,
    player_number: Optional[str] = None,
    player_name_column: str,
    player_id_column: str,
    pair_name_column: str,
    pair_id_column: str,
) -> pl.DataFrame:
    """Apply the shared name-contains and exact-player-number predicates."""
    if df.is_empty():
        return df
    if rating_type not in ("Players", "Pairs"):
        raise ValueError("rating_type must be 'Players' or 'Pairs'")

    name_token = (player_name or "").strip()
    number_token = (player_number or "").strip()
    if number_token and not number_token.isdigit():
        raise ValueError("player_number must contain digits only")

    result = df
    if name_token:
        name_column = (
            player_name_column if rating_type == "Players" else pair_name_column
        )
        if name_column not in result.columns:
            raise ValueError(f"Missing identity column {name_column!r}")
        result = filter_fuzzy_text(
            result,
            column=name_column,
            query=name_token,
        )
    if number_token and not result.is_empty():
        id_column = player_id_column if rating_type == "Players" else pair_id_column
        if id_column not in result.columns:
            raise ValueError(f"Missing identity column {id_column!r}")
        number_tokens = [number_token]
        if rating_type == "Players":
            result = result.filter(
                pl.col(id_column).cast(pl.Utf8).is_in(number_tokens)
            )
        else:
            result = result.filter(_pair_contains_number_expr(id_column, number_tokens))
    return result


def filter_acbl_leaderboard(
    df: pl.DataFrame,
    *,
    rating_type: str,
    player_name: Optional[str] = None,
    player_number: Optional[str] = None,
    masterpoints_range: Optional[str] = None,
) -> pl.DataFrame:
    """ACBL sidebar filters applied after the Top-N report query."""
    result = filter_identity_table(
        df,
        rating_type=rating_type,
        player_name=player_name,
        player_number=player_number,
        player_name_column="Player_Name",
        player_id_column="Player_ID",
        pair_name_column="Pair_Names",
        pair_id_column="Pair_IDs",
    )
    lower, upper = masterpoints_bounds(masterpoints_range)
    if rating_type == "Players" and lower is not None and "MasterPoints" in result.columns:
        points = pl.col("MasterPoints").cast(pl.Float64, strict=False)
        result = (
            result.filter(points >= lower)
            if upper is None
            else result.filter((points >= lower) & (points < upper))
        )
    return result


def filter_ffbridge_leaderboard(
    df: pl.DataFrame,
    *,
    rating_type: str,
    player_name: Optional[str] = None,
    player_number: Optional[str] = None,
) -> pl.DataFrame:
    """FFBridge sidebar identity filters applied after the Top-N query."""
    number_token = (player_number or "").strip()
    if number_token and not number_token.isdigit():
        raise ValueError("player_number must contain digits only")
    aliases = (
        expand_ffbridge_player_numbers([number_token]) if number_token else None
    )
    result = filter_identity_table(
        df,
        rating_type=rating_type,
        player_name=player_name,
        player_number=None,
        player_name_column="Player_Name",
        player_id_column="Player_ID",
        pair_name_column="Pair_Name",
        pair_id_column="Pair_ID",
    )
    if not aliases or result.is_empty():
        return result
    if rating_type == "Players":
        if "Player_ID" not in result.columns:
            raise ValueError("Missing identity column 'Player_ID'")
        return result.filter(pl.col("Player_ID").cast(pl.Utf8).is_in(aliases))
    if "Pair_ID" not in result.columns:
        raise ValueError("Missing identity column 'Pair_ID'")
    return result.filter(_pair_contains_number_expr("Pair_ID", aliases))

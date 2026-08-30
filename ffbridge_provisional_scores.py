"""Immediate provisional scores for FFBridge simultaneous events.

National Lancelot ranking rows can be present with every percentage set to zero
while the organizer's provisional Scratch and Handicap tables are already
published.  This module matches those BridgeInterNet tables back to Lancelot
team IDs.  It intentionally returns no score when a match is ambiguous.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

_ROOT = Path(__file__).resolve().parent
_MLBRIDGE_PATH = next(
    (path for path in (_ROOT / "mlBridge", _ROOT.parent / "mlBridge") if path.is_dir()),
    None,
)
if _MLBRIDGE_PATH is None:
    raise FileNotFoundError("mlBridge not found at ./mlBridge or ../mlBridge")
_PKG_ROOT = str(_MLBRIDGE_PATH.parent)
if _PKG_ROOT not in sys.path:
    sys.path.append(_PKG_ROOT)

from mlBridge import mlBridgeBILib  # noqa: E402
from mlBridge.mlBridgeBILib import (  # noqa: E402
    BI_SERIES_IDS,
    BRIDGEINTER_BASE_URL,
    OCTOPUS_SERIES_ID,
    SIMULTANET_SERIES_ID,
    fetch_session_pair_scores,
)


def _number(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def national_ranking_is_pending(ranking: Iterable[Dict[str, Any]]) -> bool:
    """Return True only when the entire ranking is an unpublished zero shell."""
    rows = [row for row in ranking if isinstance(row, dict)]
    if not rows:
        return False

    national_scores = [
        _number(row.get("sessionScore") if row.get("sessionScore") is not None else row.get("totalScore"))
        for row in rows
    ]
    if any(score not in (None, 0.0) for score in national_scores):
        return False

    publication_values = (
        "handicapPercentage",
        "scoreHandicap",
        "totalScoreWithoutHandicap",
        "rankWithoutHandicap",
        "theoreticalRank",
    )
    return all(
        all(row.get(field) is None for field in publication_values)
        and _number(row.get("peBonus")) in (None, 0.0)
        and _number(row.get("rank")) in (None, 0.0, 1.0)
        for row in rows
    )


def fetch_provisional_pair_percentages(
    ranking: Iterable[Dict[str, Any]],
    tournament_date: str,
    series_id: Optional[Any] = None,
    *,
    get_text: Optional[Callable[[str], str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Return Scratch/Handicap values keyed by Lancelot team ID."""
    return fetch_session_pair_scores(
        ranking,
        tournament_date,
        series_id,
        get_text=get_text,
    )


__all__ = [
    "BI_SERIES_IDS",
    "BRIDGEINTER_BASE_URL",
    "OCTOPUS_SERIES_ID",
    "SIMULTANET_SERIES_ID",
    "fetch_provisional_pair_percentages",
    "mlBridgeBILib",
    "national_ranking_is_pending",
]

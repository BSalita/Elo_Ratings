"""Immediate provisional scores for FFBridge simultaneous events.

National Lancelot ranking rows can be present with every percentage set to zero
while the organizer's provisional Scratch and Handicap tables are already
published.  This module reads those tables and matches them back to Lancelot
team IDs.  It intentionally returns no score when a match is ambiguous.
"""

from __future__ import annotations

import html
import re
import unicodedata
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, Optional
from urllib.parse import urljoin

import requests


OCTOPUS_SERIES_ID = 386
BRIDGEINTER_BASE_URL = "http://www.bridgeinter.net/"
_PCT_RE = re.compile(r"(\d{1,2}\.\d{2})%")
_CLUB_LINK_RE = re.compile(
    r"""href\s*=\s*["']?([^"'\s>]*restotal\.php\?[^"'\s>]+)""",
    flags=re.IGNORECASE,
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


def _normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", html.unescape(value or ""))
    ascii_text = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", ascii_text.upper()).strip()


def _page_text(page_html: str) -> str:
    without_scripts = re.sub(
        r"<(?:script|style)\b[\s\S]*?</(?:script|style)>",
        " ",
        page_html,
        flags=re.IGNORECASE,
    )
    return _normalize_text(re.sub(r"<[^>]+>", " ", without_scripts))


def _octopus_url(date_yyyy_mm_dd: str, category: str) -> Optional[str]:
    try:
        day = datetime.strptime(date_yyyy_mm_dd[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None
    route = {0: ("octopus_l", "lo"), 3: ("octopus_j", "jo")}.get(day.weekday())
    if route is None or category not in {"scratch", "handicap"}:
        return None
    directory, prefix = route
    score_code = "s" if category == "scratch" else "h"
    return (
        f"{BRIDGEINTER_BASE_URL}{directory}/resseance_{prefix[0]}.php"
        f"?v_codeseance={prefix}{day.strftime('%y%m%d')}"
        f"&v_type_classement={score_code}"
    )


def _default_get_text(url: str) -> str:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def _category_text(
    date_yyyy_mm_dd: str,
    category: str,
    get_text: Callable[[str], str],
) -> tuple[str, Optional[str]]:
    main_url = _octopus_url(date_yyyy_mm_dd, category)
    if main_url is None:
        return "", None
    main_html = get_text(main_url)
    pages = [main_html]
    for relative_url in dict.fromkeys(_CLUB_LINK_RE.findall(main_html)):
        pages.append(get_text(urljoin(main_url, html.unescape(relative_url))))
    return " ".join(_page_text(page) for page in pages), main_url


def _find_pair_percentage(page_text: str, surname1: str, surname2: str) -> Optional[float]:
    surname1 = _normalize_text(surname1)
    surname2 = _normalize_text(surname2)
    if not page_text or not surname1 or not surname2:
        return None

    matches: list[tuple[int, float]] = []
    for pct_match in _PCT_RE.finditer(page_text):
        start = max(0, pct_match.start() - 140)
        context = page_text[start:pct_match.start()]
        pos1 = context.rfind(surname1)
        pos2 = context.rfind(surname2)
        if pos1 < 0 or pos2 < 0:
            continue
        distance = max(len(context) - pos1, len(context) - pos2)
        matches.append((distance, float(pct_match.group(1))))

    if not matches:
        return None
    matches.sort()
    best_distance, best_value = matches[0]
    equally_close = {value for distance, value in matches if distance == best_distance}
    return best_value if len(equally_close) == 1 else None


def fetch_provisional_pair_percentages(
    ranking: Iterable[Dict[str, Any]],
    tournament_date: str,
    series_id: Optional[Any],
    *,
    get_text: Optional[Callable[[str], str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Return provisional Scratch/Handicap values keyed by Lancelot team ID."""
    try:
        normalized_series_id = int(series_id) if series_id is not None else None
    except (TypeError, ValueError):
        normalized_series_id = None
    if normalized_series_id != OCTOPUS_SERIES_ID:
        return {}

    fetch_text = get_text or _default_get_text
    try:
        scratch_text, scratch_url = _category_text(
            tournament_date, "scratch", fetch_text
        )
        handicap_text, handicap_url = _category_text(
            tournament_date, "handicap", fetch_text
        )
    except requests.RequestException as exc:
        print(f"[provisional] BridgeInterNet fetch failed: {exc}", flush=True)
        return {}

    scores: Dict[str, Dict[str, Any]] = {}
    for row in ranking:
        team = row.get("team") if isinstance(row, dict) else None
        if not isinstance(team, dict):
            continue
        player1 = team.get("player1")
        player2 = team.get("player2")
        if not isinstance(player1, dict) or not isinstance(player2, dict):
            continue
        team_id = str(team.get("id") or "")
        if not team_id:
            continue
        scratch = _find_pair_percentage(
            scratch_text,
            str(player1.get("lastName") or ""),
            str(player2.get("lastName") or ""),
        )
        handicap = _find_pair_percentage(
            handicap_text,
            str(player1.get("lastName") or ""),
            str(player2.get("lastName") or ""),
        )
        scores[team_id] = {
            "scratch_percentage": scratch,
            "handicap_percentage": handicap,
            "scratch_url": scratch_url,
            "handicap_url": handicap_url,
        }
    return scores

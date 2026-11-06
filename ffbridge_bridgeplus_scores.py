"""Bridge+ handicap percentages for Roy René handicap sessions.

On the first Tuesday of the month Bridge+ runs the Roy René as a handicap
tournament: each pair gets a percentage bonus that shrinks as its players'
IV rises. FFBridge's Lancelot ranking sometimes carries the handicap
percentages (6 Oct 2026: totalBonus set) and sometimes only the scratch ones
(3 Nov 2026: every totalBonus 0, field average 50%). Bridge+'s club pages
always show the handicap percentage, so this module reads them.

Each row carries both players' licenses and, for Mitchell movements, the
section, direction and table, so the caller can match it to a Lancelot team.
"""

from __future__ import annotations

import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from html import unescape
from typing import Callable, Dict, Iterable, List

import requests

RESULTS_URL = "https://www.bridgeplus.com/nos-simultanes/resultats/"
CLUB_URL = RESULTS_URL + "?p=club/{tournament}/{club}"
REQUEST_TIMEOUT = 30
_USER_AGENT = {"User-Agent": "Mozilla/5.0"}

Fetch = Callable[[str], str]

_FRENCH_MONTHS = {
    "janvier": 1, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
    "juillet": 7, "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11,
    "decembre": 12,
}
# "Mardi 3 Novembre 2026 (Après-midi)</div> <div ...> <a href='...p=clubs/S202722'"
_RESULTS_ROW_RE = re.compile(
    r"(\d{1,2})\s+([^\s<]+)\s+(\d{4})[^<]*</div>\s*<div[^>]*>\s*"
    r"<a href='[^']*p=clubs/(S\d+)'",
)
_CLUB_ROW_SPLIT = "<div class='row text-center raye"
_RANK_RE = re.compile(r"<div class='col-2 p-0'>\s*(\d+)\s*</div>")
_PCT_RE = re.compile(r"<div class='col-10 p-0'>\s*(\d{1,3},\d{2})\s*</div>")
_LICENSES_RE = re.compile(r"d-sm-block'>([^<]*)</div>")
_ROUTE_RE = re.compile(r"href='([^']*p=route/(S\d+)/(\d+)/([A-Za-z]+)/(\d+))'")


def default_fetch(url: str) -> str:
    response = requests.get(url, headers=_USER_AGENT, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    response.encoding = "utf-8"
    return response.text


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    ).casefold()


def license_key(value: object) -> str:
    """FFB license without leading zeros ("09500754" -> "9500754")."""
    return str(value or "").strip().lstrip("0")


def parse_results_list(html: str) -> List[Dict[str, str]]:
    """Every simultaneous on the Bridge+ results list, with its date."""
    rows = []
    for day, month, year, tournament_id in _RESULTS_ROW_RE.findall(html):
        number = _FRENCH_MONTHS.get(_fold(month))
        if number is None:
            raise ValueError(f"Bridge+ results list has an unknown month {month!r}")
        rows.append(
            {
                "tournament_id": tournament_id,
                "date": f"{year}-{number:02d}-{int(day):02d}",
            }
        )
    return rows


def parse_club_page(html: str) -> List[Dict[str, object]]:
    """One row per pair on a Bridge+ club results page."""
    rows = []
    for block in html.split(_CLUB_ROW_SPLIT)[1:]:
        route = _ROUTE_RE.search(block)
        pct = _PCT_RE.search(block)
        rank = _RANK_RE.search(block)
        if route is None or pct is None or rank is None:
            raise ValueError(f"Bridge+ club row is missing rank, % or route: {block[:200]!r}")
        licenses_match = _LICENSES_RE.search(block)
        licenses = (
            {
                license_key(token)
                for token in unescape(licenses_match.group(1)).split("-")
                if license_key(token)
            }
            if licenses_match
            else set()
        )
        url, tournament_id, club, section, pair_number = route.groups()
        number = int(pair_number)
        # Mitchell pairs are 1xx (NS) / 2xx (EW); Howell pairs are 1..n and
        # carry no direction.
        if number >= 300:
            raise ValueError(f"Bridge+ pair number {pair_number} is not 1xx, 2xx or a Howell pair")
        direction = {1: "NS", 2: "EW"}.get(number // 100)
        rows.append(
            {
                "tournament_id": tournament_id,
                "club": club.lstrip("0"),
                "section": section.upper(),
                "direction": direction,
                "table": number % 100 if direction else None,
                "rank": int(rank.group(1)),
                "pct": float(pct.group(1).replace(",", ".")),
                "licenses": licenses,
                "url": url,
            }
        )
    return rows


def tournaments_on(session_date: str, fetch: Fetch = default_fetch) -> List[str]:
    """Bridge+ simultaneous ids played on session_date (YYYY-MM-DD)."""
    return [
        row["tournament_id"]
        for row in parse_results_list(fetch(RESULTS_URL))
        if row["date"] == session_date[:10]
    ]


def club_rows(
    session_date: str,
    clubs: Iterable[str],
    fetch: Fetch = default_fetch,
    workers: int = 8,
) -> Dict[str, List[Dict[str, object]]]:
    """Bridge+ rows for every club, keyed by club code without leading zeros.

    Returns {} when Bridge+ lists no simultaneous that day (older seasons drop
    off the results list).
    """
    tournaments = tournaments_on(session_date, fetch)
    if not tournaments:
        return {}
    if len(tournaments) > 1:
        raise ValueError(
            f"Bridge+ lists {len(tournaments)} simultaneous on {session_date[:10]}: {tournaments}"
        )
    tournament_id = tournaments[0]
    club_list = sorted({str(club) for club in clubs if str(club)})
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pages = list(
            pool.map(
                # FFB club codes are 7 digits; Lancelot sends them as ints.
                lambda club: parse_club_page(
                    fetch(CLUB_URL.format(tournament=tournament_id, club=club.zfill(7)))
                ),
                club_list,
            )
        )
    return {club.lstrip("0"): page for club, page in zip(club_list, pages)}


__all__ = [
    "CLUB_URL",
    "RESULTS_URL",
    "club_rows",
    "default_fetch",
    "license_key",
    "parse_club_page",
    "parse_results_list",
    "tournaments_on",
]

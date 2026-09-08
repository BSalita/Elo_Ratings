"""Elo-style fuzzy player-name matching used by postmortem sidebars."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Iterable, Mapping, Sequence

FUZZY_NAME_THRESHOLD = 0.72
MIN_FUZZY_LETTERS = 3


def normalize_fuzzy_text(value: object) -> str:
    """Normalize accents, punctuation, whitespace, and case for fuzzy search."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def fuzzy_text_score(candidate: object, query: object) -> float:
    """Score a free-text candidate; substring matches count as exact."""
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


def name_match_rank(display: object, query: object) -> tuple[float, int, int]:
    """Higher is better: fuzzy score, exact last name, last-name prefix."""
    needle = normalize_fuzzy_text(query)
    display_n = normalize_fuzzy_text(display)
    last_n = display_n.split()[-1] if display_n else ""
    return (
        fuzzy_text_score(display, query),
        1 if last_n == needle else 0,
        1 if needle and last_n.startswith(needle) else 0,
    )


def name_query_matches(candidate: object, query: object) -> bool:
    needle = normalize_fuzzy_text(query)
    if not needle:
        return False
    if len(needle) < MIN_FUZZY_LETTERS:
        haystack = normalize_fuzzy_text(candidate)
        return needle in haystack.split() or haystack == needle
    return fuzzy_text_score(candidate, query) >= FUZZY_NAME_THRESHOLD


def rank_named_records(
    rows: Sequence[Mapping[str, Any]],
    query: str,
    *,
    name_key: str = "player_name",
    drop_below_threshold: bool = False,
) -> list[dict[str, Any]]:
    """Sort records by Elo-style name rank and attach match_score (0-100)."""
    ranked: list[tuple[float, int, int, dict[str, Any]]] = []
    for row in rows:
        record = dict(row)
        display = record.get(name_key) or ""
        score, exact_last, prefix_last = name_match_rank(display, query)
        if drop_below_threshold and not name_query_matches(display, query):
            continue
        record["match_score"] = round(score * 100.0, 1)
        ranked.append((score, exact_last, prefix_last, record))
    ranked.sort(
        key=lambda item: (-item[0], -item[1], -item[2], str(item[3].get(name_key) or ""))
    )
    return [item[3] for item in ranked]


def filter_name_list(names: Iterable[str], query: str) -> list[str]:
    """Return names that fuzzy-match query, best first."""
    values = [str(name) for name in names if str(name).strip()]
    token = (query or "").strip()
    if not token:
        return values
    scored = [
        (name_match_rank(name, token), name)
        for name in values
        if name_query_matches(name, token)
    ]
    scored.sort(key=lambda item: (-item[0][0], -item[0][1], -item[0][2], item[1]))
    return [name for _rank, name in scored]

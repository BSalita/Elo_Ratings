"""ACBL event strata (MP-limit) labels and normalization helpers.

Elo board parquets carry ``mp_limit`` (raw) and ``strata_bucket`` (normalized),
written by ``acbl/acbl_elo_ratings_create.py``. Keep bucket rules in sync with
``acbl/acbl_strata.py``.

Elo *update* importance (Open ≫ restricted) is applied in
``mlBridge.mlBridgeAugmentLib.EVENT_MP_LIMIT_K_SCALE`` during
``compute_matchpoint_elo_ratings`` — not here.
"""
from __future__ import annotations

import re
from typing import Optional

# UI / API selectbox labels (order = display order; Open is default).
STRATA_OPTIONS: tuple[str, ...] = (
    "Open (unlimited)",
    "0-299",
    "0-499 / NLM",
    "0-749",
    "0-1500",
    "Other restricted",
    "All",
)

STRATA_DEFAULT = "Open (unlimited)"

BUCKET_OPEN = "open"
BUCKET_0_299 = "0-299"
BUCKET_0_499_NLM = "0-499-nlm"
BUCKET_0_749 = "0-749"
BUCKET_0_1500 = "0-1500"
BUCKET_OTHER = "other-restricted"

_LABEL_TO_BUCKET: dict[str, Optional[str]] = {
    "Open (unlimited)": BUCKET_OPEN,
    "0-299": BUCKET_0_299,
    "0-499 / NLM": BUCKET_0_499_NLM,
    "0-749": BUCKET_0_749,
    "0-1500": BUCKET_0_1500,
    "Other restricted": BUCKET_OTHER,
    "All": None,
}

_CLUB_PREFIX_RE = re.compile(r"^MP\s*Limits:\s*", re.IGNORECASE)
_INT_RE = re.compile(r"[\d,]+")


def strata_label_to_bucket(label: str) -> Optional[str]:
    """Map UI label to ``strata_bucket``. ``All`` -> ``None`` (no filter)."""
    if not label:
        return BUCKET_OPEN
    if label in _LABEL_TO_BUCKET:
        return _LABEL_TO_BUCKET[label]
    if label in {
        BUCKET_OPEN, BUCKET_0_299, BUCKET_0_499_NLM,
        BUCKET_0_749, BUCKET_0_1500, BUCKET_OTHER,
    }:
        return label
    return BUCKET_OPEN


def _parse_int_token(token: str) -> Optional[int]:
    token = (token or "").strip()
    if not token or token.lower() == "none":
        return None
    m = _INT_RE.search(token.replace(",", ""))
    if not m:
        return None
    try:
        return int(m.group(0).replace(",", ""))
    except ValueError:
        return None


def club_top_mp_limit(mp_limits_raw: str | None) -> Optional[int]:
    if mp_limits_raw is None:
        return None
    text = str(mp_limits_raw).strip()
    if not text:
        return None
    text = _CLUB_PREFIX_RE.sub("", text).strip()
    if not text:
        return None
    return _parse_int_token(text.split("/")[0].strip())


def tournament_top_mp_limit(mp_limit_raw: str | None) -> tuple[Optional[int], bool]:
    if mp_limit_raw is None:
        return (None, False)
    text = str(mp_limit_raw).strip()
    if not text or text.lower() == "none":
        return (None, False)
    is_nlm = text.upper().startswith("NLM")
    if "/" in text:
        parts = text.split("/")
        for part in reversed(parts):
            if part.upper() == "NLM":
                continue
            if part.upper().startswith("0-"):
                return (_parse_int_token(part[2:]), is_nlm)
            n = _parse_int_token(part)
            if n is not None:
                return (n, is_nlm)
        return (None, is_nlm)
    if text.upper() == "NLM":
        return (500, True)
    if text.upper().startswith("0-"):
        return (_parse_int_token(text[2:]), is_nlm)
    return (_parse_int_token(text), is_nlm)


def bucket_from_top_limit(top: Optional[int], *, is_nlm: bool = False) -> str:
    if top is None and not is_nlm:
        return BUCKET_OPEN
    if is_nlm and (top is None or top <= 500):
        return BUCKET_0_499_NLM
    if top is None:
        return BUCKET_OTHER
    if top <= 300:
        return BUCKET_0_299
    if top <= 500:
        return BUCKET_0_499_NLM
    if top <= 750:
        return BUCKET_0_749
    if top <= 1500:
        return BUCKET_0_1500
    return BUCKET_OTHER


def club_strata_bucket(mp_limits_raw: str | None) -> str:
    return bucket_from_top_limit(club_top_mp_limit(mp_limits_raw))


def tournament_strata_bucket(mp_limit_raw: str | None) -> str:
    top, is_nlm = tournament_top_mp_limit(mp_limit_raw)
    return bucket_from_top_limit(top, is_nlm=is_nlm)

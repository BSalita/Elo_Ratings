"""Session-level FFBridge pair rankings from the persisted Elo results parquet."""
from __future__ import annotations

from typing import Any, Mapping

import polars as pl

import ffbridge_report_service as reports


_SCOPES = frozenset({"national", "club"})
_SESSION_COLUMNS = (
    "tournament_id",
    "date",
    "tournament_name",
    "team_id",
    "player1_id",
    "player2_id",
    "player1_lancelot_id",
    "player2_lancelot_id",
    "player1_name",
    "player2_name",
    "club_code",
    "club_name",
    "National_Scratch_Pct",
    "National_Scratch_Rank",
    "National_Handicap_Pct",
    "National_Handicap_Rank",
    "Club_Scratch_Pct",
    "Club_Scratch_Rank",
    "Club_Handicap_Pct",
    "Club_Handicap_Rank",
    "iv_bonus",
)


def _normalize_club_code(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    stripped = text.lstrip("0")
    return stripped or "0"


def _round_pct(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value), 2)


def _rank(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _row(entry: Mapping[str, Any], *, prefix: str, field_size: int) -> dict[str, Any]:
    return {
        "team_id": entry.get("team_id"),
        "player1_id": entry.get("player1_id"),
        "player2_id": entry.get("player2_id"),
        "player1_lancelot_id": entry.get("player1_lancelot_id"),
        "player2_lancelot_id": entry.get("player2_lancelot_id"),
        "player1_name": entry.get("player1_name"),
        "player2_name": entry.get("player2_name"),
        "club_code": entry.get("club_code"),
        "scratch_pct": _round_pct(entry.get(f"{prefix}_Scratch_Pct")),
        "scratch_rank": _rank(entry.get(f"{prefix}_Scratch_Rank")),
        "iv_bonus": _round_pct(entry.get("iv_bonus")),
        "handicap_pct": _round_pct(entry.get(f"{prefix}_Handicap_Pct")),
        "handicap_rank": _rank(entry.get(f"{prefix}_Handicap_Rank")),
        "field_size": field_size,
    }


def get_session_ranking(
    session_id: str | int,
    *,
    scope: str = "national",
    club_code: str | None = None,
    api_key: str | None = None,
    fetch_iv: bool = True,
    results_df: pl.DataFrame | None = None,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return official per-pair scratch and handicap ranks for one session."""
    sid = str(session_id).strip()
    if not sid.isdigit():
        raise ValueError("session_id must contain only digits")
    normalized_scope = str(scope).strip().lower()
    if normalized_scope not in _SCOPES:
        raise ValueError("scope must be either national or club")

    if results_df is None:
        results_df, loaded_meta = reports.load_results(api_key, fetch_iv)
        meta = loaded_meta
    results_df = reports.filter_valid_percentages(results_df)
    if "tournament_id" not in results_df.columns:
        raise ValueError("Elo results parquet lacks tournament_id")
    available = [column for column in _SESSION_COLUMNS if column in results_df.columns]
    session = results_df.filter(
        pl.col("tournament_id").cast(pl.Utf8) == sid
    ).select(available)
    if session.is_empty():
        raise FileNotFoundError(f"No Elo results for session {sid}")

    clubs = sorted(
        {
            _normalize_club_code(value)
            for value in session["club_code"].to_list()
            if value not in (None, "")
        }
    ) if "club_code" in session.columns else []

    wanted_club = None
    if normalized_scope == "club":
        if club_code is None or not str(club_code).strip():
            raise ValueError(
                f"club_code is required when scope=club. "
                f"Clubs in session {sid}: {clubs}"
            )
        wanted_club = _normalize_club_code(club_code)
        if wanted_club not in clubs:
            raise ValueError(
                f"Session {sid} has no club {wanted_club}. Clubs: {clubs}"
            )
        session = session.with_columns(
            pl.col("club_code")
            .cast(pl.Utf8)
            .fill_null("")
            .str.strip_chars()
            .str.replace(r"^0+", "")
            .alias("_club_code")
        ).filter(pl.col("_club_code") == wanted_club)
        if session.is_empty():
            raise FileNotFoundError(
                f"No Elo results for session {sid} club {wanted_club}"
            )

    prefix = "Club" if normalized_scope == "club" else "National"
    rank_column = f"{prefix}_Scratch_Rank"
    if rank_column in session.columns:
        session = session.sort(
            [
                pl.col(rank_column).is_null(),
                pl.col(rank_column),
                pl.col("team_id").cast(pl.Utf8),
            ]
        )
    field_size = session.height
    rows = [
        _row(entry, prefix=prefix, field_size=field_size)
        for entry in session.to_dicts()
    ]
    first = session.row(0, named=True) if field_size else {}
    return {
        "session_id": sid,
        "scope": normalized_scope,
        "club_code": wanted_club,
        "field_size": field_size,
        "date": first.get("date"),
        "tournament_name": first.get("tournament_name"),
        "rows": rows,
        "row_count": field_size,
        "dataset_built_at": (meta or {}).get("built_at"),
    }

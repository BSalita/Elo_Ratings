"""ACBL masterpoint awards by color (Platinum, Gold, Red, Black).

Tournament session JSON overalls carry ``mp_won`` / ``mp_color`` per pair.
Club ``pigment`` / ``awards_score`` parquets carry the same colors. Elo
parquets do not, so these rows are extracted into a compact sidecar and
joined in the report/detail API.

Event totals use the max award per (player, event, color) so a multi-session
final is not counted four times. Session grids show the award listed on that
session.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
from datetime import datetime
from typing import Iterable

import polars as pl
from tqdm import tqdm

AWARD_COLORS = ("Platinum", "Gold", "Red", "Black")
TOURNAMENT_AWARDS_FILENAME = "acbl_tournament_player_awards.parquet"
CLUB_AWARDS_FILENAME = "acbl_club_player_awards.parquet"

_AWARDS_SCHEMA = {
    "session_id": pl.Utf8,
    "event_id": pl.Utf8,
    "player_id": pl.Utf8,
    "pair_ids": pl.Utf8,
    "mp_color": pl.Utf8,
    "mp_won": pl.Float64,
}


def empty_awards() -> pl.DataFrame:
    return pl.DataFrame(schema=_AWARDS_SCHEMA)


def empty_award_totals(id_column: str) -> pl.DataFrame:
    return pl.DataFrame(
        schema={id_column: pl.Utf8, **{color: pl.Float64 for color in AWARD_COLORS}}
    )


def canonicalize_mp_color(value: object | None) -> str | None:
    if value is None:
        return None
    color = str(value).strip().title()
    if color not in AWARD_COLORS:
        return None
    return color


def event_id_from_session_id(session_id: str) -> str:
    session = str(session_id).strip()
    head, sep, tail = session.rpartition("-")
    if sep and tail.isdigit():
        return head
    return session


def canonical_pair_ids(player_ids: Iterable[object]) -> str:
    ids = sorted(
        {str(player).strip() for player in player_ids if player is not None and str(player).strip()}
    )
    return "-".join(ids)


def pair_member_ids(pair_ids: str) -> list[str]:
    return [part.strip() for part in str(pair_ids).split("-") if part.strip()]


def _award_row(
    *,
    session_id: str,
    event_id: str,
    player_id: object,
    pair_ids: str,
    mp_color: object,
    mp_won: object,
) -> dict | None:
    pid = str(player_id).strip() if player_id is not None else ""
    color = canonicalize_mp_color(mp_color)
    if not pid or color is None:
        return None
    try:
        amount = float(mp_won)
    except (TypeError, ValueError):
        return None
    if amount <= 0:
        return None
    return {
        "session_id": str(session_id).strip(),
        "event_id": str(event_id).strip(),
        "player_id": pid,
        "pair_ids": pair_ids,
        "mp_color": color,
        "mp_won": amount,
    }


def awards_from_tournament_overalls(
    session_id: str,
    overalls: Iterable[dict],
    *,
    event_id: str | None = None,
) -> list[dict]:
    sid = str(session_id).strip()
    eid = str(event_id).strip() if event_id else event_id_from_session_id(sid)
    rows: list[dict] = []
    for overall in overalls or []:
        players = overall.get("players") or []
        pair_ids = canonical_pair_ids(players)
        for player_id in players:
            row = _award_row(
                session_id=sid,
                event_id=eid,
                player_id=player_id,
                pair_ids=pair_ids,
                mp_color=overall.get("mp_color"),
                mp_won=overall.get("mp_won"),
            )
            if row:
                rows.append(row)
    return rows


def awards_from_club_event(event: dict) -> list[dict]:
    event_id = event.get("id")
    if event_id is None:
        return []
    sid = str(event_id).strip()
    rows: list[dict] = []
    for session in event.get("sessions") or []:
        for section in session.get("sections") or []:
            for pair in section.get("pair_summaries") or []:
                players = pair.get("players") or []
                pair_ids = canonical_pair_ids(
                    player.get("id_number") for player in players
                )
                for player in players:
                    player_id = player.get("id_number")
                    pigments: list[dict] = []
                    for award in player.get("awards_score") or []:
                        pigments.extend(award.get("pigment") or [])
                    for award in player.get("awards") or []:
                        pigments.extend(award.get("pigment") or [])
                    for pigment in pigments:
                        row = _award_row(
                            session_id=sid,
                            event_id=sid,
                            player_id=player_id,
                            pair_ids=pair_ids,
                            mp_color=pigment.get("color"),
                            mp_won=pigment.get("amount"),
                        )
                        if row:
                            rows.append(row)
    return rows


def awards_frame(rows: Iterable[dict]) -> pl.DataFrame:
    materialized = [row for row in rows]
    if not materialized:
        return empty_awards()
    return (
        pl.DataFrame(materialized, schema=_AWARDS_SCHEMA)
        .unique(
            subset=["session_id", "player_id", "mp_color", "mp_won"],
            keep="first",
        )
    )


def extract_tournament_awards(sessions_dir: pathlib.Path) -> pl.DataFrame:
    files = sorted(sessions_dir.glob("*.session.json"))
    rows: list[dict] = []
    for path in tqdm(files, desc="tournament awards", unit="file"):
        session = json.loads(path.read_text(encoding="utf-8"))
        session_id = str(session.get("id") or path.stem.replace(".session", "")).strip()
        event = session.get("event") or {}
        event_id = event.get("id") or event_id_from_session_id(session_id)
        rows.extend(
            awards_from_tournament_overalls(
                session_id,
                session.get("overalls") or [],
                event_id=str(event_id),
            )
        )
    return awards_frame(rows)


def extract_club_awards(details_files: Iterable[pathlib.Path]) -> pl.DataFrame:
    files = list(details_files)
    rows: list[dict] = []
    for path in tqdm(files, desc="club awards", unit="file"):
        event = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(awards_from_club_event(event))
    return awards_frame(rows)


def extract_club_awards_from_parquet(club_parquet_dir: pathlib.Path) -> pl.DataFrame:
    """Join club pigment/award tables into the compact sidecar schema.

    Club Elo uses ``session_id == event_id``, so both columns are the event id.
    """
    started = datetime.now()
    color = pl.col("color").cast(pl.Utf8).str.strip_chars().str.to_titlecase()
    pigment = (
        pl.scan_parquet(club_parquet_dir / "pigment.parquet")
        .select(
            pl.col("award_id").cast(pl.Utf8),
            pl.col("amount").cast(pl.Float64, strict=False).alias("mp_won"),
            color.alias("mp_color"),
        )
        .filter(pl.col("mp_color").is_in(list(AWARD_COLORS)) & (pl.col("mp_won") > 0))
    )
    award_links = pl.concat(
        [
            pl.scan_parquet(club_parquet_dir / "awards_score.parquet").select(
                pl.col("id").cast(pl.Utf8).alias("award_id"),
                pl.col("player_id").cast(pl.Utf8).alias("player_row_id"),
            ),
            pl.scan_parquet(club_parquet_dir / "awards.parquet").select(
                pl.col("id").cast(pl.Utf8).alias("award_id"),
                pl.col("player_id").cast(pl.Utf8).alias("player_row_id"),
            ),
        ]
    ).unique(subset=["award_id", "player_row_id"])
    players = pl.scan_parquet(club_parquet_dir / "players.parquet").select(
        pl.col("id").cast(pl.Utf8).alias("player_row_id"),
        pl.col("id_number").cast(pl.Utf8).str.strip_chars().alias("player_id"),
        pl.col("pair_summary_id").cast(pl.Utf8),
    )
    pair_ids = players.group_by("pair_summary_id").agg(
        pl.col("player_id")
        .filter(pl.col("player_id") != "")
        .unique()
        .sort()
        .str.join("-")
        .alias("pair_ids")
    )
    pair_summaries = pl.scan_parquet(club_parquet_dir / "pair_summaries.parquet").select(
        pl.col("id").cast(pl.Utf8).alias("pair_summary_id"),
        pl.col("section_id").cast(pl.Utf8),
    )
    sections = pl.scan_parquet(club_parquet_dir / "sections.parquet").select(
        pl.col("id").cast(pl.Utf8).alias("section_id"),
        pl.col("session_id").cast(pl.Utf8),
    )
    sessions = pl.scan_parquet(club_parquet_dir / "sessions.parquet").select(
        pl.col("id").cast(pl.Utf8).alias("session_id"),
        pl.col("event_id").cast(pl.Utf8).str.strip_chars().alias("event_id"),
    )
    frame = (
        pigment.join(award_links, on="award_id", how="inner")
        .join(players, on="player_row_id", how="inner")
        .join(pair_ids, on="pair_summary_id", how="left")
        .join(pair_summaries, on="pair_summary_id", how="inner")
        .join(sections, on="section_id", how="inner")
        .join(sessions, on="session_id", how="inner")
        .select(
            pl.col("event_id").alias("session_id"),
            pl.col("event_id"),
            pl.col("player_id"),
            pl.col("pair_ids").fill_null(""),
            pl.col("mp_color"),
            pl.col("mp_won"),
        )
        .filter(pl.col("player_id") != "")
        .unique(subset=["session_id", "player_id", "mp_color", "mp_won"])
        .collect()
    )
    elapsed = (datetime.now() - started).total_seconds()
    if elapsed > 30:
        print(
            f"extract_club_awards_from_parquet: {frame.height:,} rows in {elapsed:.1f}s",
            flush=True,
        )
    return frame


def pivot_award_totals(
    awards: pl.DataFrame,
    *,
    id_column: str,
    dedupe_events: bool = True,
) -> pl.DataFrame:
    """Sum awards by color. ``id_column`` is ``player_id`` or ``pair_ids``."""
    empty = empty_award_totals(id_column)
    if awards.is_empty() or id_column not in awards.columns:
        return empty
    frame = awards.filter(pl.col(id_column).cast(pl.Utf8).str.strip_chars() != "")
    if frame.is_empty():
        return empty
    if dedupe_events and "event_id" in frame.columns:
        frame = frame.group_by([id_column, "event_id", "mp_color"]).agg(
            pl.col("mp_won").max()
        )
    return frame.group_by(id_column).agg(
        [
            pl.col("mp_won")
            .filter(pl.col("mp_color") == color)
            .sum()
            .fill_null(0)
            .round(2)
            .alias(color)
            for color in AWARD_COLORS
        ]
    )


def pivot_session_awards(awards: pl.DataFrame) -> pl.DataFrame:
    """One row per session_id + player_id with color columns."""
    if awards.is_empty():
        return pl.DataFrame(
            schema={
                "session_id": pl.Utf8,
                "player_id": pl.Utf8,
                **{color: pl.Float64 for color in AWARD_COLORS},
            }
        )
    return awards.group_by(["session_id", "player_id"]).agg(
        [
            pl.col("mp_won")
            .filter(pl.col("mp_color") == color)
            .max()
            .fill_null(0)
            .round(2)
            .alias(color)
            for color in AWARD_COLORS
        ]
    )


def _scope_awards_to_sessions(
    awards: pl.DataFrame,
    session_ids: Iterable[str] | None,
) -> pl.DataFrame:
    if awards.is_empty() or session_ids is None:
        return awards
    sessions = pl.DataFrame(
        {"session_id": [str(session).strip() for session in session_ids]}
    ).unique()
    return awards.join(sessions, on="session_id", how="inner")


def pair_award_totals(awards: pl.DataFrame, pair_ids: Iterable[str]) -> pl.DataFrame:
    """Totals for pairs that both earned the same session/event/color award."""
    pairs = (
        pl.DataFrame({"Pair_IDs": [str(pair).strip() for pair in pair_ids]})
        .unique()
        .with_columns(pl.col("Pair_IDs").str.split("-").alias("_ids"))
        .filter(pl.col("_ids").list.len() >= 2)
        .with_columns(
            pl.col("_ids").list.get(0).alias("p1"),
            pl.col("_ids").list.get(1).alias("p2"),
        )
        .drop("_ids")
    )
    if pairs.is_empty() or awards.is_empty():
        return empty_award_totals("Pair_IDs")
    members = pairs.unpivot(
        index="Pair_IDs",
        on=["p1", "p2"],
        value_name="player_id",
    ).drop("variable")
    both = (
        members.join(awards, on="player_id", how="inner")
        .group_by(["Pair_IDs", "session_id", "event_id", "mp_color"])
        .agg(
            pl.col("player_id").n_unique().alias("n_players"),
            pl.col("mp_won").max(),
        )
        .filter(pl.col("n_players") >= 2)
        .rename({"Pair_IDs": "pair_ids"})
    )
    return pivot_award_totals(both, id_column="pair_ids").rename(
        {"pair_ids": "Pair_IDs"}
    )


def _insert_award_columns(frame: pl.DataFrame) -> pl.DataFrame:
    after = next(
        (
            column
            for column in (
                "MasterPoint_Rank",
                "MasterPoints",
                "Avg_MPs_Rank",
                "Avg_MPs",
            )
            if column in frame.columns
        ),
        None,
    )
    cols = [column for column in frame.columns if column not in AWARD_COLORS]
    award_cols = [color for color in AWARD_COLORS if color in frame.columns]
    if after and after in cols:
        index = cols.index(after) + 1
        cols = cols[:index] + award_cols + cols[index:]
    else:
        cols = cols + award_cols
    return frame.select(cols)


def attach_award_totals(
    leaderboard: pl.DataFrame,
    awards: pl.DataFrame,
    *,
    rating_type: str,
    session_ids: Iterable[str] | None = None,
) -> pl.DataFrame:
    """Add Platinum/Gold/Red/Black totals after MasterPoints / Avg_MPs."""
    zeros = leaderboard.with_columns(
        [pl.lit(0.0).alias(color) for color in AWARD_COLORS]
    )
    if leaderboard.is_empty() or awards.is_empty():
        return _insert_award_columns(zeros)
    scoped = _scope_awards_to_sessions(awards, session_ids)
    if scoped.is_empty():
        return _insert_award_columns(zeros)
    if rating_type.lower() == "pairs":
        if "Pair_IDs" not in leaderboard.columns:
            return _insert_award_columns(zeros)
        totals = pair_award_totals(
            scoped, leaderboard.get_column("Pair_IDs").cast(pl.Utf8).to_list()
        )
        join_column = "Pair_IDs"
    else:
        if "Player_ID" not in leaderboard.columns:
            return _insert_award_columns(zeros)
        totals = pivot_award_totals(scoped, id_column="player_id").rename(
            {"player_id": "Player_ID"}
        )
        join_column = "Player_ID"
    attached = (
        leaderboard.with_columns(pl.col(join_column).cast(pl.Utf8))
        .join(totals.with_columns(pl.col(join_column).cast(pl.Utf8)), on=join_column, how="left")
        .with_columns([pl.col(color).fill_null(0).cast(pl.Float64) for color in AWARD_COLORS])
    )
    return _insert_award_columns(attached)


def attach_session_awards(
    detail: pl.DataFrame,
    awards: pl.DataFrame,
    *,
    player_id: str | None = None,
    pair_ids: str | None = None,
) -> pl.DataFrame:
    """Broadcast session award colors onto board-level detail rows."""
    if detail.is_empty() or "Session" not in detail.columns:
        return detail
    zeros = detail.with_columns([pl.lit(0.0).alias(color) for color in AWARD_COLORS])
    if awards.is_empty():
        return zeros
    scoped = awards
    if player_id:
        scoped = scoped.filter(pl.col("player_id") == str(player_id).strip())
    elif pair_ids:
        scoped = scoped.filter(pl.col("player_id").is_in(pair_member_ids(pair_ids)))
    if scoped.is_empty():
        return zeros
    session_awards = pivot_session_awards(scoped).group_by("session_id").agg(
        [pl.col(color).max() for color in AWARD_COLORS]
    )
    attached = (
        detail.with_columns(pl.col("Session").cast(pl.Utf8).alias("_award_session"))
        .join(session_awards, left_on="_award_session", right_on="session_id", how="left")
        .drop("_award_session")
    )
    return attached.with_columns(
        [pl.col(color).fill_null(0).cast(pl.Float64) for color in AWARD_COLORS]
    )


def _unique_existing(paths: Iterable[pathlib.Path]) -> list[pathlib.Path]:
    seen: set[str] = set()
    out: list[pathlib.Path] = []
    for path in paths:
        resolved = str(path)
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        out.append(path)
    return out


def awards_search_paths(data_root: pathlib.Path, club_or_tournament: str) -> list[pathlib.Path]:
    filename = (
        CLUB_AWARDS_FILENAME
        if club_or_tournament.strip().lower() == "club"
        else TOURNAMENT_AWARDS_FILENAME
    )
    override = os.getenv("ACBL_PLAYER_AWARDS", "").strip()
    candidates = []
    if override:
        candidates.append(pathlib.Path(override))
    candidates.extend(
        [
            data_root / filename,
            data_root / "_wslc_host" / "acbl-stage" / "club_results_parquet" / filename,
            data_root / "club_results_parquet" / filename,
            pathlib.Path("e:/bridge/data/acbl") / filename,
        ]
    )
    return _unique_existing(candidates)


def load_player_awards(data_root: pathlib.Path, club_or_tournament: str) -> pl.DataFrame:
    paths = awards_search_paths(data_root, club_or_tournament)
    if not paths:
        return empty_awards()
    frame = pl.read_parquet(paths[0])
    missing = set(_AWARDS_SCHEMA) - set(frame.columns)
    if missing:
        raise ValueError(f"{paths[0]} is missing {sorted(missing)}")
    return frame.select(list(_AWARDS_SCHEMA))


def load_awards_for_players(
    data_root: pathlib.Path,
    club_or_tournament: str,
    player_ids: Iterable[str],
) -> pl.DataFrame:
    """Scan the sidecar for a small player set. Does not load the full table."""
    paths = awards_search_paths(data_root, club_or_tournament)
    ids = [str(player).strip() for player in player_ids if str(player).strip()]
    if not paths or not ids:
        return empty_awards()
    frame = (
        pl.scan_parquet(paths[0])
        .select(list(_AWARDS_SCHEMA))
        .filter(pl.col("player_id").is_in(ids))
        .collect()
    )
    missing = set(_AWARDS_SCHEMA) - set(frame.columns)
    if missing:
        raise ValueError(f"{paths[0]} is missing {sorted(missing)}")
    return frame


def write_player_awards_sidecar(awards: pl.DataFrame, path: pathlib.Path) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    awards.select(list(_AWARDS_SCHEMA)).write_parquet(path)
    return path


def tournament_sessions_search_paths() -> list[pathlib.Path]:
    override = os.getenv("ACBL_TOURNAMENT_SESSIONS", "").strip()
    candidates = []
    if override:
        candidates.append(pathlib.Path(override))
    candidates.extend(
        [
            pathlib.Path("e:/bridge/data/acbl/tournaments/sessions"),
            pathlib.Path("/data/_wslc_host/acbl-stage/tournaments/sessions"),
        ]
    )
    return [path for path in _unique_existing(candidates) if path.is_dir()]


def club_results_parquet_search_paths() -> list[pathlib.Path]:
    override = os.getenv("ACBL_CLUB_RESULTS_PARQUET", "").strip()
    candidates = []
    if override:
        candidates.append(pathlib.Path(override))
    candidates.extend(
        [
            pathlib.Path("e:/bridge/data/acbl/club_results_parquet"),
            pathlib.Path("/data/_wslc_host/acbl-stage/club_results_parquet"),
        ]
    )
    return [path for path in _unique_existing(candidates) if path.is_dir()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract ACBL player award sidecars.")
    parser.add_argument("--club", action="store_true")
    parser.add_argument("--tournament", action="store_true")
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("e:/bridge/data/acbl"))
    args = parser.parse_args(argv)
    if not args.club and not args.tournament:
        args.club = True
        args.tournament = True
    started = datetime.now()
    print(f"acbl_awards extract started {started.isoformat(timespec='seconds')}", flush=True)
    if args.tournament:
        sessions_dirs = tournament_sessions_search_paths()
        if not sessions_dirs:
            raise FileNotFoundError("No tournament sessions directory found.")
        awards = extract_tournament_awards(sessions_dirs[0])
        path = write_player_awards_sidecar(awards, args.out_dir / TOURNAMENT_AWARDS_FILENAME)
        print(f"wrote {path} ({awards.height:,} rows)", flush=True)
    if args.club:
        club_dirs = club_results_parquet_search_paths()
        if not club_dirs:
            raise FileNotFoundError("No club_results_parquet directory found.")
        awards = extract_club_awards_from_parquet(club_dirs[0])
        path = write_player_awards_sidecar(awards, args.out_dir / CLUB_AWARDS_FILENAME)
        print(f"wrote {path} ({awards.height:,} rows)", flush=True)
    ended = datetime.now()
    print(
        f"acbl_awards extract ended {ended.isoformat(timespec='seconds')} "
        f"elapsed {(ended - started).total_seconds():.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

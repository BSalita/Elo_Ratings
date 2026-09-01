"""Identify ACBL tournament events that award platinum masterpoints.

Board-result parquets carry ``mp_color`` (and ``mp_rating``). Elo rating
parquets do not; they join back through ``event_id``. An event awards
platinum only when ``mp_color`` is Platinum — not merely because it is
played at an NABC, and not because ``mp_rating`` is NABC+.
"""
from __future__ import annotations

import os
import pathlib
from typing import Iterable

import polars as pl

PLATINUM_MP_COLOR = "platinum"
SIDECAR_FILENAME = "acbl_tournament_platinum_events.parquet"
BOARD_RESULTS_FILENAME = "acbl_tournament_board_results_augmented.parquet"

_SIDECAR_SCHEMA = {
    "event_id": pl.Utf8,
    "event_name": pl.Utf8,
    "mp_color": pl.Utf8,
    "mp_rating": pl.Utf8,
}


def is_platinum_mp_color(value: object | None) -> bool:
    """True when ``value`` is the ACBL Platinum masterpoint color."""
    if value is None:
        return False
    return str(value).strip().lower() == PLATINUM_MP_COLOR


def platinum_mp_color_expr(column: str = "mp_color") -> pl.Expr:
    return pl.col(column).cast(pl.Utf8).str.strip_chars().str.to_lowercase() == PLATINUM_MP_COLOR


def platinum_events_from_awards(df: pl.DataFrame) -> pl.DataFrame:
    """Return unique platinum-awarding events from a board-results (or sidecar) frame."""
    if "event_id" not in df.columns:
        raise ValueError("awards frame is missing event_id")
    if "mp_color" not in df.columns:
        raise ValueError("awards frame is missing mp_color")

    event_id = pl.col("event_id").cast(pl.Utf8).str.strip_chars()
    selected = df.filter(platinum_mp_color_expr() & event_id.is_not_null() & (event_id != ""))
    if selected.is_empty():
        return pl.DataFrame(schema=_SIDECAR_SCHEMA)

    exprs = [event_id.alias("event_id")]
    for col, dtype in (
        ("event_name", pl.Utf8),
        ("mp_color", pl.Utf8),
        ("mp_rating", pl.Utf8),
    ):
        if col in selected.columns:
            exprs.append(pl.col(col).cast(dtype).alias(col))
        else:
            exprs.append(pl.lit(None, dtype=dtype).alias(col))
    return (
        selected.select(exprs)
        .unique(subset=["event_id"])
        .sort("event_id")
    )


def platinum_event_ids(events: pl.DataFrame) -> list[str]:
    if events.is_empty() or "event_id" not in events.columns:
        return []
    return (
        events.get_column("event_id")
        .cast(pl.Utf8)
        .str.strip_chars()
        .drop_nulls()
        .to_list()
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


def board_results_search_paths(data_root: pathlib.Path) -> list[pathlib.Path]:
    override = os.getenv("ACBL_TOURNAMENT_BOARD_RESULTS", "").strip()
    candidates = []
    if override:
        candidates.append(pathlib.Path(override))
    candidates.extend(
        [
            data_root / BOARD_RESULTS_FILENAME,
            data_root / "_wslc_host" / "acbl-stage" / "club_results_parquet" / BOARD_RESULTS_FILENAME,
            data_root / "club_results_parquet" / BOARD_RESULTS_FILENAME,
            pathlib.Path("/data/_wslc_host/acbl-stage/club_results_parquet") / BOARD_RESULTS_FILENAME,
        ]
    )
    return _unique_existing(candidates)


def sidecar_search_paths(data_root: pathlib.Path) -> list[pathlib.Path]:
    override = os.getenv("ACBL_PLATINUM_EVENTS_PATH", "").strip()
    candidates = []
    if override:
        candidates.append(pathlib.Path(override))
    candidates.append(data_root / SIDECAR_FILENAME)
    return _unique_existing(candidates)


def extract_platinum_events_from_board_results(path: pathlib.Path) -> pl.DataFrame:
    want = [col for col in ("event_id", "event_name", "mp_color", "mp_rating") if True]
    schema_names = set(pl.scan_parquet(path).collect_schema().names())
    missing = {"event_id", "mp_color"} - schema_names
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    cols = [col for col in want if col in schema_names]
    return platinum_events_from_awards(pl.scan_parquet(path).select(cols).collect())


def load_platinum_events(data_root: pathlib.Path) -> pl.DataFrame:
    """Load unique platinum-awarding events.

    Prefers the live tournament board-results parquet (authoritative
    ``mp_color``) and falls back to a DATA_ROOT sidecar written from that
    same extract. Missing sources yield an empty frame — callers that
    require platinum events must fail fast.
    """
    board_paths = board_results_search_paths(data_root)
    if board_paths:
        return extract_platinum_events_from_board_results(board_paths[0])
    sidecar_paths = sidecar_search_paths(data_root)
    if sidecar_paths:
        return platinum_events_from_awards(pl.read_parquet(sidecar_paths[0]))
    return pl.DataFrame(schema=_SIDECAR_SCHEMA)


def write_platinum_events_sidecar(events: pl.DataFrame, data_root: pathlib.Path) -> pathlib.Path:
    path = data_root / SIDECAR_FILENAME
    platinum_events_from_awards(events).write_parquet(path)
    return path

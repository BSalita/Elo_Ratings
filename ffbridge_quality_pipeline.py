"""Reusable historical FFBridge board-quality pipeline.

The module is deliberately free of import-time I/O.  Callers must explicitly
choose an output directory before any artifact is written.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import polars as pl
import requests


SCHEMA_VERSION = 2
DEFAULT_SOURCE_DIR = pathlib.Path(r"E:\bridge\data\ffbridge\data")
DEFAULT_CUTOFF = date.today()
DEFAULT_DISCOVERY_START = date(2026, 1, 1)
BOARD_FILENAME = "ffbridge_quality_boards.parquet"
PLAYER_FILENAME = "ffbridge_quality_players.parquet"
PAIR_FILENAME = "ffbridge_quality_pairs.parquet"
METADATA_FILENAME = "ffbridge_quality_metadata.json"
FRAGMENT_DIRNAME = "session_fragments"
HRS_CACHE_FILENAME = "ffbridge_hand_records_cache.parquet"
_EV_SUMMARY_RE = re.compile(r"^EV_(NS|EW)_[NESW]_[SHDCN]_[1-7]_(V|NV)$")
_EV_PAIR_MAX_RE = re.compile(r"^EV_(NS|EW)_(V|NV)_Max$")
SEATS = ("N", "E", "S", "W")
QUALITY_COLUMNS = (
    "Is_Par_Suit",
    "Is_Sacrifice",
    "Sacrifice_Opportunity",
    "Par_Contract_Score_NS",
    "Par_Contract_Score_EW",
    "DD_Tricks_Diff",
)
QUALITY_BOARD_COLUMNS = (
    "session_id",
    "board_id",
    "Board",
    "group_id",
    "team_id",
    "Date",
    "Pair_Declarer_Direction",
    "Declarer_Direction",
    *(f"Player_ID_{seat}" for seat in SEATS),
    "Pair_ID_NS",
    "Pair_ID_EW",
    *QUALITY_COLUMNS,
)
QUALITY_METRIC_DEFINITIONS = {
    "DD_Tricks_Diff_Avg": {
        "formula": "mean(Tricks - DD_Tricks)",
        "attribution": "declarer only; declaring partnership in pair reports",
    },
    "Par_Contract_Rate_Pct": {
        "formula": "success percentage derived from +1 when directional DD score >= directional par, otherwise -1",
        "attribution": "both partnerships on every board",
    },
    "Par_Suit_Rate_Pct": {
        "formula": "par-strain declarations / all declarations",
        "attribution": "declaring partnership",
    },
    "Sacrifice_Rate_Pct": {
        "formula": "DD score equals negative directional par / negative-par declarations",
        "attribution": "declaring partnership",
    },
    "filter_scope": "same sessions and teams selected by the leaderboard filters",
}


@dataclass(frozen=True)
class SessionAudit:
    session_id: str
    session_date: str
    metadata_path: str
    in_training: bool
    ranking_present: bool
    expected_team_ids: tuple[str, ...]
    present_team_ids: tuple[str, ...]
    missing_team_ids: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return self.ranking_present and not self.missing_team_ids


@dataclass(frozen=True)
class AuditReport:
    source_dir: str
    cutoff: str
    training_session_count: int
    cached_session_count: int
    sessions: tuple[SessionAudit, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["summary"] = {
            "sessions_through_cutoff": len(self.sessions),
            "covered_by_training": sum(s.in_training for s in self.sessions),
            "missing_from_training": sum(not s.in_training for s in self.sessions),
            "complete_raw_sessions": sum(s.complete for s in self.sessions),
            "missing_rankings": sum(not s.ranking_present for s in self.sessions),
            "missing_team_score_files": sum(len(s.missing_team_ids) for s in self.sessions),
        }
        return result


class NoQualityRowsError(ValueError):
    """The upstream session publishes no rows that can produce quality metrics."""


class LancelotDDMismatchError(ValueError):
    """Lancelot's embedded DD table does not match ddss on the audit sample."""


def _read_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}: {exc}") from exc


def _atomic_write_json(
    value: Any,
    path: pathlib.Path,
    *,
    skip_if_exists: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        for attempt in range(10):
            if skip_if_exists and path.is_file():
                return
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.1 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def _fragment_schema_is_current(path: pathlib.Path) -> bool:
    """Reuse a session fragment only when it already has schema-v2 columns."""
    return set(QUALITY_BOARD_COLUMNS).issubset(pl.read_parquet_schema(path))


def _atomic_write_parquet(frame: pl.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        frame.write_parquet(temporary)
        for attempt in range(10):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.1 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def _clean_identifier(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    normalized = str(value).strip()
    if not normalized or normalized.lower() in {"none", "null"}:
        return None
    if normalized.endswith(".0") and normalized[:-2].isdigit():
        normalized = normalized[:-2]
    if normalized.isdigit():
        normalized = normalized.lstrip("0") or "0"
    return normalized


def stable_pair_id(player_a: Any, player_b: Any) -> str | None:
    """Return the current FFBridge Elo pair key (lexical IDs joined by ``_``)."""
    first = _clean_identifier(player_a)
    second = _clean_identifier(player_b)
    if first is None or second is None:
        return None
    return "_".join(sorted((first, second)))


def _session_date(metadata: Mapping[str, Any]) -> date:
    candidates: list[Any] = [metadata.get("date"), metadata.get("startDate")]
    candidates.extend(
        group_session.get("date")
        for group_session in metadata.get("groupSessions") or []
        if isinstance(group_session, Mapping)
    )
    for candidate in candidates:
        if candidate in (None, ""):
            continue
        try:
            return datetime.fromisoformat(str(candidate).replace("Z", "+00:00")).date()
        except ValueError:
            continue
    raise ValueError(f"Session {metadata.get('id')!r} has no parseable date")


def load_session_metadata(
    source_dir: pathlib.Path = DEFAULT_SOURCE_DIR,
    cutoff: date = DEFAULT_CUTOFF,
) -> dict[str, tuple[date, pathlib.Path, dict[str, Any]]]:
    sessions_dir = pathlib.Path(source_dir) / "competitions" / "sessions"
    if not sessions_dir.is_dir():
        raise FileNotFoundError(f"Session metadata directory not found: {sessions_dir}")
    sessions: dict[str, tuple[date, pathlib.Path, dict[str, Any]]] = {}
    for path in sorted(sessions_dir.glob("*.json")):
        payload = _read_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Session metadata must be an object: {path}")
        session_id = _clean_identifier(payload.get("id") or path.stem)
        if session_id is None:
            raise ValueError(f"Session metadata has no ID: {path}")
        session_date = _session_date(payload)
        if session_date <= cutoff:
            if session_id in sessions:
                raise ValueError(f"Duplicate session metadata for {session_id}")
            sessions[session_id] = (session_date, path, payload)
    return sessions


def discover_session_metadata(
    source_dir: pathlib.Path = DEFAULT_SOURCE_DIR,
    *,
    start_date: date = DEFAULT_DISCOVERY_START,
    cutoff: date | None = None,
    timeout: float = 30.0,
    delay: float = 0.1,
) -> int:
    """Discover Lancelot sessions in a date window and cache their metadata."""
    if cutoff is None:
        cutoff = date.today()
    if start_date > cutoff:
        raise ValueError(
            f"Discovery start {start_date.isoformat()} is after cutoff "
            f"{cutoff.isoformat()}"
        )

    ffbridge = _import_ffbridge_lib()
    sessions_dir = pathlib.Path(source_dir) / "competitions" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    writes = 0
    seen_ids: set[str] = set()

    with requests.Session() as http:
        for lancelot_series_id in sorted(ffbridge.LANCELOT_TO_MIGRATION):
            page = 1
            while True:
                payload = ffbridge.get_simultaneous_sessions_page(
                    lancelot_series_id,
                    page=page,
                    per_page=80,
                    timeout=timeout,
                    rate_limit_delay=delay,
                    session=http,
                )
                if not isinstance(payload, Mapping):
                    raise ValueError(
                        f"Session discovery returned non-object for series "
                        f"{lancelot_series_id}, page {page}"
                    )
                items = payload.get("items")
                if not isinstance(items, list):
                    raise ValueError(
                        f"Session discovery lacks items for series "
                        f"{lancelot_series_id}, page {page}"
                    )
                for item in items:
                    if not isinstance(item, Mapping):
                        raise ValueError(
                            f"Malformed session metadata for series "
                            f"{lancelot_series_id}, page {page}"
                        )
                    session_id = _clean_identifier(item.get("id"))
                    if session_id is None or session_id in seen_ids:
                        continue
                    seen_ids.add(session_id)
                    session_day = _session_date(item)
                    if not start_date <= session_day <= cutoff:
                        continue
                    path = sessions_dir / f"{session_id}.json"
                    if path.is_file():
                        continue
                    metadata = dict(item)
                    metadata["series_id"] = ffbridge.LANCELOT_TO_MIGRATION[
                        lancelot_series_id
                    ]
                    metadata["lancelot_series_id"] = lancelot_series_id
                    _atomic_write_json(metadata, path, skip_if_exists=True)
                    writes += 1

                pagination = payload.get("pagination") or {}
                if not isinstance(pagination, Mapping):
                    raise ValueError(
                        f"Malformed pagination for series {lancelot_series_id}, "
                        f"page {page}"
                    )
                if not pagination.get("has_next_page", False):
                    break
                page += 1
    return writes


def training_session_ids(training_parquet: pathlib.Path) -> set[str]:
    path = pathlib.Path(training_parquet)
    if not path.is_file():
        raise FileNotFoundError(f"Training parquet not found: {path}")
    schema = pl.scan_parquet(path).collect_schema()
    if "session_id" not in schema:
        raise ValueError(f"Training parquet lacks session_id: {path}")
    values = (
        pl.scan_parquet(path)
        .select(pl.col("session_id").cast(pl.String).unique())
        .collect()["session_id"]
        .to_list()
    )
    return {value for raw in values if (value := _clean_identifier(raw)) is not None}


def _ranking_team_ids(ranking: Any, path: pathlib.Path) -> tuple[str, ...]:
    if not isinstance(ranking, list):
        raise ValueError(f"Ranking must be a JSON list: {path}")
    ids: set[str] = set()
    ns_ids: set[str] = set()
    for row in ranking:
        if not isinstance(row, dict):
            raise ValueError(f"Ranking rows must be objects: {path}")
        team = row.get("team")
        team_id = _clean_identifier(team.get("id") if isinstance(team, dict) else None)
        if team_id is not None:
            ids.add(team_id)
            orientation = str(
                row.get("orientation")
                or (team.get("orientation") if isinstance(team, dict) else "")
                or ""
            ).upper()
            if orientation == "NS":
                ns_ids.add(team_id)
    # A team-score response contains the full four-seat lineup, so one endpoint
    # per table is sufficient. In ordinary simultaneous pairs sessions the NS
    # ranking rows provide exactly that covering set.
    return tuple(sorted(ns_ids or ids))


def audit_historical_cache(
    source_dir: pathlib.Path = DEFAULT_SOURCE_DIR,
    training_parquet: pathlib.Path | None = None,
    cutoff: date = DEFAULT_CUTOFF,
) -> AuditReport:
    source_dir = pathlib.Path(source_dir)
    training_path = training_parquet or source_dir / "ffbridge_training_data_df.parquet"
    covered = training_session_ids(training_path)
    metadata = load_session_metadata(source_dir, cutoff)
    audited: list[SessionAudit] = []
    for session_id, (session_date, metadata_path, _) in metadata.items():
        ranking_path = source_dir / "results" / "sessions" / session_id / "ranking.json"
        ranking_present = ranking_path.is_file()
        team_ids = _ranking_team_ids(_read_json(ranking_path), ranking_path) if ranking_present else ()
        present = tuple(
            team_id
            for team_id in team_ids
            if (
                source_dir
                / "results"
                / "teams"
                / team_id
                / "session"
                / session_id
                / "scores.json"
            ).is_file()
        )
        missing = tuple(sorted(set(team_ids) - set(present)))
        audited.append(
            SessionAudit(
                session_id=session_id,
                session_date=session_date.isoformat(),
                metadata_path=str(metadata_path),
                in_training=session_id in covered,
                ranking_present=ranking_present,
                expected_team_ids=team_ids,
                present_team_ids=present,
                missing_team_ids=missing,
            )
        )
    audited.sort(key=lambda item: (item.session_date, item.session_id))
    return AuditReport(
        source_dir=str(source_dir),
        cutoff=cutoff.isoformat(),
        training_session_count=len(covered & set(metadata)),
        cached_session_count=len(audited),
        sessions=tuple(audited),
    )


def _import_ffbridge_lib() -> Any:
    root = pathlib.Path(__file__).resolve().parent
    mlbridge = next(
        (path for path in (root / "mlBridge", root.parent / "mlBridge") if path.is_dir()),
        None,
    )
    if mlbridge is None:
        raise FileNotFoundError("mlBridge not found at ./mlBridge or ../mlBridge")
    if str(mlbridge.parent) not in sys.path:
        sys.path.insert(0, str(mlbridge.parent))
    from mlBridge import mlBridgeFFLib  # type: ignore

    return mlBridgeFFLib


def fetch_missing_artifacts(
    report: AuditReport,
    *,
    timeout: float = 60.0,
    delay: float = 0.1,
    max_attempts: int = 6,
    workers: int = 8,
) -> int:
    """Fetch only audit-reported missing files and write them to the raw cache."""
    ffbridge = _import_ffbridge_lib()
    source_dir = pathlib.Path(report.source_dir)
    writes = 0
    pending_sessions = [
        session
        for session in report.sessions
        if not session.ranking_present or session.missing_team_ids
    ]
    iterator: Iterable[SessionAudit] = pending_sessions
    if pending_sessions:
        from tqdm import tqdm

        iterator = tqdm(pending_sessions, desc="Fetching missing FFBridge artifacts")

    def fetch(call: Any, *args: Any, **kwargs: Any) -> Any:
        for attempt in range(1, max_attempts + 1):
            try:
                return call(*args, **kwargs)
            except requests.HTTPError as exc:
                status = getattr(exc.response, "status_code", None)
                if status == 404 or attempt == max_attempts:
                    raise
                time.sleep(min(30.0, float(2 ** (attempt - 1))))
            except requests.RequestException:
                if attempt == max_attempts:
                    raise
                time.sleep(min(30.0, float(2 ** (attempt - 1))))
        raise AssertionError("unreachable")

    with requests.Session() as http:
        for session in iterator:
            ranking_path = (
                source_dir / "results" / "sessions" / session.session_id / "ranking.json"
            )
            if not session.ranking_present:
                try:
                    ranking = fetch(
                        ffbridge.get_session_ranking,
                        int(session.session_id),
                        timeout=timeout,
                        rate_limit_delay=delay,
                        session=http,
                    )
                except requests.HTTPError as exc:
                    if getattr(exc.response, "status_code", None) == 404:
                        continue
                    raise
                except requests.RequestException as exc:
                    print(
                        f"[quality-builder] skip ranking session {session.session_id}: {exc}",
                        flush=True,
                    )
                    continue
                if not isinstance(ranking, list):
                    raise ValueError(
                        f"Ranking fetch returned non-list for {session.session_id}"
                    )
                _atomic_write_json(ranking, ranking_path, skip_if_exists=True)
                writes += 1
                team_ids = _ranking_team_ids(ranking, ranking_path)
            else:
                team_ids = session.expected_team_ids
            missing_team_ids = []
            for team_id in team_ids:
                score_path = (
                    source_dir
                    / "results"
                    / "teams"
                    / team_id
                    / "session"
                    / session.session_id
                    / "scores.json"
                )
                if score_path.is_file():
                    continue
                missing_team_ids.append(team_id)

            def fetch_team_scores(team_id: str) -> int:
                score_path = (
                    source_dir
                    / "results"
                    / "teams"
                    / team_id
                    / "session"
                    / session.session_id
                    / "scores.json"
                )
                try:
                    scores = fetch(
                        ffbridge.get_team_session_scores,
                        int(team_id),
                        int(session.session_id),
                        timeout=timeout,
                        rate_limit_delay=delay,
                    )
                except requests.HTTPError as exc:
                    if getattr(exc.response, "status_code", None) == 404:
                        return 0
                    raise
                except requests.RequestException as exc:
                    print(
                        f"[quality-builder] skip team {team_id} session {session.session_id}: {exc}",
                        flush=True,
                    )
                    return 0
                if not isinstance(scores, list):
                    raise ValueError(
                        f"Scores fetch returned non-list for session={session.session_id}, "
                        f"team={team_id}"
                    )
                _atomic_write_json(scores, score_path, skip_if_exists=True)
                return 1

            if missing_team_ids:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    writes += sum(executor.map(fetch_team_scores, missing_team_ids))
    return writes


def _require_columns(frame: pl.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{context} lacks required columns: {missing}")


def _clean_id_expr(column: str) -> pl.Expr:
    value = pl.col(column).cast(pl.String, strict=False).str.strip_chars()
    without_decimal = value.str.replace(r"\.0$", "")
    without_zeroes = without_decimal.str.strip_chars_start("0")
    normalized = (
        pl.when(without_decimal.str.contains(r"^\d+$") & (without_zeroes == ""))
        .then(pl.lit("0"))
        .otherwise(without_zeroes)
    )
    return (
        pl.when(value.is_null() | value.is_in(["", "None", "null", "NaN"]))
        .then(None)
        .otherwise(normalized)
    )


def _pair_expr(column_a: str, column_b: str) -> pl.Expr:
    first = _clean_id_expr(column_a)
    second = _clean_id_expr(column_b)
    return (
        pl.when(first.is_null() | second.is_null())
        .then(None)
        .otherwise(
            pl.when(first <= second)
            .then(pl.concat_str([first, second], separator="_"))
            .otherwise(pl.concat_str([second, first], separator="_"))
        )
    )


def validate_unique_board_plays(frame: pl.DataFrame) -> None:
    """Reject repeated table-board plays; IDs and all four seats define a play."""
    key = ["session_id", "board_id", *(f"Player_ID_{seat}" for seat in SEATS)]
    _require_columns(frame, key, "Board-quality frame")
    duplicates = (
        frame.group_by(key)
        .len()
        .filter(pl.col("len") > 1)
        .sort("len", descending=True)
    )
    if duplicates.height:
        sample = duplicates.head(3).to_dicts()
        raise ValueError(
            f"Duplicate board-play rows for key {key}; "
            f"{duplicates.height} duplicate key(s), sample={sample}"
        )


def deduplicate_board_plays(frame: pl.DataFrame) -> pl.DataFrame:
    """Collapse repeated endpoint copies, rejecting conflicting quality values."""
    key = ["session_id", "board_id", *(f"Player_ID_{seat}" for seat in SEATS)]
    _require_columns(frame, [*key, *QUALITY_COLUMNS], "Board-quality frame")
    duplicate_keys = frame.group_by(key).len().filter(pl.col("len") > 1)
    if duplicate_keys.is_empty():
        return frame

    conflicts = (
        frame.join(duplicate_keys.select(key), on=key, how="inner")
        .group_by(key)
        .agg(
            *[
                pl.col(column).drop_nulls().n_unique().alias(column)
                for column in QUALITY_COLUMNS
            ]
        )
        .filter(
            pl.any_horizontal(
                [pl.col(column) > 1 for column in QUALITY_COLUMNS]
            )
        )
    )
    if conflicts.height:
        raise NoQualityRowsError(
            "Conflicting duplicate board-play quality values; "
            f"{conflicts.height} key(s), sample={conflicts.head(3).to_dicts()}"
        )
    return frame.unique(subset=key, keep="first", maintain_order=True)


def _dynamic_dd_score_expr(frame: pl.DataFrame) -> pl.Expr:
    expressions: list[pl.Expr] = []
    for level in range(1, 8):
        for suit in "SHDCN":
            for direction in SEATS:
                columns = (
                    f"DDScore_{level}{suit}_{direction}",
                    f"DD_Score_{level}{suit}_{direction}",
                )
                column = next((name for name in columns if name in frame.columns), None)
                if column is not None:
                    expressions.append(
                        pl.when(
                            (pl.col("BidLvl").cast(pl.String) == str(level))
                            & (pl.col("BidSuit") == suit)
                            & (pl.col("Declarer_Direction") == direction)
                        ).then(pl.col(column).cast(pl.Int32, strict=False))
                    )
    if not expressions:
        raise ValueError("No DDScore_{level}{suit}_{direction} columns are available")
    return pl.coalesce(expressions)


def _extract_par_strains(value: Any) -> list[str]:
    """Return canonical strains from training strings or augmented structs."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                return _extract_par_strains(json.loads(text))
            except json.JSONDecodeError:
                pass
        values: list[Any] = [part for part in text.split(",") if part.strip()]
    elif isinstance(value, Mapping):
        values = [value]
    elif isinstance(value, pl.Series):
        values = value.to_list()
    elif isinstance(value, Sequence):
        values = list(value)
    else:
        values = [value]

    strains: set[str] = set()
    for item in values:
        if isinstance(item, Mapping):
            strain = str(item.get("Strain") or item.get("strain") or "").upper()
            if strain in {"C", "D", "H", "S", "N"}:
                strains.add(strain)
            continue
        contract = str(item).strip().upper()
        for character in contract:
            if character.isdigit() or character.isspace():
                continue
            if character in {"C", "D", "H", "S", "N"}:
                strains.add(character)
            break
    return sorted(strains)


def _par_strains_expr(frame: pl.DataFrame) -> pl.Expr:
    column = next(
        (name for name in ("ParContracts", "ParContract") if name in frame.columns),
        None,
    )
    if column is None:
        raise ValueError("ParContracts or ParContract is required for Par Suit")
    return pl.col(column).map_elements(
        _extract_par_strains,
        return_dtype=pl.List(pl.String),
    )


def _par_suit_hit(value: Mapping[str, Any]) -> bool:
    suit = str(value.get("BidSuit") or "").upper()
    return suit in (value.get("_Par_Strains") or [])


def _directional_par_contract_score(
    dd_column: str,
    par_column: str,
) -> pl.Expr:
    dd_score = pl.col(dd_column).cast(pl.Int32, strict=False)
    par_score = pl.col(par_column).cast(pl.Int32, strict=False)
    return (
        pl.when(dd_score.is_null() | par_score.is_null())
        .then(None)
        .when(dd_score >= par_score)
        .then(pl.lit(1, dtype=pl.Int8))
        .otherwise(pl.lit(-1, dtype=pl.Int8))
    )


def normalize_quality_frame(
    frame: pl.DataFrame,
    *,
    session_dates: pl.DataFrame | None = None,
    reject_duplicates: bool = True,
) -> pl.DataFrame:
    """Normalize an augmented FFBridge frame to the slim quality schema."""
    renames = {}
    if "DDTricks" in frame.columns and "DD_Tricks" not in frame.columns:
        renames["DDTricks"] = "DD_Tricks"
    if "DDTricks_Diff" in frame.columns and "DD_Tricks_Diff" not in frame.columns:
        renames["DDTricks_Diff"] = "DD_Tricks_Diff"
    if "Par_NS" in frame.columns and "ParScore_NS" not in frame.columns:
        renames["Par_NS"] = "ParScore_NS"
    if "Par_EW" in frame.columns and "ParScore_EW" not in frame.columns:
        renames["Par_EW"] = "ParScore_EW"
    if (
        "Declarer_Pair_Direction" in frame.columns
        and "Pair_Declarer_Direction" not in frame.columns
    ):
        renames["Declarer_Pair_Direction"] = "Pair_Declarer_Direction"
    out = frame.rename(renames)
    _require_columns(
        out,
        [
            "session_id",
            "board_id",
            "Board",
            "ParScore_NS",
            "ParScore_EW",
            "Pair_Declarer_Direction",
            "Declarer_Direction",
            "BidLvl",
            "BidSuit",
            "DD_Tricks_Diff",
            *(f"Player_ID_{seat}" for seat in SEATS),
        ],
        "Augmented FFBridge frame",
    )
    out = out.with_columns(
        *[
            _clean_id_expr(f"Player_ID_{seat}").alias(f"Player_ID_{seat}")
            for seat in SEATS
        ],
        pl.col("session_id").cast(pl.String).alias("session_id"),
        pl.col("board_id").cast(pl.String).alias("board_id"),
        pl.col("Board").cast(pl.Int32, strict=False),
        pl.col("DD_Tricks_Diff").cast(pl.Int8, strict=False),
        pl.when(pl.col("Pair_Declarer_Direction") == "NS")
        .then(pl.col("ParScore_NS"))
        .when(pl.col("Pair_Declarer_Direction") == "EW")
        .then(pl.col("ParScore_EW"))
        .otherwise(None)
        .cast(pl.Int32, strict=False)
        .alias("Par_Declarer"),
        _dynamic_dd_score_expr(out).alias("_DD_Score_Declarer"),
        _par_strains_expr(out).alias("_Par_Strains"),
    ).with_columns(
        pl.when(pl.col("Pair_Declarer_Direction") == "NS")
        .then(pl.col("_DD_Score_Declarer"))
        .when(pl.col("Pair_Declarer_Direction") == "EW")
        .then(-pl.col("_DD_Score_Declarer"))
        .otherwise(None)
        .cast(pl.Int32, strict=False)
        .alias("_DD_Score_NS"),
        pl.struct(["BidSuit", "_Par_Strains"]).map_elements(
            _par_suit_hit,
            return_dtype=pl.Boolean,
        ).alias("Is_Par_Suit"),
    ).with_columns(
        (-pl.col("_DD_Score_NS")).cast(pl.Int32).alias("_DD_Score_EW"),
    ).with_columns(
        _directional_par_contract_score(
            "_DD_Score_NS", "ParScore_NS"
        ).alias("Par_Contract_Score_NS"),
        _directional_par_contract_score(
            "_DD_Score_EW", "ParScore_EW"
        ).alias("Par_Contract_Score_EW"),
        (
            pl.col("Par_Declarer").is_not_null()
            & (pl.col("Par_Declarer") < 0)
        ).alias("Sacrifice_Opportunity"),
        (
            (pl.col("Par_Declarer") == pl.col("_DD_Score_Declarer"))
            & (pl.col("Par_Declarer") < 0)
        ).alias("Is_Sacrifice"),
        _pair_expr("Player_ID_N", "Player_ID_S").alias("Pair_ID_NS"),
        _pair_expr("Player_ID_E", "Player_ID_W").alias("Pair_ID_EW"),
    )
    if session_dates is not None:
        _require_columns(session_dates, ["session_id", "Date"], "Session dates")
        dates = session_dates.select(
            pl.col("session_id").cast(pl.String),
            pl.col("Date").cast(pl.Date),
        ).unique("session_id")
        if "Date" in out.columns:
            out = out.drop("Date")
        out = out.join(dates, on="session_id", how="left", validate="m:1")
    elif "Date" in out.columns:
        out = out.with_columns(pl.col("Date").cast(pl.Date, strict=False))
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Date).alias("Date"))

    optional = ("group_id", "team_id")
    for column in optional:
        if column not in out.columns:
            out = out.with_columns(pl.lit(None, dtype=pl.String).alias(column))
        else:
            out = out.with_columns(pl.col(column).cast(pl.String, strict=False))
    selected = out.select(*QUALITY_BOARD_COLUMNS)
    identity_columns = [f"Player_ID_{seat}" for seat in SEATS]
    selected = selected.filter(
        pl.any_horizontal(
            [pl.col(column).is_not_null() for column in identity_columns]
        )
    )
    if selected.is_empty():
        raise NoQualityRowsError(
            "Session has no board rows with a mapped player identity"
        )
    if selected["Date"].null_count():
        missing_sessions = selected.filter(pl.col("Date").is_null())[
            "session_id"
        ].unique().head(10).to_list()
        raise ValueError(f"Missing metadata dates for sessions: {missing_sessions}")
    if reject_duplicates:
        selected = deduplicate_board_plays(selected)
        validate_unique_board_plays(selected)
    return selected.sort(["Date", "session_id", "Board", "board_id"])


def session_dates_frame(
    metadata: Mapping[str, tuple[date, pathlib.Path, dict[str, Any]]],
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "session_id": list(metadata),
            "Date": [metadata[session_id][0] for session_id in metadata],
        },
        schema={"session_id": pl.String, "Date": pl.Date},
    )


def normalize_training_parquet(
    training_parquet: pathlib.Path,
    session_dates: pl.DataFrame,
    identity_map: Mapping[str, str] | None = None,
) -> pl.DataFrame:
    scan = pl.scan_parquet(training_parquet)
    schema = scan.collect_schema()
    essential = {
        "session_id",
        "board_id",
        "Board",
        "group_id",
        "team_id",
        "Date",
        "ParScore_NS",
        "ParScore_EW",
        "Pair_Declarer_Direction",
        "Declarer_Direction",
        "BidLvl",
        "BidSuit",
        "ParContract",
        "ParContracts",
        "DDTricks",
        "DDTricks_Diff",
        "DD_Tricks",
        "DD_Tricks_Diff",
        *(f"Player_ID_{seat}" for seat in SEATS),
    }
    selected = [
        column
        for column in schema
        if column in essential
        or column.startswith("DDScore_")
        or column.startswith("DD_Score_")
    ]
    included_sessions = session_dates["session_id"].cast(pl.String).to_list()
    normalized = normalize_quality_frame(
        scan.filter(pl.col("session_id").cast(pl.String).is_in(included_sessions))
        .select(selected)
        .collect(),
        session_dates=session_dates,
    )
    return apply_identity_map(normalized, identity_map or {})


def load_identity_map(report: AuditReport) -> dict[str, str]:
    """Load Lancelot-to-Elo aliases from every cached ranking in the audit."""
    source_dir = pathlib.Path(report.source_dir)
    aliases: dict[str, str] = {}
    for session in report.sessions:
        if not session.ranking_present:
            continue
        ranking_path = (
            source_dir
            / "results"
            / "sessions"
            / session.session_id
            / "ranking.json"
        )
        ranking = _read_json(ranking_path)
        if not isinstance(ranking, list):
            raise ValueError(f"Ranking must be a list: {ranking_path}")
        for lancelot_id, stable_id in ranking_identity_map(ranking).items():
            previous = aliases.get(lancelot_id)
            if previous is not None and previous != stable_id:
                if previous == lancelot_id:
                    aliases[lancelot_id] = stable_id
                    continue
                if stable_id == lancelot_id:
                    continue
                raise ValueError(
                    f"Lancelot ID {lancelot_id} maps to both {previous} and "
                    f"{stable_id}"
                )
            aliases[lancelot_id] = stable_id
    return aliases


def apply_identity_map(
    board_quality: pl.DataFrame,
    identity_map: Mapping[str, str],
) -> pl.DataFrame:
    """Convert historical Lancelot seat IDs to current Elo stable IDs."""
    if not identity_map:
        return board_quality
    remapped = board_quality.with_columns(
        *[
            pl.col(f"Player_ID_{seat}")
            .replace_strict(identity_map, default=pl.col(f"Player_ID_{seat}"))
            .alias(f"Player_ID_{seat}")
            for seat in SEATS
        ]
    )
    return remapped.with_columns(
        _pair_expr("Player_ID_N", "Player_ID_S").alias("Pair_ID_NS"),
        _pair_expr("Player_ID_E", "Player_ID_W").alias("Pair_ID_EW"),
    )


def _rank_desc(column: str, alias: str) -> pl.Expr:
    return pl.col(column).rank(method="min", descending=True).cast(pl.Int32).alias(alias)


def _quality_aggregates(frame: pl.DataFrame, id_column: str) -> pl.DataFrame:
    return (
        frame.group_by(id_column)
        .agg(
            pl.len().cast(pl.UInt32).alias("Board_Rows"),
            pl.col("session_id").n_unique().cast(pl.UInt32).alias("Sessions"),
            pl.col("_par_suit_hit").cast(pl.Float64).mean().alias("par_suit_rate"),
            pl.col("_par_contract_score").cast(pl.Float64).mean().alias(
                "par_contract_rate"
            ),
            pl.col("_sacrifice_hit").cast(pl.Float64).mean().alias(
                "sacrifice_rate"
            ),
            pl.col("_dd_tricks_diff").cast(pl.Float64).mean().alias(
                "dd_tricks_diff_avg"
            ),
        )
        .with_columns(
            _rank_desc("par_suit_rate", "Par_Suit_Rank"),
            _rank_desc("par_contract_rate", "Par_Contract_Rank"),
            _rank_desc("sacrifice_rate", "Sacrifice_Rank"),
            _rank_desc("dd_tricks_diff_avg", "DD_Tricks_Diff_Rank"),
        )
        .sort(id_column)
    )


def build_player_sidecar(board_quality: pl.DataFrame) -> pl.DataFrame:
    appearances = pl.concat(
        [
            board_quality.select(
                pl.col(f"Player_ID_{seat}").alias("player_id"),
                "session_id",
                pl.when(pl.col("Pair_Declarer_Direction") == ("NS" if seat in "NS" else "EW"))
                .then(pl.col("Is_Par_Suit"))
                .otherwise(None)
                .alias("_par_suit_hit"),
                pl.col(
                    "Par_Contract_Score_NS" if seat in "NS" else "Par_Contract_Score_EW"
                ).alias("_par_contract_score"),
                pl.when(
                    (pl.col("Pair_Declarer_Direction") == ("NS" if seat in "NS" else "EW"))
                    & pl.col("Sacrifice_Opportunity")
                )
                .then(pl.col("Is_Sacrifice"))
                .otherwise(None)
                .alias("_sacrifice_hit"),
                pl.when(pl.col("Declarer_Direction") == seat)
                .then(pl.col("DD_Tricks_Diff"))
                .otherwise(None)
                .alias("_dd_tricks_diff"),
            )
            for seat in SEATS
        ],
        how="vertical",
    ).filter(pl.col("player_id").is_not_null())
    return _quality_aggregates(appearances, "player_id")


def build_pair_sidecar(board_quality: pl.DataFrame) -> pl.DataFrame:
    appearances = pl.concat(
        [
            board_quality.select(
                pl.col(pair_column).alias("pair_id"),
                "session_id",
                pl.when(pl.col("Pair_Declarer_Direction") == side)
                .then(pl.col("Is_Par_Suit"))
                .otherwise(None)
                .alias("_par_suit_hit"),
                pl.col(f"Par_Contract_Score_{side}").alias(
                    "_par_contract_score"
                ),
                pl.when(
                    (pl.col("Pair_Declarer_Direction") == side)
                    & pl.col("Sacrifice_Opportunity")
                )
                .then(pl.col("Is_Sacrifice"))
                .otherwise(None)
                .alias("_sacrifice_hit"),
                pl.when(pl.col("Pair_Declarer_Direction") == side)
                .then(pl.col("DD_Tricks_Diff"))
                .otherwise(None)
                .alias("_dd_tricks_diff"),
            )
            for pair_column, side in (("Pair_ID_NS", "NS"), ("Pair_ID_EW", "EW"))
        ],
        how="vertical",
    ).filter(pl.col("pair_id").is_not_null())
    return _quality_aggregates(appearances, "pair_id")


def ranking_identity_map(ranking: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Map Lancelot person IDs to the migration IDs used by current Elo."""
    mapping: dict[str, str] = {}
    for row in ranking:
        team = row.get("team")
        if not isinstance(team, Mapping):
            continue
        for player_key in ("player1", "player2", "player3", "player4"):
            player = team.get(player_key)
            if not isinstance(player, Mapping):
                continue
            lancelot_id = _clean_identifier(player.get("id"))
            if lancelot_id is None:
                continue
            migration_id = _clean_identifier(player.get("migrationId"))
            stable_id = migration_id or lancelot_id
            previous = mapping.get(lancelot_id)
            if previous is not None and previous != stable_id:
                if previous == lancelot_id:
                    mapping[lancelot_id] = stable_id
                    continue
                if stable_id == lancelot_id:
                    continue
                raise ValueError(
                    f"Lancelot ID {lancelot_id} maps to both {previous} and {stable_id}"
                )
            mapping[lancelot_id] = stable_id
    return mapping


def _historical_player_id(
    player: Mapping[str, Any] | None,
    identity_map: Mapping[str, str],
) -> str | None:
    if not isinstance(player, Mapping):
        return None
    migration_id = _clean_identifier(player.get("migrationId"))
    if migration_id is not None:
        return migration_id
    lancelot_id = _clean_identifier(player.get("id"))
    return identity_map.get(lancelot_id, lancelot_id) if lancelot_id is not None else None


def _quality_safe_frequencies(value: Any) -> list[dict[str, Any]]:
    """Remove ambiguous dual score strings while preserving notes/counts."""
    if not isinstance(value, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        row = dict(item)
        ns_score = str(row.get("nsScore") or "").strip()
        ew_score = str(row.get("ewScore") or "").strip()
        if ns_score and ew_score:
            row["nsScore"] = ""
            row["ewScore"] = ""
        cleaned.append(row)
    return cleaned


def flatten_team_scores(
    session_id: str,
    scores: Sequence[Mapping[str, Any]],
    identity_map: Mapping[str, str],
) -> tuple[pl.DataFrame, int]:
    """Flatten score JSON using only JSON primitives and Polars construction."""
    rows: list[dict[str, Any]] = []
    unmapped = 0
    seen_play_ids: dict[str, dict[str, Any]] = {}
    seat_person = _import_ffbridge_lib().lancelot_seat_person
    for score in scores:
        board = score.get("board")
        lineup = score.get("lineup")
        if not isinstance(board, Mapping) or not isinstance(lineup, Mapping):
            raise ValueError(f"Malformed score row in session {session_id}")
        play_id = _clean_identifier(score.get("id"))
        board_id = _clean_identifier(board.get("id"))
        if play_id is None or board_id is None:
            raise ValueError(f"Score row lacks id/board.id in session {session_id}")
        players = {
            seat: _historical_player_id(
                lineup.get(
                    {"N": "northPlayer", "E": "eastPlayer", "S": "southPlayer", "W": "westPlayer"}[
                        seat
                    ]
                ),
                identity_map,
            )
            for seat in SEATS
        }
        row: dict[str, Any] = {
            "play_id": play_id,
            "session_id": session_id,
            "board_id": board_id,
            "Board": score.get("boardNumber") or board.get("boardNumber"),
            "Contract": score.get("contract"),
            "Declarer_Direction": str(score.get("declarer") or "").replace("O", "W") or None,
            "Result": score.get("result"),
            "PBN": board.get("deal"),
            "group_id": score.get("groupId") or board.get("groupId"),
            **{f"Player_ID_{seat}": players[seat] for seat in SEATS},
        }
        game = ((lineup.get("segment") or {}).get("game") or {})
        home_team = game.get("homeTeam") or {}
        away_team = game.get("awayTeam") or {}
        row.update(
            {
                "team_id": home_team.get("id"),
                "boardNumber": row["Board"],
                "board_frequencies": _quality_safe_frequencies(
                    board.get("frequencies")
                ),
                "board_deal": board.get("deal"),
                "contract": score.get("contract"),
                "declarer": score.get("declarer"),
                "result": score.get("result"),
                "nsScore": score.get("nsScore"),
                "ewScore": score.get("ewScore"),
                "nsNote": score.get("nsNote"),
                "ewNote": score.get("ewNote"),
                "lineup_segment_game_homeTeam_id": home_team.get("id"),
                "lineup_segment_game_homeTeam_section": home_team.get("section"),
                "lineup_segment_game_homeTeam_orientation": home_team.get("orientation"),
                "lineup_segment_game_homeTeam_startTableNumber": home_team.get(
                    "startTableNumber"
                ),
                "lineup_segment_game_awayTeam_id": away_team.get("id"),
                "lineup_segment_game_awayTeam_section": away_team.get("section"),
                "lineup_segment_game_awayTeam_orientation": away_team.get("orientation"),
                "lineup_segment_game_awayTeam_startTableNumber": away_team.get(
                    "startTableNumber"
                ),
            }
        )
        for seat, field in {
            "N": "northPlayer",
            "E": "eastPlayer",
            "S": "southPlayer",
            "W": "westPlayer",
        }.items():
            player = seat_person(lineup.get(field)) or {}
            prefix = f"lineup_{field}_"
            row[f"{prefix}id"] = player.get("id")
            row[f"{prefix}firstName"] = player.get("firstName")
            row[f"{prefix}lastName"] = player.get("lastName")
        contract = str(score.get("contract") or "").upper().replace("NT", "N")
        if contract[:1] in "1234567" and len(contract) >= 2:
            row["BidLvl"] = contract[0]
            row["BidSuit"] = contract[1]
            row["Pair_Declarer_Direction"] = (
                "NS" if row["Declarer_Direction"] in {"N", "S"} else "EW"
            )
        else:
            row["BidLvl"] = None
            row["BidSuit"] = None
            row["Pair_Declarer_Direction"] = None
        dds = board.get("dds")
        if isinstance(dds, Mapping):
            for key, value in dds.items():
                if key in {"ParScore_NS", "ParScore_EW"} or str(key).startswith(
                    ("DDScore_", "DD_", "DDTricks")
                ):
                    row[str(key)] = value
            for direction in SEATS:
                tricks = dds.get(direction)
                if not isinstance(tricks, Mapping):
                    continue
                for raw_suit, value in tricks.items():
                    suit = "N" if str(raw_suit).upper() == "NT" else str(raw_suit).upper()
                    if suit in "SHDCN":
                        row[f"DD_{direction}_{suit}"] = value
        prior = seen_play_ids.get(play_id)
        if prior is None:
            seen_play_ids[play_id] = row
            rows.append(row)
            unmapped += sum(value is None for value in players.values())
        elif prior != row:
            raise ValueError(f"Conflicting duplicate play_id {play_id} in session {session_id}")
    if not rows:
        raise NoQualityRowsError(
            f"Raw session {session_id} score files contain no board rows"
        )
    return pl.from_dicts(rows, infer_schema_length=None), unmapped


def load_raw_session(
    source_dir: pathlib.Path,
    session: SessionAudit,
) -> tuple[pl.DataFrame, int]:
    if not session.complete:
        raise FileNotFoundError(
            f"Raw session {session.session_id} is incomplete; run audit or --fetch-missing"
        )
    ranking_path = (
        pathlib.Path(source_dir)
        / "results"
        / "sessions"
        / session.session_id
        / "ranking.json"
    )
    ranking = _read_json(ranking_path)
    if not isinstance(ranking, list):
        raise ValueError(f"Ranking must be a list: {ranking_path}")
    identity_map = ranking_identity_map(ranking)
    all_scores: list[Mapping[str, Any]] = []
    for team_id in session.expected_team_ids:
        path = (
            pathlib.Path(source_dir)
            / "results"
            / "teams"
            / team_id
            / "session"
            / session.session_id
            / "scores.json"
        )
        payload = _read_json(path)
        if not isinstance(payload, list):
            raise ValueError(f"Team scores must be a list: {path}")
        all_scores.extend(payload)
    return flatten_team_scores(session.session_id, all_scores, identity_map)


EMBEDDED_DD_COLUMNS = tuple(f"DD_{seat}_{suit}" for seat in SEATS for suit in "SHDCN")
_HCP_RANKS = {"A": 4, "K": 3, "Q": 2, "J": 1}
_AUGMENT_LOCK = threading.RLock()


def _has_quality_dd_scores(frame: pl.DataFrame) -> bool:
    required = {"ParScore_NS", "ParScore_EW", "DDTricks_Diff"}
    has_dd_scores = any(
        column.startswith(("DDScore_", "DD_Score_")) for column in frame.columns
    )
    return required.issubset(frame.columns) and has_dd_scores


def _has_embedded_dd_table(frame: pl.DataFrame) -> bool:
    if any(column not in frame.columns for column in EMBEDDED_DD_COLUMNS):
        return False
    return bool(
        frame.select(
            pl.all_horizontal(
                [pl.col(column).is_not_null() for column in EMBEDDED_DD_COLUMNS]
            ).any()
        ).item()
    )


def _pbn_audit_key(pbn: str) -> int:
    return int.from_bytes(hashlib.sha1(pbn.encode("utf-8")).digest()[:8], "big")


def _select_lancelot_dd_audit_pbns(pbns: Sequence[str], rate: float) -> list[str]:
    unique = sorted({pbn for pbn in pbns if pbn})
    if rate <= 0 or not unique:
        return []
    sample_size = min(len(unique), max(1, int(round(len(unique) * rate))))
    return sorted(unique, key=_pbn_audit_key)[:sample_size]


def _ddss_columns_from_table(table: Any) -> dict[str, int]:
    rows = table.to_list(player_major=True)
    return {
        f"DD_{seat}_{strain}": int(rows[seat_index][strain_index])
        for seat_index, seat in enumerate(SEATS)
        for strain_index, strain in enumerate("SHDCN")
    }


def _unique_embedded_dd_deals(frame: pl.DataFrame) -> pl.DataFrame:
    _require_columns(frame, ["PBN", *EMBEDDED_DD_COLUMNS], "Lancelot DD audit")
    return (
        frame.select("PBN", *EMBEDDED_DD_COLUMNS)
        .filter(
            pl.col("PBN").is_not_null()
            & pl.all_horizontal(
                [pl.col(column).is_not_null() for column in EMBEDDED_DD_COLUMNS]
            )
        )
        .unique(subset=["PBN"], maintain_order=True)
    )


def _ddss_tables_for_pbns(pbns: Sequence[str]) -> list[Any]:
    _ff_lib, augment_lib = _import_mlbridge()
    from endplay.types import Deal
    from mlBridge.dds_ddss import DDSS_AVAILABLE

    if not DDSS_AVAILABLE:
        raise RuntimeError("ddss DLL is required to audit Lancelot DD tables")
    deals = [Deal(pbn) for pbn in pbns]
    with _AUGMENT_LOCK:
        return augment_lib.solve_dd_for_deals(deals)


def _lancelot_dd_sample_mismatches(
    frame: pl.DataFrame, *, rate: float
) -> tuple[int, int, list[str]]:
    """Return (unique_deals, sample_size, mismatch lines)."""
    if rate <= 0:
        return 0, 0, []
    unique = _unique_embedded_dd_deals(frame)
    sample_pbns = _select_lancelot_dd_audit_pbns(unique["PBN"].to_list(), rate)
    if not sample_pbns:
        return unique.height, 0, []
    sample = unique.filter(pl.col("PBN").is_in(sample_pbns))
    tables = _ddss_tables_for_pbns(sample["PBN"].to_list())
    mismatches: list[str] = []
    for rec, table in zip(sample.iter_rows(named=True), tables):
        expected = _ddss_columns_from_table(table)
        diffs = [
            f"{column} Lancelot={int(rec[column])} ddss={expected[column]}"
            for column in EMBEDDED_DD_COLUMNS
            if int(rec[column]) != expected[column]
        ]
        if diffs:
            mismatches.append(f"{rec['PBN']}: {', '.join(diffs)}")
    return unique.height, len(sample_pbns), mismatches


def _replace_embedded_dd_with_ddss(frame: pl.DataFrame) -> pl.DataFrame:
    """Overwrite every unique deal's Lancelot DD table with ddss."""
    pbns = _unique_embedded_dd_deals(frame)["PBN"].to_list()
    if not pbns:
        return frame
    tables = _ddss_tables_for_pbns(pbns)
    corrections = pl.DataFrame(
        [
            {"PBN": pbn, **_ddss_columns_from_table(table)}
            for pbn, table in zip(pbns, tables)
        ]
    )
    work = frame.drop([column for column in EMBEDDED_DD_COLUMNS if column in frame.columns])
    return work.join(corrections, on="PBN", how="left")


def _apply_lancelot_dd_audit(
    frame: pl.DataFrame, *, rate: float = 1.0
) -> tuple[pl.DataFrame, set[str]]:
    """Replace every unique deal's Lancelot DD table with ddss.

    ``rate <= 0`` skips the replacement (tests). Otherwise every unique PBN is
    solved with ddss; Lancelot is never the source of truth. Mismatched PBNs
    are returned so the hand-records cache can rewrite DD/Par.
    """
    if rate <= 0:
        return frame, set()
    unique = _unique_embedded_dd_deals(frame)
    pbns = unique["PBN"].to_list()
    if not pbns:
        return frame, set()
    tables = _ddss_tables_for_pbns(pbns)
    corrections: list[dict[str, Any]] = []
    mismatch_pbns: set[str] = set()
    mismatch_lines: list[str] = []
    for rec, table in zip(unique.iter_rows(named=True), tables):
        expected = _ddss_columns_from_table(table)
        corrections.append({"PBN": rec["PBN"], **expected})
        diffs = [
            f"{column} Lancelot={int(rec[column])} ddss={expected[column]}"
            for column in EMBEDDED_DD_COLUMNS
            if int(rec[column]) != expected[column]
        ]
        if diffs:
            mismatch_pbns.add(str(rec["PBN"]))
            mismatch_lines.append(f"{rec['PBN']}: {', '.join(diffs)}")
    print(
        f"[ffbridge-quality] replaced {len(pbns)} Lancelot DD tables with ddss; "
        f"{len(mismatch_pbns)} disagreed",
        flush=True,
    )
    if mismatch_lines:
        preview = "\n  ".join(mismatch_lines[:8])
        print(f"[ffbridge-quality] Lancelot DD mismatches:\n  {preview}", flush=True)
    work = frame.drop(
        [column for column in EMBEDDED_DD_COLUMNS if column in frame.columns]
    )
    return work.join(pl.DataFrame(corrections), on="PBN", how="left"), mismatch_pbns


def _audit_lancelot_dd_sample(frame: pl.DataFrame, *, rate: float = 0.1) -> int:
    """Compare a sample of Lancelot DD tables to ddss. Raise on mismatch."""
    unique_count, sample_count, mismatches = _lancelot_dd_sample_mismatches(
        frame, rate=rate
    )
    if sample_count:
        print(
            f"[ffbridge-quality] audited {sample_count}/{unique_count} "
            f"Lancelot DD tables against ddss",
            flush=True,
        )
    if mismatches:
        raise LancelotDDMismatchError(
            f"Lancelot DD mismatch on {len(mismatches)}/{sample_count} "
            f"sampled deals:\n  " + "\n  ".join(mismatches)
        )
    return sample_count


def _needs_lancelot_convert(frame: pl.DataFrame) -> bool:
    return "board_deal" in frame.columns or "board_frequencies" in frame.columns


def _hcp_of_cards(cards: str) -> int:
    return sum(_HCP_RANKS.get(rank, 0) for rank in cards)


def _quick_tricks(cards: str) -> float:
    has_a = "A" in cards
    has_k = "K" in cards
    has_q = "Q" in cards
    if has_a and has_k:
        return 2.0
    if has_a and has_q:
        return 1.5
    if has_a:
        return 1.0
    if has_k and has_q:
        return 1.0
    if has_k:
        return 0.5
    return 0.0


def _distribution_points(length: int) -> int:
    return {0: 3, 1: 2, 2: 1}.get(length, 0)


def _hand_features_from_pbn(pbn: str) -> dict[str, Any] | None:
    body = str(pbn).split(":", 1)[-1].strip()
    hands = body.split()
    if len(hands) != 4:
        return None
    features: dict[str, Any] = {"PBN": pbn}
    hcp_ns = 0
    hcp_ew = 0
    qt_ns = 0.0
    qt_ew = 0.0
    dp_ns = 0
    dp_ew = 0
    for seat, hand in zip(SEATS, hands):
        suits = hand.split(".")
        if len(suits) != 4:
            return None
        # PBN suit order is S.H.D.C
        suit_cards = {"S": suits[0], "H": suits[1], "D": suits[2], "C": suits[3]}
        hcp = _hcp_of_cards(hand)
        qt = sum(_quick_tricks(suit_cards[suit]) for suit in "SHDC")
        dp = sum(_distribution_points(len(suit_cards[suit])) for suit in "SHDC")
        features[f"HCP_{seat}"] = hcp
        features[f"QT_{seat}"] = qt
        features[f"DP_{seat}"] = dp
        if seat == "N":
            for suit, cards in suit_cards.items():
                features[f"SL_N_{suit}"] = len(cards)
                features[f"DP_N_{suit}"] = _distribution_points(len(cards))
        if seat in {"N", "S"}:
            hcp_ns += hcp
            qt_ns += qt
            dp_ns += dp
        else:
            hcp_ew += hcp
            qt_ew += qt
            dp_ew += dp
    features["HCP_NS"] = hcp_ns
    features["HCP_EW"] = hcp_ew
    features["QT_NS"] = qt_ns
    features["QT_EW"] = qt_ew
    features["DP_NS"] = dp_ns
    features["DP_EW"] = dp_ew
    features["SL_N_ML_SJ"] = max(
        features["SL_N_S"], features["SL_N_H"], features["SL_N_D"], features["SL_N_C"]
    )
    return features


def _hand_feature_frame(pbns: Sequence[str]) -> pl.DataFrame:
    rows = []
    seen: set[str] = set()
    for pbn in pbns:
        if not pbn or pbn in seen:
            continue
        seen.add(pbn)
        features = _hand_features_from_pbn(pbn)
        if features is not None:
            rows.append(features)
    if not rows:
        return pl.DataFrame({"PBN": []})
    return pl.DataFrame(rows)


def _dd_tricks_expr() -> pl.Expr:
    expr = pl.lit(None, dtype=pl.Int32)
    for direction in SEATS:
        for suit in "SHDCN":
            expr = (
                pl.when(
                    (pl.col("Declarer_Direction") == direction)
                    & (pl.col("BidSuit") == suit)
                )
                .then(pl.col(f"DD_{direction}_{suit}").cast(pl.Int32, strict=False))
                .otherwise(expr)
            )
    return expr


def _import_mlbridge() -> tuple[Any, Any]:
    here = pathlib.Path(__file__).resolve().parent
    mlbridge_root = next(
        (path for path in (here / "mlBridge", here.parent / "mlBridge") if path.is_dir()),
        None,
    )
    if mlbridge_root is None:
        raise FileNotFoundError("mlBridge not found at ./mlBridge or ../mlBridge")
    if str(mlbridge_root.parent) not in sys.path:
        sys.path.insert(0, str(mlbridge_root.parent))
    from mlBridge import mlBridgeFFLib  # type: ignore
    from mlBridge import mlBridgeAugmentLib  # type: ignore

    return mlBridgeFFLib, mlBridgeAugmentLib


def _par_rows_from_embedded_dd(frame: pl.DataFrame, augment_lib: Any) -> pl.DataFrame:
    """Par from the Lancelot DD table. Does not re-solve the deals."""
    needed = ("PBN", "Dealer", "Vul", *EMBEDDED_DD_COLUMNS)
    _require_columns(frame, needed, "Embedded DD frame")
    unique = (
        frame.select("PBN", "Dealer", "Vul", *EMBEDDED_DD_COLUMNS)
        .filter(
            pl.col("PBN").is_not_null()
            & pl.col("Dealer").is_not_null()
            & pl.col("Vul").is_not_null()
            & pl.all_horizontal(
                [pl.col(column).is_not_null() for column in EMBEDDED_DD_COLUMNS]
            )
        )
        .unique(subset=["PBN", "Dealer", "Vul"], maintain_order=True)
    )
    if unique.is_empty():
        raise NoQualityRowsError("Embedded DD table has no complete PBN/Dealer/Vul rows")
    par_fn = augment_lib.par
    rows: list[dict[str, Any]] = []
    with _AUGMENT_LOCK:
        for rec in unique.iter_rows(named=True):
            table = augment_lib._list_to_ddtable(
                [
                    [int(rec[f"DD_{direction}_{suit}"]) for suit in "SHDCN"]
                    for direction in SEATS
                ]
            )
            parlist = par_fn(
                table,
                augment_lib.VulToEndplayVul_d[rec["Vul"]],
                augment_lib.DealerToEndPlayDealer_d[rec["Dealer"]],
            )
            contracts = [
                {
                    "Level": str(contract.level),
                    "Strain": "SHDCN"[int(contract.denom)],
                    "Doubled": contract.penalty.abbr,
                    "Pair_Direction": (
                        "NS" if contract.declarer.abbr in "NS" else "EW"
                    ),
                    "Result": contract.result,
                }
                for contract in parlist
            ]
            rows.append(
                {
                    "PBN": rec["PBN"],
                    "Dealer": rec["Dealer"],
                    "Vul": rec["Vul"],
                    "ParScore": int(parlist.score),
                    "ParContracts": contracts,
                }
            )
    return pl.DataFrame(rows)


def attach_embedded_dd_metrics(frame: pl.DataFrame) -> pl.DataFrame:
    """Derive quality/Club DD columns from an already-present DD trick table."""
    _require_columns(
        frame,
        [
            "PBN",
            "BidLvl",
            "BidSuit",
            "Declarer_Direction",
            *EMBEDDED_DD_COLUMNS,
        ],
        "Embedded DD frame",
    )
    _, augment_lib = _import_mlbridge()
    out = frame.with_columns(
        pl.col("BidLvl").cast(pl.Int32, strict=False),
        pl.col("BidSuit").cast(pl.String).str.to_uppercase(),
        pl.col("Declarer_Direction")
        .cast(pl.String)
        .str.to_uppercase()
        .replace({"O": "W"}),
    )
    if "Dealer" not in out.columns:
        board = pl.col("Board").cast(pl.Int64, strict=False)
        out = out.with_columns(
            pl.when(board.is_null() | (board <= 0))
            .then(pl.lit(None, dtype=pl.Utf8))
            .otherwise(
                pl.col("Board")
                .cast(pl.Int64, strict=False)
                .sub(1)
                .mod(4)
                .replace_strict({0: "N", 1: "E", 2: "S", 3: "W"})
            )
            .alias("Dealer")
        )
    if "Vul" not in out.columns:
        raise ValueError("Vul is required to derive par from the embedded DD table")
    if "Result" in out.columns:
        result = pl.col("Result").cast(pl.Int32, strict=False)
    else:
        result = pl.lit(0, dtype=pl.Int32)
    out = out.with_columns(
        _dd_tricks_expr().alias("DD_Tricks"),
        (pl.col("BidLvl") + 6 + result).alias("Tricks"),
        pl.when(pl.col("Declarer_Direction").is_in(["N", "S"]))
        .then(pl.lit("NS"))
        .when(pl.col("Declarer_Direction").is_in(["E", "W"]))
        .then(pl.lit("EW"))
        .otherwise(None)
        .alias("Pair_Declarer_Direction"),
    ).with_columns(
        (pl.col("Tricks") - pl.col("DD_Tricks"))
        .cast(pl.Int8, strict=False)
        .alias("DDTricks_Diff"),
        (pl.col("Tricks") - pl.col("DD_Tricks"))
        .cast(pl.Int8, strict=False)
        .alias("DD_Tricks_Diff"),
    )
    scores_d = augment_lib.precompute_contract_score_tables()[1]
    if "Vul_NS" not in out.columns:
        out = out.with_columns(
            pl.col("Vul").is_in(["N_S", "Both"]).alias("Vul_NS"),
            pl.col("Vul").is_in(["E_W", "Both"]).alias("Vul_EW"),
        )
    out = out.with_columns(
        pl.struct(
            [
                "BidLvl",
                "BidSuit",
                "DD_Tricks",
                "Declarer_Direction",
                "Vul_NS",
                "Vul_EW",
            ]
        )
        .map_elements(
            lambda rec: scores_d.get(
                (
                    rec["BidLvl"],
                    rec["BidSuit"],
                    rec["DD_Tricks"],
                    rec["Vul_NS"]
                    if rec["Declarer_Direction"] in {"N", "S"}
                    else rec["Vul_EW"],
                )
            ),
            return_dtype=pl.Int32,
        )
        .alias("DD_Score_Declarer")
    )
    score_columns: list[pl.Expr] = []
    contracts = (
        out.select("BidLvl", "BidSuit", "Declarer_Direction")
        .drop_nulls()
        .unique()
        .iter_rows(named=True)
    )
    for rec in contracts:
        level = rec["BidLvl"]
        suit = rec["BidSuit"]
        direction = rec["Declarer_Direction"]
        if level is None or suit not in set("SHDCN") or direction not in SEATS:
            continue
        name = f"DDScore_{int(level)}{suit}_{direction}"
        if name in out.columns:
            continue
        score_columns.append(
            pl.when(
                (pl.col("BidLvl") == level)
                & (pl.col("BidSuit") == suit)
                & (pl.col("Declarer_Direction") == direction)
            )
            .then(pl.col("DD_Score_Declarer"))
            .otherwise(None)
            .alias(name)
        )
    if score_columns:
        out = out.with_columns(*score_columns)
    par_rows = _par_rows_from_embedded_dd(out, augment_lib)
    if "ParScore" in out.columns:
        out = out.drop("ParScore")
    out = out.join(par_rows, on=["PBN", "Dealer", "Vul"], how="left")
    out = out.with_columns(
        pl.col("ParScore").alias("ParScore_NS"),
        (-pl.col("ParScore")).alias("ParScore_EW"),
    )
    if "Score_NS" in out.columns and "Score_Declarer" not in out.columns:
        out = out.with_columns(
            pl.when(pl.col("Pair_Declarer_Direction") == "NS")
            .then(pl.col("Score_NS"))
            .when(pl.col("Pair_Declarer_Direction") == "EW")
            .then(-pl.col("Score_NS"))
            .otherwise(None)
            .alias("Score_Declarer")
        )
    if "Pct_NS" in out.columns and "Declarer_Pct" not in out.columns:
        out = out.with_columns(
            pl.when(pl.col("Pair_Declarer_Direction") == "NS")
            .then(pl.col("Pct_NS"))
            .when(pl.col("Pair_Declarer_Direction") == "EW")
            .then(pl.col("Pct_EW") if "Pct_EW" in out.columns else (1 - pl.col("Pct_NS")))
            .otherwise(None)
            .alias("Declarer_Pct")
        )
    hands = _hand_feature_frame(out["PBN"].drop_nulls().unique().to_list())
    if hands.height:
        overlap = [column for column in hands.columns if column != "PBN" and column in out.columns]
        if overlap:
            out = out.drop(overlap)
        out = out.join(hands, on="PBN", how="left")
    return out


def _convert_and_reattach(prepared: pl.DataFrame) -> pl.DataFrame:
    ff_lib, _augment_lib = _import_mlbridge()
    converted = ff_lib.convert_ffdf_lancelot_to_mldf(prepared)
    if converted.height != prepared.height:
        raise ValueError(
            f"mlBridge conversion changed row count from {prepared.height} "
            f"to {converted.height}"
        )
    overlays = []
    for column in (
        "session_id",
        "group_id",
        "team_id",
        "board_id",
        "Date",
        "BidLvl",
        "BidSuit",
        "Declarer_Direction",
        "Result",
        *EMBEDDED_DD_COLUMNS,
        *(f"Player_ID_{seat}" for seat in SEATS),
    ):
        if column in prepared.columns:
            overlays.append(prepared[column].alias(column))
    if overlays:
        converted = converted.with_columns(*overlays)
    return converted


def _validate_converted_suits(frame: pl.DataFrame) -> None:
    if "BidSuit" not in frame.columns:
        return
    invalid_suits = sorted(
        set(
            frame["BidSuit"]
            .drop_nulls()
            .cast(pl.String)
            .str.to_uppercase()
            .to_list()
        )
        - {"C", "D", "H", "S", "N"}
    )
    if invalid_suits:
        raise NoQualityRowsError(
            f"Raw session has unsupported contract denominations: {invalid_suits}"
        )


_HRS_DD_COLUMNS = tuple(f"DD_{seat}_{suit}" for seat in "NESW" for suit in "CDHSN")
_HRS_PROB_COLUMNS = tuple(
    f"Probs_{pair}_{declarer}_{strain}_{taken}"
    for pair in ("NS", "EW")
    for declarer in "NESW"
    for strain in "CDHSN"
    for taken in range(14)
)
_HRS_PAR_CONTRACTS = pl.List(
    pl.Struct(
        {
            "Level": pl.String,
            "Strain": pl.String,
            "Doubled": pl.String,
            "Pair_Direction": pl.String,
            "Result": pl.Int16,
        }
    )
)


def _official_hrs_schema() -> dict[str, pl.DataType]:
    """ACBL hand-records cache: DD, Par, SD probs. No EV."""
    return {
        "PBN": pl.String,
        "Dealer": pl.String,
        "Vul": pl.String,
        **{column: pl.UInt8 for column in _HRS_DD_COLUMNS},
        "ParScore": pl.Int16,
        "ParNumber": pl.Int8,
        "ParContracts": _HRS_PAR_CONTRACTS,
        "Probs_Trials": pl.Int64,
        **{column: pl.Float32 for column in _HRS_PROB_COLUMNS},
    }


def _official_hrs_cache(frame: pl.DataFrame) -> pl.DataFrame:
    schema = _official_hrs_schema()
    extras = [column for column in frame.columns if column not in schema]
    if extras:
        frame = frame.drop(extras)
    additions = [
        pl.lit(None, dtype=dtype).alias(column)
        for column, dtype in schema.items()
        if column not in frame.columns
    ]
    if additions:
        frame = frame.with_columns(*additions)
    casts = []
    for column, dtype in schema.items():
        if frame.schema[column] != dtype:
            casts.append(pl.col(column).cast(dtype, strict=False))
    if casts:
        frame = frame.with_columns(*casts)
    return _dedupe_hrs_cache_keys(frame.select(list(schema)))


def _dedupe_hrs_cache_keys(cache: pl.DataFrame) -> pl.DataFrame:
    """Keep PBN+Dealer+Vul rows; drop null-key leftovers once a real key exists."""
    if cache.is_empty() or "Dealer" not in cache.columns:
        return cache
    keyed = cache.filter(pl.col("Dealer").is_not_null() & pl.col("Vul").is_not_null())
    if keyed.is_empty():
        return cache
    orphans = cache.filter(
        pl.col("Dealer").is_null() | pl.col("Vul").is_null()
    ).join(keyed.select("PBN").unique(), on="PBN", how="anti")
    if orphans.is_empty():
        return keyed
    return pl.concat([keyed, orphans], how="vertical")


def default_hrs_cache_path(source_dir: pathlib.Path | None = None) -> pathlib.Path:
    root = pathlib.Path(source_dir or DEFAULT_SOURCE_DIR).parent
    return root / HRS_CACHE_FILENAME


def load_hrs_cache(path: pathlib.Path) -> pl.DataFrame | None:
    if not pathlib.Path(path).is_file():
        return None
    return _official_hrs_cache(pl.read_parquet(path))


def save_hrs_cache(frame: pl.DataFrame, path: pathlib.Path) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    _official_hrs_cache(frame).write_parquet(temporary)
    temporary.replace(path)


def repair_hrs_cache_dd_with_ddss(
    cache_path: pathlib.Path,
    *,
    dry_run: bool = False,
    limit: int | None = None,
    batch_size: int = 1000,
) -> dict[str, int]:
    """Compare every cached DD table to ddss and rewrite mismatches plus Par."""
    from tqdm import tqdm

    started = datetime.now()
    print(
        f"[ffbridge-quality] DD repair start {started.isoformat(timespec='seconds')}",
        flush=True,
    )
    cache = load_hrs_cache(cache_path)
    if cache is None or cache.is_empty():
        print("[ffbridge-quality] DD repair: no cache", flush=True)
        return {"cache_deals": 0, "mismatched_deals": 0, "rewritten": 0}
    unique = (
        cache.filter(
            pl.col("PBN").is_not_null()
            & pl.all_horizontal(
                [pl.col(column).is_not_null() for column in EMBEDDED_DD_COLUMNS]
            )
        )
        .unique(subset=["PBN"], maintain_order=True)
    )
    if limit is not None:
        unique = unique.head(limit)
    pbns = unique["PBN"].to_list()
    by_pbn = {rec["PBN"]: rec for rec in unique.iter_rows(named=True)}
    mismatch_pbns: list[str] = []
    corrections: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(pbns), batch_size), desc="ddss DD repair"):
        chunk = pbns[start : start + batch_size]
        tables = _ddss_tables_for_pbns(chunk)
        for pbn, table in zip(chunk, tables):
            expected = _ddss_columns_from_table(table)
            rec = by_pbn[pbn]
            if any(
                int(rec[column]) != expected[column] for column in EMBEDDED_DD_COLUMNS
            ):
                mismatch_pbns.append(pbn)
                corrections.append({"PBN": pbn, **expected})
    stats = {
        "cache_deals": len(pbns),
        "mismatched_deals": len(mismatch_pbns),
        "rewritten": 0,
    }
    print(
        f"[ffbridge-quality] DD repair compared {len(pbns)} deals; "
        f"{len(mismatch_pbns)} disagreed with ddss",
        flush=True,
    )
    if corrections and not dry_run:
        corr = pl.DataFrame(corrections)
        keep = cache.filter(~pl.col("PBN").is_in(mismatch_pbns))
        fix = (
            cache.filter(pl.col("PBN").is_in(mismatch_pbns))
            .drop([column for column in EMBEDDED_DD_COLUMNS if column in cache.columns])
            .join(corr, on="PBN", how="left")
        )
        _, augment_lib = _import_mlbridge()
        par = _par_rows_from_embedded_dd(fix, augment_lib)
        drop_par = [
            column
            for column in ("ParScore", "ParNumber", "ParContracts")
            if column in fix.columns
        ]
        if drop_par:
            fix = fix.drop(drop_par)
        fix = fix.join(par, on=["PBN", "Dealer", "Vul"], how="left")
        cache = _official_hrs_cache(pl.concat([keep, fix], how="diagonal"))
        save_hrs_cache(cache, cache_path)
        stats["rewritten"] = len(mismatch_pbns)
    ended = datetime.now()
    elapsed = (ended - started).total_seconds()
    print(
        f"[ffbridge-quality] DD repair end {ended.isoformat(timespec='seconds')} "
        f"(elapsed {elapsed:.1f}s)",
        flush=True,
    )
    return stats


def _unique_deal_join_columns(frame: pl.DataFrame) -> list[str]:
    keep = ["PBN", "Dealer", "Vul"]
    for column in frame.columns:
        if _EV_SUMMARY_RE.match(column) or _EV_PAIR_MAX_RE.match(column):
            keep.append(column)
        elif column.startswith("CT_"):
            keep.append(column)
    return list(dict.fromkeys(keep))


def _join_ev_score_declarer(frame: pl.DataFrame) -> pl.DataFrame:
    """Pick the played-contract EV summary without a 280-way when/then."""
    ev_columns = [column for column in frame.columns if _EV_SUMMARY_RE.match(column)]
    if not ev_columns or "PBN" not in frame.columns:
        return frame.with_columns(_ev_score_declarer_expr(frame).alias("EV_Score_Declarer"))
    long = (
        frame.select(["PBN", *ev_columns])
        .unique(subset=["PBN"], maintain_order=True)
        .unpivot(
            index="PBN",
            on=ev_columns,
            variable_name="_ev_col",
            value_name="EV_Score_Declarer",
        )
        .with_columns(pl.col("_ev_col").str.split("_").alias("_parts"))
        .with_columns(
            pl.col("_parts").list.get(1).alias("Pair_Declarer_Direction"),
            pl.col("_parts").list.get(2).alias("Declarer_Direction"),
            pl.col("_parts").list.get(3).alias("BidSuit"),
            pl.col("_parts").list.get(4).cast(pl.Int32).alias("_ev_level"),
            (pl.col("_parts").list.get(5) == "V").alias("Vul_Declarer"),
        )
        .drop("_ev_col", "_parts")
    )
    work = frame
    if "EV_Score_Declarer" in work.columns:
        work = work.drop("EV_Score_Declarer")
    if "Vul_Declarer" not in work.columns:
        return work.with_columns(_ev_score_declarer_expr(work).alias("EV_Score_Declarer"))
    work = work.with_columns(
        pl.col("BidLvl").cast(pl.Int32, strict=False).alias("_ev_level")
    )
    work = work.join(
        long,
        on=[
            "PBN",
            "Pair_Declarer_Direction",
            "Declarer_Direction",
            "BidSuit",
            "_ev_level",
            "Vul_Declarer",
        ],
        how="left",
    )
    return work.drop("_ev_level")


def _ev_score_declarer_expr(frame: pl.DataFrame) -> pl.Expr:
    expr = pl.lit(None, dtype=pl.Float32)
    for pair in ("NS", "EW"):
        for declarer in pair:
            for strain in "SHDCN":
                for level in range(1, 8):
                    for vul_token, vul_flag in (("V", True), ("NV", False)):
                        column = f"EV_{pair}_{declarer}_{strain}_{level}_{vul_token}"
                        if column not in frame.columns:
                            continue
                        expr = (
                            pl.when(
                                (pl.col("Pair_Declarer_Direction") == pair)
                                & (pl.col("Declarer_Direction") == declarer)
                                & (pl.col("BidSuit") == strain)
                                & (pl.col("BidLvl").cast(pl.Int32, strict=False) == level)
                                & (pl.col("Vul_Declarer") == vul_flag)
                            )
                            .then(pl.col(column).cast(pl.Float32, strict=False))
                            .otherwise(expr)
                        )
    return expr


def _ev_max_declarer_expr() -> pl.Expr:
    return (
        pl.when((pl.col("Pair_Declarer_Direction") == "NS") & pl.col("Vul_NS"))
        .then(pl.col("EV_NS_V_Max"))
        .when((pl.col("Pair_Declarer_Direction") == "NS") & ~pl.col("Vul_NS"))
        .then(pl.col("EV_NS_NV_Max"))
        .when((pl.col("Pair_Declarer_Direction") == "EW") & pl.col("Vul_EW"))
        .then(pl.col("EV_EW_V_Max"))
        .when((pl.col("Pair_Declarer_Direction") == "EW") & ~pl.col("Vul_EW"))
        .then(pl.col("EV_EW_NV_Max"))
        .otherwise(None)
    )


def _matchpoint_against_field(
    frame: pl.DataFrame,
    *,
    value_col: str,
    field_col: str,
    pair: str,
) -> pl.DataFrame:
    """Matchpoint `value_col` against the board's actual `field_col` scores."""
    if value_col not in frame.columns or field_col not in frame.columns:
        return frame
    unique_cols = ["session_id", "Board"]
    if "section_name" in frame.columns:
        unique_cols.insert(1, "section_name")
    missing = [column for column in unique_cols if column not in frame.columns]
    if missing:
        return frame
    mp_col = f"MP_{value_col}"
    pct_col = f"{value_col}_Pct"
    field_scores = (
        frame.select([*unique_cols, field_col])
        .unique()
        .group_by(unique_cols)
        .agg(pl.col(field_col).alias("_field_scores"))
    )
    lookup = (
        frame.select([*unique_cols, value_col])
        .unique()
        .join(field_scores, on=unique_cols, how="left")
        .explode("_field_scores", empty_as_null=True)
        .group_by([*unique_cols, value_col])
        .agg(
            (pl.col("_field_scores") < pl.col(value_col))
            .sum()
            .cast(pl.Float32)
            .alias("beats"),
            (pl.col("_field_scores") == pl.col(value_col))
            .sum()
            .cast(pl.Float32)
            .alias("ties"),
            pl.col("_field_scores").count().alias("total_comparisons"),
        )
        .with_columns(
            (pl.col("beats") + pl.col("ties") * 0.5).alias(mp_col),
            (
                (pl.col("beats") + pl.col("ties") * 0.5)
                / pl.when(pl.col("total_comparisons") >= 1)
                .then(pl.col("total_comparisons"))
                .otherwise(pl.lit(1))
            )
            .cast(pl.Float32)
            .alias(pct_col),
        )
        .select([*unique_cols, value_col, mp_col, pct_col])
    )
    return frame.join(lookup, on=[*unique_cols, value_col], how="left")


def _attach_contract_types(frame: pl.DataFrame) -> pl.DataFrame:
    if any(column.startswith("CT_") for column in frame.columns):
        return frame
    if any(column not in frame.columns for column in EMBEDDED_DD_COLUMNS):
        return frame
    _ff_lib, augment_lib = _import_mlbridge()
    return augment_lib.add_contract_types(frame)


def _empty_hrs_cache() -> pl.DataFrame:
    return pl.DataFrame(schema=_official_hrs_schema())


def _latest_cache(
    cache: pl.DataFrame | None,
    cache_file_path: pathlib.Path | None,
) -> pl.DataFrame:
    if cache_file_path is not None:
        disk = load_hrs_cache(cache_file_path)
        if disk is not None:
            return disk
    if cache is None or cache.is_empty():
        return _empty_hrs_cache()
    return _official_hrs_cache(cache)


def _cached_sd_pbns(cache: pl.DataFrame | None) -> set[str]:
    if cache is None or cache.is_empty() or "Probs_Trials" not in cache.columns:
        return set()
    return {
        pbn
        for pbn in cache.filter(pl.col("Probs_Trials").is_not_null())["PBN"].to_list()
        if pbn
    }


def _missing_dd_par_deals(
    unique: pl.DataFrame,
    cache: pl.DataFrame,
    force_pbns: set[str] | None = None,
) -> pl.DataFrame:
    keys = ["PBN", "Dealer", "Vul"]
    if cache.is_empty() or "DD_N_C" not in cache.columns:
        return unique
    have = cache.filter(
        pl.col("DD_N_C").is_not_null() & pl.col("ParScore").is_not_null()
    )
    if force_pbns:
        have = have.filter(~pl.col("PBN").is_in(list(force_pbns)))
    return unique.join(have.select(*keys), on=keys, how="anti")


def _stale_dd_par_deals(unique: pl.DataFrame, cache: pl.DataFrame) -> pl.DataFrame:
    """Rows whose cached DD table no longer matches the frame (ddss) table."""
    keys = ["PBN", "Dealer", "Vul"]
    if cache.is_empty() or unique.is_empty() or "DD_N_C" not in cache.columns:
        return unique.head(0)
    dd_cols = [
        column
        for column in _HRS_DD_COLUMNS
        if column in unique.columns and column in cache.columns
    ]
    if not dd_cols:
        return unique.head(0)
    right = cache.select(*keys, *dd_cols).rename(
        {column: f"{column}__cache" for column in dd_cols}
    )
    joined = unique.join(right, on=keys, how="inner")
    stale_keys = joined.filter(
        pl.any_horizontal(
            [
                pl.col(column).cast(pl.Int16, strict=False)
                != pl.col(f"{column}__cache").cast(pl.Int16, strict=False)
                for column in dd_cols
            ]
        )
    ).select(*keys)
    if stale_keys.is_empty():
        return unique.head(0)
    return unique.join(stale_keys, on=keys, how="semi")


def _dd_par_deals_to_upsert(
    unique: pl.DataFrame,
    cache: pl.DataFrame,
    force_pbns: set[str] | None = None,
) -> pl.DataFrame:
    missing = _missing_dd_par_deals(unique, cache, force_pbns=force_pbns)
    stale = _stale_dd_par_deals(unique, cache)
    if stale.is_empty():
        return missing
    if missing.is_empty():
        return stale
    return pl.concat([missing, stale], how="vertical").unique(
        subset=["PBN", "Dealer", "Vul"], maintain_order=True
    )


def _upsert_dd_par_into_cache(
    unique: pl.DataFrame,
    cache: pl.DataFrame,
    augment_lib: Any,
    force_pbns: set[str] | None = None,
) -> tuple[pl.DataFrame, int]:
    """Fill ACBL cache DD/Par from unique deals. Uses the frame's DD table."""
    needed = ("PBN", "Dealer", "Vul", *_HRS_DD_COLUMNS)
    if any(column not in unique.columns for column in needed):
        return cache, 0
    complete = unique.filter(
        pl.col("PBN").is_not_null()
        & pl.col("Dealer").is_not_null()
        & pl.col("Vul").is_not_null()
        & pl.all_horizontal(
            [pl.col(column).is_not_null() for column in _HRS_DD_COLUMNS]
        )
    ).unique(subset=["PBN", "Dealer", "Vul"], maintain_order=True)
    todo = _dd_par_deals_to_upsert(complete, cache, force_pbns=force_pbns)
    if todo.is_empty():
        return cache, 0
    par = _par_rows_from_embedded_dd(todo, augment_lib)
    incoming = (
        todo.select("PBN", "Dealer", "Vul", *_HRS_DD_COLUMNS)
        .join(par, on=["PBN", "Dealer", "Vul"], how="left")
        .with_columns(
            pl.col("ParContracts").list.len().cast(pl.Int8).alias("ParNumber"),
            *[pl.col(column).cast(pl.UInt8, strict=False) for column in _HRS_DD_COLUMNS],
            pl.col("ParScore").cast(pl.Int16, strict=False),
            pl.col("ParContracts").cast(_HRS_PAR_CONTRACTS, strict=False),
        )
    )
    augment_logger = logging.getLogger("mlBridge.mlBridgeAugmentLib")
    previous_level = augment_logger.level
    augment_logger.setLevel(logging.WARNING)
    try:
        updated = _dedupe_hrs_cache_keys(
            augment_lib.update_hand_records_cache(cache, incoming)
        )
    finally:
        augment_logger.setLevel(previous_level)
    return updated, todo.height


def _drop_intermediate_ev(frame: pl.DataFrame) -> pl.DataFrame:
    drop = [
        column
        for column in frame.columns
        if column.startswith("EV_")
        and not _EV_SUMMARY_RE.match(column)
        and not _EV_PAIR_MAX_RE.match(column)
    ]
    return frame.drop(drop) if drop else frame


def _attach_pair_ev_max(frame: pl.DataFrame) -> pl.DataFrame:
    """Pair/vulnerability max EV. Avoids identify_best_contracts_by_ev (minutes)."""
    exprs: list[pl.Expr] = []
    for pair in ("NS", "EW"):
        for vul in ("V", "NV"):
            columns = [
                column
                for column in frame.columns
                if _EV_SUMMARY_RE.match(column)
                and column.startswith(f"EV_{pair}_")
                and column.endswith(f"_{vul}")
            ]
            if columns:
                exprs.append(pl.max_horizontal(pl.col(columns)).alias(f"EV_{pair}_{vul}_Max"))
    if not exprs:
        return frame
    return frame.with_columns(*exprs)


def _compute_ev_from_probs(
    unique: pl.DataFrame,
    cache: pl.DataFrame,
    augment_lib: Any,
) -> pl.DataFrame:
    """Derive EV from cached SD probs. EV is not stored in the hand-records cache."""
    prob_columns = ["PBN"] + [
        column for column in cache.columns if column.startswith("Probs_")
    ]
    probs = cache.select(prob_columns).unique(subset=["PBN"], maintain_order=True)
    unique_aug = unique.join(probs, on="PBN", how="left")
    _scores_d, _scores, scores_df = augment_lib.precompute_contract_score_tables()
    unique_aug = augment_lib.add_single_dummy_expected_values(unique_aug, scores_df)
    unique_aug = _drop_intermediate_ev(unique_aug)
    return _attach_pair_ev_max(unique_aug)


def _estimate_sd_into_cache(
    todo: pl.DataFrame,
    cache: pl.DataFrame,
    augment_lib: Any,
    *,
    sd_productions: int,
    max_sd_adds: int | None,
) -> pl.DataFrame:
    if cache.is_empty() or "Probs_Trials" not in cache.columns:
        cache = _empty_hrs_cache()
    started = time.time()
    augment_logger = logging.getLogger("mlBridge.mlBridgeAugmentLib")
    previous_level = augment_logger.level
    augment_logger.setLevel(logging.WARNING)
    try:
        _sd_dfs, sd_df = augment_lib.estimate_sd_trick_distributions_for_df(
            todo,
            cache,
            sd_productions,
            max_sd_adds,
            None,
            None,
        )
    finally:
        augment_logger.setLevel(previous_level)
    if sd_df is not None and not sd_df.is_empty():
        cache = augment_lib.update_hand_records_cache(cache, sd_df)
    elapsed = time.time() - started
    print(
        f"[ffbridge-quality] SD solved {todo.height} deals in {elapsed:.1f}s",
        flush=True,
    )
    return cache


def _unique_deal_frame(frame: pl.DataFrame) -> pl.DataFrame:
    columns = ["PBN", "Dealer", "Vul", "Board"]
    columns.extend(
        column for column in _HRS_DD_COLUMNS if column in frame.columns
    )
    return frame.select(columns).unique(
        subset=["PBN", "Dealer", "Vul"], maintain_order=True
    )


def _attach_sd_ev_from_unique_deals(
    frame: pl.DataFrame,
    *,
    hrs_cache_df: pl.DataFrame | None,
    cache_file_path: pathlib.Path | None,
    sd_productions: int,
    max_sd_adds: int | None,
    force_dd_pbns: set[str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """ACBL cache (DD/Par/SD probs) on unique deals, then derive EV and join."""
    if max_sd_adds == 0:
        return frame, hrs_cache_df
    _require_columns(frame, ["PBN", "Dealer", "Vul"], "Unique-deal SD")
    unique = _unique_deal_frame(frame)
    unique_pbn = unique.unique(subset=["PBN"], maintain_order=True)
    needed = {pbn for pbn in unique_pbn["PBN"].to_list() if pbn}
    missing_sd = needed - _cached_sd_pbns(hrs_cache_df)
    missing_dd = (
        _dd_par_deals_to_upsert(unique, hrs_cache_df, force_pbns=force_dd_pbns)
        if hrs_cache_df is not None
        else unique
    )
    print(
        f"[ffbridge-quality] unique deals {len(needed)}; "
        f"SD cache hits {len(needed) - len(missing_sd)}; "
        f"DD/Par missing {missing_dd.height}; "
        f"SD missing {len(missing_sd)}",
        flush=True,
    )
    _ff_lib, augment_lib = _import_mlbridge()
    if missing_sd or missing_dd.height:
        with _AUGMENT_LOCK:
            hrs_cache_df = _latest_cache(hrs_cache_df, cache_file_path)
            hrs_cache_df, dd_added = _upsert_dd_par_into_cache(
                unique, hrs_cache_df, augment_lib, force_pbns=force_dd_pbns
            )
            missing_sd = needed - _cached_sd_pbns(hrs_cache_df)
            if missing_sd:
                todo = unique_pbn.filter(pl.col("PBN").is_in(sorted(missing_sd)))
                if max_sd_adds is not None:
                    todo = todo.head(max_sd_adds)
                hrs_cache_df = _estimate_sd_into_cache(
                    todo,
                    hrs_cache_df,
                    augment_lib,
                    sd_productions=sd_productions,
                    max_sd_adds=max_sd_adds,
                )
            if cache_file_path is not None and (dd_added or missing_sd):
                save_hrs_cache(hrs_cache_df, cache_file_path)
    if hrs_cache_df is None or hrs_cache_df.is_empty():
        return frame, hrs_cache_df
    unique_aug = _compute_ev_from_probs(unique_pbn, hrs_cache_df, augment_lib)
    join_columns = [
        column
        for column in _unique_deal_join_columns(unique_aug)
        if column in unique_aug.columns
    ]
    slim = unique_aug.select(
        [column for column in join_columns if column not in {"Dealer", "Vul"}]
    )
    overlap = [
        column
        for column in slim.columns
        if column != "PBN" and column in frame.columns
    ]
    if overlap:
        frame = frame.drop(overlap)
    return frame.join(slim, on="PBN", how="left"), hrs_cache_df


def _attach_board_ev_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Contract-dependent EV and matchpoints after unique-deal SD is joined."""
    if "Pair_Declarer_Direction" not in frame.columns:
        return frame
    if "Vul_NS" not in frame.columns and "Vul" in frame.columns:
        frame = frame.with_columns(
            pl.col("Vul").is_in(["N_S", "Both"]).alias("Vul_NS"),
            pl.col("Vul").is_in(["E_W", "Both"]).alias("Vul_EW"),
        )
    if "Vul_Declarer" not in frame.columns and "Vul_NS" in frame.columns:
        frame = frame.with_columns(
            pl.when(pl.col("Pair_Declarer_Direction") == "NS")
            .then(pl.col("Vul_NS"))
            .when(pl.col("Pair_Declarer_Direction") == "EW")
            .then(pl.col("Vul_EW"))
            .otherwise(None)
            .alias("Vul_Declarer")
        )
    if any(_EV_SUMMARY_RE.match(column) for column in frame.columns):
        if "PBN" in frame.columns:
            frame = _join_ev_score_declarer(frame)
        else:
            frame = frame.with_columns(
                _ev_score_declarer_expr(frame).alias("EV_Score_Declarer")
            )
        frame = frame.with_columns(
            pl.when(pl.col("Pair_Declarer_Direction") == "NS")
            .then(pl.col("EV_Score_Declarer"))
            .when(pl.col("Pair_Declarer_Direction") == "EW")
            .then(-pl.col("EV_Score_Declarer"))
            .otherwise(None)
            .alias("EV_Score_NS"),
            pl.when(pl.col("Pair_Declarer_Direction") == "EW")
            .then(pl.col("EV_Score_Declarer"))
            .when(pl.col("Pair_Declarer_Direction") == "NS")
            .then(-pl.col("EV_Score_Declarer"))
            .otherwise(None)
            .alias("EV_Score_EW"),
        )
    if all(
        column in frame.columns
        for column in ("EV_NS_V_Max", "EV_NS_NV_Max", "EV_EW_V_Max", "EV_EW_NV_Max")
    ):
        frame = frame.with_columns(_ev_max_declarer_expr().alias("EV_Max_Declarer"))
        frame = frame.with_columns(
            pl.when(pl.col("Pair_Declarer_Direction") == "NS")
            .then(pl.col("EV_Max_Declarer"))
            .when(pl.col("Pair_Declarer_Direction") == "EW")
            .then(-pl.col("EV_Max_Declarer"))
            .otherwise(None)
            .alias("EV_Max_NS"),
            pl.when(pl.col("Pair_Declarer_Direction") == "EW")
            .then(pl.col("EV_Max_Declarer"))
            .when(pl.col("Pair_Declarer_Direction") == "NS")
            .then(-pl.col("EV_Max_Declarer"))
            .otherwise(None)
            .alias("EV_Max_EW"),
        )
    if "Score_NS" in frame.columns and "EV_Score_NS" in frame.columns:
        frame = _matchpoint_against_field(
            frame, value_col="EV_Score_NS", field_col="Score_NS", pair="NS"
        )
        frame = _matchpoint_against_field(
            frame, value_col="EV_Score_EW", field_col="Score_EW", pair="EW"
        )
        frame = frame.with_columns(
            pl.when(pl.col("Pair_Declarer_Direction") == "NS")
            .then(pl.col("EV_Score_NS_Pct"))
            .when(pl.col("Pair_Declarer_Direction") == "EW")
            .then(pl.col("EV_Score_EW_Pct"))
            .otherwise(None)
            .alias("MP_EV_Pct_Declarer")
        )
    if "Score_NS" in frame.columns and "EV_Max_NS" in frame.columns:
        frame = _matchpoint_against_field(
            frame, value_col="EV_Max_NS", field_col="Score_NS", pair="NS"
        )
        frame = _matchpoint_against_field(
            frame, value_col="EV_Max_EW", field_col="Score_EW", pair="EW"
        )
        frame = frame.with_columns(
            pl.when(pl.col("Pair_Declarer_Direction") == "NS")
            .then(pl.col("EV_Max_NS_Pct"))
            .when(pl.col("Pair_Declarer_Direction") == "EW")
            .then(pl.col("EV_Max_EW_Pct"))
            .otherwise(None)
            .alias("MP_EV_Max_Pct_Declarer")
        )
    return _attach_dd_matchpoints(frame)


def _attach_dd_matchpoints(frame: pl.DataFrame) -> pl.DataFrame:
    """Matchpoint DD tricks at the actual contract against the board field.

    ``MP_DD_Pct_Declarer`` is the matchpoint percentage the declaring side
    would earn by taking ``DD_Tricks`` in the table's contract.
    """
    if "DD_Score_Declarer" not in frame.columns:
        return frame
    pair_col = None
    if "Pair_Declarer_Direction" in frame.columns:
        pair_col = "Pair_Declarer_Direction"
    elif "Declarer_Pair_Direction" in frame.columns:
        pair_col = "Declarer_Pair_Direction"
    elif "Declarer_Direction" in frame.columns:
        frame = frame.with_columns(
            pl.when(pl.col("Declarer_Direction").is_in(["N", "S"]))
            .then(pl.lit("NS"))
            .when(pl.col("Declarer_Direction").is_in(["E", "W"]))
            .then(pl.lit("EW"))
            .otherwise(None)
            .alias("Pair_Declarer_Direction")
        )
        pair_col = "Pair_Declarer_Direction"
    if pair_col is None:
        return frame
    if "Score_NS" not in frame.columns:
        if "Score_Declarer" not in frame.columns:
            return frame
        frame = frame.with_columns(
            pl.when(pl.col(pair_col) == "NS")
            .then(pl.col("Score_Declarer"))
            .when(pl.col(pair_col) == "EW")
            .then(-pl.col("Score_Declarer"))
            .otherwise(None)
            .alias("Score_NS")
        )
    if "Score_EW" not in frame.columns and "Score_NS" in frame.columns:
        frame = frame.with_columns((-pl.col("Score_NS")).alias("Score_EW"))
    frame = frame.with_columns(
        pl.when(pl.col(pair_col) == "NS")
        .then(pl.col("DD_Score_Declarer"))
        .when(pl.col(pair_col) == "EW")
        .then(-pl.col("DD_Score_Declarer"))
        .otherwise(None)
        .alias("DD_Score_NS"),
        pl.when(pl.col(pair_col) == "EW")
        .then(pl.col("DD_Score_Declarer"))
        .when(pl.col(pair_col) == "NS")
        .then(-pl.col("DD_Score_Declarer"))
        .otherwise(None)
        .alias("DD_Score_EW"),
    )
    drop_cols = [
        col
        for col in (
            "MP_DD_Pct_Declarer",
            "DD_Score_NS_Pct",
            "DD_Score_EW_Pct",
            "MP_DD_Score_NS",
            "MP_DD_Score_EW",
        )
        if col in frame.columns
    ]
    if drop_cols:
        frame = frame.drop(drop_cols)
    frame = _matchpoint_against_field(
        frame, value_col="DD_Score_NS", field_col="Score_NS", pair="NS"
    )
    frame = _matchpoint_against_field(
        frame, value_col="DD_Score_EW", field_col="Score_EW", pair="EW"
    )
    return frame.with_columns(
        pl.when(pl.col(pair_col) == "NS")
        .then(pl.col("DD_Score_NS_Pct"))
        .when(pl.col(pair_col) == "EW")
        .then(pl.col("DD_Score_EW_Pct"))
        .otherwise(None)
        .alias("MP_DD_Pct_Declarer")
    )


def _resolve_hrs_cache(
    hrs_cache: list[pl.DataFrame | None] | None,
    cache_file_path: pathlib.Path | None,
) -> pl.DataFrame | None:
    if hrs_cache:
        return hrs_cache[0]
    if cache_file_path is not None:
        return load_hrs_cache(cache_file_path)
    return None


def _full_mlbridge_augment(converted: pl.DataFrame) -> pl.DataFrame:
    _ff_lib, augment_lib = _import_mlbridge()
    augment_logger = logging.getLogger("mlBridge.mlBridgeAugmentLib")
    previous_level = augment_logger.level
    augment_logger.setLevel(logging.WARNING)
    try:
        try:
            with _AUGMENT_LOCK:
                augmented, _ = augment_lib.AllAugmentations(
                    converted,
                    None,
                    sd_productions=10,
                    max_sd_adds=None,
                    output_progress=False,
                    incorporate_elo_ratings=False,
                ).perform_all_augmentations()
        except pl.exceptions.InvalidOperationError as exc:
            raise NoQualityRowsError(
                f"Unsupported declarer/contract values: {exc}"
            ) from exc
    finally:
        augment_logger.setLevel(previous_level)
    return augmented


def augment_raw_session(
    raw: pl.DataFrame,
    *,
    hrs_cache: list[pl.DataFrame | None] | None = None,
    cache_file_path: pathlib.Path | None = None,
    sd_productions: int = 10,
    max_sd_adds: int | None = None,
    dd_audit_rate: float = 1.0,
) -> pl.DataFrame:
    """Augment a session the ACBL way: cache DD/Par/SD probs, derive EV, join.

    The hand-records cache stores only deal facts (DD, Par, SD probabilities).
    ddss is the source of truth for DD/Par. Lancelot's embedded table is
    compared and discarded. EV is computed from cached Probs on unique deals
    and joined onto the board rows.
    """
    if _has_quality_dd_scores(raw):
        return raw
    _require_columns(raw, ["PBN", "Contract"], "Raw score frame")
    prepared = raw.filter(
        pl.col("PBN").is_not_null()
        & (pl.col("PBN").cast(pl.String).str.strip_chars() != "")
        & pl.col("Contract").is_not_null()
        & (pl.col("Contract").cast(pl.String).str.strip_chars() != "")
    )
    if prepared.is_empty():
        raise NoQualityRowsError(
            "Raw session has no board rows with both a PBN deal and contract"
        )
    work = (
        _convert_and_reattach(prepared)
        if _needs_lancelot_convert(prepared)
        else prepared
    )
    _validate_converted_suits(work)
    if _has_embedded_dd_table(work):
        work, force_dd_pbns = _apply_lancelot_dd_audit(work, rate=dd_audit_rate)
        work = attach_embedded_dd_metrics(work)
        work = _attach_contract_types(work)
        cache = _resolve_hrs_cache(hrs_cache, cache_file_path)
        work, cache = _attach_sd_ev_from_unique_deals(
            work,
            hrs_cache_df=cache,
            cache_file_path=cache_file_path,
            sd_productions=sd_productions,
            max_sd_adds=max_sd_adds,
            force_dd_pbns=force_dd_pbns or None,
        )
        if hrs_cache is not None:
            hrs_cache[0] = cache
        if "Score_EW" not in work.columns and "Score_NS" in work.columns:
            work = work.with_columns((-pl.col("Score_NS")).alias("Score_EW"))
        return _attach_board_ev_columns(work)
    return _full_mlbridge_augment(work)


def build_historical_fragments(
    report: AuditReport,
    output_dir: pathlib.Path,
    session_dates: pl.DataFrame,
    *,
    show_progress: bool = True,
) -> tuple[list[pl.DataFrame], int, list[dict[str, str]]]:
    candidates = [
        session for session in report.sessions if not session.in_training and session.complete
    ]
    iterator: Iterable[SessionAudit] = candidates
    if show_progress and candidates:
        from tqdm import tqdm

        iterator = tqdm(candidates, desc="Normalizing historical FFBridge sessions")
    fragment_dir = pathlib.Path(output_dir) / FRAGMENT_DIRNAME
    fragments: list[pl.DataFrame] = []
    total_unmapped = 0
    unsupported: list[dict[str, str]] = []
    cache_path = default_hrs_cache_path(pathlib.Path(report.source_dir))
    hrs_cache: list[pl.DataFrame | None] = [load_hrs_cache(cache_path)]
    for session in iterator:
        fragment_path = fragment_dir / f"{session.session_id}.parquet"
        if fragment_path.is_file() and _fragment_schema_is_current(fragment_path):
            fragments.append(pl.read_parquet(fragment_path).select(*QUALITY_BOARD_COLUMNS))
            continue
        try:
            raw, unmapped = load_raw_session(pathlib.Path(report.source_dir), session)
            augmented = augment_raw_session(
                raw,
                hrs_cache=hrs_cache,
                cache_file_path=cache_path,
            )
            fragment = normalize_quality_frame(
                augmented,
                session_dates=session_dates,
                reject_duplicates=True,
            )
        except NoQualityRowsError as exc:
            unsupported.append(
                {"session_id": session.session_id, "reason": str(exc)}
            )
            continue
        except ValueError as exc:
            if not str(exc).startswith("Malformed score row in session "):
                raise
            unsupported.append(
                {"session_id": session.session_id, "reason": str(exc)}
            )
            continue
        except KeyError as exc:
            if "DD_" not in str(exc):
                raise
            unsupported.append(
                {
                    "session_id": session.session_id,
                    "reason": f"Unsupported double-dummy lookup: {exc}",
                }
            )
            continue
        _atomic_write_parquet(fragment, fragment_path)
        fragments.append(fragment)
        total_unmapped += unmapped
    return fragments, total_unmapped, unsupported


def write_quality_artifacts(
    board_quality: pl.DataFrame,
    output_dir: pathlib.Path,
    *,
    cutoff: date,
    source_dir: pathlib.Path,
    audit: AuditReport,
    unmapped_seat_count: int | None = None,
    unsupported_sessions: Sequence[Mapping[str, str]] = (),
) -> dict[str, Any]:
    """Write sidecars first and metadata last as the atomic completion marker."""
    validate_unique_board_plays(board_quality)
    player_quality = build_player_sidecar(board_quality)
    pair_quality = build_pair_sidecar(board_quality)
    output_dir = pathlib.Path(output_dir)
    board_path = output_dir / BOARD_FILENAME
    player_path = output_dir / PLAYER_FILENAME
    pair_path = output_dir / PAIR_FILENAME
    metadata_path = output_dir / METADATA_FILENAME
    _atomic_write_parquet(board_quality, board_path)
    _atomic_write_parquet(player_quality, player_path)
    _atomic_write_parquet(pair_quality, pair_path)
    authoritative_unmapped = sum(
        board_quality[f"Player_ID_{seat}"].null_count() for seat in SEATS
    )
    # Raw unmapped seats can exceed output nulls: identity mapping fills some
    # IDs, and rows with no usable identity are dropped before this write.
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_dir": str(pathlib.Path(source_dir)),
        "cutoff": cutoff.isoformat(),
        "board_rows": board_quality.height,
        "player_rows": player_quality.height,
        "pair_rows": pair_quality.height,
        "unmapped_seat_count": authoritative_unmapped,
        "raw_unmapped_seat_count": unmapped_seat_count,
        "unsupported_session_count": len(unsupported_sessions),
        "unsupported_sessions": [dict(item) for item in unsupported_sessions],
        "metric_definitions": QUALITY_METRIC_DEFINITIONS,
        "audit_summary": audit.to_dict()["summary"],
        "files": {
            "board": board_path.name,
            "player": player_path.name,
            "pair": pair_path.name,
        },
    }
    _atomic_write_json(metadata, metadata_path)
    return metadata


def resolve_output_dir(explicit: pathlib.Path | None) -> pathlib.Path:
    if explicit is not None:
        return pathlib.Path(explicit)
    cache_root = os.environ.get("FFBRIDGE_CACHE_DIR", "").strip()
    if not cache_root:
        raise ValueError(
            "--output-dir is required when FFBRIDGE_CACHE_DIR is not set"
        )
    return pathlib.Path(cache_root) / "quality_cache"


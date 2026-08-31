"""Reusable historical FFBridge board-quality pipeline.

The module is deliberately free of import-time I/O.  Callers must explicitly
choose an output directory before any artifact is written.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
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
    timeout: float = 30.0,
    delay: float = 0.1,
    max_attempts: int = 4,
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
            player = lineup.get(field)
            if not isinstance(player, Mapping):
                player = {}
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


def augment_raw_session(raw: pl.DataFrame) -> pl.DataFrame:
    """Run mlBridge augmentation when embedded DD/par data is insufficient."""
    required = {
        "ParScore_NS",
        "ParScore_EW",
        "DDTricks_Diff",
    }
    has_dd_scores = any(
        column.startswith(("DDScore_", "DD_Score_")) for column in raw.columns
    )
    if required.issubset(raw.columns) and has_dd_scores:
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
    _here = pathlib.Path(__file__).resolve().parent
    mlbridge_root = next(
        (path for path in (_here / "mlBridge", _here.parent / "mlBridge") if path.is_dir()),
        None,
    )
    if mlbridge_root is None:
        raise FileNotFoundError("mlBridge not found at ./mlBridge or ../mlBridge")
    if str(mlbridge_root.parent) not in sys.path:
        sys.path.insert(0, str(mlbridge_root.parent))
    from mlBridge import mlBridgeFFLib  # type: ignore
    from mlBridge.mlBridgeAugmentLib import AllAugmentations  # type: ignore

    converted = mlBridgeFFLib.convert_ffdf_lancelot_to_mldf(prepared)
    if converted.height != prepared.height:
        raise ValueError(
            f"mlBridge conversion changed row count from {prepared.height} "
            f"to {converted.height}"
        )
    if "BidSuit" in converted.columns:
        invalid_suits = sorted(
            set(
                converted["BidSuit"]
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
    converted = converted.with_columns(
        prepared["session_id"].alias("session_id"),
        prepared["group_id"].alias("group_id"),
        prepared["team_id"].alias("team_id"),
        *[
            prepared[f"Player_ID_{seat}"].alias(f"Player_ID_{seat}")
            for seat in SEATS
        ],
    )
    augment_logger = logging.getLogger("mlBridge.mlBridgeAugmentLib")
    previous_level = augment_logger.level
    augment_logger.setLevel(logging.WARNING)
    try:
        augmented, _ = AllAugmentations(
            converted,
            None,
            sd_productions=0,
            max_sd_adds=0,
            output_progress=False,
            incorporate_elo_ratings=False,
        ).perform_all_augmentations()
    finally:
        augment_logger.setLevel(previous_level)
    return augmented


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
    for session in iterator:
        fragment_path = fragment_dir / f"{session.session_id}.parquet"
        if fragment_path.is_file() and _fragment_schema_is_current(fragment_path):
            fragments.append(pl.read_parquet(fragment_path).select(*QUALITY_BOARD_COLUMNS))
            continue
        try:
            raw, unmapped = load_raw_session(pathlib.Path(report.source_dir), session)
            augmented = augment_raw_session(raw)
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


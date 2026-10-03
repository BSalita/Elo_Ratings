"""Download recent FFBridge sessions and upsert their board rows into the recent store.

The store is ``e:/bridge/data/ffbridge/recent/ffbridge_boards_recent.parquet``.
Rows are the published board scores, with ``session_id`` and ``Date``. Double
dummy, par, and Elo stay on the historical quality parquet until the full
quality build absorbs these sessions. Each run drops sessions whose date is
already in that historical file.

``--every`` chooses how far the discovery window reaches:

- hour: yesterday through today
- day: the last 2 days
- week: the last 8 days
- quarter: the last 95 days

The window also opens to the day after the newest historical ``Date`` when
that day is earlier. Schedule examples, from ``ffbridge-pipeline``::

    ffbridge_recent.bat hour
    ffbridge_recent.bat day
    ffbridge_recent.bat week
    ffbridge_recent.bat quarter
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import date, datetime, timedelta
from typing import Optional

import polars as pl

from ffbridge_quality_pipeline import (
    DEFAULT_SOURCE_DIR,
    AuditReport,
    audit_historical_cache,
    discover_session_metadata,
    fetch_missing_artifacts,
    load_raw_session,
)

LOOKBACK_DAYS = {"hour": 2, "day": 2, "week": 8, "quarter": 95}
DEFAULT_RECENT = pathlib.Path(
    r"e:/bridge/data/ffbridge/recent/ffbridge_boards_recent.parquet"
)
DEFAULT_HISTORICAL = pathlib.Path(
    r"e:/bridge/data/ffbridge/quality_cache/ffbridge_quality_boards.parquet"
)


def coverage_start(
    historical_max: Optional[date], every: str, today: date
) -> date:
    """First session date this run should discover and keep."""
    lookback = today - timedelta(days=LOOKBACK_DAYS[every] - 1)
    if historical_max is None:
        return lookback
    return min(historical_max + timedelta(days=1), lookback)


def historical_max_date(path: pathlib.Path) -> Optional[date]:
    """Newest Date in the historical quality boards file, if it exists."""
    if not path.is_file():
        return None
    schema = pl.scan_parquet(path).collect_schema()
    if "Date" not in schema.names():
        return None
    value = pl.scan_parquet(path).select(pl.col("Date").max()).collect().item()
    if value is None or str(value).strip() == "":
        return None
    return date.fromisoformat(str(value)[:10])


def upsert_boards(
    path: pathlib.Path,
    incoming: pl.DataFrame,
    *,
    historical_max: Optional[date],
) -> int:
    """Replace incoming session ids and drop sessions already in history."""
    frames = []
    if path.is_file():
        old = pl.read_parquet(path)
        if incoming.height and "session_id" in old.columns:
            ids = incoming["session_id"].cast(pl.String).unique().to_list()
            old = old.filter(~pl.col("session_id").cast(pl.String).is_in(ids))
        frames.append(old)
    if incoming.height:
        frames.append(incoming)
    if not frames:
        return 0
    frame = frames[0] if len(frames) == 1 else pl.concat(frames, how="diagonal_relaxed")
    if historical_max is not None and "Date" in frame.columns:
        frame = frame.filter(pl.col("Date").cast(pl.String).str.slice(0, 10) > historical_max.isoformat())
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.is_empty():
        if path.is_file():
            path.unlink()
        return 0
    temporary = path.with_suffix(".parquet.tmp")
    frame.write_parquet(temporary, compression="zstd")
    temporary.replace(path)
    return frame["session_id"].n_unique()


def _with_keys(frame: pl.DataFrame, session_id: str, session_day: str) -> pl.DataFrame:
    return frame.with_columns(
        pl.lit(session_id).alias("session_id"),
        pl.lit(session_day).alias("Date"),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--every", choices=tuple(LOOKBACK_DAYS), required=True)
    parser.add_argument("--today", type=date.fromisoformat, default=date.today())
    parser.add_argument("--source-dir", type=pathlib.Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--recent", type=pathlib.Path, default=DEFAULT_RECENT)
    parser.add_argument("--historical", type=pathlib.Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--fetch-workers", type=int, default=8)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Upsert complete sessions already in the local cache; do not call Lancelot.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    historical_max = historical_max_date(args.historical)
    coverage = coverage_start(historical_max, args.every, args.today)
    print(
        f"[recent] every={args.every} coverage_start={coverage.isoformat()} "
        f"historical_max={historical_max.isoformat() if historical_max else 'none'}",
        flush=True,
    )
    if not args.no_download:
        discovered = discover_session_metadata(
            args.source_dir,
            start_date=coverage,
            cutoff=args.today,
        )
        print(f"[recent] new session metadata files: {discovered}", flush=True)
    audit = audit_historical_cache(args.source_dir, cutoff=args.today)
    recent_sessions = tuple(
        session
        for session in audit.sessions
        if date.fromisoformat(session.session_date) >= coverage
    )
    print(f"[recent] sessions in window: {len(recent_sessions)}", flush=True)
    if not args.no_download and recent_sessions:
        fetch_missing_artifacts(
            AuditReport(
                source_dir=audit.source_dir,
                cutoff=audit.cutoff,
                training_session_count=audit.training_session_count,
                cached_session_count=len(recent_sessions),
                sessions=recent_sessions,
            ),
            workers=args.fetch_workers,
        )
        audit = audit_historical_cache(args.source_dir, cutoff=args.today)
        recent_sessions = tuple(
            session
            for session in audit.sessions
            if date.fromisoformat(session.session_date) >= coverage and session.complete
        )
    frames = []
    for session in recent_sessions:
        if not session.complete:
            print(f"[recent] skip incomplete {session.session_id}", flush=True)
            continue
        try:
            raw, _unmapped = load_raw_session(args.source_dir, session)
        except Exception as exc:
            print(f"[recent] ERROR {session.session_id}: {exc}", flush=True)
            continue
        frames.append(_with_keys(raw, session.session_id, session.session_date))
    incoming = (
        pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
    )
    stored = upsert_boards(
        args.recent, incoming, historical_max=historical_max
    )
    manifest = args.recent.with_name("manifest.json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "every": args.every,
                "coverage_start": coverage.isoformat(),
                "historical_max": historical_max.isoformat() if historical_max else None,
                "sessions": stored,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[recent] sessions in recent store: {stored}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

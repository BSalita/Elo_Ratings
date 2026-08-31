"""Build historical FFBridge board, player, and pair quality parquets."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from datetime import date, datetime, timezone

import polars as pl

from ffbridge_quality_pipeline import (
    DEFAULT_CUTOFF,
    DEFAULT_DISCOVERY_START,
    DEFAULT_SOURCE_DIR,
    QUALITY_BOARD_COLUMNS,
    SCHEMA_VERSION,
    audit_historical_cache,
    build_historical_fragments,
    discover_session_metadata,
    fetch_missing_artifacts,
    load_identity_map,
    load_session_metadata,
    normalize_training_parquet,
    resolve_output_dir,
    session_dates_frame,
    write_quality_artifacts,
)


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected YYYY-MM-DD, got {value!r}") from exc


def _quality_cache_is_fresh(
    output_dir: pathlib.Path,
    cutoff: date,
    max_age_hours: float,
) -> tuple[bool, str]:
    metadata_path = output_dir / "ffbridge_quality_metadata.json"
    if not metadata_path.is_file():
        return False, "metadata missing"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        generated_at = datetime.fromisoformat(
            str(metadata["generated_at"]).replace("Z", "+00:00")
        )
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        return False, f"metadata invalid: {exc}"
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=timezone.utc)
    age_hours = (
        datetime.now(timezone.utc) - generated_at.astimezone(timezone.utc)
    ).total_seconds() / 3600
    if metadata.get("cutoff") != cutoff.isoformat():
        return False, f"cutoff is {metadata.get('cutoff')!r}"
    if metadata.get("schema_version") != SCHEMA_VERSION:
        return False, (
            f"schema is {metadata.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION}"
        )
    if age_hours >= max_age_hours:
        return False, f"cache is {age_hours:.1f}h old"
    return True, f"cache is fresh ({age_hours:.1f}h old)"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=pathlib.Path,
        default=DEFAULT_SOURCE_DIR,
        help=r"Raw FFBridge data root (default: E:\bridge\data\ffbridge\data).",
    )
    parser.add_argument(
        "--training-parquet",
        type=pathlib.Path,
        default=None,
        help="Training parquet (default: SOURCE/ffbridge_training_data_df.parquet).",
    )
    parser.add_argument(
        "--cutoff",
        type=_parse_date,
        default=DEFAULT_CUTOFF,
        help="Inclusive historical cutoff (default: today).",
    )
    parser.add_argument(
        "--discover-since",
        type=_parse_date,
        default=DEFAULT_DISCOVERY_START,
        help="Discover Lancelot session metadata from this date (default: 2026-01-01).",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Artifact directory. Required unless FFBRIDGE_CACHE_DIR is set.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Print the structured cache/training audit and write nothing.",
    )
    parser.add_argument(
        "--fetch-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch audit-reported missing rankings and team scores (default: enabled).",
    )
    parser.add_argument(
        "--if-stale",
        action="store_true",
        help="Skip discovery and rebuilding when today's cache is still fresh.",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=20.0,
        help="Freshness threshold used with --if-stale (default: 20).",
    )
    parser.add_argument(
        "--fetch-workers",
        type=int,
        default=8,
        help="Concurrent team-score downloads (default: 8).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm session progress.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = datetime.now()
    started_clock = time.perf_counter()
    print(f"[quality-builder] start {started.isoformat(timespec='seconds')}", flush=True)
    try:
        source_dir = args.source_dir.resolve()
        training_path = (
            args.training_parquet.resolve()
            if args.training_parquet is not None
            else source_dir / "ffbridge_training_data_df.parquet"
        )
        output_dir = (
            resolve_output_dir(args.output_dir).resolve()
            if not args.audit_only
            else None
        )
        if args.if_stale and not args.audit_only:
            assert output_dir is not None
            fresh, reason = _quality_cache_is_fresh(
                output_dir, args.cutoff, args.max_age_hours
            )
            if fresh:
                print(f"[quality-builder] {reason}; skipping", flush=True)
                return 0
            print(f"[quality-builder] rebuilding: {reason}", flush=True)

        if not args.audit_only:
            discovered = discover_session_metadata(
                source_dir,
                start_date=args.discover_since,
                cutoff=args.cutoff,
            )
            print(
                f"[quality-builder] discovered {discovered} new session metadata file(s)",
                flush=True,
            )
        audit = audit_historical_cache(source_dir, training_path, args.cutoff)
        if args.fetch_missing and not args.audit_only:
            writes = fetch_missing_artifacts(audit, workers=args.fetch_workers)
            print(f"[quality-builder] fetched {writes} missing artifact(s)", flush=True)
            audit = audit_historical_cache(source_dir, training_path, args.cutoff)
        if args.audit_only:
            print(json.dumps(audit.to_dict(), indent=2, sort_keys=True), flush=True)
            return 0

        assert output_dir is not None
        incomplete_unsupported = [
            {
                "session_id": session.session_id,
                "reason": "Incomplete raw cache after fetch (missing ranking or team scores)",
            }
            for session in audit.sessions
            if not session.in_training and not session.complete
        ]
        if incomplete_unsupported:
            print(
                f"[quality-builder] skipping {len(incomplete_unsupported)} "
                "incomplete session(s)",
                flush=True,
            )

        metadata = load_session_metadata(source_dir, args.cutoff)
        dates = session_dates_frame(metadata)
        identity_map = load_identity_map(audit)
        phase_clock = time.perf_counter()
        board_quality = normalize_training_parquet(
            training_path,
            dates,
            identity_map=identity_map,
        )
        elapsed = time.perf_counter() - phase_clock
        if elapsed > 30:
            print(
                f"[quality-builder] normalized training parquet in {elapsed:.1f}s",
                flush=True,
            )

        fragments, unmapped, unsupported = build_historical_fragments(
            audit,
            output_dir,
            dates,
            show_progress=not args.no_progress,
        )
        if fragments:
            board_quality = pl.concat(
                [
                    board_quality.select(*QUALITY_BOARD_COLUMNS),
                    *[fragment.select(*QUALITY_BOARD_COLUMNS) for fragment in fragments],
                ],
                how="vertical",
            ).sort(["Date", "session_id", "Board", "board_id"])
        metadata_result = write_quality_artifacts(
            board_quality,
            output_dir,
            cutoff=args.cutoff,
            source_dir=source_dir,
            audit=audit,
            unmapped_seat_count=unmapped,
            unsupported_sessions=[*unsupported, *incomplete_unsupported],
        )
        print(
            f"[quality-builder] wrote {metadata_result['board_rows']} board rows, "
            f"{metadata_result['player_rows']} players, "
            f"{metadata_result['pair_rows']} pairs to {output_dir}; "
            f"skipped {metadata_result['unsupported_session_count']} "
            "sessions without usable board data",
            flush=True,
        )
        return 0
    finally:
        ended = datetime.now()
        elapsed = time.perf_counter() - started_clock
        print(
            f"[quality-builder] end {ended.isoformat(timespec='seconds')} "
            f"(elapsed {elapsed:.1f}s)",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())


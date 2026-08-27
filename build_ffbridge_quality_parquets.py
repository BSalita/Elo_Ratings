"""Build historical FFBridge board, player, and pair quality parquets."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from datetime import date, datetime

import polars as pl

from ffbridge_quality_pipeline import (
    DEFAULT_CUTOFF,
    DEFAULT_SOURCE_DIR,
    audit_historical_cache,
    build_historical_fragments,
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
        help="Inclusive historical cutoff (default: 2025-01-01).",
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
        action="store_true",
        help="Explicitly fetch audit-reported missing rankings and team scores.",
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
        audit = audit_historical_cache(source_dir, training_path, args.cutoff)
        if args.fetch_missing:
            writes = fetch_missing_artifacts(audit)
            print(f"[quality-builder] fetched {writes} missing artifact(s)", flush=True)
            audit = audit_historical_cache(source_dir, training_path, args.cutoff)
        if args.audit_only:
            print(json.dumps(audit.to_dict(), indent=2, sort_keys=True), flush=True)
            return 0

        output_dir = resolve_output_dir(args.output_dir).resolve()
        uncovered_incomplete = [
            session.session_id
            for session in audit.sessions
            if not session.in_training and not session.complete
        ]
        if uncovered_incomplete:
            raise FileNotFoundError(
                "Sessions absent from training have incomplete raw caches. "
                "Run --audit-only, then explicitly use --fetch-missing if desired. "
                f"First IDs: {uncovered_incomplete[:20]}"
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
                [board_quality, *fragments],
                how="vertical_relaxed",
            ).sort(["Date", "session_id", "Board", "board_id"])
        metadata_result = write_quality_artifacts(
            board_quality,
            output_dir,
            cutoff=args.cutoff,
            source_dir=source_dir,
            audit=audit,
            unmapped_seat_count=unmapped,
            unsupported_sessions=unsupported,
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


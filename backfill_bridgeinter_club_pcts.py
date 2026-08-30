"""Backfill Club_Scratch_Pct, Club_Handicap_Pct, and Theoretical_Rank from bridgeinter.net.

Reads an Elo results parquet, fills null club percentages for Monday Octopus,
Thursday Octopus, and Friday Simultanet, and writes the updated frame.

Usage:
    python backfill_bridgeinter_club_pcts.py --results path/to/results.parquet
    python backfill_bridgeinter_club_pcts.py --results path/to/results.parquet --date-from 2025-01-01 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

_ROOT = Path(__file__).resolve().parent
_MLBRIDGE = next(
    path
    for path in (_ROOT / "mlBridge", _ROOT.parent / "mlBridge")
    if path.is_dir()
)
if str(_MLBRIDGE.parent) not in sys.path:
    sys.path.append(str(_MLBRIDGE.parent))

from mlBridge import mlBridgeBILib as bi  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path, help="Results parquet to update")
    parser.add_argument("--output", type=Path, default=None, help="Destination parquet (default: overwrite --results)")
    parser.add_argument("--date-from", default=None)
    parser.add_argument("--date-to", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    started = datetime.now()
    print(f"start {started.isoformat(timespec='seconds')}", flush=True)

    results_path = args.results.resolve()
    if not results_path.exists():
        raise FileNotFoundError(results_path)
    df = pl.read_parquet(results_path)
    before = {
        "club_scratch": df.filter(pl.col("Club_Scratch_Pct").is_not_null()).height
        if "Club_Scratch_Pct" in df.columns
        else 0,
        "club_handicap": df.filter(pl.col("Club_Handicap_Pct").is_not_null()).height
        if "Club_Handicap_Pct" in df.columns
        else 0,
        "theoretical_rank": df.filter(pl.col("Theoretical_Rank").is_not_null()).height
        if "Theoretical_Rank" in df.columns
        else 0,
        "rows": df.height,
    }
    filled = bi.fill_missing_club_pcts(
        df,
        date_from=args.date_from,
        date_to=args.date_to,
        show_progress=True,
    )
    after = {
        "club_scratch": filled.filter(pl.col("Club_Scratch_Pct").is_not_null()).height,
        "club_handicap": filled.filter(pl.col("Club_Handicap_Pct").is_not_null()).height,
        "theoretical_rank": filled.filter(pl.col("Theoretical_Rank").is_not_null()).height,
        "rows": filled.height,
    }
    print(
        f"club_scratch {before['club_scratch']} -> {after['club_scratch']}  "
        f"club_handicap {before['club_handicap']} -> {after['club_handicap']}  "
        f"theoretical_rank {before['theoretical_rank']} -> {after['theoretical_rank']}",
        flush=True,
    )
    if args.dry_run:
        print("dry-run: parquet not written", flush=True)
    else:
        output = (args.output or results_path).resolve()
        filled.write_parquet(output)
        print(f"wrote {output}", flush=True)

    elapsed = (datetime.now() - started).total_seconds()
    print(f"end {datetime.now().isoformat(timespec='seconds')} elapsed={elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

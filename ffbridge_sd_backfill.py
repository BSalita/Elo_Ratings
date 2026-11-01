"""Re-solve FFBridge single dummy for cached deals below the current sample count.

The daily build solves new deals with SD_PRODUCTIONS samples per side. Deals
cached earlier with fewer samples keep them until this script upgrades them.
The cache is saved after every batch; rerun to resume. Do not run it while
ffbridge_all.bat runs, since both rewrite the hand-records cache.

After it finishes, delete E:\\bridge\\data\\ffbridge\\club_session_fragments
(or run a stage 2 rebuild) so the board file picks up EV from the new probabilities.

    python ffbridge_sd_backfill.py --limit 400        # timing trial
    python ffbridge_sd_backfill.py                    # everything below 40
"""
from __future__ import annotations

import argparse
import json
import pathlib
from datetime import datetime

from ffbridge_quality_pipeline import (
    DEFAULT_SOURCE_DIR,
    SD_PRODUCTIONS,
    backfill_sd_trials,
    default_hrs_cache_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=pathlib.Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--cache", type=pathlib.Path, default=None,
                        help="hand-records cache parquet (default: next to --source-dir)")
    parser.add_argument("--sd-productions", type=int, default=SD_PRODUCTIONS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-deals", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=None, help="solve at most this many deals")
    args = parser.parse_args()

    cache_path = args.cache or default_hrs_cache_path(args.source_dir)
    started = datetime.now()
    print(f"start {started:%Y-%m-%d %H:%M:%S}  cache {cache_path}", flush=True)
    summary = backfill_sd_trials(
        cache_path,
        sd_productions=args.sd_productions,
        workers=args.workers,
        batch_deals=args.batch_deals,
        limit=args.limit,
    )
    ended = datetime.now()
    print(json.dumps(summary, indent=2))
    print(f"end {ended:%Y-%m-%d %H:%M:%S}  elapsed {ended - started}", flush=True)


if __name__ == "__main__":
    main()

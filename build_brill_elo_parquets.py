"""Build or update the Brill Elo dataset (data/brill, or BRILL_ELO_CACHE_DIR).

Fetches every closed Brill event that has no fragment yet, scores its boards
and replays Elo over all events. The Streamlit app runs the same update when
its data is stale; use this for the first build or a full rebuild.

    python build_brill_elo_parquets.py            # add what is missing
    python build_brill_elo_parquets.py --rebuild  # refetch every event
    python build_brill_elo_parquets.py --limit 5  # only the 5 oldest missing events
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

from tqdm import tqdm

import elo_brill


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rebuild", action="store_true", help="discard the fragments and refetch every event")
    parser.add_argument("--limit", type=int, default=None, help="build at most this many missing events")
    args = parser.parse_args()

    started = datetime.now()
    t0 = time.time()
    print(f"Start: {started:%Y-%m-%d %H:%M:%S}  cache: {elo_brill.CACHE_DIR}", flush=True)
    metadata = elo_brill.update_dataset(
        progress=lambda items: tqdm(items, desc="Brill events", unit="event"),
        limit=args.limit,
        rebuild=args.rebuild,
    )
    print(json.dumps({k: v for k, v in metadata.items() if k != "skipped"}, indent=2))
    if metadata["skipped"]:
        print(f"Skipped {len(metadata['skipped'])} events without board results.")
    print(f"End: {datetime.now():%Y-%m-%d %H:%M:%S}  elapsed: {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

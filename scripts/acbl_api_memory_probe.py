"""Step-by-step memory probe of acbl_api_server, in process (no HTTP).

Loads a frame the way the server does, then calls the /acbl/report handler
directly, printing the frame's own size and the process's anonymous memory
after each step. Run inside the API's Linux container from /code/elo:

    python scripts/acbl_api_memory_probe.py --sequence club,club,club

See ACBL_API_MEMORY.md.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import sys
import time
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import acbl_api_server as s  # noqa: E402


def anon_gb() -> float:
    for line in pathlib.Path("/proc/self/smaps_rollup").read_text().splitlines():
        if line.startswith("Anonymous:"):
            return int(line.split()[1]) / 1024 ** 2
    raise RuntimeError("no Anonymous line")


def report(kind: str, rating_type: str, date_range: str | None) -> dict:
    return s.acbl_report(
        club_or_tournament=kind,
        rating_type=rating_type,
        top_n=100,
        min_sessions=10,
        rating_method="Latest",
        moving_avg_days=10,
        elo_rating_type="Current Rating (End of Session)",
        date_from=None,
        date_range=date_range,
        online_filter="All",
        strata="Open (unlimited)",
        prior_sessions=50,
        min_skill_z=None,
        player_name=None,
        player_number=None,
        masterpoints_range="All",
        platinum_events=False,
        foresight=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence",
        default="club",
        help="comma list of kind[:Pairs][:1y] report calls, e.g. club,tournament,club:Pairs:1y",
    )
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--dump-dir", type=pathlib.Path, help="write each report's rows as JSON here")
    args = parser.parse_args()
    started = datetime.now()
    print(f"probe started {started.isoformat(timespec='seconds')} "
          f"MALLOC_ARENA_MAX={os.getenv('MALLOC_ARENA_MAX')} "
          f"_RJEM_MALLOC_CONF={os.getenv('_RJEM_MALLOC_CONF')}", flush=True)
    print(f"start anon {anon_gb():.2f} GB", flush=True)

    first_kind = args.sequence.split(",")[0].split(":")[0]
    t0 = time.perf_counter()
    df = s._load_full_frame(first_kind)
    est = df.estimated_size() / 1024 ** 3
    chunks = {c: df[c].n_chunks() for c in df.columns[:5]}
    print(f"loaded {first_kind}: {df.height:,} rows x {df.width} cols, estimated_size {est:.2f} GB, "
          f"anon {anon_gb():.2f} GB, {time.perf_counter() - t0:.0f}s, chunks {chunks}", flush=True)
    by_col = sorted(((df[c].estimated_size(), c, str(df[c].dtype)) for c in df.columns), reverse=True)[:12]
    for size, col, dtype in by_col:
        print(f"   {size / 1024 ** 3:6.2f} GB  {col} ({dtype})", flush=True)
    del df
    gc.collect()
    s._malloc_trim()
    print(f"after gc+malloc_trim anon {anon_gb():.2f} GB", flush=True)
    if args.load_only:
        return 0

    for step in args.sequence.split(","):
        parts = step.split(":")
        kind = parts[0]
        rating_type = "Pairs" if "Pairs" in parts else "Players"
        date_range = "Last 1 year" if "1y" in parts else None
        before = anon_gb()
        t1 = time.perf_counter()
        out = report(kind, rating_type, date_range)
        print(
            f"report {step}: {time.perf_counter() - t1:.1f}s rows {out['row_count']} "
            f"anon {before:.2f} -> {anon_gb():.2f} GB  perf {out['perf']}",
            flush=True,
        )
        if args.dump_dir is not None:
            args.dump_dir.mkdir(parents=True, exist_ok=True)
            path = args.dump_dir / f"{step.replace(':', '_')}.json"
            path.write_text(json.dumps(out["rows"], sort_keys=True, default=str), encoding="utf-8")
        del out
    ended = datetime.now()
    print(f"probe ended {ended.isoformat(timespec='seconds')} "
          f"elapsed {(ended - started).total_seconds():.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

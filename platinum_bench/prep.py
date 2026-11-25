"""Build the row tables the platinum benchmark and the Elo replays read.

Writes to WORK (default D:/bridge-elo-plat):
  <side>_rows.parquet  one row per board result, in the Elo file's order (the
                       order the Elo kernels process), with the inputs the
                       kernel needs, the stored pre-session ratings, and
                       section_id for rows from EVAL_FROM on.

Run with the acbl-pipeline venv (polars 2, numba):
  acbl-pipeline/.venv/Scripts/python.exe elo/platinum_bench/prep.py --side tournament
"""
from __future__ import annotations

import argparse
import os
import pathlib
import time
from datetime import date, datetime

import polars as pl

ACBL = pathlib.Path(os.environ.get("ACBL_DIR", r"E:\bridge\data\acbl"))
WORK = pathlib.Path(os.environ.get("ELO_PLAT_WORK", r"D:\bridge-elo-plat"))
SEATS = "NESW"
KEYS = ["session_id", "Board", "Pair_Number_NS", "Pair_Number_EW", "Player_ID_N"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=("tournament", "club"), required=True)
    ap.add_argument("--section-from", default="2016-01-01",
                    help="attach section_id to rows from this date (club: 2025-01-01 keeps it small)")
    args = ap.parse_args()
    started = datetime.now()
    print(f"START {started:%Y-%m-%d %H:%M:%S} side={args.side}")
    WORK.mkdir(parents=True, exist_ok=True)
    sec_from = date.fromisoformat(args.section_from)

    elo_file = ACBL / f"acbl_{args.side}_elo_ratings.parquet"
    names = pl.scan_parquet(elo_file).collect_schema().names()
    want = [
        "Date", "session_id", "event_id", "Board", "Pair_Number_NS", "Pair_Number_EW", "Pct_NS", "MP_Top",
        "strata_bucket", "mp_color", "mp_rating", "Club",
        *[f"Player_ID_{s}" for s in SEATS],
        *[f"MasterPoints_{s}" for s in SEATS],
        *[f"Elo_R_{s}_EventStart" for s in SEATS],
        *[f"Elo_R_{s}_Before" for s in SEATS],
        *[f"Elo_N_{s}" for s in SEATS],
    ]
    cols = [c for c in want if c in names]
    print("missing (ok if club-only/tournament-only):", sorted(set(want) - set(cols)))
    t = time.perf_counter()
    rows = pl.read_parquet(elo_file, columns=cols).with_row_index("row")
    print(f"read {elo_file.name}: {rows.height:,} rows ({time.perf_counter()-t:.0f}s)")

    t = time.perf_counter()
    br = ACBL / f"acbl_{args.side}_board_results_augmented.parquet"
    sec = (
        pl.scan_parquet(br)
        .filter(pl.col("Date") >= sec_from)
        .select(*KEYS, "section_id")
        .unique(subset=KEYS, keep="none")  # drop ambiguous keys rather than guess
        .collect(engine="streaming")
    )
    print(f"section keys from {br.name}: {sec.height:,} ({time.perf_counter()-t:.0f}s)")
    # Cast keys to the Elo file's dtypes.
    sec = sec.with_columns([pl.col(k).cast(rows.schema[k]) for k in KEYS])
    rows = rows.join(sec, on=KEYS, how="left", maintain_order="left")
    got = rows.filter(pl.col("Date") >= sec_from).select(pl.col("section_id").is_not_null().mean()).item()
    print(f"section_id attached to {got:.4f} of rows since {sec_from}")
    rows = rows.sort("row").drop("row")
    out = WORK / f"{args.side}_rows.parquet"
    rows.write_parquet(out)
    print(f"wrote {out} {rows.shape}")
    ended = datetime.now()
    print(f"END {ended:%Y-%m-%d %H:%M:%S} elapsed {(ended-started).total_seconds():.0f}s")


if __name__ == "__main__":
    main()

"""Score rebuilt Elo files (a folder written by acbl_elo_ratings_create.py --out-dir)
against the current files, on the benchmark sets, using the stored
Elo_R_{seat}_EventStart columns.

  acbl-pipeline/.venv/Scripts/python.exe elo/platinum_bench/score_rebuilt.py E:/bridge/data/acbl/elo_candidate_20261125
"""
from __future__ import annotations

import json
import pathlib
import sys
from datetime import date

import numpy as np
import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import evalr  # noqa: E402
from common import PLAT_TEST_FROM, SEATS, TEST_FROM, WORK, gap_of, load_rows  # noqa: E402

KEYS = ["session_id", "Board", "Pair_Number_NS", "Pair_Number_EW", "Player_ID_N"]


def main() -> None:
    new_dir = pathlib.Path(sys.argv[1])
    out = {}
    for side in ("club", "tournament"):
        rows = load_rows(side)
        cols = KEYS + [f"Elo_R_{s}_EventStart" for s in SEATS] + [f"Elo_R_{s}_EventEnd" for s in SEATS]
        new = pl.read_parquet(new_dir / f"acbl_{side}_elo_ratings.parquet", columns=cols)
        if new.height != rows.height:
            raise RuntimeError(f"{side}: {new.height:,} rows, current file {rows.height:,}")
        same = all((new[k].cast(pl.Utf8) == rows[k].cast(pl.Utf8)).fill_null(True).all() for k in KEYS)
        if not same:
            raise RuntimeError(f"{side}: rebuilt rows are not in the current file's order")
        d = rows["Date"].to_numpy()
        hs = rows["section_id"].is_not_null().to_numpy()
        sets = {f"{side}_2025": hs & (d >= np.datetime64(date(2025, 1, 1))) & (d < np.datetime64(TEST_FROM)),
                f"{side}_2026": hs & (d >= np.datetime64(TEST_FROM))}
        if side == "tournament":
            p = rows["is_plat"].to_numpy()
            sets["Plat_2016_2024"] = hs & p & (d < np.datetime64(PLAT_TEST_FROM))
            sets["Plat_2025_2026"] = hs & p & (d >= np.datetime64(PLAT_TEST_FROM))
        g = {
            "current": gap_of({s: rows[f"Elo_R_{s}_EventStart"].cast(pl.Float64).fill_nan(None).fill_null(1500.0).to_numpy() for s in SEATS}),
            "rebuilt": gap_of({s: new[f"Elo_R_{s}_EventStart"].cast(pl.Float64).fill_nan(None).fill_null(1500.0).to_numpy() for s in SEATS}),
        }
        res = evalr.evaluate(rows, g, sets, ref="current")
        print(evalr.table(res, list(g)), flush=True)
        latest = pl.concat([new.select(pl.col(f"Elo_R_{s}_EventEnd").fill_nan(None).alias("x")) for s in SEATS]).drop_nulls()
        out[side] = {"results": res, "nan_or_null_eventstart": int(sum(new[f"Elo_R_{s}_EventStart"].fill_nan(None).null_count() for s in SEATS)),
                     "eventend_sd": float(latest["x"].std())}
        print(side, {k: v for k, v in out[side].items() if k != "results"}, flush=True)
    (WORK / "results_rebuilt.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

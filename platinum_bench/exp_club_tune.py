"""Club K / boost grid for the candidate, chosen on club 2025, reported on 2026.

  acbl-pipeline/.venv/Scripts/python.exe elo/platinum_bench/exp_club_tune.py
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from datetime import date, datetime

import numpy as np
import polars as pl

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import evalr  # noqa: E402
import kernel  # noqa: E402
from common import SEATS, TEST_FROM, WORK, gap_of, load_rows  # noqa: E402
from exp_club_joint import fit_init, slot_first_last  # noqa: E402


def main() -> None:
    started = datetime.now()
    print(f"START {started:%Y-%m-%d %H:%M:%S}", flush=True)
    df = load_rows("club")
    lay = kernel.Layout(df)
    d = df["Date"].to_numpy()
    hs = df["section_id"].is_not_null().to_numpy()
    sets = {"C2025": hs & (d >= np.datetime64(date(2025, 1, 1))) & (d < np.datetime64(TEST_FROM)),
            "C2026": hs & (d >= np.datetime64(TEST_FROM))}
    ev = sets["C2025"] | sets["C2026"]
    base_m = kernel.replay(lay, merged=True)
    slots = slot_first_last(lay, df, True, base_m)
    init, w = fit_init(slots, lay.slots[True][4], False)
    print("init weights", w, flush=True)
    g = {"base": gap_of({s: kernel.replay(lay)[f"r_{s}"].cast(pl.Float64).to_numpy()[ev] for s in SEATS})}
    for k, b, u in itertools.product((6.0, 8.0, 10.0, 12.0), (3.0, 4.0, 6.0), (100, 200)):
        t = time.perf_counter()
        r = kernel.replay(lay, merged=True, init=init, k_base=k, boost=b, boost_until=u)
        g[f"k{k:g}_b{b:g}_u{u}"] = gap_of({s: r[f"r_{s}"].cast(pl.Float64).to_numpy()[ev] for s in SEATS})
        print(f"k{k:g}_b{b:g}_u{u} {time.perf_counter()-t:.0f}s", flush=True)
    sub = df.select("session_id", "section_id", "Pair_Number_NS", "Pair_Number_EW", "Pct_NS").filter(pl.Series(ev))
    res = evalr.evaluate(sub, g, {k: v[ev] for k, v in sets.items()}, ref="base")
    ranked = sorted(g, key=lambda k: res["C2025"][k]["spearman"], reverse=True)
    md = evalr.table(res, ranked)
    print(md, flush=True)
    (WORK / "results_club_tune.md").write_text(md)
    (WORK / "results_club_tune.json").write_text(json.dumps({"results": res, "init_weights": w, "chosen": ranked[0]}, indent=1))
    ended = datetime.now()
    print(f"END {ended:%Y-%m-%d %H:%M:%S} elapsed {(ended-started).total_seconds():.0f}s")


if __name__ == "__main__":
    main()

"""Tune K and the provisional boost of the candidate (one rating per player,
start from masterpoints and platinum points) on development sets only
(tournament 2025 and platinum 2016-2024), then report the chosen setting and
platinum blends on the test sets (tournament 2026, platinum 2025-2026).

  acbl-pipeline/.venv/Scripts/python.exe elo/platinum_bench/exp_tournament_tune.py
"""
from __future__ import annotations

import itertools
import json
import sys
from datetime import date, datetime

import numpy as np
import polars as pl

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import evalr  # noqa: E402
import kernel  # noqa: E402
from common import PLAT_TEST_FROM, SEATS, TEST_FROM, WORK, gap_of, load_rows, seat_features  # noqa: E402
from exp_club_joint import first_plat, fit_init, slot_first_last  # noqa: E402


def main() -> None:
    started = datetime.now()
    print(f"START {started:%Y-%m-%d %H:%M:%S}")
    df = load_rows("tournament")
    sf = seat_features(df)
    lay = kernel.Layout(df)
    d = df["Date"].to_numpy()
    plat = df["is_plat"].to_numpy()
    hs = df["section_id"].is_not_null().to_numpy()
    pct = df["Pct_NS"].cast(pl.Float64).to_numpy()
    dev_sets = {"T2025": hs & (d >= np.datetime64(date(2025, 1, 1))) & (d < np.datetime64(TEST_FROM)),
                "Plat_2016_2024": hs & plat & (d < np.datetime64(PLAT_TEST_FROM))}
    test_sets = {"T2026": hs & (d >= np.datetime64(TEST_FROM)),
                 "Plat_2025_2026": hs & plat & (d >= np.datetime64(PLAT_TEST_FROM))}
    base_m = kernel.replay(lay, merged=True)
    slots = first_plat(slot_first_last(lay, df, True, base_m))
    init, w = fit_init(slots, lay.slots[True][4], True)
    print("init weights (log mp, log plat, const):", w)

    g = {"base": gap_of({s: kernel.replay(lay)[f"r_{s}"].cast(pl.Float64).to_numpy() for s in SEATS})}
    ks = tuple(float(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "10,12,16,20").split(","))
    bs = tuple(float(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1.5,3,4").split(","))
    us = tuple(int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "100,200,400").split(","))
    grid = list(itertools.product(ks, bs, us))
    for k, b, u in grid:
        r = kernel.replay(lay, merged=True, init=init, k_base=k, boost=b, boost_until=u)
        g[f"k{k:g}_b{b:g}_u{u}"] = gap_of({s: r[f"r_{s}"].cast(pl.Float64).to_numpy() for s in SEATS})
    res = evalr.evaluate(df, g, dev_sets, ref="base")
    score = {k: (res["T2025"][k]["spearman"] + res["Plat_2016_2024"][k]["spearman"]) / 2 for k in g}
    ranked = sorted(score, key=score.get, reverse=True)
    print("dev ranking (mean of T2025 and platinum 2016-2024):")
    for k in ranked[:10] + ["base"]:
        print(f"  {k}: T2025 {res['T2025'][k]['spearman']:.4f}  plat {res['Plat_2016_2024'][k]['spearman']:.4f}")
    best = ranked[0]
    # Blends on the chosen rating: with platinum points and platinum performance to date.
    lp = gap_of({s: np.log1p(np.nan_to_num(sf[s]["plat_pts"])) for s in SEATS})
    pf = gap_of({s: np.nan_to_num(sf[s]["plat_pct_sum"]) / (np.nan_to_num(sf[s]["plat_n"]) + 10) for s in SEATS})
    lmp = gap_of({s: np.log1p(np.nan_to_num(sf[s]["mp"], nan=0.0)) for s in SEATS})
    fit_all = (d >= np.datetime64(date(2023, 1, 1))) & (d < np.datetime64(PLAT_TEST_FROM))
    fit_plat = dev_sets["Plat_2016_2024"]
    t = {"base": g["base"], "candidate": g[best]}
    weights = {}
    for name, parts, mask in (
        ("cand+plat_allfit", [g[best], lp, pf], fit_all),
        ("cand+plat_platfit", [g[best], lp, pf], fit_plat),
        ("cand+plat+mp_allfit", [g[best], lp, pf, lmp], fit_all),
        ("cand+logplat_platfit", [g[best], lp], fit_plat),
    ):
        ww = evalr.fit_blend(parts, pct, mask)
        weights[name] = list(map(float, ww))
        t[name] = sum(a * p for a, p in zip(ww, parts))
    tres = evalr.evaluate(df, t, test_sets, ref="base")
    md = evalr.table(tres, list(t))
    print(f"chosen on dev: {best}\nblend weights (per unit gap: Elo, log plat pts, plat perf[, log mp]): {weights}")
    print(md)
    tag = "_".join(sys.argv[1:4]).replace(",", "-") if len(sys.argv) > 1 else ""
    (WORK / f"results_tournament_tune{tag}.json").write_text(json.dumps(
        {"dev": res, "test": tres, "chosen": best, "init_weights": w, "blend_weights": weights}, indent=1))
    (WORK / f"results_tournament_tune{tag}.md").write_text(f"chosen on dev: {best}\n\n" + md)
    ended = datetime.now()
    print(f"END {ended:%Y-%m-%d %H:%M:%S} elapsed {(ended-started).total_seconds():.0f}s")


if __name__ == "__main__":
    main()

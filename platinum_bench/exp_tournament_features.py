"""Ground truth 1 for today's ratings: rank pairs in tournament fields, overall
and in platinum fields, by pre-session Elo, masterpoints, platinum results to
date, and blends.

  acbl-pipeline/.venv/Scripts/python.exe elo/platinum_bench/exp_tournament_features.py
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime

import numpy as np
import polars as pl

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import evalr  # noqa: E402
from common import LINE_FROM, PLAT_TEST_FROM, SEATS, TEST_FROM, WORK, gap_of, load_rows, seat_features  # noqa: E402


def main() -> None:
    started = datetime.now()
    print(f"START {started:%Y-%m-%d %H:%M:%S}")
    df = load_rows("tournament")
    sf = seat_features(df)
    d = df["Date"].to_numpy()
    plat = df["is_plat"].to_numpy()
    has_sec = df["section_id"].is_not_null().to_numpy()
    pct = df["Pct_NS"].cast(pl.Float64).to_numpy()
    t26 = has_sec & (d >= np.datetime64(TEST_FROM))
    sets = {
        "T2026_all": t26,
        "T2026_nonplat": t26 & ~plat,
        "Plat_2016_2024": has_sec & plat & (d < np.datetime64(PLAT_TEST_FROM)),
        "Plat_2025_2026": has_sec & plat & (d >= np.datetime64(PLAT_TEST_FROM)),
    }
    fit = (d >= np.datetime64(date(2023, 1, 1))) & (d < np.datetime64(PLAT_TEST_FROM))

    def seats(f):
        return {s: f(s) for s in SEATS}

    elo = seats(lambda s: df[f"Elo_R_{s}_EventStart"].cast(pl.Float64).fill_null(1500.0).to_numpy())
    mp_med = float(np.nanmedian(np.concatenate([sf[s]["mp"] for s in SEATS])))
    g = {}
    g["elo"] = gap_of(elo)
    n = seats(lambda s: np.nan_to_num(sf[s]["n_prior"], nan=0.0))
    g["elo_shrunk50"] = gap_of({s: (n[s] * elo[s] + 50 * 1498.5) / (n[s] + 50) for s in SEATS})
    g["log_mp"] = gap_of(seats(lambda s: np.log1p(np.nan_to_num(sf[s]["mp"], nan=mp_med))))
    g["log_plat_pts"] = gap_of(seats(lambda s: np.log1p(np.nan_to_num(sf[s]["plat_pts"]))))
    g["log_gold_pts"] = gap_of(seats(lambda s: np.log1p(np.nan_to_num(sf[s]["gold_pts"]))))
    g["plat_rate"] = gap_of(seats(lambda s: np.nan_to_num(sf[s]["plat_pairs_pts"]) / (np.nan_to_num(sf[s]["plat_n"]) + 5)))
    g["plat_perf"] = gap_of(seats(lambda s: np.nan_to_num(sf[s]["plat_pct_sum"]) / (np.nan_to_num(sf[s]["plat_n"]) + 10)))
    blends = {
        "blend_elo_mp": ["elo", "log_mp"],
        "blend_elo_plat": ["elo", "log_plat_pts"],
        "blend_elo_mp_plat": ["elo", "log_mp", "log_plat_pts"],
        "blend_all": ["elo", "log_mp", "log_plat_pts", "log_gold_pts", "plat_rate", "plat_perf"],
    }
    weights = {}
    for name, parts in blends.items():
        w = evalr.fit_blend([g[p] for p in parts], pct, fit)
        weights[name] = dict(zip(parts, map(float, w)))
        g[name] = sum(wi * g[p] for wi, p in zip(w, parts))
    # Fitted on platinum fields only (2016-2024): does a strong field weigh the signals differently?
    pdev = sets["Plat_2016_2024"]
    w = evalr.fit_blend([g[p] for p in blends["blend_all"]], pct, pdev)
    weights["blend_all_platfit"] = dict(zip(blends["blend_all"], map(float, w)))
    g["blend_all_platfit"] = sum(wi * g[p] for wi, p in zip(w, blends["blend_all"]))
    print("blend weights (Pct per unit gap):", json.dumps(weights, indent=1))
    a, b = evalr.fit_line(g["elo"], pct, (d >= np.datetime64(LINE_FROM)) & (d < np.datetime64(TEST_FROM)))
    print(f"Elo line fitted on 2025: intercept {a:.4f}, slope {b*100:.2f} points per 100 Elo")

    res = evalr.evaluate(df, g, sets, ref="elo")
    names = list(g)
    md = evalr.table(res, names)
    print(md)
    out = WORK / "results_tournament_features.json"
    out.write_text(json.dumps({"results": res, "weights": weights, "slope_per_100": b * 100}, indent=1))
    (WORK / "results_tournament_features.md").write_text(md)
    ended = datetime.now()
    print(f"END {ended:%Y-%m-%d %H:%M:%S} elapsed {(ended-started).total_seconds():.0f}s")


if __name__ == "__main__":
    main()

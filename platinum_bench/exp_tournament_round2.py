"""Round 2 on the tournament stream: combine the round-1 winners (one rating
per player, start from masterpoints and platinum points) with lower K, a
longer provisional boost, and mean reversion toward an informed anchor that
moves with masterpoints and platinum/gold points to date (team events
included). Post-hoc shrinkage toward that anchor is scored as well.

  acbl-pipeline/.venv/Scripts/python.exe elo/platinum_bench/exp_tournament_round2.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date, datetime

import numpy as np
import polars as pl

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import evalr  # noqa: E402
import kernel  # noqa: E402
from common import PLAT_TEST_FROM, SEATS, TEST_FROM, WORK, gap_of, load_rows, seat_features  # noqa: E402
from exp_tournament_kernel import fit_anchor  # noqa: E402


def anchor_features(sf, s):
    return np.column_stack([
        np.log1p(np.nan_to_num(sf[s]["mp"], nan=0.0)),
        np.log1p(np.nan_to_num(sf[s]["plat_pts"])),
        np.log1p(np.nan_to_num(sf[s]["gold_pts"])),
    ])


def fit_dynamic_anchor(r, sf, mask):
    xs, ys = [], []
    for s in SEATS:
        m = mask & (r[f"n_{s}"].to_numpy() >= 50)
        xs.append(anchor_features(sf, s)[m])
        ys.append(r[f"r_{s}"].to_numpy()[m])
    X = np.vstack(xs)
    w, *_ = np.linalg.lstsq(np.column_stack([X, np.ones(len(X))]), np.concatenate(ys), rcond=None)
    return w


def main() -> None:
    started = datetime.now()
    print(f"START {started:%Y-%m-%d %H:%M:%S}")
    df = load_rows("tournament")
    sf = seat_features(df)
    lay = kernel.Layout(df)
    d = df["Date"].to_numpy()
    plat = df["is_plat"].to_numpy()
    has_sec = df["section_id"].is_not_null().to_numpy()
    pct = df["Pct_NS"].cast(pl.Float64).to_numpy()
    t26 = has_sec & (d >= np.datetime64(TEST_FROM))
    sets = {
        "T2026_all": t26,
        "Plat_2016_2024": has_sec & plat & (d < np.datetime64(PLAT_TEST_FROM)),
        "Plat_2025_2026": has_sec & plat & (d >= np.datetime64(PLAT_TEST_FROM)),
    }
    dev = d < np.datetime64(PLAT_TEST_FROM)
    null_id = {s: df[f"Player_ID_{s}"].is_null().to_numpy() for s in SEATS}

    base = kernel.replay(lay)
    merged = kernel.replay(lay, merged=True)
    w0 = fit_anchor(df, merged, sf, dev, True)
    mp1 = lay.first_values(True, {s: np.log1p(np.nan_to_num(sf[s]["mp"], nan=0.0)) for s in SEATS})
    p1 = lay.first_values(True, {s: np.log1p(np.nan_to_num(sf[s]["plat_pts"])) for s in SEATS})
    init = w0[0] * np.nan_to_num(mp1) + w0[1] * np.nan_to_num(p1) + w0[2]
    M = dict(merged=True, init=init)
    rm = kernel.replay(lay, **M)
    wd = fit_dynamic_anchor(rm, sf, dev)
    print("start anchor", w0, "dynamic anchor", wd)
    anc = {}
    for s in SEATS:
        a = anchor_features(sf, s) @ wd[:-1] + wd[-1]
        a[null_id[s]] = np.nan
        anc[s] = a.astype(np.float32)

    variants = {
        "base": (dict(), None),
        "M": (M, rm),
        "M_k16": ({**M, "k_base": 16.0}, None),
        "M_k20": ({**M, "k_base": 20.0}, None),
        "M_boost3u200": ({**M, "boost": 3.0, "boost_until": 200}, None),
        "M_k16_boost3u200": ({**M, "k_base": 16.0, "boost": 3.0, "boost_until": 200}, None),
        "M_rev0.01": ({**M, "anchor": anc, "alpha": 0.01}, None),
        "M_rev0.03": ({**M, "anchor": anc, "alpha": 0.03}, None),
        "M_rev0.1": ({**M, "anchor": anc, "alpha": 0.1}, None),
        "M_k16_rev0.03": ({**M, "k_base": 16.0, "anchor": anc, "alpha": 0.03}, None),
    }
    g, info, reps = {}, {}, {}
    for name, (kw, pre) in variants.items():
        t = time.perf_counter()
        r = pre if pre is not None else (base if name == "base" else kernel.replay(lay, **kw))
        reps[name] = r
        g[name] = gap_of({s: r[f"r_{s}"].cast(pl.Float64).to_numpy() for s in SEATS})
        a, b = evalr.fit_line(g[name], pct, (d >= np.datetime64(date(2025, 1, 1))) & (d < np.datetime64(TEST_FROM)))
        sd = float(np.std(np.concatenate([r[f"r_{s}"].to_numpy()[t26] for s in SEATS])))
        info[name] = {"player_sd_2026": sd, "board_pct_per_100": b * 1e4}
        print(f"{name}: {time.perf_counter()-t:.0f}s sd={sd:.1f} slope={b*1e4:.2f}")
    # Post-hoc shrinkage of the M ratings: today's published form (global anchor, 50
    # sessions) and toward the dynamic anchor.
    for src in ("base", "M", "M_rev0.03"):
        r = reps[src]
        for P, kind in ((50, "global"), (5, "anchor"), (20, "anchor")):
            vals = {}
            for s in SEATS:
                x = r[f"r_{s}"].cast(pl.Float64).to_numpy()
                n = r[f"n_{s}"].cast(pl.Float64).to_numpy()
                tgt = 1498.5 if kind == "global" else np.nan_to_num(anc[s].astype(np.float64), nan=1500.0)
                vals[s] = (n * x + P * tgt) / (n + P)
            g[f"{src}_shrink_{kind}{P}"] = gap_of(vals)
    g["anchor_only"] = gap_of({s: np.nan_to_num(anc[s].astype(np.float64), nan=1500.0) for s in SEATS})

    res = evalr.evaluate(df, g, sets, ref="base")
    md = evalr.table(res, list(g))
    print(md)
    (WORK / "results_tournament_round2.json").write_text(json.dumps(
        {"results": res, "info": info, "start_anchor": list(map(float, w0)), "dynamic_anchor": list(map(float, wd))}, indent=1))
    (WORK / "results_tournament_round2.md").write_text(md)
    np.save(WORK / "tourn_dynamic_anchor_w.npy", wd)
    ended = datetime.now()
    print(f"END {ended:%Y-%m-%d %H:%M:%S} elapsed {(ended-started).total_seconds():.0f}s")


if __name__ == "__main__":
    main()

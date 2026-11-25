"""Elo formula variants replayed on the tournament stream, scored on the
overall 2026 rank eval and on platinum fields.

  acbl-pipeline/.venv/Scripts/python.exe elo/platinum_bench/exp_tournament_kernel.py [--only name,name]
"""
from __future__ import annotations

import argparse
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

COLOUR = {"platinum": "plat", "gold": "gold", "red": "red", "silver": "silver"}


def colour_kmul(df: pl.DataFrame, mult: dict[str, float]) -> np.ndarray:
    c = df["mp_color"].str.to_lowercase().fill_null("none")
    m = pl.Series([mult.get(COLOUR.get(x, "none"), 1.0) for x in c.unique().to_list()], dtype=pl.Float32)
    return c.replace_strict(c.unique(), m, return_dtype=pl.Float32).to_numpy()


def fit_anchor(df, r, sf, fit_mask, use_plat: bool):
    """Regress established ratings on log masterpoints (and log platinum to date)."""
    xs, ys = [], []
    for s in SEATS:
        m = fit_mask & (r[f"n_{s}"].to_numpy() >= 50)
        mp = np.log1p(np.nan_to_num(sf[s]["mp"][m], nan=0.0))
        cols = [mp]
        if use_plat:
            cols.append(np.log1p(np.nan_to_num(sf[s]["plat_pts"][m])))
        xs.append(np.column_stack(cols))
        ys.append(r[f"r_{s}"].to_numpy()[m])
    X = np.vstack(xs)
    y = np.concatenate(ys)
    X1 = np.column_stack([X, np.ones(len(y))])
    w, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
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
    fit = (d >= np.datetime64(date(2023, 1, 1))) & (d < np.datetime64(PLAT_TEST_FROM))
    anchor_fit = d < np.datetime64(PLAT_TEST_FROM)
    mp_seat = {s: np.nan_to_num(sf[s]["mp"], nan=0.0) for s in SEATS}
    pl_seat = {s: np.nan_to_num(sf[s]["plat_pts"]) for s in SEATS}
    log_mp_gap = gap_of({s: np.log1p(mp_seat[s]) for s in SEATS})

    base = {False: kernel.replay(lay), True: kernel.replay(lay, merged=True)}
    anchors = {}
    for merged in (False, True):
        for use_plat in (False, True):
            w = fit_anchor(df, base[merged], sf, anchor_fit, use_plat)
            anchors[(merged, use_plat)] = w
            print(f"anchor merged={merged} plat={use_plat}: {w}")

    def init_for(merged: bool, use_plat: bool, lam: float = 1.0) -> np.ndarray:
        w = anchors[(merged, use_plat)]
        mp1 = lay.first_values(merged, {s: np.log1p(mp_seat[s]) for s in SEATS})
        v = w[0] * np.nan_to_num(mp1, nan=0.0)
        if use_plat:
            p1 = lay.first_values(merged, {s: np.log1p(pl_seat[s]) for s in SEATS})
            v = v + w[1] * np.nan_to_num(p1, nan=0.0)
        v = v + w[-1]
        return 1500.0 + lam * (v - 1500.0)

    variants = {
        "base": dict(),
        "merged": dict(merged=True),
        "init_mp": dict(init=("sep", False, 1.0)),
        "init_mp_plat": dict(init=("sep", True, 1.0)),
        "init_mp_plat_x2": dict(init=("sep", True, 2.0)),
        "merged_init_mp_plat": dict(merged=True, init=("mer", True, 1.0)),
        "colour_plat2": dict(colour={"plat": 2.0}),
        "colour_plat3": dict(colour={"plat": 3.0}),
        "colour_plat2_gold1.25": dict(colour={"plat": 2.0, "gold": 1.25}),
        "colour_plat0.5": dict(colour={"plat": 0.5}),
        "k16": dict(k_base=16.0),
        "k32": dict(k_base=32.0),
        "k48": dict(k_base=48.0),
        "boost3_until200": dict(boost=3.0, boost_until=200),
        "scale200": dict(elo_scale=200.0),
    }
    if args.only:
        keep = args.only.split(",")
        variants = {k: v for k, v in variants.items() if k in keep or k == "base"}
    g, info = {}, {}
    for name, v in variants.items():
        t = time.perf_counter()
        v = dict(v)
        kw = {}
        merged = v.pop("merged", False)
        if "init" in v:
            _, use_plat, lam = v.pop("init")
            kw["init"] = init_for(merged, use_plat, lam)
        if "colour" in v:
            kw["kmul"] = colour_kmul(df, v.pop("colour"))
        r = kernel.replay(lay, merged=merged, **kw, **v)
        g[name] = gap_of({s: r[f"r_{s}"].cast(pl.Float64).to_numpy() for s in SEATS})
        w = evalr.fit_blend([g[name], log_mp_gap], pct, fit)
        g[name + "+mp"] = w[0] * g[name] + w[1] * log_mp_gap
        sd = float(np.std(np.concatenate([r[f"r_{s}"].to_numpy()[t26] for s in SEATS])))
        a, b = evalr.fit_line(g[name], pct, (d >= np.datetime64(date(2025, 1, 1))) & (d < np.datetime64(TEST_FROM)))
        info[name] = {"player_sd_2026": sd, "board_pct_per_100": b * 100 * 100, "mp_weight_ratio": float(w[1] / w[0])}
        print(f"{name}: replay+fit {time.perf_counter()-t:.0f}s  sd={sd:.1f}  slope={b*1e4:.2f} pts/100")
    res = evalr.evaluate(df, g, sets, ref="base")
    md = evalr.table(res, list(g))
    print(md)
    tag = "_" + args.only.replace(",", "_") if args.only else ""
    (WORK / f"results_tournament_kernel{tag}.json").write_text(json.dumps({"results": res, "info": info,
        "anchors": {str(k): list(map(float, w)) for k, w in anchors.items()}}, indent=1))
    (WORK / f"results_tournament_kernel{tag}.md").write_text(md)
    ended = datetime.now()
    print(f"END {ended:%Y-%m-%d %H:%M:%S} elapsed {(ended-started).total_seconds():.0f}s")


if __name__ == "__main__":
    main()

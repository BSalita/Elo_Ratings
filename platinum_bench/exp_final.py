"""Final comparison: joint club + tournament ratings at the dev-chosen K and
boost settings, the tournament-only candidate, and platinum-field blends,
scored on the dev and test sets of both streams. Also writes the per-player
latest ratings for the top-100 checks.

  acbl-pipeline/.venv/Scripts/python.exe elo/platinum_bench/exp_final.py
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
from exp_club_joint import JOINT_COLS, first_plat, fit_init, slot_first_last  # noqa: E402

JOINT_CONFIGS = {
    "J_k12_b3_u200": dict(k_base=12.0, boost=3.0, boost_until=200),
    "J_k10_b4_u100": dict(k_base=10.0, boost=4.0, boost_until=100),
    "J_k8_b4_u200": dict(k_base=8.0, boost=4.0, boost_until=200),
    "J_k6_b6_u200": dict(k_base=6.0, boost=6.0, boost_until=200),
}
TOURN_CONFIG = dict(k_base=6.0, boost=6.0, boost_until=200)


def joint_frame():
    club, tourn = load_rows("club"), load_rows("tournament")
    parts = []
    for x, p in ((club, "C"), (tourn, "T")):
        if "mp_color" not in x.columns:
            x = x.with_columns(pl.lit(None, pl.Utf8).alias("mp_color"))
        parts.append(x.with_row_index("src_row").select(["src_row", *JOINT_COLS]).with_columns(
            (pl.lit(p) + pl.col("session_id").cast(pl.Utf8)).alias("session_id"),
            *[pl.col(f"Player_ID_{s}").cast(pl.Utf8) for s in SEATS],
            pl.col("Board").cast(pl.Int32), pl.col("Pct_NS").cast(pl.Float32), pl.col("MP_Top").cast(pl.Float32),
            *[pl.col(f"MasterPoints_{s}").cast(pl.Float64) for s in SEATS],
            pl.col("Pair_Number_NS").cast(pl.Utf8), pl.col("Pair_Number_EW").cast(pl.Utf8),
            pl.col("section_id").cast(pl.Utf8), pl.col("strata_bucket").cast(pl.Utf8),
            pl.col("Elo_R_N_Before").cast(pl.Float32), pl.col("Elo_R_E_Before").cast(pl.Float32),
            pl.lit(p == "T").alias("is_tourn")))
    j = pl.concat(parts).sort("Date", maintain_order=True)
    j = j.with_columns((pl.col("mp_color").str.to_lowercase() == "platinum").fill_null(False).alias("is_plat"))
    return j, tourn


def main() -> None:
    started = datetime.now()
    print(f"START {started:%Y-%m-%d %H:%M:%S}", flush=True)
    joint, tourn = joint_frame()
    it = joint["is_tourn"].to_numpy()
    d = joint["Date"].to_numpy()
    hs = joint["section_id"].is_not_null().to_numpy()
    p = joint["is_plat"].to_numpy()
    pct = joint["Pct_NS"].cast(pl.Float64).to_numpy()
    y25 = (d >= np.datetime64(date(2025, 1, 1))) & (d < np.datetime64(TEST_FROM))
    y26 = d >= np.datetime64(TEST_FROM)
    sets = {"C2025": hs & ~it & y25, "C2026": hs & ~it & y26, "T2025": hs & it & y25, "T2026": hs & it & y26,
            "Plat_2016_2024": hs & p & (d < np.datetime64(PLAT_TEST_FROM)),
            "Plat_2025_2026": hs & p & (d >= np.datetime64(PLAT_TEST_FROM))}
    ev = np.zeros(len(d), bool)
    for m in sets.values():
        ev |= m
    tpos = np.flatnonzero(it)  # joint positions of tournament rows
    tsrc = joint["src_row"].to_numpy()[tpos]  # their tournament row numbers

    g = {}
    t = time.perf_counter()
    jl = kernel.Layout(joint)
    print(f"joint layout {time.perf_counter()-t:.0f}s", flush=True)
    jb = kernel.replay(jl)
    g["base_joint"] = gap_of({s: jb[f"r_{s}"].cast(pl.Float64).to_numpy() for s in SEATS})
    jm = kernel.replay(jl, merged=True)
    jslots = first_plat(slot_first_last(jl, joint, True, jm))
    jinit, jw = fit_init(jslots, jl.slots[True][4], True)
    print("joint init weights", jw, flush=True)
    latest = {}
    for name, kw in JOINT_CONFIGS.items():
        t = time.perf_counter()
        r = kernel.replay(jl, merged=True, init=jinit, **kw)
        g[name] = gap_of({s: r[f"r_{s}"].cast(pl.Float64).to_numpy() for s in SEATS})
        sd = float(np.std(np.concatenate([r[f"r_{s}"].to_numpy()[ev] for s in SEATS])))
        a, b = evalr.fit_line(g[name], pct, hs & y25)
        print(f"{name}: {time.perf_counter()-t:.0f}s sd={sd:.1f} slope={b*1e4:.2f} board-% points per 100", flush=True)
        latest[name] = r
    del jl

    # Tournament-only: production formula and the candidate, placed in joint row positions.
    tl = kernel.Layout(tourn)

    def full(arr: np.ndarray) -> np.ndarray:
        out = np.full(len(d), np.nan)
        out[tpos] = arr[tsrc]
        return out
    tb = kernel.replay(tl)
    g["base_tourn_only"] = full(gap_of({s: tb[f"r_{s}"].cast(pl.Float64).to_numpy() for s in SEATS}))
    tm = kernel.replay(tl, merged=True)
    tslots = first_plat(slot_first_last(tl, tourn, True, tm))
    tinit, tw = fit_init(tslots, tl.slots[True][4], True)
    tc = kernel.replay(tl, merged=True, init=tinit, **TOURN_CONFIG)
    g["T_k6_b6_u200"] = full(gap_of({s: tc[f"r_{s}"].cast(pl.Float64).to_numpy() for s in SEATS}))

    # Platinum-field blends, fitted on platinum 2016-2024 only.
    sf = seat_features(tourn)
    lp = full(gap_of({s: np.log1p(np.nan_to_num(sf[s]["plat_pts"])) for s in SEATS}))
    best_j = "J_k8_b4_u200"
    fitm = sets["Plat_2016_2024"]
    weights = {}
    for name, parts in (("J+logplat_platfit", [g[best_j], lp]), ("T+logplat_platfit", [g["T_k6_b6_u200"], lp]),
                        ("J+T_platfit", [g[best_j], g["T_k6_b6_u200"]]),
                        ("J+T+logplat_platfit", [g[best_j], g["T_k6_b6_u200"], lp])):
        w = evalr.fit_blend(parts, pct, fitm)
        weights[name] = list(map(float, w))
        g[name] = sum(a * q for a, q in zip(w, parts))
    # Tournament gaps are NaN on club rows; club sets only read joint predictors.
    for k in ("base_tourn_only", "T_k6_b6_u200", "T+logplat_platfit", "J+T_platfit", "J+T+logplat_platfit", "J+logplat_platfit"):
        g[k] = np.where(np.isnan(g[k]) & ~it, 0.0, g[k])
    sub = joint.select("session_id", "section_id", "Pair_Number_NS", "Pair_Number_EW", "Pct_NS").filter(pl.Series(ev))
    res = evalr.evaluate(sub, {k: v[ev] for k, v in g.items()}, {k: v[ev] for k, v in sets.items()}, ref="base_joint")
    md = evalr.table(res, list(g))
    print(f"blend weights: {weights}\n{md}", flush=True)
    (WORK / "results_final.md").write_text(md)
    (WORK / "results_final.json").write_text(json.dumps({"results": res, "weights": weights,
                                                          "joint_init": jw, "tourn_init": tw}, indent=1))

    # Latest rating per player (joint best and tournament candidate) for the top-100 checks.
    def latest_by_player(df, r):
        parts = [pl.DataFrame({"Player_ID": df[f"Player_ID_{s}"].cast(pl.Utf8), "Date": df["Date"],
                               "x": r[f"x_{s}"], "n": r[f"n_{s}"] + 1}) for s in SEATS]
        return pl.concat(parts).drop_nulls("Player_ID").sort("Date").group_by("Player_ID").agg(
            pl.col("x").last(), pl.col("n").max())
    lj = latest_by_player(joint, latest[best_j]).rename({"x": "elo_joint", "n": "n_joint"})
    lt = latest_by_player(tourn, tc).rename({"x": "elo_tourn_cand", "n": "n_tourn_cand"})
    lj.join(lt, on="Player_ID", how="full", coalesce=True).write_parquet(WORK / "latest_candidates.parquet")
    ended = datetime.now()
    print(f"END {ended:%Y-%m-%d %H:%M:%S} elapsed {(ended-started).total_seconds():.0f}s")


if __name__ == "__main__":
    main()

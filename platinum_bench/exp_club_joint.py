"""Club stream variants (the acceptance test) and a joint club + tournament
rating (one rating per player fed by both streams).

Starting ratings ("init") are fitted per player slot: the slot's final rating
(slots with 50+ sessions that first appeared before 2025) regressed on log
masterpoints and log platinum points to date at the slot's first appearance.

  acbl-pipeline/.venv/Scripts/python.exe elo/platinum_bench/exp_club_joint.py [--skip-joint] [--skip-club]
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
from common import PLAT_TEST_FROM, SEATS, TEST_FROM, WORK, gap_of, load_rows  # noqa: E402

DEV_FROM = date(2025, 1, 1)
JOINT_COLS = ["Date", "session_id", "Board", "Pct_NS", "MP_Top", "strata_bucket", "mp_color",
              *[f"Player_ID_{s}" for s in SEATS], *[f"MasterPoints_{s}" for s in SEATS],
              "Elo_R_N_Before", "Elo_R_E_Before", "Pair_Number_NS", "Pair_Number_EW", "section_id"]


def slot_first_last(lay: kernel.Layout, df: pl.DataFrame, merged: bool, r: pl.DataFrame):
    """Per slot: Player_ID, first date, first log masterpoints, last rating, final session count."""
    a, b, c, dd, n_slots, _ = lay.slots[merged]
    rank = np.empty(lay.n, dtype=np.int64)
    rank[lay.proc_order] = np.arange(lay.n)
    date_i = df["Date"].cast(pl.Int32).to_numpy()
    frames = []
    for s, idx in zip("NSEW", (a, b, c, dd)):
        frames.append(pl.DataFrame({
            "slot": idx, "rank": rank, "pid": df[f"Player_ID_{s}"].cast(pl.Utf8),
            "date": date_i, "mp": df[f"MasterPoints_{s}"].cast(pl.Float64),
            "x": r[f"x_{s}"].cast(pl.Float64), "n": r[f"n_{s}"].cast(pl.Int64),
        }))
    # Rating and session count as of the end of the development period, so the
    # fitted starting-rating weights never see the 2025-2026 test results.
    dev_i = (DEV_FROM - date(1970, 1, 1)).days
    before = pl.col("date") < dev_i
    g = (pl.concat(frames).sort("rank").group_by("slot")
         .agg(pl.col("pid").first(), pl.col("date").first().alias("first_date"),
              pl.col("mp").drop_nulls().first().alias("mp_first"),
              pl.col("x").filter(before).last().alias("x_last"),
              (pl.col("n").filter(before).last() + 1).alias("n_final")))
    return g.sort("slot")


def first_plat(slots: pl.DataFrame) -> pl.DataFrame:
    aw = pl.read_parquet(WORK / "awards_cum.parquet").select(
        pl.col("Player_ID").alias("pid"), pl.col("Date").alias("d"), "plat_pts").sort("d")
    q = slots.select("slot", "pid", pl.col("first_date").cast(pl.Date).alias("d")).sort("d")
    j = q.join_asof(aw, on="d", by="pid", strategy="backward", allow_exact_matches=False)
    return slots.join(j.select("slot", "plat_pts"), on="slot", how="left").with_columns(
        pl.col("plat_pts").fill_null(0.0))


def fit_init(slots: pl.DataFrame, n_slots: int, use_plat: bool) -> tuple[np.ndarray, list[float]]:
    x = [np.log1p(slots["mp_first"].fill_null(0.0).to_numpy())]
    if use_plat:
        x.append(np.log1p(slots["plat_pts"].to_numpy()))
    X = np.column_stack(x + [np.ones(slots.height)])
    fit = ((slots["n_final"].fill_null(0) >= 50) &(slots["first_date"] < (DEV_FROM - date(1970, 1, 1)).days)).to_numpy()
    fit &= slots["pid"].is_not_null().to_numpy()
    w, *_ = np.linalg.lstsq(X[fit], slots["x_last"].to_numpy()[fit], rcond=None)
    init = np.full(n_slots, 1500.0)
    init[slots["slot"].to_numpy()] = X @ w
    return init, list(map(float, w))


def run_stream(tag: str, df: pl.DataFrame, sets: dict, variants: dict, use_plat_init: bool) -> dict:
    t = time.perf_counter()
    lay = kernel.Layout(df)
    print(f"[{tag}] layout {time.perf_counter()-t:.0f}s rows={lay.n:,}")
    base_m = kernel.replay(lay, merged=True)
    slots = slot_first_last(lay, df, True, base_m)
    if use_plat_init:
        slots = first_plat(slots)
    inits = {}
    for up in ((False, True) if use_plat_init else (False,)):
        inits[up], w = fit_init(slots, lay.slots[True][4], up)
        print(f"[{tag}] init weights plat={up}: {w}")
    g, info = {}, {}
    ev_mask = np.zeros(lay.n, dtype=bool)
    for m in sets.values():
        ev_mask |= m
    for name, kw in variants.items():
        t = time.perf_counter()
        kw = dict(kw)
        if "init" in kw:
            kw["init"] = inits[kw["init"]]
        r = base_m if (kw == {"merged": True}) else kernel.replay(lay, **kw)
        g[name] = gap_of({s: r[f"r_{s}"].cast(pl.Float64).to_numpy() for s in SEATS})
        sd = float(np.std(np.concatenate([r[f"r_{s}"].to_numpy()[ev_mask] for s in SEATS])))
        info[name] = {"player_sd_eval": sd}
        print(f"[{tag}] {name}: {time.perf_counter()-t:.0f}s sd={sd:.1f}")
    # Evaluate only on rows of the sets (smaller frame).
    sub = df.select("session_id", "section_id", "Pair_Number_NS", "Pair_Number_EW", "Pct_NS").filter(pl.Series(ev_mask))
    gs = {k: v[ev_mask] for k, v in g.items()}
    ss = {k: v[ev_mask] for k, v in sets.items()}
    res = evalr.evaluate(sub, gs, ss, ref="base")
    md = evalr.table(res, list(g))
    print(md)
    (WORK / f"results_{tag}.md").write_text(md)
    (WORK / f"results_{tag}.json").write_text(json.dumps({"results": res, "info": info}, indent=1))
    return res


VARIANTS = {
    "base": {},
    "merged": {"merged": True},
    "k16": {"k_base": 16.0},
    "merged_init": {"merged": True, "init": False},
    "M_k16": {"merged": True, "init": False, "k_base": 16.0},
    "M_k16_boost3u200": {"merged": True, "init": False, "k_base": 16.0, "boost": 3.0, "boost_until": 200},
    "M_k12_boost3u200": {"merged": True, "init": False, "k_base": 12.0, "boost": 3.0, "boost_until": 200},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-club", action="store_true")
    ap.add_argument("--skip-joint", action="store_true")
    args = ap.parse_args()
    started = datetime.now()
    print(f"START {started:%Y-%m-%d %H:%M:%S}")
    club = load_rows("club")

    def dsets(df, prefix, plat_sets: bool):
        d = df["Date"].to_numpy()
        hs = df["section_id"].is_not_null().to_numpy()
        out = {f"{prefix}2025": hs & (d >= np.datetime64(DEV_FROM)) & (d < np.datetime64(TEST_FROM)),
               f"{prefix}2026": hs & (d >= np.datetime64(TEST_FROM))}
        if plat_sets:
            p = df["is_plat"].to_numpy()
            out["Plat_2016_2024"] = hs & p & (d < np.datetime64(PLAT_TEST_FROM))
            out["Plat_2025_2026"] = hs & p & (d >= np.datetime64(PLAT_TEST_FROM))
        return out

    if not args.skip_club:
        v = dict(VARIANTS)
        v["merged_init_plat"] = {"merged": True, "init": True}
        v["M_k16_boost3u200_plat"] = {"merged": True, "init": True, "k_base": 16.0, "boost": 3.0, "boost_until": 200}
        run_stream("club", club, dsets(club, "C", False), v, True)
    if not args.skip_joint:
        tourn = load_rows("tournament")
        parts = []
        for df, p in ((club, "C"), (tourn, "T")):
            parts.append(df.with_columns(pl.lit(None, pl.Utf8).alias("mp_color")) if "mp_color" not in df.columns else df)
        parts = [
            x.select([pl.col(c) for c in JOINT_COLS]).with_columns(
                (pl.lit(p) + pl.col("session_id").cast(pl.Utf8)).alias("session_id"),
                *[pl.col(f"Player_ID_{s}").cast(pl.Utf8) for s in SEATS],
                pl.col("Board").cast(pl.Int32), pl.col("Pct_NS").cast(pl.Float32), pl.col("MP_Top").cast(pl.Float32),
                *[pl.col(f"MasterPoints_{s}").cast(pl.Float64) for s in SEATS],
                pl.col("Pair_Number_NS").cast(pl.Utf8), pl.col("Pair_Number_EW").cast(pl.Utf8),
                pl.col("section_id").cast(pl.Utf8), pl.col("strata_bucket").cast(pl.Utf8),
                pl.col("Elo_R_N_Before").cast(pl.Float32), pl.col("Elo_R_E_Before").cast(pl.Float32),
            ).with_columns(pl.lit(p == "T").alias("is_tourn"))
            for x, p in zip(parts, ("C", "T"))
        ]
        joint = pl.concat(parts).sort("Date", maintain_order=True)
        joint = joint.with_columns((pl.col("mp_color").str.to_lowercase() == "platinum").fill_null(False).alias("is_plat"))
        del club, tourn
        it = joint["is_tourn"].to_numpy()
        sets = {}
        for k, m in dsets(joint, "J", True).items():
            if k.startswith("J"):
                sets["C" + k[1:]] = m & ~it
                sets["T" + k[1:]] = m & it
            else:
                sets[k] = m & it
        v = dict(VARIANTS)
        v["merged_init_plat"] = {"merged": True, "init": True}
        v["M_k16_boost3u200_plat"] = {"merged": True, "init": True, "k_base": 16.0, "boost": 3.0, "boost_until": 200}
        run_stream("joint", joint, sets, v, True)
    ended = datetime.now()
    print(f"END {ended:%Y-%m-%d %H:%M:%S} elapsed {(ended-started).total_seconds():.0f}s")


if __name__ == "__main__":
    main()

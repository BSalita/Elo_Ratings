"""Confirmation run of the recommended rating, with the starting-rating weights
fitted only on ratings before 2025.

Recommended: one rating per player, fed by club and tournament results
together, new players start from log masterpoints and log platinum points,
K 6 with a 6x provisional boost for the first 200 boards.

Reports
  1. ground truth 1 (section rank Spearman) on club/tournament 2025 and 2026
     and platinum 2016-2024 / 2025-2026, as an ablation ladder against
     production (today's formula, club and tournament rated separately);
  2. ground truth 2 (rating before T against platinum results after T);
  3. agreement with platinum points among established players today;
  4. spread (SD) and board-% per 100 points.

  acbl-pipeline/.venv/Scripts/python.exe elo/platinum_bench/exp_confirm.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date, datetime

import numpy as np
import polars as pl
from scipy.stats import spearmanr

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import evalr  # noqa: E402
import kernel  # noqa: E402
from common import PLAT_TEST_FROM, SEATS, TEST_FROM, WORK, gap_of, load_rows  # noqa: E402
from exp_club_joint import first_plat, fit_init, slot_first_last  # noqa: E402
from exp_final import joint_frame  # noqa: E402

ACBL = __import__("pathlib").Path(r"E:\bridge\data\acbl")
CAND = dict(k_base=6.0, boost=6.0, boost_until=200)


def by_player(df: pl.DataFrame, r: pl.DataFrame, col: str = "x") -> pl.DataFrame:
    parts = [pl.DataFrame({"Player_ID": df[f"Player_ID_{s}"].cast(pl.Utf8), "Date": df["Date"],
                           "session_id": df["session_id"].cast(pl.Utf8), "v": r[f"{col}_{s}"].cast(pl.Float64)})
             for s in SEATS]
    return pl.concat(parts).drop_nulls("Player_ID").unique(["Player_ID", "session_id"])


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
    src = joint["src_row"].to_numpy()
    g = {}

    # Production: each stream rated separately with today's formula.
    prod = np.full(len(d), np.nan)
    for side, mask in (("club", ~it), ("tournament", it)):
        df = load_rows(side) if side == "club" else tourn
        lay = kernel.Layout(df)
        r = kernel.replay(lay)
        gg = gap_of({s: r[f"r_{s}"].cast(pl.Float64).to_numpy() for s in SEATS})
        pos = np.flatnonzero(mask)
        prod[pos] = gg[src[pos]]
        if side == "tournament":
            prod_t_players = by_player(df, r)
        del lay, r, df
    g["production"] = prod

    t = time.perf_counter()
    jl = kernel.Layout(joint)
    print(f"joint layout {time.perf_counter()-t:.0f}s", flush=True)
    g["joint"] = gap_of({s: kernel.replay(jl)[f"r_{s}"].cast(pl.Float64).to_numpy() for s in SEATS})
    jm = kernel.replay(jl, merged=True)
    g["joint+one_rating"] = gap_of({s: jm[f"r_{s}"].cast(pl.Float64).to_numpy() for s in SEATS})
    slots = first_plat(slot_first_last(jl, joint, True, jm))
    init, w = fit_init(slots, jl.slots[True][4], True)
    print("init weights (log mp, log plat, const), fitted on pre-2025 ratings:", w, flush=True)
    ri = kernel.replay(jl, merged=True, init=init)
    g["joint+one_rating+start"] = gap_of({s: ri[f"r_{s}"].cast(pl.Float64).to_numpy() for s in SEATS})
    rc = kernel.replay(jl, merged=True, init=init, **CAND)
    g["candidate"] = gap_of({s: rc[f"r_{s}"].cast(pl.Float64).to_numpy() for s in SEATS})
    rk = kernel.replay(jl, merged=True, **CAND)
    g["candidate_without_start"] = gap_of({s: rk[f"r_{s}"].cast(pl.Float64).to_numpy() for s in SEATS})

    spread = {}
    for name in ("production", "candidate"):
        a, b = evalr.fit_line(g[name], pct, hs & y25)
        spread[name] = {"board_pct_per_100": b * 1e4}
    spread["candidate"]["player_sd"] = float(np.std(np.concatenate([rc[f"r_{s}"].to_numpy()[ev] for s in SEATS])))
    print("spread", spread, flush=True)

    sub = joint.select("session_id", "section_id", "Pair_Number_NS", "Pair_Number_EW", "Pct_NS").filter(pl.Series(ev))
    res = evalr.evaluate(sub, {k: v[ev] for k, v in g.items()}, {k: v[ev] for k, v in sets.items()}, ref="production")
    md = evalr.table(res, list(g))
    print(md, flush=True)

    # Ground truth 2 with the candidate.
    cand_players = by_player(joint, rc)
    feats = pl.read_parquet(WORK / "tourn_player_session_feats.parquet")
    awards = pl.read_parquet(ACBL / "acbl_tournament_player_awards.parquet").filter(pl.col("mp_color") == "Platinum")
    plat_ps = feats.filter("is_plat").select("Player_ID", "session_id", "Date", "pct").join(
        awards.group_by(pl.col("player_id").alias("Player_ID"), "session_id").agg(pl.col("mp_won").sum()),
        on=["Player_ID", "session_id"], how="left").with_columns(pl.col("mp_won").fill_null(0.0))
    last_award = date(2026, 7, 18)
    tp = pl.read_parquet(ACBL / "acbl_tournament_player_elo_ratings.parquet").with_columns(pl.col("Player_ID").cast(pl.Utf8))
    aw = pl.read_parquet(WORK / "awards_cum.parquet")

    def last_before(tbl, cut, cols):
        return tbl.filter(pl.col("Date") < cut).sort("Date").group_by("Player_ID").agg(
            [pl.col(c).last() for c in cols] + [pl.len().alias("_n")])

    gt2 = {}
    for cut in (date(2022, 1, 1), date(2024, 1, 1)):
        m = plat_ps.filter((pl.col("Date") >= cut) & (pl.col("Date") <= last_award)).group_by("Player_ID").agg(
            pl.len().alias("n_after"), pl.col("mp_won").mean().alias("rate_after"), pl.col("pct").mean().alias("pct_after")
        ).filter(pl.col("n_after") >= 6)
        m = m.join(last_before(tp, cut, ["Elo_R_EventEnd", "Elo_N", "MasterPoints"]).rename(
            {"Elo_R_EventEnd": "production_tourn", "_n": "sessions_before"}), on="Player_ID", how="left")
        m = m.join(last_before(prod_t_players, cut, ["v"]).rename({"v": "production_replay"}).drop("_n"), on="Player_ID", how="left")
        m = m.join(last_before(cand_players, cut, ["v"]).rename({"v": "candidate"}).drop("_n"), on="Player_ID", how="left")
        m = m.join(last_before(aw, cut, ["plat_pts"]).drop("_n"), on="Player_ID", how="left").with_columns(
            pl.col("plat_pts").fill_null(0.0),
            ((pl.col("Elo_N") + 1) * pl.col("production_tourn") + 50 * 1498.5).truediv(pl.col("Elo_N") + 51).alias("published"))
        r = {"players": m.height}
        for target in ("rate_after", "pct_after"):
            for k in ("production_tourn", "published", "candidate", "MasterPoints", "plat_pts"):
                v = m.select(k, target).drop_nulls()
                r[f"{target}|{k}"] = round(float(spearmanr(v[k], v[target]).statistic), 4)
        gt2[str(cut)] = r
    print("ground truth 2:", json.dumps(gt2, indent=1), flush=True)

    # Agreement with platinum points among established tournament players today.
    plat_tot = awards.group_by("player_id", "event_id").agg(pl.col("mp_won").max()).group_by(
        pl.col("player_id").alias("Player_ID")).agg(pl.col("mp_won").sum().alias("plat_total"))
    tn = tp.group_by("Player_ID").agg(pl.len().alias("t_sessions"), pl.col("Elo_R_EventEnd").sort_by("Date").last().alias("production"))
    cl = cand_players.sort("Date").group_by("Player_ID").agg(pl.col("v").last().alias("candidate"))
    est = tn.filter(pl.col("t_sessions") >= 30).join(cl, on="Player_ID", how="left").join(plat_tot, on="Player_ID", how="left").with_columns(
        pl.col("plat_total").fill_null(0.0)).drop_nulls(["production", "candidate"])
    agree = {"players": est.height}
    for k in ("production", "candidate"):
        agree[f"spearman_plat_total|{k}"] = round(float(spearmanr(est[k], est["plat_total"]).statistic), 4)
        e = est.with_columns((pl.col(k).rank() / est.height * 100).alias("pctile"))
        top_plat = e.sort("plat_total", descending=True).head(100)
        top_k = e.sort(k, descending=True).head(100)
        agree[k] = {"top100_plat_median_pctile": float(top_plat["pctile"].median()),
                    "top100_plat_in_top1pct": float((top_plat["pctile"] >= 99).mean()),
                    "top100_rating_share_with_plat": float((top_k["plat_total"] > 0).mean()),
                    "top100_rating_median_plat": float(top_k["plat_total"].median())}
    print("agreement:", json.dumps(agree, indent=1), flush=True)
    (WORK / "results_confirm.md").write_text(md)
    (WORK / "results_confirm.json").write_text(json.dumps(
        {"results": res, "init_weights": w, "spread": spread, "gt2": gt2, "agreement": agree}, indent=1))
    ended = datetime.now()
    print(f"END {ended:%Y-%m-%d %H:%M:%S} elapsed {(ended-started).total_seconds():.0f}s")


if __name__ == "__main__":
    main()

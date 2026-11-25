"""Ground truth 2: does a rating at date T predict platinum results after T?

Players with MIN_AFTER+ platinum pairs sessions after T. Targets:
  rate_after  platinum points won in those pairs sessions / sessions played
  pct_after   mean session % in those sessions
Predictors are values known before T. Spearman across players, all and
for players with 20+ tournament sessions before T.

  acbl-pipeline/.venv/Scripts/python.exe elo/platinum_bench/exp_future_platinum.py
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime

import numpy as np
import polars as pl
from scipy.stats import spearmanr

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import kernel  # noqa: E402
from common import SEATS, WORK, load_rows  # noqa: E402
from exp_club_joint import first_plat, fit_init, slot_first_last  # noqa: E402

ACBL = __import__("pathlib").Path(r"E:\bridge\data\acbl")
MIN_AFTER = 6
CUTS = [date(2022, 1, 1), date(2024, 1, 1)]


def latest_before(tbl: pl.DataFrame, cut: date, val: list[str]) -> pl.DataFrame:
    return (tbl.filter(pl.col("Date") < cut).sort("Date")
            .group_by("Player_ID").agg([pl.col(v).last() for v in val] + [pl.len().alias("_n")]))


def candidate_by_player(df: pl.DataFrame, lay: kernel.Layout, r: pl.DataFrame, merged: bool) -> pl.DataFrame:
    """(Player_ID, Date, rating after session, sessions) from a replay, one row per player-session."""
    parts = []
    for s in SEATS:
        parts.append(pl.DataFrame({
            "Player_ID": df[f"Player_ID_{s}"].cast(pl.Utf8), "session_id": df["session_id"],
            "Date": df["Date"], "x": r[f"x_{s}"].cast(pl.Float64)}))
    return pl.concat(parts).filter(pl.col("Player_ID").is_not_null()).unique(["Player_ID", "session_id"])


def main() -> None:
    started = datetime.now()
    print(f"START {started:%Y-%m-%d %H:%M:%S}")
    df = load_rows("tournament")
    lay = kernel.Layout(df)
    base_m = kernel.replay(lay, merged=True)
    slots = first_plat(slot_first_last(lay, df, True, base_m))
    init, w = fit_init(slots, lay.slots[True][4], True)
    cand = kernel.replay(lay, merged=True, init=init, k_base=16.0, boost=3.0, boost_until=200)
    cand_ps = candidate_by_player(df, lay, cand, True)
    base_ps = candidate_by_player(df, lay, kernel.replay(lay), False)  # per direction: mixes NS/EW like prod

    tp = pl.read_parquet(ACBL / "acbl_tournament_player_elo_ratings.parquet").with_columns(pl.col("Player_ID").cast(pl.Utf8))
    cp = pl.read_parquet(ACBL / "acbl_club_player_elo_ratings.parquet",
                         columns=["Player_ID", "Date", "Elo_R_EventEnd", "Elo_N"]).with_columns(pl.col("Player_ID").cast(pl.Utf8))
    feats = pl.read_parquet(WORK / "tourn_player_session_feats.parquet")
    aw = pl.read_parquet(WORK / "awards_cum.parquet")
    pp = pl.read_parquet(WORK / "plat_pairs_cum.parquet")
    # Platinum pairs sessions with points won, per player.
    awards = pl.read_parquet(ACBL / "acbl_tournament_player_awards.parquet").filter(pl.col("mp_color") == "Platinum")
    plat_ps = feats.filter("is_plat").select("Player_ID", "session_id", "Date", "pct").join(
        awards.group_by(pl.col("player_id").alias("Player_ID"), "session_id").agg(pl.col("mp_won").sum()), on=["Player_ID", "session_id"], how="left"
    ).with_columns(pl.col("mp_won").fill_null(0.0))
    award_last = pl.read_parquet(WORK / "tourn_sessions.parquet").filter(
        pl.col("session_id").is_in(awards["session_id"].unique().implode()))["Date"].max()
    print("last platinum pairs session with awards:", award_last)

    out = {}
    for cut in CUTS:
        after = plat_ps.filter((pl.col("Date") >= cut) & (pl.col("Date") <= award_last)).group_by("Player_ID").agg(
            pl.len().alias("n_after"), pl.col("mp_won").mean().alias("rate_after"), pl.col("pct").mean().alias("pct_after"))
        after = after.filter(pl.col("n_after") >= MIN_AFTER)
        t = latest_before(tp, cut, ["Elo_R_EventEnd", "Elo_N", "MasterPoints"]).rename(
            {"Elo_R_EventEnd": "elo_tourn", "Elo_N": "n_tourn", "_n": "sessions_before"})
        c = latest_before(cp, cut, ["Elo_R_EventEnd", "Elo_N"]).rename({"Elo_R_EventEnd": "elo_club", "Elo_N": "n_club"}).drop("_n")
        a = latest_before(aw.rename({"Date": "Date"}), cut, ["plat_pts", "gold_pts"]).drop("_n")
        p = latest_before(pp, cut, ["plat_n", "plat_pct_sum"]).drop("_n")
        cb = latest_before(cand_ps, cut, ["x"]).rename({"x": "elo_candidate"}).drop("_n")
        bb = latest_before(base_ps, cut, ["x"]).rename({"x": "elo_replay_base"}).drop("_n")
        m = after
        for x in (t, c, a, p, cb, bb):
            m = m.join(x, on="Player_ID", how="left")
        m = m.with_columns(
            ((pl.col("n_tourn") + 1) * pl.col("elo_tourn") + 50 * 1498.5).truediv(pl.col("n_tourn") + 51).alias("elo_tourn_published"),
            pl.col("plat_pts").fill_null(0), pl.col("gold_pts").fill_null(0), pl.col("plat_n").fill_null(0),
            pl.col("plat_pct_sum").fill_null(0),
        ).with_columns(
            (pl.col("plat_pct_sum") / (pl.col("plat_n") + 10)).alias("plat_perf"),
            pl.col("MasterPoints").alias("masterpoints"),
        )
        preds = ["elo_tourn", "elo_tourn_published", "elo_replay_base", "elo_candidate", "elo_club",
                 "masterpoints", "plat_pts", "gold_pts", "plat_perf"]
        res = {}
        for sub, mm in (("all", m), ("20plus_sessions", m.filter(pl.col("sessions_before") >= 20))):
            res[sub] = {"players": mm.height}
            for target in ("rate_after", "pct_after"):
                for k in preds:
                    v = mm.select(k, target).drop_nulls()
                    res[sub][f"{target}|{k}"] = (round(float(spearmanr(v[k], v[target]).statistic), 4), v.height)
        # Top-100 by future rate: share also in each predictor's top 100 (players with all values).
        full = m.drop_nulls(["elo_tourn", "elo_candidate", "masterpoints"])
        top = set(full.sort("rate_after", descending=True).head(100)["Player_ID"])
        res["top100_overlap"] = {k: len(top & set(full.sort(k, descending=True, nulls_last=True).head(100)["Player_ID"]))
                                 for k in preds if k != "elo_club"}
        out[str(cut)] = res
        print(f"cut {cut}: players {m.height}")
        print(json.dumps(res, indent=1))
    (WORK / "results_future_platinum.json").write_text(json.dumps(out, indent=1))
    ended = datetime.now()
    print(f"END {ended:%Y-%m-%d %H:%M:%S} elapsed {(ended-started).total_seconds():.0f}s")


if __name__ == "__main__":
    main()

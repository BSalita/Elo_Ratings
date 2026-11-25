"""Shared loading for the benchmark scripts."""
from __future__ import annotations

import os
import pathlib
from datetime import date

import numpy as np
import polars as pl

WORK = pathlib.Path(os.environ.get("ELO_PLAT_WORK", r"D:\bridge-elo-plat"))
SEATS = "NESW"
TEST_FROM = date(2026, 1, 1)      # the rank eval's held-out year
LINE_FROM = date(2025, 1, 1)      # the rank eval fits its Elo line on 2025
PLAT_TEST_FROM = date(2025, 1, 1)  # platinum sessions from here are the platinum test set
FEATS = ["mp", "plat_pts", "gold_pts", "red_pts", "plat_n", "plat_pct_sum", "plat_pairs_pts", "n_prior"]


def load_rows(side: str) -> pl.DataFrame:
    df = pl.read_parquet(WORK / f"{side}_rows.parquet")
    plat = (pl.col("mp_color").str.to_lowercase() == "platinum").fill_null(False) if "mp_color" in df.columns else pl.lit(False)
    return df.with_columns(plat.alias("is_plat"))


def seat_features(df: pl.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """Per seat, the pre-session player features of features.py, row-aligned to df."""
    f = pl.read_parquet(WORK / "tourn_player_session_feats.parquet")
    f = f.sort("Player_ID", "Date", "session_id").with_columns(
        (pl.col("Player_ID").cum_count().over("Player_ID") - 1).alias("n_prior")
    ).select("Player_ID", "session_id", *FEATS)
    out = {}
    base = df.select("session_id", *[pl.col(f"Player_ID_{s}").cast(pl.Utf8) for s in SEATS]).with_row_index("row")
    for s in SEATS:
        j = base.select("row", "session_id", pl.col(f"Player_ID_{s}").alias("Player_ID")).join(
            f, on=["Player_ID", "session_id"], how="left").sort("row")
        assert j.height == df.height
        out[s] = {k: j[k].cast(pl.Float64).to_numpy() for k in FEATS}
    return out


def gap_of(seat_vals: dict[str, np.ndarray]) -> np.ndarray:
    return ((seat_vals["N"] + seat_vals["S"]) - (seat_vals["E"] + seat_vals["W"])) / 2.0

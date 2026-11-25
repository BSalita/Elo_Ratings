"""Pre-session player features for ACBL tournament sessions.

One row per (Player_ID, session_id) of the tournament Elo rows, with values
known before that session's date (strictly earlier dates only):

  mp            MasterPoints listed for the player in this session (ACBL total
                at entry time)
  plat_pts      platinum points won to date, all events (pairs AND teams:
                Spingold, Vanderbilt, Reisinger ... from the awards file)
  gold_pts, red_pts   same for gold and red
  plat_n        platinum pairs sessions played to date (from board results)
  plat_pct_sum  sum over those sessions of (session % - 0.5)
  plat_pairs_pts  platinum points won in those pairs sessions

Award dates: the session's date when the session is in the board results;
otherwise the last session date of that NABC (team events at an NABC), an
approximate NABC end date for NABCs before the board results start, or the
end of the month in the session id (yymm...) for regional events.

  acbl-pipeline/.venv/Scripts/python.exe elo/platinum_bench/features.py
"""
from __future__ import annotations

import calendar
import os
import pathlib
from datetime import date, datetime

import polars as pl

ACBL = pathlib.Path(os.environ.get("ACBL_DIR", r"E:\bridge\data\acbl"))
WORK = pathlib.Path(os.environ.get("ELO_PLAT_WORK", r"D:\bridge-elo-plat"))
NABC_APPROX_END = {"1": (3, 31), "2": (8, 5), "3": (12, 10)}


def player_sessions(rows: pl.DataFrame) -> pl.DataFrame:
    parts = []
    for seat, ns in (("N", True), ("S", True), ("E", False), ("W", False)):
        pct = pl.col("Pct_NS") if ns else (1.0 - pl.col("Pct_NS"))
        parts.append(rows.select(
            pl.col(f"Player_ID_{seat}").cast(pl.Utf8).alias("Player_ID"),
            "session_id", "Date", "is_plat",
            pct.cast(pl.Float64).alias("pct"),
            pl.col(f"MasterPoints_{seat}").cast(pl.Float64).alias("mp"),
        ))
    return (
        pl.concat(parts)
        .filter(pl.col("Player_ID").is_not_null())
        .group_by("Player_ID", "session_id")
        .agg(pl.col("Date").first(), pl.col("is_plat").first(),
             pl.col("pct").filter(pl.col("pct").is_between(0, 1)).mean(), pl.col("mp").max())
        .sort("Player_ID", "Date", "session_id")
    )


def award_dates(awards: pl.DataFrame, sessions: pl.DataFrame) -> pl.DataFrame:
    smap = sessions.select("session_id", pl.col("Date").alias("sdate"))
    nabc_end = (
        sessions.filter(pl.col("session_id").str.starts_with("NABC"))
        .with_columns(pl.col("session_id").str.extract(r"^(NABC\d+)").alias("nabc"))
        .group_by("nabc").agg(pl.col("Date").max().alias("nabc_end"))
    )
    a = (
        awards.join(smap, on="session_id", how="left")
        .with_columns(pl.col("event_id").str.extract(r"^(NABC\d+)").alias("nabc"))
        .join(nabc_end, on="nabc", how="left")
    )

    def approx(row_nabc: str | None, sid: str) -> date | None:
        if row_nabc:
            yy, n = int(row_nabc[4:6]), row_nabc[6:7]
            m, d = NABC_APPROX_END[n]
            return date(2000 + yy, m, d)
        head = sid[:4]
        if head.isdigit():
            y, m = 2000 + int(head[:2]), int(head[2:4])
            if 1 <= m <= 12:
                return date(y, m, calendar.monthrange(y, m)[1])
        return None

    need = a.filter(pl.col("sdate").is_null() & pl.col("nabc_end").is_null()).select("nabc", "session_id").unique()
    need = need.with_columns(pl.Series("approx", [approx(n, s) for n, s in need.iter_rows()], dtype=pl.Date))
    a = a.join(need.drop("nabc"), on="session_id", how="left")
    # Board-results session date when known. NABC awards without a board
    # session (team events) become known at the NABC's end.
    return a.with_columns(
        pl.coalesce("sdate", "nabc_end", "approx").alias("avail")
    ).drop("sdate", "nabc_end", "approx")


def main() -> None:
    started = datetime.now()
    print(f"START {started:%Y-%m-%d %H:%M:%S}")
    rows = pl.read_parquet(WORK / "tournament_rows.parquet",
                           columns=["Date", "session_id", "Pct_NS", "mp_color",
                                    *[f"Player_ID_{s}" for s in "NESW"], *[f"MasterPoints_{s}" for s in "NESW"]])
    rows = rows.with_columns((pl.col("mp_color").str.to_lowercase() == "platinum").fill_null(False).alias("is_plat"))
    sessions = rows.group_by("session_id").agg(pl.col("Date").first(), pl.col("is_plat").first())
    ps = player_sessions(rows)
    print(f"player-sessions {ps.height:,}, platinum {ps['is_plat'].sum():,}")

    awards = pl.read_parquet(ACBL / "acbl_tournament_player_awards.parquet")
    a = award_dates(awards, sessions)
    print("awards without a date:", a["avail"].null_count(), "of", a.height)
    a = a.filter(pl.col("avail").is_not_null())
    plat_sessions = sessions.filter("is_plat").select("session_id")
    a = a.with_columns(pl.col("session_id").is_in(plat_sessions["session_id"].implode()).alias("in_plat_pairs"))
    # One value per player and date: totals by colour.
    daily = (
        a.group_by("player_id", "avail")
        .agg(
            pl.col("mp_won").filter(pl.col("mp_color") == "Platinum").sum().alias("d_plat"),
            pl.col("mp_won").filter(pl.col("mp_color") == "Gold").sum().alias("d_gold"),
            pl.col("mp_won").filter(pl.col("mp_color") == "Red").sum().alias("d_red"),
            pl.col("mp_won").filter((pl.col("mp_color") == "Platinum") & pl.col("in_plat_pairs")).sum().alias("d_pp"),
        )
        .sort("player_id", "avail")
        .with_columns(
            pl.col("d_plat").cum_sum().over("player_id").alias("plat_pts"),
            pl.col("d_gold").cum_sum().over("player_id").alias("gold_pts"),
            pl.col("d_red").cum_sum().over("player_id").alias("red_pts"),
            pl.col("d_pp").cum_sum().over("player_id").alias("plat_pairs_pts"),
        )
        .select(pl.col("player_id").alias("Player_ID"), pl.col("avail").alias("Date"),
                "plat_pts", "gold_pts", "red_pts", "plat_pairs_pts")
    )
    # Platinum pairs sessions played and performance to date, by date.
    pday = (
        ps.filter("is_plat")
        .group_by("Player_ID", "Date")
        .agg(pl.len().alias("d_n"), (pl.col("pct") - 0.5).fill_null(0).sum().alias("d_pct"))
        .sort("Player_ID", "Date")
        .with_columns(pl.col("d_n").cum_sum().over("Player_ID").alias("plat_n"),
                      pl.col("d_pct").cum_sum().over("Player_ID").alias("plat_pct_sum"))
        .select("Player_ID", "Date", "plat_n", "plat_pct_sum")
    )
    daily.write_parquet(WORK / "awards_cum.parquet")
    pday.write_parquet(WORK / "plat_pairs_cum.parquet")
    out = ps.sort("Date")
    for tbl in (daily.sort("Date"), pday.sort("Date")):
        out = out.join_asof(tbl, on="Date", by="Player_ID", strategy="backward", allow_exact_matches=False)
    fill = ["plat_pts", "gold_pts", "red_pts", "plat_pairs_pts", "plat_n", "plat_pct_sum"]
    out = out.with_columns([pl.col(c).fill_null(0) for c in fill])
    path = WORK / "tourn_player_session_feats.parquet"
    out.write_parquet(path)
    print(f"wrote {path} {out.shape}")
    print(out.filter(pl.col("is_plat")).select(fill + ["mp"]).describe())
    ended = datetime.now()
    print(f"END {ended:%Y-%m-%d %H:%M:%S} elapsed {(ended-started).total_seconds():.0f}s")


if __name__ == "__main__":
    main()

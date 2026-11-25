"""FFBridge Elo: replay variants of the session Elo and score them.

Data: the Lancelot pair-session results parquet the FFBridge Elo app builds
(``elo_full_v11_FFBridge_Lancelot_API_iv_1.results.parquet``, one row per
pair per simultaneous tournament, 2023-11 to 2026-10).

Production formula (streamlit_app_ffbridge_elo_ratings.py / elo_common.py):
each player rating r moves by K * fs * (pct/100 - E), E = 1/(1+10^((F - r)/400)),
F = mean pair rating of the whole tournament field (self included), K 32,
start 1200, fs = field-strength damping (k_min 0.7, sigma 6).

Ground truth: within each club session (tournament_id, club_id) with 4+ pairs,
Spearman of the pair's pre-session strength (mean of the two players) against
its session %. Dev = 2025, test = 2026. Same idea as the rank eval, at
session level (the results parquet has no boards).

  acbl-pipeline/.venv/Scripts/python.exe elo/ffbridge_bench/ffb_bench.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
from datetime import datetime

import numba
import numpy as np
import polars as pl

SRC = pathlib.Path(__file__).resolve().parents[2]
RESULTS = SRC / "data" / "ffbridge" / "elo_cache" / "elo_full_v11_FFBridge_Lancelot_API_iv_1.results.parquet"
WORK = pathlib.Path(r"D:\bridge-elo-plat\ffbridge")
DEFAULT_ELO = 1200.0
SIGMA = 6.0
K_MIN = 0.7
Z_FLOOR = -2.0


def load() -> pl.DataFrame:
    d = pl.read_parquet(RESULTS, columns=[
        "tournament_id", "date", "club_id", "player1_id", "player2_id", "Pct_Used", "scratch_score_status",
        "pe", "player1_scratch_elo_before", "player2_scratch_elo_before"])
    d = d.with_columns(
        pl.col("date").str.slice(0, 10).str.to_date().alias("Date"),
        pl.when(pl.col("scratch_score_status") != "unresolved").then(pl.col("Pct_Used")).alias("pct"),
    )
    # Production processes tournaments by date string, then file order within a tournament.
    d = d.with_row_index("file_row").sort(["date", "file_row"], maintain_order=True)
    return d


@numba.njit(cache=False)
def _replay(t_start, t_end, p1, p2, pct, club, n_players, n_clubs, k_base, boost, boost_n,
            pair_expect, exclude_self, damp, club_start, scale, out1, out2, cnt1, cnt2):
    r = np.full(n_players, DEFAULT_ELO)
    games = np.zeros(n_players, dtype=np.int64)
    seen = np.zeros(n_players, dtype=np.bool_)
    club_sum = np.zeros(n_clubs)
    club_n = np.zeros(n_clubs)
    for t in range(t_start.shape[0]):
        a = t_start[t]
        b = t_end[t]
        # Starting rating for newcomers: mean pre-session rating of this club
        # session's established players (20+ sessions), when 3+ of them.
        if club_start:
            for i in range(a, b):
                c = club[i]
                for p in (p1[i], p2[i]):
                    if seen[p] and games[p] >= 20:
                        club_sum[c] += r[p]
                        club_n[c] += 1.0
            for i in range(a, b):
                c = club[i]
                for p in (p1[i], p2[i]):
                    if not seen[p] and club_n[c] >= 3:
                        r[p] = club_sum[c] / club_n[c]
            for i in range(a, b):
                club_sum[club[i]] = 0.0
                club_n[club[i]] = 0.0
        for i in range(a, b):
            seen[p1[i]] = True
            seen[p2[i]] = True
        tot = 0.0
        n = 0
        for i in range(a, b):
            tot += (r[p1[i]] + r[p2[i]]) / 2.0
            n += 1
        favg = tot / n if n > 0 else DEFAULT_ELO
        fs = 1.0
        if damp:
            z = (favg - DEFAULT_ELO) / SIGMA
            if z < 0.0:
                fs = K_MIN if z <= Z_FLOOR else K_MIN + (1.0 - K_MIN) * (z - Z_FLOOR) / (0.0 - Z_FLOOR)
        for i in range(a, b):
            out1[i] = r[p1[i]]
            out2[i] = r[p2[i]]
            cnt1[i] = games[p1[i]]
            cnt2[i] = games[p2[i]]
        new1 = np.empty(b - a)
        new2 = np.empty(b - a)
        for i in range(a, b):
            x = pct[i]
            r1 = r[p1[i]]
            r2 = r[p2[i]]
            if np.isnan(x):
                new1[i - a] = r1
                new2[i - a] = r2
                continue
            s = x / 100.0
            pr = (r1 + r2) / 2.0
            f = favg
            if exclude_self and n > 1:
                f = (tot - pr) / (n - 1)
            if pair_expect:
                e1 = 1.0 / (1.0 + 10.0 ** ((f - pr) / scale))
                e2 = e1
            else:
                e1 = 1.0 / (1.0 + 10.0 ** ((f - r1) / scale))
                e2 = 1.0 / (1.0 + 10.0 ** ((f - r2) / scale))
            k1 = k_base * fs * (boost if games[p1[i]] < boost_n else 1.0)
            k2 = k_base * fs * (boost if games[p2[i]] < boost_n else 1.0)
            new1[i - a] = r1 + k1 * (s - e1)
            new2[i - a] = r2 + k2 * (s - e2)
        # Production updates players one row at a time; a player appears once per
        # session, so simultaneous commit is the same.
        for i in range(a, b):
            if not np.isnan(pct[i]):
                games[p1[i]] += 1
                games[p2[i]] += 1
            r[p1[i]] = new1[i - a]
            r[p2[i]] = new2[i - a]


class Data:
    def __init__(self):
        d = load()
        self.d = d
        ids = pl.concat([d["player1_id"], d["player2_id"]]).unique().sort()
        lut = pl.DataFrame({"pid": ids, "k": pl.int_range(0, ids.len(), eager=True)})
        self.p1 = d.select(pl.col("player1_id").alias("pid")).join(lut, on="pid", how="left", maintain_order="left")["k"].to_numpy()
        self.p2 = d.select(pl.col("player2_id").alias("pid")).join(lut, on="pid", how="left", maintain_order="left")["k"].to_numpy()
        self.n_players = ids.len()
        clubs = d["club_id"].fill_null("").unique().sort()
        clut = pl.DataFrame({"club_id": clubs, "c": pl.int_range(0, clubs.len(), eager=True)})
        self.club = d.select(pl.col("club_id").fill_null("")).join(clut, on="club_id", how="left", maintain_order="left")["c"].to_numpy()
        self.n_clubs = clubs.len()
        self.pct = d["pct"].cast(pl.Float64).to_numpy()
        tid = d["tournament_id"].to_numpy()
        change = np.flatnonzero(tid[1:] != tid[:-1]) + 1
        self.t_start = np.concatenate(([0], change)).astype(np.int64)
        self.t_end = np.concatenate((change, [len(tid)])).astype(np.int64)
        if len(np.unique(tid[self.t_start])) != len(self.t_start):
            raise RuntimeError("a tournament's rows are not contiguous in date order")

    def replay(self, k_base=32.0, boost=1.0, boost_n=0, pair_expect=False, exclude_self=False, damp=True,
               club_start=False, scale=400.0):
        n = len(self.pct)
        o1, o2 = np.empty(n), np.empty(n)
        c1, c2 = np.empty(n, np.int64), np.empty(n, np.int64)
        _replay(self.t_start, self.t_end, self.p1, self.p2, self.pct, self.club, self.n_players, self.n_clubs,
                float(k_base), float(boost), int(boost_n), bool(pair_expect), bool(exclude_self), bool(damp),
                bool(club_start), float(scale), o1, o2, c1, c2)
        return o1, o2, c1, c2


def field_spearman(d: pl.DataFrame, preds: dict[str, np.ndarray], mask: np.ndarray, field=("tournament_id", "club_id")):
    f = d.select(*field, "pct").with_columns([pl.Series(k, v) for k, v in preds.items()]).filter(
        pl.Series(mask) & pl.col("pct").is_not_null())
    g = f.group_by(*field).agg(pl.len().alias("pairs"), *[pl.corr(k, "pct", method="spearman").alias(k) for k in preds]).filter(
        pl.col("pairs") >= 4).sort(*field)
    return g.with_columns([pl.col(k).fill_nan(0.0).fill_null(0.0) for k in preds])


def summarize(g: pl.DataFrame, names: list[str], ref: str) -> dict:
    out = {"fields": g.height}
    for k in names:
        v = g[k].to_numpy()
        row = {"spearman": float(v.mean())}
        if k != ref:
            dlt = v - g[ref].to_numpy()
            row["vs_ref"] = float(dlt.mean())
            row["ci95"] = float(1.96 * dlt.std(ddof=1) / np.sqrt(len(dlt)))
        out[k] = row
    return out


def main() -> None:
    started = datetime.now()
    print(f"START {started:%Y-%m-%d %H:%M:%S}", flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    data = Data()
    d = data.d
    print(f"rows {d.height:,}, tournaments {len(data.t_start):,}, players {data.n_players:,}", flush=True)
    year = d["Date"].dt.year().to_numpy()
    sets = {"2025": year == 2025, "2026": year == 2026}

    stored = (d["player1_scratch_elo_before"].fill_null(0).to_numpy() + d["player2_scratch_elo_before"].fill_null(0).to_numpy()) / 2
    variants = {
        "production": {},
        "pair_expect": dict(pair_expect=True),
        "pair_expect_excl_self": dict(pair_expect=True, exclude_self=True),
        "no_damp": dict(damp=False),
        "k16": dict(k_base=16.0),
        "k24": dict(k_base=24.0),
        "k48": dict(k_base=48.0),
        "k16_boost3_n20": dict(k_base=16.0, boost=3.0, boost_n=20),
        "k24_boost2_n20": dict(k_base=24.0, boost=2.0, boost_n=20),
        "club_start": dict(club_start=True),
        "scale200": dict(scale=200.0),
        "scale800": dict(scale=800.0),
    }
    preds = {"stored_production": stored}
    for name, kw in variants.items():
        t = time.perf_counter()
        o1, o2, _, _ = data.replay(**kw)
        preds[name] = (o1 + o2) / 2
        print(f"{name}: {time.perf_counter()-t:.1f}s", flush=True)
    m = np.isfinite(stored) & (stored > 0)
    print("stored vs replayed production: Spearman of pair ratings",
          float(pl.DataFrame({"a": stored[m], "b": preds["production"][m]}).select(pl.corr("a", "b", method="spearman")).item()))
    res = {}
    for sname, mask in sets.items():
        g = field_spearman(d, preds, mask)
        res[sname] = summarize(g, list(preds), "production")
    names = list(preds)
    lines = ["| predictor | " + " | ".join(f"{s} ({res[s]['fields']:,} club sessions)" for s in res) + " |",
             "|---|" + "---|" * len(res)]
    for k in names:
        cells = []
        for s in res:
            r = res[s][k]
            c = f"{r['spearman']:.4f}"
            if "vs_ref" in r:
                c += f" ({r['vs_ref']:+.4f} ±{r['ci95']:.4f})"
            cells.append(c)
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    md = "\n".join(lines)
    print(md)
    (WORK / "results_round1.md").write_text(md)
    (WORK / "results_round1.json").write_text(json.dumps(res, indent=1))
    ended = datetime.now()
    print(f"END {ended:%Y-%m-%d %H:%M:%S} elapsed {(ended-started).total_seconds():.0f}s")


if __name__ == "__main__":
    main()

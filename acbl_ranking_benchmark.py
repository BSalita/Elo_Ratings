"""ACBL ranking predictiveness benchmark + skill-anchored rating prototype.

Goal
----
Define "rating quality" empirically as head-to-head predictiveness:

    Within a single session+direction (the matchpoint *field* where pairs are
    actually ranked against each other), how often does the pair with the
    higher PRE-session rating finish with the higher session matchpoint %?

This is a leakage-free, interpretable metric (pairwise concordance / AUC):
the rating is taken as of *event start* (``Elo_R_*_EventStart``), so it only
uses information available before the session is played.

We then compare candidate ratings on this metric:

  * ``raw``     - the stored Elo (what the headline derives from)
  * ``skill``   - a field-independent prior from card play + bidding
                  (DD_Tricks_Diff + par-suit + par-contract), learned on a
                  TRAIN period only
  * ``anchored(w)`` - convex blend ``(1-w)*raw + w*skill_anchor`` for a sweep
                  of weights w in [0,1]

If blending in field-independent skill raises concordance, that is direct
evidence it makes the rating a better predictor (and, as a side effect, pulls
weak-field-inflated "intermediate" players out of the top).

Usage
-----
    python acbl_ranking_benchmark.py --club club --test-months 12
    python acbl_ranking_benchmark.py --club tournament

Fail-fast: no fallbacks; missing files / columns raise immediately.
"""
from __future__ import annotations

import argparse
import pathlib
import time
from datetime import datetime

import duckdb
import polars as pl

DATA_DIR = pathlib.Path(r"E:\bridge\data\acbl")

# Blend weights to sweep: 0.0 == raw Elo, 1.0 == pure skill prior.
WEIGHT_SWEEP = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]

# Map a combined skill z-score onto an Elo-like scale for blending.
SKILL_ANCHOR_MEAN = 1500.0
SKILL_ANCHOR_SD = 400.0


def _elapsed(t0: float) -> str:
    return f"{time.perf_counter() - t0:.1f}s"


def build_pair_rows(con: duckdb.DuckDBPyConnection, parquet: str) -> None:
    """Materialize one row per (pair, direction, board) with the board's
    outcome %, the pair's event-start rating, and field-independent skill
    signals. Pair id is canonical (direction-agnostic, sorted player ids)."""
    con.execute(
        f"""
        CREATE OR REPLACE TABLE pair_rows AS
        WITH base AS (
            SELECT * FROM read_parquet('{parquet}')
            WHERE Pct_NS IS NOT NULL AND Pct_NS >= 0 AND Pct_NS <= 1
        )
        SELECT
            CASE WHEN Player_ID_N <= Player_ID_S
                 THEN Player_ID_N || '-' || Player_ID_S
                 ELSE Player_ID_S || '-' || Player_ID_N END AS pair_id,
            'NS' AS dir,
            session_id,
            Date,
            Pct_NS AS pct_board,
            Elo_R_NS_EventStart AS rating_eventstart,
            CAST(DD_Tricks_Diff AS DOUBLE) AS dd,
            CAST(Is_Par_Suit AS INTEGER) AS par_suit,
            CAST(Is_Par_Contract AS INTEGER) AS par_contract
        FROM base
        WHERE Player_ID_N IS NOT NULL AND Player_ID_S IS NOT NULL
          AND Elo_R_NS_EventStart IS NOT NULL
        UNION ALL
        SELECT
            CASE WHEN Player_ID_E <= Player_ID_W
                 THEN Player_ID_E || '-' || Player_ID_W
                 ELSE Player_ID_W || '-' || Player_ID_E END AS pair_id,
            'EW' AS dir,
            session_id,
            Date,
            1.0 - Pct_NS AS pct_board,
            Elo_R_EW_EventStart AS rating_eventstart,
            CAST(DD_Tricks_Diff AS DOUBLE) AS dd,
            CAST(Is_Par_Suit AS INTEGER) AS par_suit,
            CAST(Is_Par_Contract AS INTEGER) AS par_contract
        FROM base
        WHERE Player_ID_E IS NOT NULL AND Player_ID_W IS NOT NULL
          AND Elo_R_EW_EventStart IS NOT NULL
        """
    )


def cutoff_date(con: duckdb.DuckDBPyConnection, test_months: int) -> str:
    max_date = con.execute("SELECT MAX(Date) FROM pair_rows").fetchone()[0]
    cutoff = con.execute(
        "SELECT (MAX(Date) - INTERVAL '%d months')::DATE FROM pair_rows" % test_months
    ).fetchone()[0]
    print(f"  max date={max_date}  test cutoff={cutoff}  (test = last {test_months} months)")
    return str(cutoff)


def build_test_table(con: duckdb.DuckDBPyConnection, cutoff: str, min_train_boards: int) -> None:
    """Per (session, dir, pair) test rows with session %, raw rating, and a
    TRAIN-period skill anchor (z-scored over pairs, mapped to an Elo scale)."""
    con.execute(
        f"""
        CREATE OR REPLACE TABLE train_skill AS
        WITH per_pair AS (
            SELECT pair_id,
                   AVG(dd) AS dd, AVG(par_suit) AS ps, AVG(par_contract) AS pc,
                   COUNT(*) AS n
            FROM pair_rows
            WHERE Date < DATE '{cutoff}'
            GROUP BY pair_id
            HAVING COUNT(*) >= {min_train_boards}
        ),
        stats AS (
            SELECT AVG(dd) m_dd, STDDEV_POP(dd) s_dd,
                   AVG(ps) m_ps, STDDEV_POP(ps) s_ps,
                   AVG(pc) m_pc, STDDEV_POP(pc) s_pc
            FROM per_pair
        )
        SELECT p.pair_id, p.n,
               ( (p.dd - s.m_dd) / NULLIF(s.s_dd, 0)
               + (p.ps - s.m_ps) / NULLIF(s.s_ps, 0)
               + (p.pc - s.m_pc) / NULLIF(s.s_pc, 0) ) / 3.0 AS skill_z
        FROM per_pair p CROSS JOIN stats s
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE test_rows AS
        WITH sess AS (
            SELECT pair_id, dir, session_id,
                   ANY_VALUE(Date) AS Date,
                   AVG(pct_board) AS sess_pct,
                   AVG(rating_eventstart) AS raw
            FROM pair_rows
            WHERE Date >= DATE '{cutoff}'
            GROUP BY pair_id, dir, session_id
        )
        SELECT s.session_id, s.dir, s.pair_id, s.sess_pct,
               s.raw,
               {SKILL_ANCHOR_MEAN} + t.skill_z * {SKILL_ANCHOR_SD} AS skill_anchor
        FROM sess s
        JOIN train_skill t USING (pair_id)
        WHERE s.raw IS NOT NULL AND s.sess_pct IS NOT NULL
        """
    )

    # Tag each test pair-session with its field's strength (mean raw rating of
    # the session+direction) and a quartile, so we can see whether skill
    # anchoring helps specifically in strong fields (where weak-field-inflated
    # players are actually exposed to genuine experts).
    con.execute(
        """
        CREATE OR REPLACE TABLE test_rows2 AS
        WITH field AS (
            SELECT session_id, dir, AVG(raw) AS field_mean, COUNT(*) AS field_n
            FROM test_rows GROUP BY session_id, dir
        ),
        q AS (
            SELECT *, NTILE(4) OVER (ORDER BY field_mean) AS field_q FROM field
        )
        SELECT r.*, q.field_mean, q.field_q
        FROM test_rows r JOIN q USING (session_id, dir)
        """
    )


def concordance(con: duckdb.DuckDBPyConnection, rating_expr: str, where: str = "TRUE") -> tuple[float, int]:
    """Pairwise accuracy: of all same-field pair-vs-pair matchups, fraction
    where the higher ``rating_expr`` finished with the higher session %.

    ``where`` filters the (test_rows2) population, e.g. ``field_q = 4`` to
    restrict to the strongest-field quartile."""
    row = con.execute(
        f"""
        WITH r AS (
            SELECT session_id, dir, pair_id, sess_pct, ({rating_expr}) AS rating
            FROM test_rows2 WHERE {where}
        )
        SELECT
            SUM(CASE WHEN (a.rating - b.rating) * (a.sess_pct - b.sess_pct) > 0 THEN 1 ELSE 0 END) AS concordant,
            SUM(CASE WHEN (a.rating - b.rating) * (a.sess_pct - b.sess_pct) < 0 THEN 1 ELSE 0 END) AS discordant
        FROM r a
        JOIN r b
          ON a.session_id = b.session_id AND a.dir = b.dir AND a.pair_id < b.pair_id
        WHERE a.rating <> b.rating AND a.sess_pct <> b.sess_pct
        """
    ).fetchone()
    concordant, discordant = int(row[0] or 0), int(row[1] or 0)
    total = concordant + discordant
    acc = concordant / total if total else float("nan")
    return acc, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--club", choices=["club", "tournament"], default="club")
    ap.add_argument("--test-months", type=int, default=12)
    ap.add_argument("--min-train-boards", type=int, default=20)
    ap.add_argument("--memory-limit", default="40GB")
    args = ap.parse_args()

    parquet = str(DATA_DIR / f"acbl_{args.club}_elo_ratings.parquet")
    if not pathlib.Path(parquet).exists():
        raise FileNotFoundError(parquet)

    started = datetime.now()
    print(f"START {started:%Y-%m-%d %H:%M:%S}  club={args.club}  parquet={parquet}")

    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute(f"PRAGMA memory_limit='{args.memory_limit}';")
    con.execute("PRAGMA preserve_insertion_order=false;")
    tmp = pathlib.Path.cwd() / "_bench_spill"
    tmp.mkdir(exist_ok=True)
    con.execute(f"PRAGMA temp_directory='{tmp}';")

    t0 = time.perf_counter()
    print("[1/4] building per-pair board rows ...")
    build_pair_rows(con, parquet)
    n_rows = con.execute("SELECT COUNT(*) FROM pair_rows").fetchone()[0]
    print(f"      pair_rows={n_rows:,}  ({_elapsed(t0)})")

    t1 = time.perf_counter()
    print("[2/4] train/test split + skill anchor ...")
    cutoff = cutoff_date(con, args.test_months)
    build_test_table(con, cutoff, args.min_train_boards)
    n_test = con.execute("SELECT COUNT(*) FROM test_rows").fetchone()[0]
    n_pairs = con.execute("SELECT COUNT(DISTINCT pair_id) FROM test_rows").fetchone()[0]
    print(f"      test pair-sessions={n_test:,}  distinct pairs={n_pairs:,}  ({_elapsed(t1)})")

    t2 = time.perf_counter()
    print("[3/4] concordance sweep (higher = better predictor) ...")
    results = []
    for w in WEIGHT_SWEEP:
        expr = f"(1-{w})*raw + {w}*skill_anchor"
        acc, total = concordance(con, expr)
        label = "raw Elo" if w == 0.0 else ("pure skill" if w == 1.0 else f"anchored w={w}")
        results.append((w, label, acc, total))
        print(f"      w={w:<4}  {label:<16}  accuracy={acc*100:.2f}%  (n={total:,})")

    t3 = time.perf_counter()
    best = max(results, key=lambda r: r[2])
    base = results[0]
    best_w = best[0]
    print("[4/4] summary")
    print(f"      baseline raw Elo : {base[2]*100:.2f}%")
    print(f"      best blend       : {best[1]}  {best[2]*100:.2f}%  "
          f"(+{(best[2]-base[2])*100:.2f} pts)")

    # Field-strength stratified: does skill anchoring help where it matters
    # (strong fields, where inflated weak-field players meet real experts)?
    print("      field-strength quartiles (Q1=weakest .. Q4=strongest):")
    print(f"        {'quartile':<10}{'raw Elo':>10}{'best blend':>12}{'delta':>8}{'n':>14}")
    for q in (1, 2, 3, 4):
        acc_raw, n_raw = concordance(con, "raw", where=f"field_q = {q}")
        acc_blend, _ = concordance(con, f"(1-{best_w})*raw + {best_w}*skill_anchor", where=f"field_q = {q}")
        print(f"        Q{q:<9}{acc_raw*100:>9.2f}%{acc_blend*100:>11.2f}%"
              f"{(acc_blend-acc_raw)*100:>+7.2f}{n_raw:>14,}")
    print(f"      sweep+strata done in {_elapsed(t2)}; total {_elapsed(t0)}")

    ended = datetime.now()
    print(f"END   {ended:%Y-%m-%d %H:%M:%S}  elapsed={(ended-started).total_seconds():.1f}s")


if __name__ == "__main__":
    main()

"""Clone Elo: partner-adjusted pair rating for FFBridge and ACBL sessions."""

from __future__ import annotations

import math
import unittest
from datetime import date, timedelta

import polars as pl

from clone_elo import (
    K_SCALE_400,
    _sessions_from_acbl_boards,
    compute_clone_ratings,
    holdout_clone_mae,
    logistic,
)


def _fit(**kwargs):
    kwargs.setdefault("tau_days", None)
    kwargs.setdefault("r0", 1500.0)
    kwargs.setdefault("k_scale", K_SCALE_400)
    kwargs.setdefault("min_sessions", 20)
    kwargs.setdefault("min_partners", 3)
    return compute_clone_ratings(**kwargs)


def _star_sessions() -> pl.DataFrame:
    """A star scoring 60% against a field that scores 50% with itself."""
    field = [f"F{i}" for i in range(6)]
    rows = []
    day = date(2026, 1, 1)
    for i, a in enumerate(field):
        for b in field[i + 1 :]:
            for _ in range(12):
                rows.append(_pair(a, b, 50.0, day))
    for partner in field:
        for _ in range(20):
            rows.append(_pair("S", partner, 60.0, day, elo_a=1500.0, elo_b=1500.0))
    return pl.DataFrame(rows)


def _pair(a, b, pct, day, elo_a=1500.0, elo_b=1500.0) -> dict:
    return {
        "player_a": a,
        "player_b": b,
        "date": day,
        "pct": pct,
        "elo_a": elo_a,
        "elo_b": elo_b,
    }


class CloneFitTests(unittest.TestCase):
    def test_star_clone_is_about_sixty_nine_percent(self) -> None:
        ratings = _fit(sessions_df=_star_sessions())
        star = ratings.filter(pl.col("player_id") == "S").row(0, named=True)
        expected = 100.0 * float(logistic(2.0 * math.log(0.6 / 0.4)))
        self.assertEqual(star["Clone_Status"], "ok")
        self.assertGreater(star["Clone_Pct"], expected - 4)
        self.assertLess(star["Clone_Pct"], expected + 4)
        self.assertGreater(star["Partner_Effect"], 0)
        self.assertGreaterEqual(star["Clone_N"], 20)
        self.assertGreaterEqual(star["Clone_Partners"], 3)
        self.assertIsNotNone(star["Clone_SD"])

    def test_isolated_partnership_is_not_identifiable(self) -> None:
        day = date(2026, 3, 1)
        rows = [_pair("A", "B", 62.0, day) for _ in range(30)]
        ratings = _fit(sessions_df=pl.DataFrame(rows), min_sessions=10, min_partners=1)
        row = ratings.filter(pl.col("player_id") == "A").row(0, named=True)
        self.assertEqual(row["Clone_Status"], "not identifiable")
        self.assertIsNone(row["Clone_Elo"])
        self.assertEqual(row["Clone_N"], 30)
        self.assertEqual(row["Clone_Partners"], 1)

    def test_unknown_partner_is_fixed_at_zero(self) -> None:
        day = date(2026, 4, 1)
        rows = [_pair("A", "", 60.0, day, elo_a=1500.0, elo_b=None) for _ in range(40)]
        ratings = _fit(
            sessions_df=pl.DataFrame(rows),
            min_sessions=8,
            min_partners=1,
            ridge_lambda=0.01,
        )
        row = ratings.filter(pl.col("player_id") == "A").row(0, named=True)
        expected = round(100.0 * float(logistic(2.0 * math.log(0.6 / 0.4))), 1)
        self.assertEqual(row["Clone_Status"], "ok")
        self.assertEqual(row["Clone_Pct"], expected)
        self.assertEqual(row["Clone_Partners"], 1)

    def test_recent_sessions_outweigh_old_ones(self) -> None:
        partners = ["P1", "P2", "P3", "P4"]
        rows = []
        for partner in partners:
            for _ in range(10):
                rows.append(_pair("S", partner, 40.0, date(2020, 1, 1)))
                rows.append(_pair("S", partner, 70.0, date(2026, 1, 1)))
        frame = pl.DataFrame(rows)
        career = _fit(sessions_df=frame, tau_days=None, min_sessions=8, min_partners=3)
        recent = _fit(sessions_df=frame, tau_days=365, min_sessions=8, min_partners=3)
        career_pct = career.filter(pl.col("player_id") == "S")["Clone_Pct"][0]
        recent_pct = recent.filter(pl.col("player_id") == "S")["Clone_Pct"][0]
        self.assertGreater(recent_pct, career_pct)

    def test_ffbridge_uses_national_score_only(self) -> None:
        day = date(2026, 5, 1)
        rows = []
        field = ["F1", "F2", "F3", "F4"]
        for partner in field:
            for _ in range(8):
                rows.append(
                    {
                        "player1_id": "S",
                        "player2_id": partner,
                        "date": day,
                        "National_Scratch_Pct": 60.0,
                        "National_Handicap_Pct": 70.0,
                        "Club_Scratch_Pct": 90.0,
                        "player1_scratch_elo_after": 1500.0,
                        "player2_scratch_elo_after": 1500.0,
                        "player1_handicap_elo_after": 1500.0,
                        "player2_handicap_elo_after": 1500.0,
                    }
                )
        frame = pl.DataFrame(rows)
        scratch = _fit(sessions_df=frame, score="Scratch", min_sessions=8, min_partners=3)
        handicap = _fit(sessions_df=frame, score="Handicap", min_sessions=8, min_partners=3)
        scratch_pct = scratch.filter(pl.col("player_id") == "S")["Clone_Pct"][0]
        handicap_pct = handicap.filter(pl.col("player_id") == "S")["Clone_Pct"][0]
        self.assertLess(scratch_pct, 75)
        self.assertGreater(handicap_pct, scratch_pct)

    def test_null_percentage_is_skipped(self) -> None:
        day = date(2026, 6, 1)
        rows = [_pair("A", "", 55.0, day) for _ in range(6)]
        rows.append(_pair("A", "", None, day))
        ratings = _fit(sessions_df=pl.DataFrame(rows), min_sessions=1, min_partners=1)
        self.assertEqual(ratings.filter(pl.col("player_id") == "A")["Clone_N"][0], 6)

    def test_bad_score_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_clone_ratings(pl.DataFrame({"player_a": ["A"]}), score="Bogus")

    def test_acbl_boards_become_both_directions(self) -> None:
        boards = pl.DataFrame(
            {
                "Date": [date(2026, 7, 1), date(2026, 7, 1)],
                "session_id": ["s1", "s1"],
                "Player_ID_N": ["N", "N"],
                "Player_ID_S": ["S", "S"],
                "Player_ID_E": ["E", "E"],
                "Player_ID_W": ["W", "W"],
                "Pct_NS": [0.6, 0.6],
                "Elo_R_N": [1510.0, 1510.0],
                "Elo_R_S": [1490.0, 1490.0],
                "Elo_R_E": [1480.0, 1480.0],
                "Elo_R_W": [1520.0, 1520.0],
            }
        )
        sessions = _sessions_from_acbl_boards(boards)
        pct = {
            tuple(sorted((row["player_a"], row["player_b"]))): row["pct"]
            for row in sessions.to_dicts()
        }
        self.assertAlmostEqual(pct[("N", "S")], 60.0)
        self.assertAlmostEqual(pct[("E", "W")], 40.0)
        ratings = _fit(sessions_df=boards, min_sessions=1, min_partners=1)
        self.assertEqual(
            ratings.filter(pl.col("player_id") == "N")["Clone_Status"][0],
            "not identifiable",
        )

    def test_holdout_beats_partner_polluted_elo(self) -> None:
        players = {
            "A": 80.0,
            "B": 40.0,
            "C": -20.0,
            "D": -30.0,
            "E": 10.0,
            "F": -10.0,
        }
        ids = list(players)
        rows = []
        day = date(2024, 1, 1)
        n = 0
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                for _ in range(8):
                    c_sum = players[a] + players[b]
                    pct = 100.0 * float(logistic(c_sum / K_SCALE_400))
                    rows.append(
                        _pair(
                            a,
                            b,
                            pct,
                            day + timedelta(days=n % 400),
                            elo_a=1500.0 + 0.25 * players[a],
                            elo_b=1500.0 + 0.25 * players[b],
                        )
                    )
                    n += 1
        report = holdout_clone_mae(
            pl.DataFrame(rows),
            tau_days=None,
            r0=1500.0,
            k_scale=K_SCALE_400,
            min_sessions=1,
            min_partners=1,
        )
        self.assertLess(report["clone_mae"], 2.0)
        self.assertLess(report["clone_mae"], report["elo_mae"])


if __name__ == "__main__":
    unittest.main()

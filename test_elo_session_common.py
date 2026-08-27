from __future__ import annotations

import unittest
from datetime import datetime

import polars as pl

from elo_session_common import summarize_acbl_sessions


class AcblSessionSummaryTests(unittest.TestCase):
    def test_collapses_boards_into_newest_first_sessions(self) -> None:
        detail = pl.DataFrame(
            {
                "Date": [
                    datetime(2026, 1, 1),
                    datetime(2026, 1, 2),
                    datetime(2026, 1, 2),
                ],
                "Session": ["old", "new", "new"],
                "Board": [1, 2, 1],
                "Partner": ["Old Partner", "New Partner", "New Partner"],
                "Opponents": ["A - B", "C - D", "A - B"],
                "Pct": [50.0, 70.0, 60.0],
                "Elo_Before": [900, 1010, 1000],
                "Elo_After": [910, 1020, 1010],
            }
        )

        actual = summarize_acbl_sessions(detail)

        self.assertEqual(actual["Session"].to_list(), ["new", "old"])
        newest = actual.row(0, named=True)
        self.assertEqual(newest["Partner"], "New Partner")
        self.assertEqual(newest["Boards"], 2)
        self.assertEqual(newest["Opponent_Pairs"], 2)
        self.assertEqual(newest["Avg_Pct"], 65.0)
        self.assertEqual(newest["Elo_Start"], 1000)
        self.assertEqual(newest["Elo_End"], 1020)
        self.assertEqual(newest["Elo_Delta"], 20)

    def test_requires_session_column(self) -> None:
        with self.assertRaisesRegex(ValueError, "Session"):
            summarize_acbl_sessions(pl.DataFrame({"Board": [1]}))


if __name__ == "__main__":
    unittest.main()

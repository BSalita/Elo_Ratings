from __future__ import annotations

import unittest
from datetime import datetime

import polars as pl

from elo_session_common import results_url_status, summarize_acbl_sessions


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

    def test_adds_club_and_tournament_results_urls(self) -> None:
        club_detail = pl.DataFrame(
            {
                "Date": [datetime(2026, 1, 2), datetime(2026, 1, 2)],
                "Session": ["1493280", "1493280"],
                "Event_ID": [None, 1033402],
                "Board": [1, 2],
            }
        )
        tournament_detail = pl.DataFrame(
            {
                "Date": [datetime(2026, 1, 2)],
                "Session": ["2509101-271B-2"],
                "Board": [1],
            }
        )

        club = summarize_acbl_sessions(club_detail, "Club")
        tournament = summarize_acbl_sessions(tournament_detail, "Tournament")

        self.assertEqual(
            club["Results_URL"][0],
            "https://my.acbl.org/club-results/details/1033402",
        )
        self.assertEqual(
            tournament["Results_URL"][0],
            "https://live.acbl.org/event/2509101/271B/2/summary",
        )
        self.assertEqual(club.columns[-1], "Results_URL")
        self.assertEqual(tournament.columns[-1], "Results_URL")

    def test_club_summary_survives_legacy_detail_without_event_id(self) -> None:
        detail = pl.DataFrame(
            {
                "Date": [datetime(2026, 1, 2)],
                "Session": ["1493280"],
                "Board": [1],
            }
        )

        summary = summarize_acbl_sessions(detail, "Club")

        self.assertEqual(summary["Session"][0], "1493280")
        self.assertIsNone(summary["Results_URL"][0])

    def test_results_url_status_reports_partial_coverage(self) -> None:
        status = results_url_status(
            pl.DataFrame({"Results_URL": ["https://example.test", None, ""]})
        )
        self.assertEqual(
            status,
            {
                "status": "incomplete",
                "total_rows": 3,
                "linked_rows": 1,
                "missing_rows": 2,
            },
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from datetime import datetime

import polars as pl

from elo_session_common import (
    acbl_tournament_results_url,
    results_url_status,
    summarize_acbl_sessions,
)


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
        regional_detail = pl.DataFrame(
            {
                "Date": [datetime(2026, 1, 2)],
                "Session": ["2509101-271B-2"],
                "Board": [1],
            }
        )
        nabc_detail = pl.DataFrame(
            {
                "Date": [datetime(2026, 3, 6)],
                "Session": ["NABC261-SILO-4"],
                "Board": [1],
            }
        )

        club = summarize_acbl_sessions(club_detail, "Club")
        regional = summarize_acbl_sessions(regional_detail, "Tournament")
        nabc = summarize_acbl_sessions(nabc_detail, "Tournament")

        self.assertEqual(
            club["Results_URL"][0],
            "https://my.acbl.org/club-results/details/1033402",
        )
        self.assertEqual(
            regional["Results_URL"][0],
            "https://web2.acbl.org/tournaments/results/2025/09/2509101.htm",
        )
        self.assertEqual(
            nabc["Results_URL"][0],
            "https://web2.acbl.org/tournaments/Results/NABC/NABC261.HTM",
        )
        self.assertEqual(club.columns[-1], "Results_URL")
        self.assertEqual(regional.columns[-1], "Results_URL")
        self.assertEqual(nabc.columns[-1], "Results_URL")

    def test_tournament_results_url_uses_public_web2_pages(self) -> None:
        self.assertEqual(
            acbl_tournament_results_url("NABC261-SILO-4"),
            "https://web2.acbl.org/tournaments/Results/NABC/NABC261.HTM",
        )
        self.assertEqual(
            acbl_tournament_results_url("2509101-271B-2"),
            "https://web2.acbl.org/tournaments/results/2025/09/2509101.htm",
        )
        self.assertIsNone(acbl_tournament_results_url(""))

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

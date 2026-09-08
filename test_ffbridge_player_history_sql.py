import unittest
from unittest.mock import patch

import polars as pl

import ffbridge_report_service as reports


def _history_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [
                "2026-08-25",
                "2026-08-17",
                "2026-08-06",
                "2026-07-09",
                "2026-05-11",
                "2025-12-30",
                "2025-12-29",
                "2025-12-22",
                "2025-12-03",
                "2025-10-31",
                "2025-09-01",
                "2025-08-01",
            ],
            "tournament_id": [str(280000 + index) for index in range(12)],
            "group_id": ["21333"] * 12,
            "club_name": ["BC Levallois"] * 12,
            "pair_name": ["SALITA – partner"] * 12,
            "player1_id": ["246273"] * 12,
            "player2_id": ["1"] * 12,
            "National_Scratch_Pct": [
                65.67, 61.57, 71.25, 75.83, 62.35, 64.52,
                65.02, 64.73, 66.22, 65.64, 52.00, 48.00,
            ],
            "Club_Scratch_Pct": [None] * 12,
            "Club_Handicap_Pct": [None] * 12,
            "National_Handicap_Pct": [None] * 12,
            "Club_Scratch_Rank": [None] * 12,
            "Club_Handicap_Rank": [None] * 12,
            "National_Scratch_Rank": [2, 7, 2, 1, 9, 9, 6, 3, 7, 4, 36, 55],
            "National_Handicap_Rank": [None] * 12,
        }
    )


class PlayerHistorySqlTests(unittest.TestCase):
    def test_schema_lists_national_scratch_rank(self) -> None:
        with patch.object(
            reports,
            "load_results",
            return_value=(_history_frame(), {"built_at": "2026-09-08T00:00:00Z"}),
        ):
            payload = reports.player_history_schema("246273")

        names = [column["name"] for column in payload["columns"]]
        self.assertIn("National_Scratch_Rank", names)
        self.assertEqual(payload["table"], "self")
        self.assertEqual(payload["total_sessions"], 12)

    def test_sql_rank_filter_returns_ten_rows(self) -> None:
        with patch.object(
            reports,
            "load_results",
            return_value=(_history_frame(), {"built_at": "2026-09-08T00:00:00Z"}),
        ):
            payload = reports.run_player_history_sql(
                "246273",
                """
                SELECT date, National_Scratch_Rank, National_Scratch_Pct, pair_name, club_name, Results_URL
                FROM self
                WHERE National_Scratch_Rank <= 10
                ORDER BY date DESC
                LIMIT 10
                """,
            )

        self.assertEqual(payload["row_count"], 10)
        self.assertFalse(payload["truncated"])
        self.assertEqual(payload["rows"][0]["date"], "2026-08-25")
        self.assertEqual(payload["rows"][0]["National_Scratch_Rank"], 2)
        self.assertLessEqual(max(row["National_Scratch_Rank"] for row in payload["rows"]), 10)

    def test_structured_filters_match_sql(self) -> None:
        with patch.object(
            reports,
            "load_results",
            return_value=(_history_frame(), {"built_at": "2026-09-08T00:00:00Z"}),
        ):
            payload = reports.run_player_history(
                "246273",
                max_national_rank=10,
                date_from="2025-10-01",
                limit=20,
            )

        self.assertEqual(payload["matched_sessions"], 10)
        self.assertEqual(len(payload["sessions"]), 10)
        self.assertEqual(payload["sessions"][0]["National_Scratch_Rank"], 2)

    def test_illegal_sql_is_rejected(self) -> None:
        with patch.object(
            reports,
            "load_results",
            return_value=(_history_frame(), {"built_at": "2026-09-08T00:00:00Z"}),
        ):
            with self.assertRaises(ValueError):
                reports.run_player_history_sql("246273", "DROP TABLE self")
            with self.assertRaises(ValueError):
                reports.run_player_history_sql(
                    "246273", "SELECT missing_column FROM self"
                )


if __name__ == "__main__":
    unittest.main()

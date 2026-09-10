import tempfile
import unittest
from pathlib import Path
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
            "tournament_name": ["Rondes de France"] * 8 + ["Simultané Octopus"] * 4,
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
        self.assertIn("tournament_name", names)
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
        self.assertEqual(payload["sessions"][0]["tournament_name"], "Rondes de France")

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

    def test_values_literals_are_rejected(self) -> None:
        with patch.object(
            reports,
            "load_results",
            return_value=(_history_frame(), {"built_at": "2026-09-08T00:00:00Z"}),
        ):
            with self.assertRaises(ValueError) as exc:
                reports.run_player_history_sql(
                    "246273",
                    "WITH sessions(session_id) AS (VALUES ('280000')) "
                    "SELECT * FROM sessions",
                )
        self.assertIn("VALUES", str(exc.exception))

    def test_join_club_board_results(self) -> None:
        boards = pl.DataFrame(
            {
                "session_id": ["280000", "280000", "280001"],
                "Declarer": ["246273", "1", "246273"],
                "Declarer_Pct": [0.8, 0.4, 0.5],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / reports.CLUB_BOARD_RESULTS_FILENAME
            boards.write_parquet(path)
            with patch.object(
                reports,
                "load_results",
                return_value=(_history_frame(), {"built_at": "2026-09-08T00:00:00Z"}),
            ), patch.dict(
                "os.environ",
                {"FFBRIDGE_STATS_CLUB_BOARD_RESULTS": str(path)},
                clear=False,
            ):
                payload = reports.run_player_history_sql(
                    "246273",
                    """
                    SELECT h.tournament_id,
                           AVG(CASE WHEN s.Declarer = '246273'
                                    THEN s.Declarer_Pct END) AS mean_declarer_pct
                    FROM self h
                    LEFT JOIN club_board_results s
                      ON s.session_id = h.tournament_id
                    GROUP BY h.tournament_id
                    ORDER BY h.tournament_id
                    LIMIT 2
                    """,
                )
        self.assertEqual(payload["row_count"], 2)
        self.assertAlmostEqual(payload["rows"][0]["mean_declarer_pct"], 0.8)

    def test_join_fabricates_contract_on_request(self) -> None:
        boards = pl.DataFrame(
            {
                "session_id": ["280000", "280000"],
                "BidLvl": [4, 1],
                "BidSuit": ["H", "N"],
                "Dbl": ["", "X"],
                "Declarer_Direction": ["S", "W"],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / reports.CLUB_BOARD_RESULTS_FILENAME
            boards.write_parquet(path)
            with patch.object(
                reports,
                "load_results",
                return_value=(_history_frame(), {"built_at": "2026-09-08T00:00:00Z"}),
            ), patch.dict(
                "os.environ",
                {"FFBRIDGE_STATS_CLUB_BOARD_RESULTS": str(path)},
                clear=False,
            ):
                payload = reports.run_player_history_sql(
                    "246273",
                    """
                    SELECT s.Contract
                    FROM self h
                    LEFT JOIN club_board_results s
                      ON s.session_id = h.tournament_id
                    WHERE h.tournament_id = '280000'
                    ORDER BY s.Contract
                    LIMIT 2
                    """,
                )
        self.assertEqual(
            [row["Contract"] for row in payload["rows"]],
            ["1NXW", "4HS"],
        )

    def test_join_uses_stats_api_when_parquet_is_missing(self) -> None:
        class _Response:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {
                    "rows": [{"tournament_id": "280000", "mean_declarer_pct": 0.8}],
                    "truncated": False,
                }

        with patch.object(
            reports,
            "load_results",
            return_value=(_history_frame(), {"built_at": "2026-09-08T00:00:00Z"}),
        ), patch.object(
            reports, "resolve_club_board_results_path", return_value=None
        ), patch(
            "requests.post", return_value=_Response()
        ) as post:
            payload = reports.run_player_history_sql(
                "246273",
                "SELECT h.tournament_id FROM self h "
                "LEFT JOIN club_board_results s ON s.session_id = h.tournament_id",
            )
        self.assertEqual(payload["joined_via"], "ffbridge-stats")
        self.assertEqual(payload["row_count"], 1)
        body = post.call_args.kwargs.get("json") or post.call_args[1].get("json")
        self.assertEqual(body["source"], "club_board_results")
        self.assertEqual(body["tables"]["self"][0]["tournament_id"], "280000")


if __name__ == "__main__":
    unittest.main()

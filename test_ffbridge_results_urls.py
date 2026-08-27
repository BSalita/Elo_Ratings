from __future__ import annotations

import pathlib
import tempfile
import unittest
from unittest.mock import patch

import polars as pl

import elo_ffbridge_lancelot as lancelot
import ffbridge_report_service as reports
from streamlit_app_ffbridge_elo_ratings import (
    _ffbridge_results_url_expr,
    _move_url_columns_to_end,
    _results_cache_has_group_links,
)


class FFBridgeResultsUrlTests(unittest.TestCase):
    def test_uses_public_group_and_session_route(self) -> None:
        rows = pl.DataFrame(
            {
                "tournament_id": ["282839", "282840"],
                "group_id": ["21333", None],
            }
        )

        urls = rows.select(_ffbridge_results_url_expr()).to_series().to_list()

        self.assertEqual(
            urls[0],
            "https://www.ffbridge.fr/competitions/results/groups/"
            "21333/sessions/282839/ranking",
        )
        self.assertIsNone(urls[1])

    def test_extracts_club_code_to_group_id_mapping(self) -> None:
        payload = {
            "groupSessions": [
                {
                    "group": {
                        "id": 21333,
                        "phase": {
                            "stade": {
                                "organization": {"ffbCode": "5802079"}
                            }
                        },
                    }
                }
            ]
        }
        with (
            patch.object(lancelot, "load_from_disk_cache", return_value=None),
            patch.object(lancelot, "save_to_disk_cache"),
            patch.object(lancelot, "lancelot_get", return_value=payload),
        ):
            actual = lancelot.fetch_session_group_ids("282839")

        self.assertEqual(actual, {"5802079": "21333"})

    def test_moves_all_url_columns_to_right_edge(self) -> None:
        display = pl.DataFrame(
            {
                "Results_URL": ["https://example.test/results"],
                "Date": ["2025-01-01"],
                "Score_Source_URL": [None],
                "Rank": [1],
            }
        ).to_pandas()

        reordered = _move_url_columns_to_end(display)

        self.assertEqual(
            reordered.columns.to_list(),
            ["Date", "Rank", "Results_URL", "Score_Source_URL"],
        )

    def test_player_history_api_rows_include_results_url_last(self) -> None:
        persisted = pl.DataFrame(
            {
                "date": ["2025-01-01"],
                "tournament_id": ["282839"],
                "group_id": ["21333"],
                "player1_id": ["101"],
                "player2_id": ["202"],
                "scratch_percentage": [55.0],
            }
        )
        with patch.object(
            reports,
            "load_results",
            return_value=(persisted, {"built_at": "2025-01-02T00:00:00Z"}),
        ):
            response = reports.run_player_history("101")

        row = response["sessions"][0]
        self.assertEqual(
            row["Results_URL"],
            "https://www.ffbridge.fr/competitions/results/groups/"
            "21333/sessions/282839/ranking",
        )
        self.assertEqual(list(row)[-1], "Results_URL")
        self.assertEqual(response["results_links"]["status"], "available")
        self.assertEqual(response["results_links"]["linked_rows"], 1)

    def test_empty_group_id_cache_requires_link_backfill(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "results.parquet"
            pl.DataFrame(
                {"group_id": [None, ""], "tournament_id": ["1", "2"]},
                schema={"group_id": pl.String, "tournament_id": pl.String},
            ).write_parquet(path)
            self.assertFalse(_results_cache_has_group_links(path))

            pl.DataFrame(
                {"group_id": [None, "21333"], "tournament_id": ["1", "2"]},
                schema={"group_id": pl.String, "tournament_id": pl.String},
            ).write_parquet(path)
            self.assertTrue(_results_cache_has_group_links(path))


if __name__ == "__main__":
    unittest.main()

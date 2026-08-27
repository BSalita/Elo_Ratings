from __future__ import annotations

import unittest
from unittest.mock import patch

import polars as pl

import elo_ffbridge_lancelot as lancelot
from streamlit_app_ffbridge_elo_ratings import (
    _ffbridge_results_url_expr,
    _move_url_columns_to_end,
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


if __name__ == "__main__":
    unittest.main()

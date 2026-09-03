from __future__ import annotations

import unittest

import polars as pl

from elo_common import (
    aggrid_sort_model,
    apply_aggrid_sort_model,
    apply_sort_model_to_grid_options,
    default_leaderboard_sort_model,
    remember_leaderboard_sort,
)


class _GridResponse:
    def __init__(self, grid_state=None, columns_state=None):
        self.grid_state = grid_state
        self.columns_state = columns_state


class GridSortTests(unittest.TestCase):
    def test_default_prefers_player_elo_rank(self) -> None:
        self.assertEqual(
            default_leaderboard_sort_model(["Player_Name", "Quality_Rank", "Player_Elo_Rank"]),
            [{"colId": "Player_Elo_Rank", "sort": "asc"}],
        )

    def test_default_uses_rank_for_ffbridge(self) -> None:
        self.assertEqual(
            default_leaderboard_sort_model(["Player_Name", "Rank", "Quality_Rank"]),
            [{"colId": "Rank", "sort": "asc"}],
        )

    def test_extracts_ag_grid_state_sort_model(self) -> None:
        response = _GridResponse(
            grid_state={
                "sort": {
                    "sortModel": [
                        {"colId": "Quality_Rank", "sort": "asc"},
                    ]
                }
            }
        )
        self.assertEqual(
            aggrid_sort_model(response),
            [{"colId": "Quality_Rank", "sort": "asc"}],
        )

    def test_extracts_column_state_sort(self) -> None:
        response = _GridResponse(
            columns_state=[
                {"colId": "Player_Elo_Rank"},
                {"colId": "Quality_Rank", "sort": "desc", "sortIndex": 0},
            ]
        )
        self.assertEqual(
            aggrid_sort_model(response),
            [{"colId": "Quality_Rank", "sort": "desc"}],
        )

    def test_apply_sort_to_polars_frame(self) -> None:
        frame = pl.DataFrame({
            "Player_Elo_Rank": [1, 2, 3],
            "Quality_Rank": [3, 1, 2],
            "Name": ["A", "B", "C"],
        })
        out = apply_aggrid_sort_model(
            frame, [{"colId": "Quality_Rank", "sort": "asc"}],
        )
        self.assertEqual(out["Name"].to_list(), ["B", "C", "A"])

    def test_remember_resets_when_grid_key_changes(self) -> None:
        session: dict = {}
        default = [{"colId": "Rank", "sort": "asc"}]
        remember_leaderboard_sort(
            session,
            "players_v1",
            _GridResponse(grid_state={"sort": {"sortModel": [{"colId": "Quality_Rank", "sort": "desc"}]}}),
            default,
        )
        reset = remember_leaderboard_sort(session, "pairs_v1", None, default)
        self.assertEqual(reset, default)

    def test_grid_options_keep_last_sort(self) -> None:
        options = {
            "columnDefs": [
                {"field": "Rank", "sort": "asc"},
                {"field": "Quality_Rank"},
            ]
        }
        apply_sort_model_to_grid_options(
            options, [{"colId": "Quality_Rank", "sort": "desc"}],
        )
        by_field = {col["field"]: col for col in options["columnDefs"]}
        self.assertNotIn("sort", by_field["Rank"])
        self.assertEqual(by_field["Quality_Rank"]["sort"], "desc")


if __name__ == "__main__":
    unittest.main()

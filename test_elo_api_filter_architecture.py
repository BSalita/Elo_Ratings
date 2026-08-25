from __future__ import annotations

import ast
import inspect
import pathlib
import unittest
from unittest.mock import Mock, patch

import polars as pl

import elo_mcp_server
import ffbridge_api_server
from ffbridge_report_service import filter_results, resolve_series_id
from elo_filter_common import (
    filter_acbl_leaderboard,
    filter_ffbridge_leaderboard,
)


class SharedFilterTests(unittest.TestCase):
    def test_acbl_filters_name_number_and_masterpoints(self) -> None:
        source = pl.DataFrame(
            {
                "Player_Name": ["Alice Alpha", "Alice Beta", "Bob"],
                "Player_ID": ["123", "456", "789"],
                "MasterPoints": [75.0, 150.0, 75.0],
            }
        )
        actual = filter_acbl_leaderboard(
            source,
            rating_type="Players",
            player_name="Alce Alpa",
            player_number="123",
            masterpoints_range="50-100",
        )
        self.assertEqual(actual["Player_ID"].to_list(), ["123"])

    def test_pair_number_matches_whole_member_id(self) -> None:
        source = pl.DataFrame(
            {
                "Pair_Name": ["A / B", "C / D"],
                "Pair_ID": ["12-345", "123-45"],
            }
        )
        actual = filter_ffbridge_leaderboard(
            source,
            rating_type="Pairs",
            player_number="12",
        )
        self.assertEqual(actual["Pair_ID"].to_list(), ["12-345"])

    def test_ffbridge_tournament_and_club_filters_are_fuzzy(self) -> None:
        source = pl.DataFrame(
            {
                "series_id": [386, 3],
                "club_name": ["Bridge Club Levallois Perret", "Other Club"],
            }
        )
        actual = filter_results(
            source,
            series_id="Octpus",
            club="Levalois Peret",
        )
        self.assertEqual(resolve_series_id("Octpus"), 386)
        self.assertEqual(actual.height, 1)

    def test_tournament_name_filters_apply_to_result_rows(self) -> None:
        source = pl.DataFrame(
            {
                "tournament_name": [
                    "Simultané Octopus - 12 Janvier",
                    "Rondes de France",
                ],
                "club_name": [
                    "Bridge Club Levallois Perret",
                    "Bridge Club Levallois Perret",
                ],
            }
        )
        contains = filter_results(
            source,
            tournament_contains="simultane octopus",
            club="Levalois Peret",
        )
        exact = filter_results(
            source,
            tournament="Rondes de France",
        )
        self.assertEqual(contains.height, 1)
        self.assertEqual(exact["tournament_name"].to_list(), ["Rondes de France"])


class FirstPartyApiBoundaryTests(unittest.TestCase):
    def test_mcp_has_no_report_or_data_library_imports(self) -> None:
        source_path = pathlib.Path(elo_mcp_server.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        forbidden = {
            "polars",
            "duckdb",
            "ffbridge_report_service",
            "acbl_api_server",
            "elo_ffbridge_classic",
            "elo_ffbridge_lancelot",
        }
        self.assertTrue(imported.isdisjoint(forbidden), imported & forbidden)

    def test_mcp_schema_exposes_tournament_filters(self) -> None:
        for tool in (
            elo_mcp_server.ffbridge_top_players,
            elo_mcp_server.ffbridge_top_pairs,
            elo_mcp_server.ffbridge_top_players_v2,
            elo_mcp_server.ffbridge_top_pairs_v2,
        ):
            parameters = inspect.signature(tool).parameters
            self.assertIn("tournament", parameters)
            self.assertIn("tournament_contains", parameters)
        self.assertEqual(
            elo_mcp_server.MCP_SCHEMA_VERSION,
            "2026-08-25-tournament-v2",
        )

    @patch("elo_mcp_server.requests.get")
    def test_ffbridge_mcp_calls_our_rest_api(self, mock_get: Mock) -> None:
        response = Mock()
        response.json.return_value = {"rows": []}
        mock_get.return_value = response

        elo_mcp_server.ffbridge_top_players(
            tournament_contains="Simultane Octopus",
            player_number="123",
        )

        url = mock_get.call_args.args[0]
        params = mock_get.call_args.kwargs["params"]
        self.assertEqual(
            url,
            f"{elo_mcp_server.FFBRIDGE_API_BASE_URL}/ffbridge/report",
        )
        self.assertEqual(params["tournament_contains"], "Simultane Octopus")
        self.assertEqual(params["player_number"], "123")

    @patch("elo_mcp_server.requests.get")
    def test_acbl_mcp_passes_all_sidebar_filters_to_our_api(
        self,
        mock_get: Mock,
    ) -> None:
        response = Mock()
        response.json.return_value = {"rows": []}
        mock_get.return_value = response

        elo_mcp_server.acbl_report(
            date_range="Last 1 year",
            player_name="Alice",
            player_number="123",
            masterpoints_range="50-100",
        )

        url = mock_get.call_args.args[0]
        params = mock_get.call_args.kwargs["params"]
        self.assertEqual(
            url,
            f"{elo_mcp_server.ACBL_API_BASE_URL}/acbl/report",
        )
        self.assertEqual(params["date_range"], "Last 1 year")
        self.assertEqual(params["player_number"], "123")
        self.assertEqual(params["masterpoints_range"], "50-100")

    @patch("ffbridge_api_server.reports.run_leaderboard_report")
    def test_ffbridge_api_delegates_to_shared_report_service(
        self,
        run_report: Mock,
    ) -> None:
        run_report.return_value = {"rows": []}
        ffbridge_api_server.leaderboard_report(
            rating_type="Pairs",
            score="Handicap",
            top_n=100,
            min_games=4,
            prior_sessions=50,
            api_backend="classic",
            series_id="10",
            tournament_name=None,
            tournament=None,
            tournament_contains=None,
            club="Octopus Club",
            player_name="Alice",
            player_number="123",
            date_range="Current FFBridge year",
            date_from=None,
            date_to=None,
        )
        self.assertEqual(run_report.call_args.kwargs["rating"], "Pairs")
        self.assertEqual(run_report.call_args.kwargs["series_id"], "10")
        self.assertEqual(run_report.call_args.kwargs["player_number"], "123")


if __name__ == "__main__":
    unittest.main()

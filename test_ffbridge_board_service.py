from __future__ import annotations

import pathlib
import tempfile
import unittest

import ffbridge_api_server
import ffbridge_board_service as boards


def _ranking_row(team_id: int, orientation: str, club_code: int) -> dict:
    return {
        "orientation": orientation,
        "rank": team_id,
        "simultaneousId": club_code,
        "team": {
            "id": team_id,
            "label": f"Pair {team_id}",
            "orientation": orientation,
        },
    }


def _score(result_id: int, board_number: int, ns_id: int, ew_id: int) -> dict:
    players = {
        "northPlayer": {
            "id": ns_id * 10 + 1,
            "migrationId": ns_id * 100 + 1,
            "firstName": "North",
            "lastName": str(ns_id),
        },
        "southPlayer": {
            "id": ns_id * 10 + 2,
            "migrationId": ns_id * 100 + 2,
            "firstName": "South",
            "lastName": str(ns_id),
        },
        "eastPlayer": {
            "id": ew_id * 10 + 1,
            "migrationId": ew_id * 100 + 1,
            "firstName": "East",
            "lastName": str(ew_id),
        },
        "westPlayer": {
            "id": ew_id * 10 + 2,
            "migrationId": ew_id * 100 + 2,
            "firstName": "West",
            "lastName": str(ew_id),
        },
    }
    return {
        "id": result_id,
        "boardNumber": board_number,
        "board": {
            "id": board_number * 1000,
            "boardNumber": board_number,
            "deal": "N:AKQ.JT9.876.543 ...",
        },
        "contract": "4S",
        "declarer": "N",
        "result": "=",
        "tricks": 10,
        "nsScore": "420",
        "ewScore": "",
        "nsNote": 65.5,
        "ewNote": 34.5,
        "lineup": {
            **players,
            "segment": {
                "game": {
                    "homeTeam": {
                        "id": ns_id,
                        "label": f"Pair {ns_id}",
                        "orientation": "NS",
                    },
                    "awayTeam": {
                        "id": ew_id,
                        "label": f"Pair {ew_id}",
                        "orientation": "EW",
                    },
                }
            },
        },
    }


class FakeClient:
    ranking_calls = 0
    score_calls: list[int] = []

    @classmethod
    def get_session_ranking(cls, _session_id: int) -> list[dict]:
        cls.ranking_calls += 1
        return [
            _ranking_row(10, "NS", 1001),
            _ranking_row(20, "EW", 1002),
            _ranking_row(30, "NS", 1003),
            _ranking_row(40, "EW", 1004),
        ]

    @classmethod
    def get_team_session_scores(cls, team_id: int, _session_id: int) -> list[dict]:
        cls.score_calls.append(team_id)
        opponent = {10: 20, 30: 40}[team_id]
        return [
            _score(team_id * 100 + 4, 4, team_id, opponent),
            _score(team_id * 100 + 5, 5, team_id, opponent),
        ]


class FFBridgeBoardServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeClient.ranking_calls = 0
        FakeClient.score_calls = []

    def test_returns_requested_board_across_all_clubs_and_reuses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = pathlib.Path(temporary)
            first = boards.get_board_results(
                "123",
                4,
                cache_dir=cache,
                workers=2,
                client=FakeClient,
            )
            second = boards.get_board_results(
                "123",
                4,
                cache_dir=cache,
                workers=2,
                client=FakeClient,
            )

        self.assertEqual(first, second)
        self.assertEqual(first["row_count"], 2)
        self.assertEqual(first["covering_team_count"], 2)
        self.assertEqual(
            {row["Club_Code_NS"] for row in first["rows"]},
            {1001, 1003},
        )
        self.assertEqual(
            {row["Club_Code_EW"] for row in first["rows"]},
            {1002, 1004},
        )
        self.assertTrue(all(row["Board"] == 4 for row in first["rows"]))
        self.assertEqual(FakeClient.ranking_calls, 1)
        self.assertCountEqual(FakeClient.score_calls, [10, 30])

    def test_api_registers_nationwide_board_results_route(self) -> None:
        paths = {route.path for route in ffbridge_api_server.app.routes}
        self.assertIn("/ffbridge/board-results", paths)


if __name__ == "__main__":
    unittest.main()

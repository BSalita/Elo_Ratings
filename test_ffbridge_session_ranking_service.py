from __future__ import annotations

import unittest

import polars as pl

import ffbridge_api_server
import ffbridge_session_ranking_service as rankings


SESSION_ID = "300749"
CLUB_CODE = "5802079"
OTHER_CLUB = "1234567"


def _pair(
    index: int,
    *,
    club_code: str,
    national_scratch_pct: float,
    national_scratch_rank: int,
    national_handicap_pct: float,
    national_handicap_rank: int,
    club_scratch_pct: float,
    club_scratch_rank: int,
    club_handicap_pct: float,
    club_handicap_rank: int,
    iv_bonus: float,
    player1_id: str | None = None,
    player2_id: str | None = None,
    player1_lancelot_id: str | None = None,
    player2_lancelot_id: str | None = None,
    player1_name: str | None = None,
    player2_name: str | None = None,
    team_id: str | None = None,
) -> dict:
    return {
        "tournament_id": SESSION_ID,
        "date": "2026-08-17T00:00:00+02:00",
        "tournament_name": "Simultané Octopus",
        "team_id": team_id or str(15_000_000 + index),
        "player1_id": player1_id or str(1000 + index),
        "player2_id": player2_id or str(2000 + index),
        "player1_lancelot_id": player1_lancelot_id or str(3000 + index),
        "player2_lancelot_id": player2_lancelot_id or str(4000 + index),
        "player1_name": player1_name or f"Player {index}A",
        "player2_name": player2_name or f"Player {index}B",
        "club_code": club_code,
        "National_Scratch_Pct": national_scratch_pct,
        "National_Scratch_Rank": national_scratch_rank,
        "National_Handicap_Pct": national_handicap_pct,
        "National_Handicap_Rank": national_handicap_rank,
        "Club_Scratch_Pct": club_scratch_pct,
        "Club_Scratch_Rank": club_scratch_rank,
        "Club_Handicap_Pct": club_handicap_pct,
        "Club_Handicap_Rank": club_handicap_rank,
        "iv_bonus": iv_bonus,
    }


def _session_frame() -> pl.DataFrame:
    rows = [
        _pair(
            1,
            club_code=CLUB_CODE,
            national_scratch_pct=62.45,
            national_scratch_rank=6,
            national_handicap_pct=69.45,
            national_handicap_rank=4,
            club_scratch_pct=62.5,
            club_scratch_rank=1,
            club_handicap_pct=69.5,
            club_handicap_rank=2,
            iv_bonus=7.0,
        ),
        _pair(
            2,
            club_code=CLUB_CODE,
            national_scratch_pct=61.739999999999995,
            national_scratch_rank=7,
            national_handicap_pct=71.74,
            national_handicap_rank=1,
            club_scratch_pct=61.57,
            club_scratch_rank=2,
            club_handicap_pct=71.57,
            club_handicap_rank=1,
            iv_bonus=10.0,
            player1_id="597539",
            player2_id="99497",
            player1_lancelot_id="246273",
            player2_lancelot_id="33351",
            player1_name="Robert SALITA",
            player2_name="Christian Jacoupy",
            team_id="15106224",
        ),
    ]
    for index in range(3, 13):
        rows.append(
            _pair(
                index,
                club_code=CLUB_CODE,
                national_scratch_pct=60.0 - index,
                national_scratch_rank=10 + index,
                national_handicap_pct=65.0 - index,
                national_handicap_rank=10 + index,
                club_scratch_pct=60.0 - index,
                club_scratch_rank=index,
                club_handicap_pct=65.0 - index,
                club_handicap_rank=index,
                iv_bonus=5.0,
            )
        )
    for index in range(13, 93):
        rows.append(
            _pair(
                index,
                club_code=OTHER_CLUB,
                national_scratch_pct=50.0 - (index / 100),
                national_scratch_rank=20 + index,
                national_handicap_pct=55.0 - (index / 100),
                national_handicap_rank=20 + index,
                club_scratch_pct=48.0 - (index / 100),
                club_scratch_rank=index - 11,
                club_handicap_pct=53.0 - (index / 100),
                club_handicap_rank=index - 11,
                iv_bonus=5.0,
            )
        )
    return pl.DataFrame(rows)


def _ranking(**kwargs):
    return rankings.get_session_ranking(
        SESSION_ID,
        results_df=_session_frame(),
        meta={"built_at": "2026-08-29T12:15:54+00:00"},
        **kwargs,
    )


def _salita(rows: list[dict]) -> dict:
    matches = [
        row
        for row in rows
        if row["player1_lancelot_id"] == "246273"
        and row["player2_lancelot_id"] == "33351"
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one SALITA row, got {len(matches)}")
    return matches[0]


class FFBridgeSessionRankingServiceTests(unittest.TestCase):
    def test_national_field_matches_published_salita_row(self) -> None:
        payload = _ranking(scope="national")
        self.assertEqual(payload["field_size"], 92)
        self.assertEqual(payload["row_count"], 92)
        self.assertEqual(payload["scope"], "national")
        self.assertIsNone(payload["club_code"])
        salita = _salita(payload["rows"])
        self.assertEqual(salita["scratch_pct"], 61.74)
        self.assertEqual(salita["scratch_rank"], 7)
        self.assertEqual(salita["handicap_pct"], 71.74)
        self.assertEqual(salita["handicap_rank"], 1)
        self.assertEqual(salita["iv_bonus"], 10.0)
        self.assertEqual(salita["player1_id"], "597539")
        self.assertEqual(salita["player2_id"], "99497")
        self.assertEqual(salita["field_size"], 92)

    def test_club_scope_requires_club_code_and_returns_club_field(self) -> None:
        with self.assertRaisesRegex(ValueError, "club_code is required"):
            _ranking(scope="club")
        payload = _ranking(scope="club", club_code="05802079")
        self.assertEqual(payload["field_size"], 12)
        self.assertEqual(payload["club_code"], CLUB_CODE)
        salita = _salita(payload["rows"])
        self.assertEqual(salita["scratch_pct"], 61.57)
        self.assertEqual(salita["scratch_rank"], 2)
        self.assertEqual(salita["handicap_pct"], 71.57)
        self.assertEqual(salita["handicap_rank"], 1)
        self.assertEqual(salita["field_size"], 12)

    def test_api_registers_session_ranking_route(self) -> None:
        paths = {route.path for route in ffbridge_api_server.app.routes}
        self.assertIn("/ffbridge/session-ranking", paths)


if __name__ == "__main__":
    unittest.main()

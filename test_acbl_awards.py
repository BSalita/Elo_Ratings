from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import polars as pl

from acbl_awards import (
    AWARD_COLORS,
    attach_award_totals,
    attach_session_awards,
    awards_from_tournament_overalls,
    empty_awards,
    event_id_from_session_id,
    load_player_awards,
    pair_award_totals,
    pivot_award_totals,
    write_player_awards_sidecar,
)
from elo_common import is_numeric_column_name
from elo_session_common import summarize_acbl_sessions


def _award(
    session_id: str,
    player_id: str,
    mp_won: float,
    *,
    color: str = "Platinum",
    event_id: str | None = None,
    pair_ids: str = "",
) -> dict:
    return {
        "session_id": session_id,
        "event_id": event_id or event_id_from_session_id(session_id),
        "player_id": player_id,
        "pair_ids": pair_ids,
        "mp_color": color,
        "mp_won": mp_won,
    }


class AcblAwardTests(unittest.TestCase):
    def test_event_id_strips_trailing_session_number(self) -> None:
        self.assertEqual(event_id_from_session_id("NABC261-SILO-4"), "NABC261-SILO")
        self.assertEqual(event_id_from_session_id("2509101-271B-2"), "2509101-271B")
        self.assertEqual(event_id_from_session_id("1493280"), "1493280")

    def test_overalls_emit_one_row_per_player(self) -> None:
        rows = awards_from_tournament_overalls(
            "NABC261-SILO-4",
            [
                {
                    "mp_won": 26.32,
                    "mp_color": "Platinum",
                    "players": [1709925, 6811434],
                }
            ],
            event_id="NABC261-SILO",
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["player_id"] for row in rows}, {"1709925", "6811434"})
        self.assertEqual(rows[0]["pair_ids"], "1709925-6811434")
        self.assertEqual(rows[0]["event_id"], "NABC261-SILO")
        self.assertEqual(rows[0]["mp_won"], 26.32)

    def test_event_totals_use_max_not_session_sum(self) -> None:
        awards = pl.DataFrame(
            [
                _award("NABC261-SILO-1", "1709925", 2.57),
                _award("NABC261-SILO-2", "1709925", 2.57),
                _award("NABC261-SILO-3", "1709925", 3.86),
                _award("NABC261-SILO-4", "1709925", 26.32),
                _award("NABC261-NAIL-1", "1709925", 10.0),
            ]
        )
        totals = pivot_award_totals(awards, id_column="player_id")
        row = totals.filter(pl.col("player_id") == "1709925").row(0, named=True)
        self.assertEqual(row["Platinum"], 36.32)
        self.assertEqual(row["Gold"], 0.0)

    def test_session_grid_keeps_listed_award(self) -> None:
        awards = pl.DataFrame(
            [
                _award("NABC261-SILO-1", "1709925", 2.57),
                _award("NABC261-SILO-4", "1709925", 26.32),
            ]
        )
        detail = pl.DataFrame(
            {
                "Session": ["NABC261-SILO-1", "NABC261-SILO-4"],
                "Board": [1, 1],
            }
        )
        attached = attach_session_awards(detail, awards, player_id="1709925")
        self.assertEqual(attached["Platinum"].to_list(), [2.57, 26.32])

    def test_pair_totals_require_both_players(self) -> None:
        awards = pl.DataFrame(
            [
                _award("NABC261-SILO-4", "1709925", 26.32, pair_ids="1709925-6811434"),
                _award("NABC261-SILO-4", "6811434", 26.32, pair_ids="1709925-6811434"),
                _award("NABC261-NAIL-1", "1709925", 10.0, pair_ids="111-1709925"),
            ]
        )
        totals = pair_award_totals(awards, ["1709925-6811434"])
        row = totals.row(0, named=True)
        self.assertEqual(row["Pair_IDs"], "1709925-6811434")
        self.assertEqual(row["Platinum"], 26.32)

    def test_leaderboard_inserts_colors_after_masterpoints(self) -> None:
        leaderboard = pl.DataFrame(
            {
                "Player_ID": ["1709925"],
                "Player_Name": ["Silverman"],
                "MasterPoints": [8000],
                "MasterPoint_Rank": [1],
                "Sessions_Played": [20],
            }
        )
        awards = pl.DataFrame([_award("NABC261-SILO-4", "1709925", 26.32)])
        attached = attach_award_totals(
            leaderboard,
            awards,
            rating_type="Players",
            session_ids=["NABC261-SILO-4"],
        )
        self.assertEqual(
            attached.columns[attached.columns.index("MasterPoint_Rank") + 1 : attached.columns.index("MasterPoint_Rank") + 5],
            list(AWARD_COLORS),
        )
        self.assertEqual(attached["Platinum"][0], 26.32)

    def test_filtered_sessions_exclude_other_events(self) -> None:
        leaderboard = pl.DataFrame({"Player_ID": ["1709925"], "MasterPoints": [1]})
        awards = pl.DataFrame(
            [
                _award("NABC261-SILO-4", "1709925", 26.32),
                _award("NABC261-NAIL-1", "1709925", 10.0),
            ]
        )
        attached = attach_award_totals(
            leaderboard,
            awards,
            rating_type="Players",
            session_ids=["NABC261-SILO-4"],
        )
        self.assertEqual(attached["Platinum"][0], 26.32)

    def test_empty_awards_still_add_zero_columns(self) -> None:
        leaderboard = pl.DataFrame({"Player_ID": ["1"], "MasterPoints": [10]})
        attached = attach_award_totals(leaderboard, empty_awards(), rating_type="Players")
        self.assertEqual(attached["Platinum"][0], 0.0)
        self.assertTrue(all(color in attached.columns for color in AWARD_COLORS))

    def test_sidecar_round_trip(self) -> None:
        awards = pl.DataFrame([_award("NABC261-SILO-4", "1709925", 26.32)])
        with tempfile.TemporaryDirectory() as tmp:
            path = write_player_awards_sidecar(
                awards, Path(tmp) / "acbl_tournament_player_awards.parquet"
            )
            loaded = load_player_awards(Path(tmp), "tournament")
            self.assertEqual(path.name, "acbl_tournament_player_awards.parquet")
            self.assertEqual(loaded["mp_won"][0], 26.32)

    def test_award_colors_sort_numerically(self) -> None:
        for color in AWARD_COLORS:
            self.assertTrue(is_numeric_column_name(color))

    def test_session_summary_keeps_award_columns(self) -> None:
        detail = pl.DataFrame(
            {
                "Date": [datetime(2026, 3, 6), datetime(2026, 3, 6)],
                "Session": ["NABC261-SILO-4", "NABC261-SILO-4"],
                "Board": [1, 2],
                "Platinum": [26.32, 26.32],
                "Gold": [0.0, 0.0],
                "Red": [0.0, 0.0],
                "Black": [0.0, 0.0],
            }
        )
        summary = summarize_acbl_sessions(detail, "Tournament")
        self.assertEqual(summary["Platinum"][0], 26.32)
        self.assertEqual(
            summary.columns[-5:],
            ["Platinum", "Gold", "Red", "Black", "Results_URL"],
        )


if __name__ == "__main__":
    unittest.main()

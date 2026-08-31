from __future__ import annotations

from datetime import datetime
import unittest

import duckdb
import polars as pl

from acbl_api_server import (
    QUALITY_METRIC_DEFINITIONS,
    _required_columns_for_mode,
    generate_top_pairs_sql,
    generate_top_players_sql,
)


def _quality_fixture() -> pl.DataFrame:
    rows = [
        {
            "Date": datetime(2026, 1, 1),
            "session_id": "session-1",
            "Round": 1,
            "Board": 1,
            "Declarer_Direction": "N",
            "Declarer_Pair_Direction": "NS",
            "BidSuit": "S",
            "ParContracts": [{"Strain": "S"}, {"Strain": "N"}],
            "DD_Tricks_Diff": 2,
            "DD_Score_NS": -100,
            "DD_Score_EW": 100,
            "Par_NS": -100,
            "Par_EW": 100,
            "DD_Score_Declarer": -100,
            "Par_Declarer": -100,
        },
        {
            "Date": datetime(2026, 1, 1),
            "session_id": "session-1",
            "Round": 1,
            "Board": 2,
            "Declarer_Direction": "E",
            "Declarer_Pair_Direction": "EW",
            "BidSuit": "H",
            "ParContracts": [{"Strain": "S"}],
            "DD_Tricks_Diff": -1,
            "DD_Score_NS": 50,
            "DD_Score_EW": -50,
            "Par_NS": 100,
            "Par_EW": -100,
            "DD_Score_Declarer": 40,
            "Par_Declarer": -50,
        },
        {
            "Date": datetime(2026, 1, 1),
            "session_id": "session-1",
            "Round": 1,
            "Board": 3,
            "Declarer_Direction": None,
            "Declarer_Pair_Direction": None,
            "BidSuit": None,
            "ParContracts": None,
            "DD_Tricks_Diff": None,
            "DD_Score_NS": None,
            "DD_Score_EW": 0,
            "Par_NS": 0,
            "Par_EW": 0,
            "DD_Score_Declarer": None,
            "Par_Declarer": None,
        },
    ]
    for row in rows:
        for seat, player_id in zip("NESW", ("1", "2", "3", "4"), strict=True):
            row[f"Player_ID_{seat}"] = player_id
            row[f"Player_Name_{seat}"] = f"Player {seat}"
            row[f"MasterPoints_{seat}"] = 100.0 + int(player_id)
            row[f"Elo_R_{seat}"] = 1500.0 + int(player_id)
        row["Elo_R_NS"] = 1510.0
        row["Elo_R_EW"] = 1490.0
    return pl.DataFrame(rows)


def _run(sql: str) -> pl.DataFrame:
    con = duckdb.connect()
    try:
        con.register("self", _quality_fixture())
        return con.execute(sql).pl()
    finally:
        con.close()


class AcblQualityMetricSqlTests(unittest.TestCase):
    def test_player_sql_applies_pair_metrics_to_members_and_tdd_to_declarer(self) -> None:
        result = _run(
            generate_top_players_sql(
                top_n=10,
                min_sessions=1,
                rating_method="Latest",
                elo_rating_type="Current Rating (End of Session)",
            )
        )
        players = {row["Player_ID"]: row for row in result.to_dicts()}

        for player_id in ("1", "3"):
            self.assertEqual(players[player_id]["Par_Suit_Rate_Pct"], 100.0)
            self.assertEqual(players[player_id]["Sacrifice_Rate_Pct"], 100.0)
        for player_id in ("2", "4"):
            self.assertEqual(players[player_id]["Par_Suit_Rate_Pct"], 0.0)
            self.assertEqual(players[player_id]["Sacrifice_Rate_Pct"], 0.0)
        self.assertEqual(players["1"]["Par_Contract_Rate_Pct"], 50.0)
        self.assertEqual(players["2"]["Par_Contract_Rate_Pct"], 100.0)
        self.assertEqual(players["1"]["DD_Tricks_Diff_Avg"], 2.0)
        self.assertEqual(players["2"]["DD_Tricks_Diff_Avg"], -1.0)
        self.assertIsNone(players["3"]["DD_Tricks_Diff_Avg"])
        self.assertIsNone(players["4"]["DD_Tricks_Diff_Avg"])

    def test_pair_sql_uses_directional_and_declaration_denominators(self) -> None:
        result = _run(
            generate_top_pairs_sql(
                top_n=10,
                min_sessions=1,
                rating_method="Latest",
                elo_rating_type="Current Rating (End of Session)",
            )
        )
        pairs = {row["Pair_IDs"]: row for row in result.to_dicts()}
        ns = pairs["1-3"]
        ew = pairs["2-4"]

        self.assertEqual(ns["Par_Contract_Rate_Pct"], 50.0)
        self.assertEqual(ew["Par_Contract_Rate_Pct"], 100.0)
        self.assertEqual(ns["Par_Suit_Rate_Pct"], 100.0)
        self.assertEqual(ew["Par_Suit_Rate_Pct"], 0.0)
        self.assertEqual(ns["Sacrifice_Rate_Pct"], 100.0)
        self.assertEqual(ew["Sacrifice_Rate_Pct"], 0.0)
        self.assertEqual(ns["DD_Tricks_Diff_Avg"], 2.0)
        self.assertEqual(ew["DD_Tricks_Diff_Avg"], -1.0)

    def test_required_raw_columns_and_metadata_are_explicit(self) -> None:
        required = set(
            _required_columns_for_mode(
                "Players", "Current Rating (End of Session)"
            )
        )
        self.assertTrue(
            {
                "Declarer_Direction",
                "Declarer_Pair_Direction",
                "BidSuit",
                "ParContracts",
                "DD_Score_NS",
                "DD_Score_EW",
                "Par_NS",
                "Par_EW",
                "DD_Score_Declarer",
                "Par_Declarer",
            }.issubset(required)
        )
        self.assertEqual(
            set(QUALITY_METRIC_DEFINITIONS),
            {
                "DD_Tricks_Diff_Avg",
                "Par_Contract_Rate_Pct",
                "Par_Suit_Rate_Pct",
                "Sacrifice_Rate_Pct",
            },
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from datetime import datetime
import unittest

import duckdb
import polars as pl

from acbl_api_server import acbl_favorites_meta
from elo_favorites import (
    ACBL_MACRO_KEYS,
    FFBRIDGE_MACRO_KEYS,
    button_prompt_ids,
    extract_macros,
    flatten_favorites,
    lint_favorites,
    load_favorites,
    process_sql_macros,
    run_favorite,
    vetted_prompt_sql,
)
from ffbridge_report_service import (
    ffbridge_favorites_meta,
    list_favorites as ffbridge_list_favorites,
    run_top_players_favorite,
    run_top_pairs_favorite,
)


def _acbl_quality_fixture() -> pl.DataFrame:
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


def _acbl_meta(**overrides) -> dict:
    meta = acbl_favorites_meta(
        top_n=10,
        min_sessions=1,
        rating_method="Latest",
        elo_rating_type="Current Rating (End of Session)",
        rating_type="Players",
        prior_anchor=None,
        prior_sessions=0,
        min_skill_z=-90.0,
    )
    meta.update(overrides)
    return meta


def _run_acbl(prompt_id: str, meta: dict) -> pl.DataFrame:
    favorites = load_favorites("acbl")
    con = duckdb.connect()
    try:
        con.register("self", _acbl_quality_fixture())
        result, _sql = run_favorite(con, favorites, prompt_id, meta)
        return result
    finally:
        con.close()


class MacroTests(unittest.TestCase):
    def test_replaces_known_keys_and_leaves_missing(self) -> None:
        sql = "SELECT Elo_R_N{Elo_Suffix}, {Missing} FROM self LIMIT {Top_N}"
        out = process_sql_macros(sql, {"Elo_Suffix": "_Before", "Top_N": 5})
        self.assertEqual(out, "SELECT Elo_R_N_Before, {Missing} FROM self LIMIT 5")

    def test_skips_none_values(self) -> None:
        sql = "SELECT {Prior_Anchor} AS x"
        self.assertEqual(process_sql_macros(sql, {"Prior_Anchor": None}), sql)

    def test_button_prompt_uses_rating_type(self) -> None:
        favorites = load_favorites("acbl")
        self.assertEqual(
            button_prompt_ids(favorites, "Leaderboard", {"Rating_Type": "Players"}),
            ["Top_Players"],
        )
        self.assertEqual(
            button_prompt_ids(favorites, "Leaderboard", {"Rating_Type": "Pairs"}),
            ["Top_Pairs"],
        )

    def test_extract_macros(self) -> None:
        self.assertEqual(extract_macros("a {Elo_Kind} b {Top_N}"), {"Elo_Kind", "Top_N"})


class FavoritesLintTests(unittest.TestCase):
    def test_acbl_and_ffbridge_lint_clean(self) -> None:
        self.assertEqual(lint_favorites("acbl"), [])
        self.assertEqual(lint_favorites("ffbridge"), [])

    def test_documented_macros_cover_sql(self) -> None:
        for org, allowed in (("acbl", ACBL_MACRO_KEYS), ("ffbridge", FFBRIDGE_MACRO_KEYS)):
            sql = vetted_prompt_sql(load_favorites(org), "Top_Players")
            self.assertTrue(extract_macros(sql) <= allowed)


class AcblGoldenRowTests(unittest.TestCase):
    def test_player_quality_denominators(self) -> None:
        result = _run_acbl("Top_Players", _acbl_meta())
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

    def test_pair_quality_denominators(self) -> None:
        result = _run_acbl("Top_Pairs", _acbl_meta(Rating_Type="Pairs"))
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

    def test_skill_gate_sql_stays_in_favorites(self) -> None:
        favorites = load_favorites("acbl")
        sql = vetted_prompt_sql(favorites, "Top_Players")
        self.assertIn("Skill_Z >= {Min_Skill_Z}", sql)
        ungated = process_sql_macros(sql, _acbl_meta(Min_Skill_Z=-90.0))
        self.assertIn("(-90.0 <= -90 OR Skill_Z >= -90.0)", ungated)
        gated = process_sql_macros(sql, _acbl_meta(Min_Skill_Z=0.0))
        self.assertIn("(0.0 <= -90 OR Skill_Z >= 0.0)", gated)

    def test_substituted_sql_contains_rating_columns(self) -> None:
        favorites = load_favorites("acbl")
        meta = _acbl_meta(
            elo_rating_type="Rating at Start of Session",
        )
        meta = acbl_favorites_meta(
            top_n=10,
            min_sessions=1,
            rating_method="Latest",
            elo_rating_type="Rating at Start of Session",
            rating_type="Players",
            prior_anchor=None,
            prior_sessions=0,
            min_skill_z=-90.0,
        )
        sql = process_sql_macros(vetted_prompt_sql(favorites, "Top_Players"), meta)
        self.assertIn("Elo_R_N_Before", sql)
        self.assertNotIn("{Elo_Col_N}", sql)


class FfbridgeGoldenRowTests(unittest.TestCase):
    def test_top_players_order_and_averages(self) -> None:
        rows = []
        for i in range(20):
            rows.append(
                {
                    "player1_id": "1",
                    "player2_id": "2",
                    "player1_name": "One",
                    "player2_name": "Two",
                    "player1_scratch_elo_after": 1400.0,
                    "player2_scratch_elo_after": 1300.0,
                    "player1_handicap_elo_after": 1400.0,
                    "player2_handicap_elo_after": 1300.0,
                    "Club_Scratch_Pct": None,
                    "Club_Handicap_Pct": None,
                    "National_Scratch_Pct": 60.0,
                    "National_Handicap_Pct": 60.0,
                    "iv_bonus": 0.0,
                    "score_status": "official",
                    "date": f"2026-01-{i + 1:02d}",
                }
            )
            rows.append(
                {
                    "player1_id": "3",
                    "player2_id": "4",
                    "player1_name": "Three",
                    "player2_name": "Four",
                    "player1_scratch_elo_after": 1200.0,
                    "player2_scratch_elo_after": 1100.0,
                    "player1_handicap_elo_after": 1200.0,
                    "player2_handicap_elo_after": 1100.0,
                    "Club_Scratch_Pct": None,
                    "Club_Handicap_Pct": None,
                    "National_Scratch_Pct": 54.0,
                    "National_Handicap_Pct": 54.0,
                    "iv_bonus": 0.0,
                    "score_status": "official",
                    "date": f"2026-01-{i + 1:02d}",
                }
            )
        table, sql, _anchor = run_top_players_favorite(
            pl.DataFrame(rows), top_n=2, min_games=10, prior_sessions=0
        )
        self.assertEqual(table.get_column("Player_ID").to_list(), ["1", "2"])
        self.assertEqual(table.get_column("Player_Elo").to_list(), [1400, 1300])
        self.assertIn("player1_scratch_elo_after", sql)
        self.assertNotIn("{Elo_Kind}", sql)

    def test_top_pairs_quality_rank_can_differ(self) -> None:
        results = pl.DataFrame(
            {
                "pair_id": ["B", "C", "A"],
                "pair_name": ["Bee", "See", "Aye"],
                "player1_id": ["1", "3", "5"],
                "player2_id": ["2", "4", "6"],
                "scratch_pair_elo": [1400.0, 1300.0, 1200.0],
                "handicap_pair_elo": [1400.0, 1300.0, 1200.0],
                "Club_Scratch_Pct": [None, None, None],
                "Club_Handicap_Pct": [None, None, None],
                "National_Scratch_Pct": [60.0, 58.0, 56.0],
                "National_Handicap_Pct": [60.0, 58.0, 56.0],
                "iv_bonus": [0.0, 0.0, 0.0],
                "score_status": ["official", "official", "official"],
                "date": ["2026-01-03", "2026-01-03", "2026-01-03"],
            }
        )
        quality = pl.DataFrame(
            {
                "pair_id": ["A", "B", "C"],
                "par_suit_rate": [0.9, 0.8, 0.7],
                "par_contract_rate": [0.9, 0.8, 0.7],
                "sacrifice_rate": [0.1, 0.9, 0.5],
                "dd_tricks_diff_avg": [0.9, 0.8, 0.7],
            }
        )
        table, _sql, _anchor = run_top_pairs_favorite(
            results, top_n=3, min_games=1, quality_df=quality
        )
        self.assertEqual(table.get_column("Pair_ID").to_list(), ["B", "C", "A"])
        self.assertEqual(table.get_column("Quality_Rank").to_list(), [2, 3, 1])

    def test_ffbridge_meta_kind(self) -> None:
        meta = ffbridge_favorites_meta(
            rating_type="Players",
            score="Handicap",
            top_n=10,
            min_games=5,
            prior_sessions=50,
        )
        self.assertEqual(meta["Elo_Kind"], "handicap")
        self.assertEqual(meta["Elo_Col_Name"], "HC_Player_Elo")


class CatalogTests(unittest.TestCase):
    def test_acbl_and_ffbridge_favorite_ids(self) -> None:
        acbl = flatten_favorites(load_favorites("acbl"))
        self.assertEqual({item["id"] for item in acbl}, {"Top_Players", "Top_Pairs"})
        acbl_sql = next(
            item["statements"][0]["sql"] for item in acbl if item["id"] == "Top_Players"
        )
        self.assertIn("{Elo_Col_N}", acbl_sql)
        self.assertIn("{Min_Skill_Z}", acbl_sql)
        listed = ffbridge_list_favorites()
        self.assertEqual(listed["organization"], "ffbridge")
        ids = {item["id"] for item in listed["favorites"]}
        self.assertEqual(ids, {"Top_Players", "Top_Pairs"})
        ff_sql = next(
            item["statements"][0]["sql"]
            for item in listed["favorites"]
            if item["id"] == "Top_Players"
        )
        self.assertIn("{Elo_Kind}", ff_sql)
        self.assertIn("{Min_Games}", ff_sql)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np
import polars as pl

import elo_brill
from elo_common import DEFAULT_ELO


def _board_rows(scores: dict[str, list[int]], tournament_id: str = "t1", event_time: str = "2026-11-01T11:00:00+00:00",
                declarers: list[str] | None = None, types: dict[str, str] | None = None) -> pl.DataFrame:
    """Augmented-looking board rows: one per player per board, scores listed by board."""
    rows = []
    for player, per_board in scores.items():
        for index, score in enumerate(per_board):
            declarer = (declarers or ["S"] * len(per_board))[index]
            rows.append({
                "tournament_id": tournament_id, "Event": "Daily MP — 2026-11-01", "Event_Type": "Daily MP",
                "Scoring": "MP", "Event_Time": event_time, "Date": event_time[:10],
                "Boards_In_Event": len(per_board), "Board": index + 1,
                "Player_ID": player, "Player_Name": player.split(":")[1],
                "Player_Type": (types or {}).get(player, elo_brill.HUMAN), "Table": 1, "Site_Rank": None,
                "Declarer_Direction": declarer, "Score_NS": score,
                "DD_Tricks_Diff": 1 if score > 0 else -1,
                "DD_Lead_Cost": 0 if score > 0 else 2, "Lead_Is_DD_Best": score > 0,
                "Is_Par_Contract": score > 0, "Is_Par_Suit": True, "Is_Par_Score": score >= 100,
                "Par_Diff_NS": score - 100, "Sacrifice_Opportunity": False, "Is_Sacrifice": False,
            })
    frame = pl.DataFrame(rows, schema_overrides={"Site_Rank": pl.Int32})
    return elo_brill.add_field_scores(frame)


class IdentityTests(unittest.TestCase):
    def test_human_id_is_stable_and_hides_the_account_id(self) -> None:
        a = elo_brill.player_identity({"participant_type": "human", "user_id": "abc-123", "display_name": "Ann"})
        b = elo_brill.player_identity({"participant_type": "human", "user_id": "abc-123", "display_name": "Ann B"})
        self.assertEqual(a[0], b[0])
        self.assertNotIn("abc-123", a[0])
        self.assertEqual(a[1:], ("Ann", elo_brill.HUMAN))

    def test_robot_is_its_robot_name(self) -> None:
        self.assertEqual(
            elo_brill.player_identity({"participant_type": "robot", "robot_name": "WBridge5", "display_name": "WBridge5"}),
            ("R:WBridge5", "WBridge5", elo_brill.ROBOT),
        )

    def test_event_type_drops_the_date(self) -> None:
        self.assertEqual(elo_brill.event_type("Daily MP — 2026-11-13"), "Daily MP")
        self.assertEqual(elo_brill.event_type("Daily IMP - strongest hand — 2026-11-13"), "Daily IMP - strongest hand")
        self.assertEqual(elo_brill.event_type("EBU Sunday"), "EBU Sunday")


class FieldScoreTests(unittest.TestCase):
    def test_imp_scale_is_the_wbf_table_at_every_band_edge(self) -> None:
        # (difference, IMPs) at each band's bottom and top edge: WBF table and Brill's js/scoring.js.
        top = [10, 40, 80, 120, 160, 210, 260, 310, 360, 420, 490, 590, 740, 890, 1090, 1290,
               1490, 1740, 1990, 2240, 2490, 2990, 3490, 3990]
        bottom = [0, 20, 50, 90, 130, 170, 220, 270, 320, 370, 430, 500, 600, 750, 900, 1100,
                  1300, 1500, 1750, 2000, 2250, 2500, 3000, 3500, 4000]
        for n, lo in enumerate(bottom):
            self.assertEqual(int(elo_brill.imps(lo)), n, lo)
            self.assertEqual(int(elo_brill.imps(-lo)), -n, -lo)
        for n, hi in enumerate(top):
            self.assertEqual(int(elo_brill.imps(hi)), n, hi)
            self.assertEqual(int(elo_brill.imps(-hi)), -n, -hi)
        self.assertEqual(int(elo_brill.imps(7600)), 24)
        self.assertEqual(elo_brill.imps(np.array([0, 10, 20, 40, 50, 80, 90])).tolist(), [0, 0, 1, 1, 2, 2, 3])

    def test_cross_imps_at_band_edges(self) -> None:
        # 10 points is a push in IMPs; 40 is 1; 80 is 2.
        self.assertEqual(elo_brill.cross_imps(np.array([420, 410])).tolist(), [0.0, 0.0])
        self.assertEqual(elo_brill.cross_imps(np.array([140, 100])).tolist(), [1.0, -1.0])
        self.assertEqual(elo_brill.cross_imps(np.array([-50, -130])).tolist(), [2.0, -2.0])

    def test_cross_imps_are_the_mean_against_the_other_tables(self) -> None:
        imps = elo_brill.cross_imps(np.array([620, 170, -100]))
        # 620 v 170 = 450 -> 10 IMPs; 620 v -100 = 720 -> 12; 170 v -100 = 270 -> 7.
        self.assertEqual(imps.tolist(), [11.0, -1.5, -9.5])
        self.assertAlmostEqual(float(imps.sum()), 0.0)

    def test_matchpoint_percentage_shares_ties(self) -> None:
        frame = _board_rows({"H:a": [100], "H:b": [100], "H:c": [-50]})
        pct = dict(zip(frame["Player_ID"], frame["Pct_NS"]))
        self.assertEqual(pct, {"H:a": 75.0, "H:b": 75.0, "H:c": 0.0})

    def test_single_table_board_has_no_percentage(self) -> None:
        frame = _board_rows({"H:a": [100]})
        self.assertIsNone(frame["Pct_NS"][0])


class EloTests(unittest.TestCase):
    def test_first_event_moves_ratings_by_k_times_score_minus_half(self) -> None:
        boards = _board_rows({"H:a": [100, 100], "H:b": [0, 0], "H:c": [-100, -100]})
        rated = elo_brill.rate_sessions(elo_brill.session_table(boards), k_factor=32)
        after = dict(zip(rated["Player_ID"], rated["Elo_After"]))
        self.assertAlmostEqual(after["H:a"], DEFAULT_ELO + 16)
        self.assertAlmostEqual(after["H:b"], DEFAULT_ELO)
        self.assertAlmostEqual(after["H:c"], DEFAULT_ELO - 16)
        self.assertEqual(rated["Elo_Before"].to_list(), [DEFAULT_ELO] * 3)

    def test_ratings_carry_into_the_next_event_in_time_order(self) -> None:
        first = _board_rows({"H:a": [100], "H:b": [-100]}, "t1", "2026-11-01T11:00:00+00:00")
        second = _board_rows({"H:a": [100], "H:b": [-100]}, "t2", "2026-11-02T11:00:00+00:00")
        rated = elo_brill.rate_sessions(elo_brill.session_table(pl.concat([second, first])))
        a = rated.filter(pl.col("Player_ID") == "H:a").sort("Event_Time")
        self.assertEqual(a["tournament_id"].to_list(), ["t1", "t2"])
        self.assertAlmostEqual(a["Elo_Before"][1], a["Elo_After"][0])
        # The favourite gains less the second time.
        self.assertLess(a["Elo_Delta"][1], a["Elo_Delta"][0])

    def test_partial_event_counts_in_proportion_to_boards_played(self) -> None:
        full = _board_rows({"H:a": [100, 100], "H:b": [-100, -100]})
        partial = full.filter(~((pl.col("Player_ID") == "H:a") & (pl.col("Board") == 2)))
        partial = elo_brill.add_field_scores(partial.drop("Pct_NS", "IMP_NS"))
        rated = elo_brill.rate_sessions(elo_brill.session_table(partial), k_factor=32)
        delta = dict(zip(rated["Player_ID"], rated["Elo_Delta"]))
        self.assertAlmostEqual(delta["H:a"], 8.0)  # half the boards, half the K


class LeaderboardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.types = {"R:bot": elo_brill.ROBOT}
        self.boards = _board_rows(
            {"H:ann": [420, 100, 200], "H:bob": [-50, 100, -100], "R:bot": [170, -100, 50]},
            declarers=["S", "W", "E"], types=self.types,
        )
        self.sessions = elo_brill.rate_sessions(elo_brill.session_table(self.boards))
        self.scale = elo_brill.elo_scale(self.sessions, min_games=1)

    def test_type_column_and_ranks(self) -> None:
        table = elo_brill.leaderboard(self.boards, self.sessions, self.scale, min_games=1)
        rows = {row["Player_Name"]: row for row in table.to_dicts()}
        self.assertEqual(rows["bot"]["Type"], elo_brill.ROBOT)
        self.assertEqual(rows["ann"]["Type"], elo_brill.HUMAN)
        self.assertEqual(rows["ann"]["Elo_Rank"], 1)
        # Board 1 is declared by South, board 2 defended, board 3 led against.
        self.assertEqual(rows["ann"]["DD_Tricks_Declared"], 1)
        self.assertEqual(rows["ann"]["Boards_Defended"], 2)
        self.assertEqual(rows["ann"]["Leads_Scored"], 1)
        self.assertEqual(rows["ann"]["Lead_Cost_Avg"], 0.0)
        self.assertEqual(rows["bob"]["Lead_Cost_Avg"], 2.0)
        self.assertEqual(rows["ann"]["Leader_Rank"], 1)
        self.assertEqual(rows["ann"]["Par_Contract_Rate_Pct"], 100.0)
        # Defence is the declarer's double-dummy difference with the sign turned.
        self.assertEqual(rows["ann"]["Defense_Tricks_Diff_Avg"], -1.0)

    def test_min_games_drops_players_before_ranking(self) -> None:
        self.assertTrue(elo_brill.leaderboard(self.boards, self.sessions, self.scale, min_games=2).is_empty())

    def test_ranking_order_and_display_filters(self) -> None:
        table = elo_brill.leaderboard(self.boards, self.sessions, self.scale, min_games=1)
        leaders = elo_brill.order_for_ranking(table, "Opening Lead")
        self.assertEqual(leaders.columns[:3], ["Leader_Rank", "Player_Name", "Type"])
        self.assertEqual(leaders["Leader_Rank"].to_list(), sorted(leaders["Leader_Rank"].to_list()))
        robots = elo_brill.filter_players(table, "Robots")
        self.assertEqual(robots["Player_Name"].to_list(), ["bot"])
        # A display filter leaves the rank among all listed players.
        self.assertEqual(robots["Elo_Rank"][0], table.filter(pl.col("Player_Name") == "bot")["Elo_Rank"][0])
        self.assertEqual(elo_brill.filter_players(table, "All", "AN")["Player_Name"].to_list(), ["ann"])

    def test_event_filters(self) -> None:
        self.assertTrue(elo_brill.filter_frame(self.sessions, "Daily IMP").is_empty())
        self.assertEqual(elo_brill.filter_frame(self.sessions, scoring="MP").height, 3)
        self.assertTrue(elo_brill.filter_frame(self.sessions, date_from="2026-11-02").is_empty())

    def test_player_detail_tables(self) -> None:
        history = elo_brill.player_sessions(self.sessions, "H:ann")
        self.assertEqual(history.height, 1)
        self.assertIn("tournament=t1&entrant=ann", history["Postmortem_URL"][0])
        standings = elo_brill.event_standings(self.sessions, "t1")
        self.assertEqual(standings["Player_Name"][0], "ann")


class SelectionAndScaleTests(unittest.TestCase):
    """An MP-only or IMP-only view is its own rating, and every view shares one scale."""

    def setUp(self) -> None:
        # ann wins the MP event and loses the IMP event to bob; cat is always in the middle.
        mp = _board_rows({"H:ann": [100], "H:bob": [-100], "H:cat": [0]}, "t1", "2026-11-01T11:00:00+00:00")
        imp = _board_rows({"H:ann": [-100], "H:bob": [100], "H:cat": [0]}, "t2", "2026-11-02T11:00:00+00:00")
        imp = imp.with_columns(pl.lit("IMP").alias("Scoring"), pl.lit("Daily IMP").alias("Event_Type"))
        self.boards = pl.concat([mp, imp])
        self.sessions = elo_brill.rate_sessions(elo_brill.session_table(self.boards))
        self.scale = elo_brill.elo_scale(self.sessions, min_games=1)

    def _elo_raw(self, scoring: str) -> dict[str, float]:
        boards, sessions = elo_brill.rated_selection(self.boards, scoring=scoring)
        table = elo_brill.leaderboard(boards, sessions, self.scale, min_games=1)
        return dict(zip(table["Player_Name"], table["Elo_Raw"]))

    def test_one_scoring_is_replayed_over_its_own_events(self) -> None:
        mp, imp, everything = self._elo_raw("MP"), self._elo_raw("IMP"), self._elo_raw("All")
        self.assertAlmostEqual(mp["ann"], DEFAULT_ELO + 16)
        self.assertAlmostEqual(imp["ann"], DEFAULT_ELO - 16)
        # The IMP event starts from 1200 in its own replay, not from ann's MP gain.
        self.assertGreater(imp["ann"], everything["ann"] - 16 - 1e-9)
        self.assertTrue(imp["ann"] < everything["ann"] < mp["ann"])

    def test_selection_keeps_only_its_events(self) -> None:
        boards, sessions = elo_brill.rated_selection(self.boards, scoring="IMP")
        self.assertEqual(sessions["tournament_id"].unique().to_list(), ["t2"])
        self.assertEqual(boards["Scoring"].unique().to_list(), ["IMP"])
        self.assertEqual(sessions["Elo_Before"].to_list(), [DEFAULT_ELO] * 3)

    def test_empty_selection_gives_an_empty_leaderboard(self) -> None:
        boards, sessions = elo_brill.rated_selection(self.boards, "No such series")
        self.assertTrue(elo_brill.leaderboard(boards, sessions, self.scale).is_empty())

    def test_scale_is_the_mean_and_spread_of_latest_ratings(self) -> None:
        mean, sd = elo_brill.elo_scale(self.sessions, min_games=1)
        latest = self.sessions.sort("Event_Time").group_by("Player_ID").agg(pl.col("Elo_After").last())["Elo_After"]
        self.assertAlmostEqual(mean, latest.mean())
        self.assertAlmostEqual(sd, latest.std(ddof=0))
        with self.assertRaises(ValueError):
            elo_brill.elo_scale(self.sessions, min_games=3)

    def test_elo_does_not_depend_on_who_is_listed(self) -> None:
        # Two events for everyone, then one more for ann and bob only.
        extra = _board_rows({"H:ann": [100], "H:bob": [-100]}, "t3", "2026-11-03T11:00:00+00:00")
        boards = pl.concat([self.boards, extra])
        sessions = elo_brill.rate_sessions(elo_brill.session_table(boards))
        everyone = elo_brill.leaderboard(boards, sessions, self.scale, min_games=1)
        regulars = elo_brill.leaderboard(boards, sessions, self.scale, min_games=3)
        self.assertEqual(sorted(regulars["Player_Name"].to_list()), ["ann", "bob"])
        elo = dict(zip(everyone["Player_Name"], everyone["Elo"]))
        for row in regulars.to_dicts():
            self.assertEqual(row["Elo"], elo[row["Player_Name"]])
        # The rank is among the listed players.
        self.assertEqual(sorted(regulars["Elo_Rank"].to_list()), [1, 2])

    def test_scale_maps_its_mean_to_1500_and_one_sd_to_400(self) -> None:
        table = elo_brill.leaderboard(self.boards, self.sessions, (DEFAULT_ELO, 16.0), min_games=1)
        raw = dict(zip(table["Player_Name"], zip(table["Elo_Raw"], table["Elo"])))
        self.assertEqual(raw["cat"][1], 1500)  # cat's rating is the scale's mean
        for rating, shown in raw.values():
            # Elo_Raw is shown to one decimal, worth up to 1.25 points on this scale.
            self.assertLessEqual(abs(shown - (1500 + (rating - DEFAULT_ELO) / 16.0 * 400)), 1.5)


if __name__ == "__main__":
    unittest.main()

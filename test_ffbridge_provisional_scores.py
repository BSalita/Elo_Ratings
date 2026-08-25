from __future__ import annotations

import unittest
from unittest.mock import patch

import polars as pl

import elo_ffbridge_lancelot as lancelot
import ffbridge_report_service as reports
from ffbridge_provisional_scores import (
    fetch_provisional_pair_percentages,
    national_ranking_is_pending,
)


def _ranking_row(
    team_id: int,
    surname1: str,
    surname2: str,
    *,
    score: float = 0.0,
    pe_bonus: float = 0,
    total_bonus: float = 0.0,
) -> dict:
    return {
        "sessionScore": score,
        "totalScore": score,
        "handicapPercentage": None,
        "scoreHandicap": None,
        "totalScoreWithoutHandicap": None,
        "rankWithoutHandicap": None,
        "theoreticalRank": None,
        "peBonus": pe_bonus,
        "totalBonus": total_bonus,
        "rank": 1,
        "simultaneousId": 5802079,
        "team": {
            "id": team_id,
            "player1": {
                "id": 11,
                "migrationId": 101,
                "ffbId": 1001,
                "firstName": "One",
                "lastName": surname1,
            },
            "player2": {
                "id": 22,
                "migrationId": 202,
                "ffbId": 2002,
                "firstName": "Two",
                "lastName": surname2,
            },
        },
    }


class PublicationStateTests(unittest.TestCase):
    def test_collective_zero_shell_is_pending(self) -> None:
        self.assertTrue(
            national_ranking_is_pending(
                [_ranking_row(1, "Alpha", "Beta"), _ranking_row(2, "Gamma", "Delta")]
            )
        )

    def test_individual_real_zero_in_published_field_is_not_pending(self) -> None:
        rows = [
            _ranking_row(1, "Alpha", "Beta", score=0.0),
            _ranking_row(2, "Gamma", "Delta", score=55.0),
        ]
        self.assertFalse(national_ranking_is_pending(rows))

    def test_octopus_scrape_returns_independent_categories(self) -> None:
        ranking = [_ranking_row(10, "Salita", "Jacoupy")]
        main_s = (
            '<a href=restotal.php?v_codeclub=LEV&v_type_classement=s'
            '&v_codeseance=lo260824>Levallois</a>'
        )
        main_h = main_s.replace("classement=s", "classement=h")
        club_s = "JACOUPY Christian SALITA Robert 54.41%"
        club_h = "JACOUPY Christian SALITA Robert 64.41%"

        def get_text(url: str) -> str:
            if "resseance" in url:
                return main_h if "classement=h" in url else main_s
            return club_h if "classement=h" in url else club_s

        scores = fetch_provisional_pair_percentages(
            ranking, "2026-08-24", 386, get_text=get_text
        )
        self.assertEqual(scores["10"]["scratch_percentage"], 54.41)
        self.assertEqual(scores["10"]["handicap_percentage"], 64.41)

    def test_missing_handicap_does_not_copy_scratch(self) -> None:
        ranking = [_ranking_row(10, "Salita", "Jacoupy")]
        main_s = (
            '<a href=restotal.php?v_codeclub=LEV&v_type_classement=s'
            '&v_codeseance=lo260824>Levallois</a>'
        )
        main_h = main_s.replace("classement=s", "classement=h")

        def get_text(url: str) -> str:
            if "resseance" in url:
                return main_h if "classement=h" in url else main_s
            if "classement=h" in url:
                return "OTHER Pair 51.00%"
            return "JACOUPY Christian SALITA Robert 54.41%"

        scores = fetch_provisional_pair_percentages(
            ranking, "2026-08-24", 386, get_text=get_text
        )
        self.assertEqual(scores["10"]["scratch_percentage"], 54.41)
        self.assertIsNone(scores["10"]["handicap_percentage"])

    @patch("elo_ffbridge_lancelot.fetch_provisional_pair_percentages")
    def test_pending_zero_is_replaced_before_elo(self, provisional) -> None:
        provisional.return_value = {
            "10": {
                "scratch_percentage": 54.41,
                "handicap_percentage": 64.41,
                "scratch_url": "http://example/s",
                "handicap_url": "http://example/h",
            }
        }
        rows = lancelot._normalize_ranking_results(
            [_ranking_row(10, "Salita", "Jacoupy")],
            series_id=386,
            tournament_date="2026-08-24",
        )
        self.assertEqual(rows[0]["scratch_percentage"], 54.41)
        self.assertEqual(rows[0]["handicap_percentage"], 64.41)
        self.assertEqual(rows[0]["score_status"], "provisional")

    @patch("elo_ffbridge_lancelot.fetch_provisional_pair_percentages")
    def test_pending_without_club_match_is_unresolved(self, provisional) -> None:
        provisional.return_value = {}
        rows = lancelot._normalize_ranking_results(
            [_ranking_row(10, "Unknown", "Pair")],
            series_id=386,
            tournament_date="2026-08-24",
        )
        self.assertIsNone(rows[0]["scratch_percentage"])
        self.assertIsNone(rows[0]["handicap_percentage"])
        self.assertEqual(rows[0]["score_status"], "unresolved")

    @patch("elo_ffbridge_lancelot.save_to_disk_cache")
    @patch("elo_ffbridge_lancelot.lancelot_get")
    @patch("elo_ffbridge_lancelot.load_from_disk_cache")
    def test_expired_pending_cache_refetches_official_result(
        self,
        load_cache,
        get_api,
        _save_cache,
    ) -> None:
        pending = [_ranking_row(10, "Salita", "Jacoupy")]
        official = [_ranking_row(10, "Salita", "Jacoupy", score=54.41)]
        load_cache.side_effect = [pending, None]
        get_api.return_value = official
        rows, was_cached = lancelot.fetch_tournament_results(
            "300751",
            tournament_date="2026-08-24",
            series_id=386,
        )
        self.assertFalse(was_cached)
        self.assertEqual(rows[0]["score_status"], "official")
        self.assertEqual(rows[0]["scratch_percentage"], 54.41)


class OfficialCategoryMappingTests(unittest.TestCase):
    def test_rondes_de_france_keeps_session_score_as_scratch(self) -> None:
        rows = lancelot._normalize_ranking_results(
            [_ranking_row(10, "Salita", "Collins", score=65.67, pe_bonus=52)],
            series_id=3,
            tournament_date="2026-08-25",
        )
        self.assertEqual(rows[0]["scratch_percentage"], 65.67)
        self.assertIsNone(rows[0]["handicap_percentage"])
        self.assertEqual(rows[0]["handicap_score_status"], "scratch_only")
        self.assertEqual(rows[0]["iv_bonus"], 0.0)
        self.assertEqual(rows[0]["pe_bonus"], "52.0")

    def test_octopus_uses_total_bonus_not_pe_bonus(self) -> None:
        rows = lancelot._normalize_ranking_results(
            [
                _ranking_row(
                    10,
                    "Salita",
                    "Jacoupy",
                    score=71.74,
                    pe_bonus=0,
                    total_bonus=10.0,
                )
            ],
            series_id=386,
            tournament_date="2026-08-17",
        )
        self.assertAlmostEqual(rows[0]["scratch_percentage"], 61.74)
        self.assertEqual(rows[0]["handicap_percentage"], 71.74)
        self.assertEqual(rows[0]["iv_bonus"], 10.0)
        self.assertEqual(rows[0]["handicap_score_status"], "official")


class ScoreAvailabilityTests(unittest.TestCase):
    def test_schema_version_forces_full_elo_replay(self) -> None:
        self.assertTrue(
            reports.elo_cache_key("FFBridge_Lancelot_API", True).startswith(
                "elo_full_v6_"
            )
        )

    def test_scratch_only_does_not_fill_handicap_average(self) -> None:
        frame = pl.DataFrame(
            {
                "player1_id": ["1", "1"],
                "player2_id": ["2", "3"],
                "player1_name": ["Salita", "Salita"],
                "player2_name": ["Collins", "Other"],
                "player1_scratch_elo_after": [1600.0, 1610.0],
                "player2_scratch_elo_after": [1500.0, 1500.0],
                "player1_handicap_elo_after": [1600.0, 1610.0],
                "player2_handicap_elo_after": [1500.0, 1500.0],
                "scratch_percentage": [65.67, 54.41],
                "handicap_percentage": [None, 64.41],
                "iv_bonus": [0.0, 10.0],
                "score_status": ["official", "official"],
                "date": ["2026-08-25", "2026-08-24"],
            }
        )
        players = reports.aggregate_players_from_results(frame, use_handicap=False)
        salita = players.filter(pl.col("player_id") == "1")
        self.assertAlmostEqual(salita.item(0, "avg_scratch_pct"), 60.04)
        self.assertAlmostEqual(salita.item(0, "avg_handicap_pct"), 64.41)

    def test_unresolved_categories_are_filtered_independently(self) -> None:
        frame = pl.DataFrame(
            {
                "scratch_score_status": ["provisional", "unresolved"],
                "handicap_score_status": ["unresolved", "provisional"],
                "score_status": ["provisional", "provisional"],
            }
        )
        scratch = reports.filter_score_available(frame, use_handicap=False)
        handicap = reports.filter_score_available(frame, use_handicap=True)
        self.assertEqual(scratch.height, 1)
        self.assertEqual(handicap.height, 1)
        self.assertEqual(
            reports.score_provenance_counts(frame)["provisional_rows"], 2
        )


if __name__ == "__main__":
    unittest.main()

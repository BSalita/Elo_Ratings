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


def _club_row(player1: str, player2: str, pct: str) -> str:
    return (
        "<tr class=text_res>"
        "<td align=right>1</td><td align=right>1</td><td align=right>1</td>"
        "<td align=center>EO</td>"
        f"<td align=left>{player1}</td><td align=left>{player2}</td>"
        f"<td align=right> {pct}%</td>"
        "<td align=right>0</td><td align=right>0</td>"
        "</tr>"
    )


def _ranking_row(
    team_id: int,
    surname1: str,
    surname2: str,
    *,
    score: float = 0.0,
    pe_bonus: float = 0,
    total_bonus: float = 0.0,
    theoretical_rank: int | None = None,
) -> dict:
    return {
        "sessionScore": score,
        "totalScore": score,
        "handicapPercentage": None,
        "scoreHandicap": None,
        "totalScoreWithoutHandicap": None,
        "rankWithoutHandicap": None,
        "theoreticalRank": theoretical_rank,
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
        club_s = _club_row("JACOUPY Christian", "SALITA Robert", "54.41")
        club_h = _club_row("JACOUPY Christian", "SALITA Robert", "64.41")

        def get_text(url: str) -> str:
            if "resseance" in url:
                return main_h if "classement=h" in url else main_s
            return club_h if "classement=h" in url else club_s

        scores = fetch_provisional_pair_percentages(
            ranking, "2026-08-24", 386, get_text=get_text
        )
        self.assertEqual(scores["10"]["scratch_percentage"], 54.41)
        self.assertEqual(scores["10"]["handicap_percentage"], 64.41)
        self.assertIsNone(scores["10"]["national_scratch_percentage"])
        self.assertEqual(scores["10"]["club_scratch_percentage"], 54.41)

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
                return _club_row("OTHER Pair", "UNKNOWN Name", "51.00")
            return _club_row("JACOUPY Christian", "SALITA Robert", "54.41")

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
        self.assertEqual(rows[0]["Club_Scratch_Pct"], 54.41)
        self.assertEqual(rows[0]["Club_Handicap_Pct"], 64.41)
        self.assertEqual(rows[0]["Club_Scratch_Rank"], 1)
        self.assertEqual(rows[0]["Club_Handicap_Rank"], 1)
        self.assertIsNone(rows[0]["National_Scratch_Pct"])
        self.assertEqual(rows[0]["score_status"], "provisional")

    @patch("elo_ffbridge_lancelot.fetch_provisional_pair_percentages")
    def test_pending_without_club_match_is_unresolved(self, provisional) -> None:
        provisional.return_value = {}
        rows = lancelot._normalize_ranking_results(
            [_ranking_row(10, "Unknown", "Pair")],
            series_id=386,
            tournament_date="2026-08-24",
        )
        self.assertIsNone(rows[0]["Club_Scratch_Pct"])
        self.assertIsNone(rows[0]["Club_Handicap_Pct"])
        self.assertEqual(rows[0]["score_status"], "unresolved")

    @patch("elo_ffbridge_lancelot._fetch_organizer_scores", return_value={})
    @patch("elo_ffbridge_lancelot.save_to_disk_cache")
    @patch("elo_ffbridge_lancelot.lancelot_get")
    @patch("elo_ffbridge_lancelot.load_from_disk_cache")
    def test_expired_pending_cache_refetches_official_result(
        self,
        load_cache,
        get_api,
        _save_cache,
        _organizer_scores,
    ) -> None:
        pending = [_ranking_row(10, "Salita", "Jacoupy")]
        official = [_ranking_row(10, "Salita", "Jacoupy", score=54.41)]
        # Ranking cache check, expired-cache reload, then session-group cache.
        load_cache.side_effect = [pending, None, None]
        get_api.return_value = official
        rows, was_cached = lancelot.fetch_tournament_results(
            "300751",
            tournament_date="2026-08-24",
            series_id=386,
        )
        self.assertFalse(was_cached)
        self.assertEqual(rows[0]["score_status"], "official")
        self.assertEqual(rows[0]["National_Scratch_Pct"], 54.41)


class OfficialCategoryMappingTests(unittest.TestCase):
    def test_rondes_de_france_keeps_session_score_as_scratch(self) -> None:
        rows = lancelot._normalize_ranking_results(
            [_ranking_row(10, "Salita", "Collins", score=65.67, pe_bonus=52)],
            series_id=3,
            tournament_date="2026-08-25",
        )
        self.assertEqual(rows[0]["National_Scratch_Pct"], 65.67)
        self.assertIsNone(rows[0]["National_Handicap_Pct"])
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
                    theoretical_rank=42,
                )
            ],
            series_id=386,
            tournament_date="2026-08-17",
        )
        self.assertAlmostEqual(rows[0]["National_Scratch_Pct"], 61.74)
        self.assertEqual(rows[0]["National_Handicap_Pct"], 71.74)
        self.assertEqual(rows[0]["iv_bonus"], 10.0)
        self.assertEqual(rows[0]["handicap_score_status"], "official")
        self.assertEqual(rows[0]["Theoretical_Rank"], 42)

    def test_missing_octopus_bonus_does_not_copy_handicap_into_scratch(self) -> None:
        ranking_row = _ranking_row(
            10,
            "Salita",
            "Jacoupy",
            score=71.74,
        )
        ranking_row.pop("totalBonus")
        rows = lancelot._normalize_ranking_results(
            [ranking_row],
            series_id=386,
            tournament_date="2026-06-08",
        )
        self.assertIsNone(rows[0]["National_Scratch_Pct"])
        self.assertEqual(rows[0]["National_Handicap_Pct"], 71.74)
        self.assertIsNone(rows[0]["iv_bonus"])

    def test_organizer_theoretical_rank_fills_when_lancelot_omits_it(self) -> None:
        ranking_row = _ranking_row(10, "Salita", "Jacoupy", score=54.41)
        ranking_row["theoreticalRank"] = None
        rows = lancelot._normalize_ranking_results(
            [ranking_row],
            series_id=386,
            tournament_date="2026-06-08",
            organizer_scores={"10": {"theoretical_rank": 80}},
        )
        self.assertEqual(rows[0]["Theoretical_Rank"], 80)

    def test_lancelot_theoretical_rank_is_not_replaced_by_organizer(self) -> None:
        ranking_row = _ranking_row(
            10, "Salita", "Jacoupy", score=54.41, theoretical_rank=42
        )
        rows = lancelot._normalize_ranking_results(
            [ranking_row],
            series_id=386,
            tournament_date="2026-07-02",
            organizer_scores={"10": {"theoretical_rank": 80}},
        )
        self.assertEqual(rows[0]["Theoretical_Rank"], 42)

    def test_historical_organizer_score_restores_missing_national_scratch(
        self,
    ) -> None:
        ranking_row = _ranking_row(
            10,
            "Salita",
            "Jacoupy",
            score=67.62,
        )
        ranking_row.pop("totalBonus")
        rows = lancelot._normalize_ranking_results(
            [ranking_row],
            series_id=386,
            tournament_date="2026-06-08",
            organizer_scores={
                "10": {
                    "national_scratch_percentage": 54.61,
                    "national_handicap_percentage": 67.61,
                    "club_scratch_percentage": 54.4,
                    "club_handicap_percentage": 67.4,
                    "scratch_url": "http://example/s",
                    "handicap_url": "http://example/h",
                }
            },
        )
        row = rows[0]
        self.assertEqual(row["National_Scratch_Pct"], 54.61)
        self.assertEqual(row["National_Handicap_Pct"], 67.62)
        self.assertEqual(row["Club_Scratch_Pct"], 54.4)
        self.assertEqual(row["Club_Handicap_Pct"], 67.4)
        self.assertAlmostEqual(row["iv_bonus"], 13.01)

    @patch("elo_ffbridge_lancelot._fetch_organizer_scores", return_value={})
    @patch("elo_ffbridge_lancelot.fetch_session_group_ids", return_value={})
    @patch("elo_ffbridge_lancelot.save_to_disk_cache")
    @patch("elo_ffbridge_lancelot.lancelot_get")
    @patch("elo_ffbridge_lancelot.load_from_disk_cache")
    def test_recent_finalized_ranking_is_revalidated(
        self,
        load_cache,
        get_api,
        _save_cache,
        _group_ids,
        _organizer_scores,
    ) -> None:
        cached = [
            _ranking_row(
                10,
                "Salita",
                "Jacoupy",
                score=71.74,
                total_bonus=10.0,
                theoretical_rank=42,
            )
        ]
        refreshed = cached + [
            _ranking_row(
                11,
                "Other",
                "Pair",
                score=65.0,
                total_bonus=5.0,
                theoretical_rank=30,
            )
        ]
        load_cache.side_effect = [cached, None]
        get_api.return_value = refreshed

        rows, was_cached = lancelot.fetch_tournament_results(
            "300751",
            tournament_date="2026-08-24",
            series_id=386,
        )

        self.assertFalse(was_cached)
        self.assertEqual(len(rows), 2)


class ScoreAvailabilityTests(unittest.TestCase):
    def test_schema_version_forces_full_elo_replay(self) -> None:
        self.assertTrue(
            reports.elo_cache_key("FFBridge_Lancelot_API", True).startswith(
                "elo_full_v11_"
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
                "Club_Scratch_Pct": [None, 54.41],
                "Club_Handicap_Pct": [None, 64.41],
                "National_Scratch_Pct": [65.67, None],
                "National_Handicap_Pct": [None, None],
                "iv_bonus": [0.0, 10.0],
                "score_status": ["official", "official"],
                "date": ["2026-08-25", "2026-08-24"],
            }
        )
        players = reports.aggregate_players_from_results(frame, use_handicap=False)
        salita = players.filter(pl.col("player_id") == "1")
        self.assertAlmostEqual(salita.item(0, "avg_scratch_pct"), 60.04)
        self.assertAlmostEqual(salita.item(0, "avg_handicap_pct"), 64.41)

    def test_roy_rene_first_tuesday_is_handicap_only(self) -> None:
        rows = lancelot._normalize_ranking_results(
            [_ranking_row(10, "Salita", "Collins", score=82.37)],
            series_id=5,
            tournament_date="2025-10-07",
        )
        row = rows[0]
        self.assertEqual(row["National_Handicap_Pct"], 82.37)
        self.assertIsNone(row["National_Scratch_Pct"])
        self.assertEqual(row["National_Handicap_Rank"], 1)
        self.assertIsNone(row["National_Scratch_Rank"])
        self.assertEqual(row["scoring_mode"], "handicap")

    def test_ordinary_roy_rene_session_is_scratch_only(self) -> None:
        rows = lancelot._normalize_ranking_results(
            [_ranking_row(10, "Salita", "Collins", score=61.25)],
            series_id=5,
            tournament_date="2025-10-14",
        )
        row = rows[0]
        self.assertEqual(row["National_Scratch_Pct"], 61.25)
        self.assertIsNone(row["National_Handicap_Pct"])
        self.assertEqual(row["National_Scratch_Rank"], 1)
        self.assertIsNone(row["National_Handicap_Rank"])
        self.assertEqual(row["scoring_mode"], "scratch")

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

    def test_score_provenance_reports_source_counts(self) -> None:
        frame = pl.DataFrame(
            {
                "score_source": [
                    "national_official",
                    "national_and_organizer_official",
                    "national_and_organizer_official",
                ]
            }
        )
        self.assertEqual(
            reports.score_provenance_counts(frame)["score_sources"],
            {
                "national_official": 1,
                "national_and_organizer_official": 2,
            },
        )

    def test_category_filters_do_not_count_the_other_score_type(self) -> None:
        frame = pl.DataFrame(
            {
                "Club_Scratch_Pct": [None, None],
                "Club_Handicap_Pct": [None, None],
                "National_Scratch_Pct": [61.0, None],
                "National_Handicap_Pct": [None, 82.37],
                "scratch_score_status": ["official", "unresolved"],
                "handicap_score_status": ["scratch_only", "official"],
            }
        )
        scratch = reports.filter_score_available(frame, use_handicap=False)
        handicap = reports.filter_score_available(frame, use_handicap=True)
        self.assertEqual(scratch["National_Scratch_Pct"].to_list(), [61.0])
        self.assertEqual(handicap["National_Handicap_Pct"].to_list(), [82.37])


if __name__ == "__main__":
    unittest.main()

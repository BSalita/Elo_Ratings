"""Roy René handicap sessions whose FFBridge ranking carries scratch scores."""

from __future__ import annotations

import unittest

import elo_ffbridge_lancelot as lancelot
import ffbridge_bridgeplus_scores as bridgeplus

TOURNAMENT = "S202722"
DATE = "2026-11-03"  # first Tuesday: handicap schedule

RESULTS_LIST = f"""
<div class='col-4 p-0 text-left h5'>Mardi 3 Novembre 2026 (Après-midi)</div>
<div class='col h4'>
  <a href='https://www.bridgeplus.com/nos-simultanes/resultats/?p=clubs/{TOURNAMENT}'>n&deg;22</a>
</div>
<div class='col-4 p-0 text-left h5'>Mardi 3 Février 2026 (Après-midi)</div>
<div class='col h4'>
  <a href='https://www.bridgeplus.com/nos-simultanes/resultats/?p=clubs/S202650'>n&deg;50</a>
</div>
"""


def _club_row(rank: int, pct: str, licenses: str, club: str, pair: int) -> str:
    return f"""
	<div class='row text-center raye p-0 py-1'>
			<div class='col-md-5 p-0'><div class='row'>
					<div class='col-2 p-0'>{rank}</div>
					<div class='col-10 p-0'>{pct}</div>
			</div></div>
			<div class='col-md-7 p-0'><div class='row'>
					<div class='col-4 p-0 d-none d-sm-block'>
						{licenses}
					</div>
					<div class='col p-0'>
						<a href='https://www.bridgeplus.com/nos-simultanes/resultats/?p=route/{TOURNAMENT}/{club}/A/{pair}'>
							PAIR
						</a>
					</div>
			</div></div>
		</div>"""


def _club_page(*rows: str) -> str:
    return "<div class='row text-center bg-primary text-white'></div>" + "".join(rows)


def _entry(
    team_id: int,
    score: float,
    *,
    club: int = 5802079,
    orientation: str = "EW",
    table: int = 4,
    licenses: tuple[int, int] = (9500754, 1353276),
    rank: int = 1,
) -> dict:
    return {
        "sessionScore": score,
        "totalScore": score,
        "handicapPercentage": None,
        "scoreHandicap": None,
        "totalScoreWithoutHandicap": None,
        "rankWithoutHandicap": None,
        "theoreticalRank": 100,
        "peBonus": 0,
        "totalBonus": 0.0,
        "rank": rank,
        "section": "A",
        "orientation": orientation,
        "tableNumber": table,
        "simultaneousId": club,
        "team": {
            "id": team_id,
            "player1": {"id": 1, "migrationId": team_id * 10 + 1, "ffbId": licenses[0],
                        "firstName": "A", "lastName": "One"},
            "player2": {"id": 2, "migrationId": team_id * 10 + 2, "ffbId": licenses[1],
                        "firstName": "B", "lastName": "Two"},
        },
    }


def _scratch_ranking() -> list[dict]:
    """Mitchell club 5802079 and Howell club 4600071; scores average 50%."""
    return [
        _entry(10, 60.16, table=4, orientation="EW", rank=1),
        _entry(20, 39.84, table=4, orientation="NS", licenses=(1111111, 2222222), rank=3),
        _entry(30, 55.0, club=4600071, table=9, orientation="NS",
               licenses=(3333333, 4444444), rank=2),
        _entry(40, 45.0, club=4600071, table=7, orientation="EW",
               licenses=(5555555, 6666666), rank=4),
    ]


def _fetch(pages: dict[str, str]):
    def fetch(url: str) -> str:
        if url == bridgeplus.RESULTS_URL:
            return RESULTS_LIST
        for club, html in pages.items():
            if url == bridgeplus.CLUB_URL.format(tournament=TOURNAMENT, club=club):
                return html
        raise AssertionError(f"unexpected url {url}")
    return fetch


HANDICAP_PAGES = {
    "5802079": _club_page(
        _club_row(10, "70,66", "09500754-01353276", "5802079", 204),
        # No licenses shown: matched by Mitchell table NS4.
        _club_row(200, "44,01", "", "5802079", 104),
    ),
    # Howell pair numbers do not follow Lancelot tables; licenses decide.
    "4600071": _club_page(
        _club_row(50, "62,00", "03333333-04444444", "4600071", 2),
        _club_row(150, "52,50", "05555555-06666666", "4600071", 5),
    ),
}


class ParseTests(unittest.TestCase):
    def test_results_list_dates_handle_accented_months(self) -> None:
        self.assertEqual(
            bridgeplus.parse_results_list(RESULTS_LIST),
            [
                {"tournament_id": TOURNAMENT, "date": "2026-11-03"},
                {"tournament_id": "S202650", "date": "2026-02-03"},
            ],
        )

    def test_club_page_reads_mitchell_and_howell_pairs(self) -> None:
        mitchell, = bridgeplus.parse_club_page(
            _club_page(_club_row(10, "70,66", "09500754-01353276", "5802079", 204))
        )
        self.assertEqual(
            (mitchell["direction"], mitchell["table"], mitchell["pct"], mitchell["rank"]),
            ("EW", 4, 70.66, 10),
        )
        self.assertEqual(mitchell["licenses"], {"9500754", "1353276"})
        howell, = bridgeplus.parse_club_page(
            _club_page(_club_row(50, "62,00", "03333333-04444444", "4600071", 2))
        )
        self.assertIsNone(howell["direction"])
        self.assertIsNone(howell["table"])


class DetectionTests(unittest.TestCase):
    def test_first_tuesday_scores_averaging_50_are_scratch(self) -> None:
        self.assertTrue(lancelot.handicap_published_as_scratch(_scratch_ranking(), DATE, 5))

    def test_first_tuesday_scores_above_50_are_handicap(self) -> None:
        ranking = [_entry(10, 64.32), _entry(20, 54.0, licenses=(1, 2))]
        self.assertFalse(lancelot.handicap_published_as_scratch(ranking, DATE, 5))

    def test_published_bonus_means_handicap(self) -> None:
        ranking = _scratch_ranking()
        ranking[0]["totalBonus"] = 10.5
        self.assertFalse(lancelot.handicap_published_as_scratch(ranking, DATE, 5))

    def test_ordinary_roy_rene_day_is_not_checked(self) -> None:
        self.assertFalse(
            lancelot.handicap_published_as_scratch(_scratch_ranking(), "2026-11-10", 5)
        )


class BridgeplusScoreTests(unittest.TestCase):
    def test_matches_by_license_and_by_mitchell_table(self) -> None:
        scores = lancelot.bridgeplus_handicap_scores(
            _scratch_ranking(), DATE, fetch=_fetch(HANDICAP_PAGES)
        )
        self.assertEqual(
            {team: row["national_handicap_percentage"] for team, row in scores.items()},
            {"10": 70.66, "20": 44.01, "30": 62.0, "40": 52.5},
        )
        self.assertEqual(scores["10"]["national_handicap_rank"], 10)

    def test_normal_game_with_equal_percentages_has_no_handicap(self) -> None:
        pages = {
            "5802079": _club_page(
                _club_row(1, "60,16", "09500754-01353276", "5802079", 204),
                _club_row(3, "39,84", "01111111-02222222", "5802079", 104),
            ),
            "4600071": _club_page(
                _club_row(2, "55,00", "03333333-04444444", "4600071", 2),
                _club_row(4, "45,00", "05555555-06666666", "4600071", 5),
            ),
        }
        self.assertEqual(
            lancelot.bridgeplus_handicap_scores(_scratch_ranking(), DATE, fetch=_fetch(pages)),
            {},
        )

    def test_conflicting_licenses_at_same_table_fail(self) -> None:
        pages = dict(HANDICAP_PAGES)
        pages["5802079"] = _club_page(
            _club_row(10, "70,66", "07777777-08888888", "5802079", 204),
        )
        with self.assertRaises(ValueError):
            lancelot.bridgeplus_handicap_scores(_scratch_ranking(), DATE, fetch=_fetch(pages))

    def test_day_missing_from_bridgeplus_gives_no_scores(self) -> None:
        self.assertEqual(
            lancelot.bridgeplus_handicap_scores(
                _scratch_ranking(), "2024-08-06", fetch=_fetch(HANDICAP_PAGES)
            ),
            {},
        )


class NormalizeTests(unittest.TestCase):
    def test_scratch_from_ffbridge_and_handicap_from_bridgeplus(self) -> None:
        ranking = _scratch_ranking()
        scores = lancelot.bridgeplus_handicap_scores(ranking, DATE, fetch=_fetch(HANDICAP_PAGES))
        rows = lancelot._normalize_ranking_results(
            ranking, series_id=5, tournament_date=DATE, organizer_scores=scores
        )
        row = next(r for r in rows if r["team_id"] == "10")
        self.assertEqual(row["National_Scratch_Pct"], 60.16)
        self.assertEqual(row["National_Handicap_Pct"], 70.66)
        self.assertEqual(row["National_Scratch_Rank"], 1)
        self.assertEqual(row["National_Handicap_Rank"], 10)
        self.assertAlmostEqual(row["iv_bonus"], 10.5)
        self.assertEqual(row["scoring_mode"], "handicap")
        self.assertNotEqual(row["handicap_score_status"], "unresolved")

    def test_without_bridgeplus_the_session_is_scratch(self) -> None:
        rows = lancelot._normalize_ranking_results(
            _scratch_ranking(), series_id=5, tournament_date="2024-08-06", organizer_scores={}
        )
        row = next(r for r in rows if r["team_id"] == "10")
        self.assertEqual(row["National_Scratch_Pct"], 60.16)
        self.assertIsNone(row["National_Handicap_Pct"])
        self.assertIsNone(row["National_Handicap_Rank"])
        self.assertEqual(row["National_Scratch_Rank"], 1)
        self.assertEqual(row["scoring_mode"], "scratch")


if __name__ == "__main__":
    unittest.main()

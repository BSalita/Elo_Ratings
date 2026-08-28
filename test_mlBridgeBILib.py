from __future__ import annotations

import sys
import unittest
from pathlib import Path

import polars as pl

_ROOT = Path(__file__).resolve().parent
_MLBRIDGE = next(
    path
    for path in (_ROOT / "mlBridge", _ROOT.parent / "mlBridge")
    if path.is_dir()
)
if str(_MLBRIDGE) not in sys.path:
    sys.path.append(str(_MLBRIDGE))

import mlBridgeBILib as bi  # noqa: E402


CLUB_ROW = (
    "<tr class=text_res>"
    "<td align=right>5</td><td align=right>12</td><td align=right>80</td>"
    "<td align=center>EO</td>"
    "<td align=left><a href=feuilleroute.php?v_numpaire=lo260824LEVA08EO"
    "&v_codeseance=lo260824 target=_blank>JACOUPY Christian</a></td>"
    "<td align=left><a href=feuilleroute.php?v_numpaire=lo260824LEVA08EO"
    "&v_codeseance=lo260824 target=_blank>SALITA Robert</a></td>"
    "<td align=right> 54.41%</td>"
    "<td align=right>188</td><td align=right>0</td>"
    "</tr>"
)
NATIONAL_ROW = (
    "<tr class=text_res>"
    "<td align=right>29</td><td align=right>80</td>"
    "<td align=center>EO</td>"
    "<td align=left>JACOUPY Christian</td>"
    "<td align=left>SALITA Robert</td>"
    "<td align=right> 54.41%</td>"
    "<td align=right>188</td><td align=right>0</td>"
    "<td>Levallois-Perret</td>"
    "</tr>"
)
HANDICAP_CLUB_ROW = CLUB_ROW.replace("54.41%", "64.41%")
CLUB_LINK = (
    "<a href=restotal.php?v_codeclub=LEV&v_type_classement=s"
    "&v_codeseance=lo260824>Levallois-Perret</a>"
)


class UrlCatalogTests(unittest.TestCase):
    def test_monday_octopus(self) -> None:
        self.assertEqual(bi.game_for_date("2026-08-24").key, "octopus_monday")
        self.assertEqual(bi.session_code_for_date("2026-08-24"), "lo260824")
        self.assertEqual(
            bi.ranking_url("2026-08-24", "s"),
            "http://www.bridgeinter.net/octopus_l/resseance_l.php"
            "?v_codeseance=lo260824&v_type_classement=s",
        )

    def test_thursday_octopus(self) -> None:
        self.assertEqual(bi.game_for_date("2026-08-27").key, "octopus_thursday")
        self.assertEqual(
            bi.ranking_url("2026-08-27", "handicap"),
            "http://www.bridgeinter.net/octopus_j/resseance_j.php"
            "?v_codeseance=jo260827&v_type_classement=h",
        )

    def test_friday_simultanet(self) -> None:
        self.assertEqual(bi.game_for_date("2026-08-21").key, "simultanet")
        self.assertEqual(bi.session_code_for_date("2026-08-21"), "vi260821")
        self.assertEqual(
            bi.club_url("2026-08-21", "PLM", "scratch"),
            "http://www.bridgeinter.net/simultanet/restotal.php"
            "?v_codeclub=PLM&v_type_classement=s&v_codeseance=vi260821",
        )

    def test_non_game_day_is_none(self) -> None:
        self.assertIsNone(bi.game_for_date("2026-08-25"))
        self.assertIsNone(bi.ranking_url("2026-08-25", "s"))

    def test_history_urls(self) -> None:
        self.assertTrue(
            bi.history_url("octopus_monday").endswith(
                "octopus_l/SeancesPrecedantes_l.php"
            )
        )
        self.assertTrue(
            bi.history_url("simultanet").endswith(
                "simultanet/seancesprecedentes_vi.php"
            )
        )


class ParseTests(unittest.TestCase):
    def test_parse_club_row(self) -> None:
        rows = bi.parse_result_rows(
            CLUB_ROW, page_kind="club", club_code="LEV", club_name="Levallois-Perret"
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["player1_name"], "JACOUPY Christian")
        self.assertEqual(row["player2_name"], "SALITA Robert")
        self.assertEqual(row["percentage"], 54.41)
        self.assertEqual(row["local_rank"], 5)
        self.assertEqual(row["global_rank"], 12)
        self.assertEqual(row["theoretical_rank"], 80)
        self.assertEqual(row["direction"], "EO")
        self.assertEqual(row["pair_id"], "lo260824LEVA08EO")

    def test_parse_national_row(self) -> None:
        rows = bi.parse_result_rows(NATIONAL_ROW, page_kind="national")
        self.assertEqual(rows[0]["percentage"], 54.41)
        self.assertEqual(rows[0]["global_rank"], 29)
        self.assertEqual(rows[0]["club_name"], "Levallois-Perret")
        self.assertIsNone(rows[0]["local_rank"])

    def test_parse_club_links(self) -> None:
        clubs = bi.parse_club_links(
            CLUB_LINK,
            "http://www.bridgeinter.net/octopus_l/resseance_l.php?v_codeseance=lo260824",
        )
        self.assertEqual(clubs[0]["club_code"], "LEV")
        self.assertIn("restotal.php", clubs[0]["url"])

    def test_parse_history(self) -> None:
        html = (
            '<a href=resseance.php?v_codeseance=vi260821&v_type_classement=h>'
            "21 Aout 2026</a>"
            '<a href=resseance.php?v_codeseance=vi260814&v_type_classement=h>'
            "14 Aout 2026</a>"
        )
        sessions = bi.parse_history(html)
        self.assertEqual([row["session_code"] for row in sessions], ["vi260821", "vi260814"])
        self.assertEqual(str(sessions[0]["date"]), "2026-08-21")


class MatchAndBackfillTests(unittest.TestCase):
    def test_fetch_session_pair_scores_uses_club_tables(self) -> None:
        ranking = [
            {
                "team": {
                    "id": 10,
                    "player1": {"lastName": "Salita"},
                    "player2": {"lastName": "Jacoupy"},
                }
            }
        ]

        def get_text(url: str) -> str:
            if "resseance" in url:
                return CLUB_LINK.replace(
                    "classement=s",
                    "classement=h" if "classement=h" in url else "classement=s",
                )
            return HANDICAP_CLUB_ROW if "classement=h" in url else CLUB_ROW

        scores = bi.fetch_session_pair_scores(
            ranking, "2026-08-24", 386, get_text=get_text
        )
        self.assertEqual(scores["10"]["club_scratch_percentage"], 54.41)
        self.assertEqual(scores["10"]["club_handicap_percentage"], 64.41)
        self.assertEqual(scores["10"]["club_scratch_rank"], 5)
        self.assertEqual(scores["10"]["theoretical_rank"], 80)
        self.assertIsNone(scores["10"]["national_scratch_percentage"])

    def test_simultanet_series_is_accepted(self) -> None:
        ranking = [
            {
                "team": {
                    "id": "7",
                    "player1": {"lastName": "Mahe"},
                    "player2": {"lastName": "Mahe"},
                }
            }
        ]
        club_link = (
            "<a href=restotal.php?v_codeclub=PLM&v_type_classement=s"
            "&v_codeseance=vi260821>Paris Club PLM</a>"
        )
        club_row = (
            "<tr class=text_res>"
            "<td>1</td><td>1</td><td>6</td><td>NS</td>"
            "<td>MAHE Andre</td><td>MAHE Marie-dominique</td>"
            "<td>67.35%</td><td>292</td><td>0</td>"
            "</tr>"
        )

        def get_text(url: str) -> str:
            if "resseance" in url:
                return club_link
            return club_row

        scores = bi.fetch_session_pair_scores(
            ranking, "2026-08-21", 384, get_text=get_text
        )
        self.assertEqual(scores["7"]["club_scratch_percentage"], 67.35)
        self.assertEqual(scores["7"]["club_handicap_percentage"], 67.35)

    def test_wrong_series_for_weekday_is_empty(self) -> None:
        scores = bi.fetch_session_pair_scores([], "2026-08-24", 384)
        self.assertEqual(scores, {})

    def test_fill_missing_club_pcts_leaves_existing_values(self) -> None:
        df = pl.DataFrame(
            {
                "date": ["2026-08-24", "2026-08-24"],
                "series_id": [386, 386],
                "player1_name": ["Christian Jacoupy", "Someone Else"],
                "player2_name": ["Robert Salita", "Other Person"],
                "Club_Scratch_Pct": [None, 50.0],
                "Club_Handicap_Pct": [None, None],
                "Theoretical_Rank": [None, 9],
            }
        )

        def get_text(url: str) -> str:
            if "resseance" in url:
                return CLUB_LINK
            return HANDICAP_CLUB_ROW if "classement=h" in url else CLUB_ROW

        filled = bi.fill_missing_club_pcts(df, get_text=get_text, show_progress=False)
        self.assertEqual(filled["Club_Scratch_Pct"].to_list(), [54.41, 50.0])
        self.assertEqual(filled["Club_Handicap_Pct"].to_list(), [64.41, None])
        self.assertEqual(filled["Club_Scratch_Rank"].to_list(), [5, None])
        self.assertEqual(filled["Theoretical_Rank"].to_list(), [80, 9])


if __name__ == "__main__":
    unittest.main()

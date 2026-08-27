from __future__ import annotations

import unittest

import polars as pl

from streamlit_app_ffbridge_elo_ratings import _ffbridge_results_url_expr


class FFBridgeResultsUrlTests(unittest.TestCase):
    def test_uses_classic_club_id_or_lancelot_club_code(self) -> None:
        rows = pl.DataFrame(
            {
                "tournament_id": ["300751", "300751", "300751"],
                "team_id": ["15128020", "15128021", "15128022"],
                "club_id": ["123", None, None],
                "club_code": ["ignored", "4100081", ""],
            }
        )

        urls = rows.select(_ffbridge_results_url_expr()).to_series().to_list()

        self.assertEqual(
            urls[0],
            "https://licencie.ffbridge.fr/#/resultats/simultane/"
            "300751/details/15128020?orgId=123",
        )
        self.assertEqual(
            urls[1],
            "https://licencie.ffbridge.fr/#/resultats/simultane/"
            "300751/details/15128021?orgId=4100081",
        )
        self.assertIsNone(urls[2])


if __name__ == "__main__":
    unittest.main()

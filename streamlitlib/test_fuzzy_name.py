import unittest

from streamlitlib.fuzzy_name import (
    filter_name_list,
    fuzzy_text_score,
    name_query_matches,
    rank_named_records,
)


class FuzzyNameTests(unittest.TestCase):
    def test_accent_and_typo_match_salita(self) -> None:
        self.assertGreaterEqual(fuzzy_text_score("Robert SALITA", "salita"), 0.72)
        self.assertTrue(name_query_matches("Robert SALITA", "salitta"))
        self.assertTrue(name_query_matches("Robert SALITA", "Salità"))

    def test_rank_puts_last_name_first(self) -> None:
        rows = rank_named_records(
            [
                {"player_name": "Marie Salitas", "player_number": "1"},
                {"player_name": "Robert Salita", "player_number": "2"},
            ],
            "salita",
        )
        self.assertEqual([row["player_number"] for row in rows], ["2", "1"])
        self.assertGreaterEqual(rows[0]["match_score"], 72)

    def test_filter_name_list_best_first(self) -> None:
        names = filter_name_list(
            ["Jean Balleroy", "Robert Salita", "Marie Salitas"],
            "salita",
        )
        self.assertEqual(names[0], "Robert Salita")
        self.assertIn("Marie Salitas", names)


if __name__ == "__main__":
    unittest.main()

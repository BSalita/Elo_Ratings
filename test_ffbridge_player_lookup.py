import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl

_ROOT = Path(__file__).resolve().parent
_MLBRIDGE = next(
    path
    for path in (_ROOT / "mlBridge", _ROOT.parent / "mlBridge")
    if path.is_dir()
)
if str(_MLBRIDGE.parent) not in sys.path:
    sys.path.insert(0, str(_MLBRIDGE.parent))

import ffbridge_report_service as reports


def _persons() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "lancelot_person_id": ["246273", "136662", "244120"],
            "classic_person_id": ["597539", "322582", "244120"],
            "license_number": ["9500754", "4958370", "1"],
            "display_name": ["Robert SALITA", "Guy Laumond", "Solita Duplan"],
            "first_session_date": ["2025-01-01", "2025-01-01", "2025-01-01"],
            "last_session_date": ["2026-09-07", "2025-02-03", "2026-01-01"],
        }
    )


class PlayerLookupTests(unittest.TestCase):
    def _patch_persons(self):
        from mlBridge.mlBridgeFFIndexLib import lookup_person, lookup_persons_by_name

        persons = _persons()
        return patch.object(
            reports,
            "_index_lookup_helpers",
            return_value=(lambda: persons, lookup_person, lookup_persons_by_name),
        )

    def test_unique_last_name_sets_player_id(self) -> None:
        with self._patch_persons():
            payload = reports.run_player_lookup("SALITA")

        self.assertTrue(payload["unique"])
        self.assertEqual(payload["player_id"], "246273")
        self.assertEqual(payload["rows"][0]["classic_person_id"], "597539")
        self.assertEqual(payload["rows"][0]["license_number"], "9500754")

    def test_number_resolves_through_index(self) -> None:
        with self._patch_persons():
            payload = reports.run_player_lookup("597539")

        self.assertEqual(payload["player_id"], "246273")

    def test_empty_name_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "name is required"):
            reports.run_player_lookup("  ")


def _colliding_persons() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "lancelot_person_id": ["167894", "65159"],
            "classic_person_id": ["492430", "167894"],
            "license_number": ["9932204", "1951715"],
            "display_name": ["Yves Mercier", "Arlette Abiker"],
            "first_session_date": ["2023-11-22", "2023-12-11"],
            "last_session_date": ["2024-10-02", "2026-10-14"],
        }
    )


class AmbiguousPlayerIdTests(unittest.TestCase):
    def _history(self, player_id: str) -> dict:
        import elo_filter_common
        from mlBridge.mlBridgeFFIndexLib import expand_player_aliases

        persons = _colliding_persons()
        results = pl.DataFrame(
            {
                "date": ["2026-10-14", "2024-10-02"],
                "tournament_id": ["282858", "200000"],
                "pair_name": ["Abiker – partner", "Mercier – partner"],
                "player1_id": ["167894", "492430"],
                "player2_id": ["1", "2"],
                "National_Scratch_Pct": [55.0, 50.0],
                "Club_Scratch_Pct": [None, None],
            }
        )
        with patch.object(
            elo_filter_common,
            "_ffbridge_index_helpers",
            return_value=(expand_player_aliases, lambda: persons),
        ), patch.object(reports, "load_results", return_value=(results, {})):
            return reports.run_player_history(player_id, limit=10)

    def test_bare_colliding_id_lists_prefixed_choices(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"classic:167894 = Arlette Abiker; lancelot:167894 = Yves Mercier",
        ):
            self._history("167894")

    def test_classic_prefix_selects_the_migrated_player(self) -> None:
        payload = self._history("classic:167894")
        self.assertEqual(
            [row["pair_name"] for row in payload["sessions"]], ["Abiker – partner"]
        )

    def test_lancelot_prefix_selects_the_lancelot_player(self) -> None:
        payload = self._history("lancelot:167894")
        self.assertEqual(
            [row["pair_name"] for row in payload["sessions"]], ["Mercier – partner"]
        )

    def test_api_accepts_prefixed_ids_and_rejects_junk(self) -> None:
        from pydantic import BaseModel, Field, ValidationError

        import ffbridge_api_server

        class Query(BaseModel):
            player_id: str = Field(pattern=reports.PLAYER_ID_PATTERN)

        for good in ("65159", "classic:167894", "Lancelot:167894", "license:1951715"):
            self.assertEqual(Query(player_id=good).player_id, good)
        for bad in ("name:abc", "classic:", "167894x", "classic:12:3"):
            with self.assertRaises(ValidationError):
                Query(player_id=bad)
        with self.assertRaisesRegex(ValueError, "optionally prefixed"):
            ffbridge_api_server.PlayerHistorySqlBody(player_id="167894x", sql="select 1")
        body = ffbridge_api_server.PlayerHistorySqlBody(player_id="License:1951715", sql="select 1")
        self.assertEqual(body.player_id, "License:1951715")


if __name__ == "__main__":
    unittest.main()

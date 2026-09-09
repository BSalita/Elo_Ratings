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


if __name__ == "__main__":
    unittest.main()

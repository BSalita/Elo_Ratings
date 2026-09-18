from __future__ import annotations

import unittest

from export_lancelot_dd_mismatches import _cards_key, _compact_dd, _mismatch_cells


class ExportLancelotDdMismatchesTests(unittest.TestCase):
    def test_cards_key_ignores_seat_rotation(self) -> None:
        north = (
            "N:QJ862.Q7.J84.K76 94.AT52.Q93.J943 AK753..AKT762.A2 T.KJ98643.5.QT85"
        )
        east = (
            "E:94.AT52.Q93.J943 AK753..AKT762.A2 T.KJ98643.5.QT85 QJ862.Q7.J84.K76"
        )
        self.assertEqual(_cards_key(north), _cards_key(east))
        self.assertEqual(len(_cards_key(north) or ""), 52)
        self.assertNotEqual(_cards_key(north), _cards_key(north.replace("QJ862", "QJ863")))

    def test_compact_and_mismatch_cells(self) -> None:
        rec = {
            "Lancelot_DD_N_S": 8,
            "ddss_DD_N_S": 7,
            "Lancelot_DD_N_H": 6,
            "ddss_DD_N_H": 6,
        }
        for seat in "NESW":
            for suit in "SHDCN":
                rec.setdefault(f"Lancelot_DD_{seat}_{suit}", 5)
                rec.setdefault(f"ddss_DD_{seat}_{suit}", 5)
        rec["Lancelot_DD_N_S"] = 8
        rec["ddss_DD_N_S"] = 7
        self.assertIn("N:S=8", _compact_dd(rec, "Lancelot_"))
        self.assertEqual(_mismatch_cells(rec), ["DD_N_S:8->7"])


if __name__ == "__main__":
    unittest.main()

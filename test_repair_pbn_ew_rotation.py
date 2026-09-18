from __future__ import annotations

import unittest

from endplay.types import Deal

from repair_pbn_ew_rotation import legacy_pbn_to_n


class LegacyPbnToNTests(unittest.TestCase):
    def test_legacy_east_is_the_old_180_rotation(self) -> None:
        raw = (
            "E:.KT875.AQ3.QJ984 AKQ98543.J3.T7.5 T2.Q.986542.K632 J76.A9642.KJ.AT7"
        )
        old_wrong = Deal(
            "N:AKQ98543.J3.T7.5 T2.Q.986542.K632 J76.A9642.KJ.AT7 .KT875.AQ3.QJ984"
        ).to_pbn()
        self.assertEqual(legacy_pbn_to_n(raw), old_wrong)

    def test_legacy_west_is_the_old_180_rotation(self) -> None:
        raw = (
            "W:875.AK6.T75.QT92 Q42.2.A8642.A753 JT9.J54.KJ9.K864 AK63.QT9873.Q3.J"
        )
        old_wrong = Deal(
            "N:AK63.QT9873.Q3.J 875.AK6.T75.QT92 Q42.2.A8642.A753 JT9.J54.KJ9.K864"
        ).to_pbn()
        self.assertEqual(legacy_pbn_to_n(raw), old_wrong)


if __name__ == "__main__":
    unittest.main()

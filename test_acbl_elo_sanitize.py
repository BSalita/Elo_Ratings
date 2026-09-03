from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

import polars as pl

_ROOT = Path(__file__).resolve().parent
_MLBRIDGE = next(
    path
    for path in (_ROOT / "mlBridge", _ROOT.parent / "mlBridge")
    if path.is_dir()
)
if str(_MLBRIDGE.parent) not in sys.path:
    sys.path.append(str(_MLBRIDGE.parent))
_PIPELINE = _ROOT.parent / "acbl-pipeline"
if str(_PIPELINE) not in sys.path:
    sys.path.append(str(_PIPELINE))

from acbl_elo_ratings_create import _sanitize_session_mp_top  # noqa: E402
from mlBridge.mlBridgeAugmentLib import (  # noqa: E402
    compute_pair_matchpoint_elo_ratings,
    compute_player_matchpoint_elo_ratings,
)


def _one_session(*, poison_pct: float | None = 9.99) -> pl.DataFrame:
    """Four-table session of 0.50 boards, with one optional corrupt Pct_NS."""
    ns = ["A", "B", "C", "D"]
    ew = ["E", "F", "G", "H"]
    rows = []
    for board in range(1, 4):
        for table, ns_id in enumerate(ns):
            ew_id = ew[(table + board) % len(ew)]
            pct = 0.50
            if poison_pct is not None and board == 2 and table == 0:
                pct = poison_pct
            rows.append({
                "Date": date(2026, 4, 18),
                "session_id": "2604318-18O1-2",
                "Round": board,
                "Board": board,
                "Player_ID_N": f"{ns_id}N",
                "Player_ID_S": f"{ns_id}S",
                "Player_ID_E": f"{ew_id}E",
                "Player_ID_W": f"{ew_id}W",
                "Pct_NS": pct,
                "MP_Top": 16.0,
            })
    return pl.DataFrame(rows)


class EloSanitizeTests(unittest.TestCase):
    def test_session_median_replaces_poison_mp_top(self) -> None:
        df = pl.DataFrame({
            "session_id": ["s1", "s1", "s1", "s2", "s2"],
            "MP_Top": [16.0, 16.0, 320.0, 8.0, 12.0],
        })
        out = _sanitize_session_mp_top(df)
        self.assertEqual(out["MP_Top"].to_list(), [16.0, 16.0, 16.0, 10.0, 10.0])

    def test_session_median_is_noop_without_mp_top(self) -> None:
        df = pl.DataFrame({"session_id": ["s1"], "Pct_NS": [0.5]})
        self.assertEqual(_sanitize_session_mp_top(df).columns, df.columns)

    def test_pair_elo_skips_pct_ns_outside_unit_interval(self) -> None:
        clean = compute_pair_matchpoint_elo_ratings(
            _one_session(poison_pct=None),
            minimum_sessions=1,
            provisional_boost_until=0,
        )
        poisoned = compute_pair_matchpoint_elo_ratings(
            _one_session(poison_pct=9.99),
            minimum_sessions=1,
            provisional_boost_until=0,
        )
        clean_after = clean.get_column("Elo_R_NS").to_list()
        poisoned_after = poisoned.get_column("Elo_R_NS").to_list()
        self.assertTrue(all(abs(v - 1500.0) < 1e-3 for v in clean_after))
        self.assertEqual(poisoned_after, clean_after)

    def test_player_elo_skips_pct_ns_outside_unit_interval(self) -> None:
        clean = compute_player_matchpoint_elo_ratings(
            _one_session(poison_pct=None),
            minimum_sessions=1,
            provisional_boost_until=0,
        )
        poisoned = compute_player_matchpoint_elo_ratings(
            _one_session(poison_pct=-0.2),
            minimum_sessions=1,
            provisional_boost_until=0,
        )
        self.assertEqual(
            poisoned.get_column("Elo_R_N").to_list(),
            clean.get_column("Elo_R_N").to_list(),
        )

    def test_boundary_pct_ns_is_kept(self) -> None:
        out = compute_pair_matchpoint_elo_ratings(
            _one_session(poison_pct=1.0),
            minimum_sessions=1,
            provisional_boost_until=0,
        )
        # A perfect 1.0 board must move the NS pair off the initial 1500.
        poison_row = out.filter((pl.col("Board") == 2) & (pl.col("Pct_NS") == 1.0))
        self.assertGreater(float(poison_row["Elo_R_NS"][0]), 1500.0)


if __name__ == "__main__":
    unittest.main()

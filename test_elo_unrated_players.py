from __future__ import annotations

import sys
import unittest
from datetime import date, timedelta
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

from mlBridge.mlBridgeAugmentLib import (  # noqa: E402
    compute_pair_matchpoint_elo_ratings,
    compute_player_matchpoint_elo_ratings,
    pair_key_expr,
)

NS_PAIRS = [(1, 2), (3, 4), (5, 6), (7, None)]
EW_PAIRS = [(11, 12), (13, 14), (15, 16), (17, 18)]


def _sessions(count: int = 12) -> pl.DataFrame:
    """Club-style Int32 ids. NS 7 always plays with a partner who has no ACBL number and scores 70%."""
    rows = []
    for s in range(count):
        for board in range(1, 4):
            for table, (n, south) in enumerate(NS_PAIRS):
                e, w = EW_PAIRS[(table + board) % len(EW_PAIRS)]
                rows.append({
                    "Date": date(2026, 1, 1) + timedelta(days=s),
                    "session_id": f"s{s:02d}",
                    "Board": board,
                    "Player_ID_N": n,
                    "Player_ID_S": south,
                    "Player_ID_E": e,
                    "Player_ID_W": w,
                    "Pct_NS": 0.70 if n == 7 else 0.45,
                    "MP_Top": 6.0,
                })
    return pl.DataFrame(rows, schema_overrides={f"Player_ID_{d}": pl.Int32 for d in "NESW"})


class UnratedPlayerEloTests(unittest.TestCase):
    def test_player_without_id_stays_at_initial_rating_and_is_not_reported(self) -> None:
        out = compute_player_matchpoint_elo_ratings(_sessions(), minimum_sessions=0)
        guest = out.filter(pl.col("Player_ID_S").is_null())
        self.assertGreater(guest.height, 0)
        self.assertEqual(guest["Elo_R_S"].unique().to_list(), [1500.0])
        self.assertEqual(guest["Elo_N_S"].unique().to_list(), [0])
        self.assertGreater(guest["Elo_R_N"].max(), 1500.0)

        reported = compute_player_matchpoint_elo_ratings(_sessions(), minimum_sessions=10)
        last = reported.filter(pl.col("session_id") == "s11")
        self.assertTrue(last.filter(pl.col("Player_ID_S").is_null())["Elo_R_S"].is_nan().all())
        self.assertFalse(last.filter(pl.col("Player_ID_N") == 1)["Elo_R_N"].is_nan().any())

    def test_pair_with_unknown_partner_is_unrated(self) -> None:
        out = compute_pair_matchpoint_elo_ratings(_sessions(), minimum_sessions=0)
        guest_pair = out.filter(pl.col("Player_ID_S").is_null())
        self.assertEqual(guest_pair["Elo_R_NS"].unique().to_list(), [1500.0])
        self.assertEqual(guest_pair["Elo_N_NS"].unique().to_list(), [0])
        self.assertNotIn("_Pair_ID_NS", out.columns)


class PairKeyTests(unittest.TestCase):
    def test_orders_as_strings_and_is_null_without_both_ids(self) -> None:
        frame = pl.DataFrame(
            {"a": [123, 45, 7, None, 9], "b": [45, 123, None, None, 9]},
            schema={"a": pl.Int32, "b": pl.Int32},
        )
        keys = frame.select(pair_key_expr("a", "b").alias("key"))["key"].to_list()
        self.assertEqual(keys, ["123-45", "123-45", None, None, None])


if __name__ == "__main__":
    unittest.main()

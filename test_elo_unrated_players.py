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
    compute_event_start_end_elo_columns,
    compute_matchpoint_elo_ratings,
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


# Two sections in one session reuse pair numbers 1 and 2. Section B comes
# first in the frame and its pair 1 has a partner without an ACBL number.
SECTIONS = {
    "B": {"NS": [(7, None), (5, 6)], "EW": [(15, 16), (17, 18)]},
    "A": {"NS": [(1, 2), (3, 4)], "EW": [(11, 12), (13, 14)]},
}
NS_PCT = {1: 0.70, 3: 0.30, 5: 0.60, 7: 0.40}


def _two_section_sessions(count: int = 4) -> pl.DataFrame:
    rows = []
    for s in range(count):
        for board in range(1, 4):
            for pairs in SECTIONS.values():
                for table, (n, south) in enumerate(pairs["NS"]):
                    ew_number = (table + board) % 2
                    e, w = pairs["EW"][ew_number]
                    rows.append({
                        "Date": date(2026, 1, 1) + timedelta(days=s),
                        "session_id": f"s{s:02d}",
                        "Board": board,
                        "Pair_Number_NS": table + 1,
                        "Pair_Number_EW": ew_number + 1,
                        "Player_ID_N": n,
                        "Player_ID_S": south,
                        "Player_ID_E": e,
                        "Player_ID_W": w,
                        "Pct_NS": NS_PCT[n],
                        "MP_Top": 2.0,
                    })
    return pl.DataFrame(rows, schema_overrides={f"Player_ID_{d}": pl.Int32 for d in "NESW"})


class EventStartEndPairTests(unittest.TestCase):
    """Session start and end pair ratings belong to the pair, not to its pair number."""

    def setUp(self) -> None:
        self.out = compute_matchpoint_elo_ratings(_two_section_sessions(), minimum_sessions=1)

    def test_each_pair_gets_its_own_start_and_end_rating(self) -> None:
        for direction, a, b in (("NS", "Player_ID_N", "Player_ID_S"), ("EW", "Player_ID_E", "Player_ID_W")):
            known = self.out.filter(pair_key_expr(a, b).is_not_null())
            self.assertGreater(known.height, 0)
            start, end = known[f"Elo_R_{direction}_EventStart"], known[f"Elo_R_{direction}_EventEnd"]
            self.assertFalse(start.is_nan().any())
            self.assertFalse(end.is_nan().any())
            self.assertEqual(start.to_list(), known[f"Elo_R_{direction}_Before"].to_list())
            self.assertEqual(end.to_list(), known[f"Elo_R_{direction}"].to_list())

    def test_pairs_that_share_a_pair_number_differ(self) -> None:
        last = self.out.filter(pl.col("session_id") == "s03")
        by_north = {
            n: last.filter(pl.col("Player_ID_N") == n)["Elo_R_NS_EventStart"].unique().to_list()
            for n in (1, 3, 5)
        }
        self.assertTrue(all(len(v) == 1 for v in by_north.values()), by_north)
        # 3-4 (section A) and 5-6 (section B) are both pair number 2.
        self.assertLess(by_north[3][0], 1500.0)
        self.assertGreater(by_north[5][0], 1500.0)
        # 1-2 shares pair number 1 with the pair that has no rating.
        self.assertGreater(by_north[1][0], 1500.0)

    def test_pair_with_unknown_partner_stays_unrated(self) -> None:
        guest = self.out.filter(pl.col("Player_ID_S").is_null())
        self.assertGreater(guest.height, 0)
        self.assertTrue(guest["Elo_R_NS_EventStart"].is_nan().all())
        self.assertTrue(guest["Elo_R_NS_EventEnd"].is_nan().all())

    def test_no_helper_columns_are_left(self) -> None:
        self.assertEqual([c for c in self.out.columns if c.startswith("_")], [])

    def test_pair_number_is_the_key_without_player_ids(self) -> None:
        frame = self.out.drop(
            [f"Player_ID_{d}" for d in "NESW"]
            + [c for c in self.out.columns if c.endswith(("_EventStart", "_EventEnd"))]
        )
        out = compute_event_start_end_elo_columns(frame)
        per_number = out.group_by("session_id", "Pair_Number_NS").agg(
            pl.col("Elo_R_NS_EventStart").n_unique().alias("n")
        )
        self.assertEqual(per_number["n"].unique().to_list(), [1])


if __name__ == "__main__":
    unittest.main()

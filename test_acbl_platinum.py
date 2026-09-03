from __future__ import annotations

import unittest
from pathlib import Path

import polars as pl

from acbl_api_server import (
    _reject_club_platinum,
    _self_filter_sql_clauses,
    _sql_string_list,
)
from elo_common import coerce_bool
from acbl_platinum import (
    is_platinum_mp_color,
    load_platinum_events,
    platinum_event_ids,
    platinum_events_from_awards,
    write_platinum_events_sidecar,
)
from fastapi import HTTPException


class PlatinumAwardTests(unittest.TestCase):
    def test_mp_color_is_strictly_platinum(self) -> None:
        self.assertTrue(is_platinum_mp_color("Platinum"))
        self.assertTrue(is_platinum_mp_color(" platinum "))
        self.assertFalse(is_platinum_mp_color("Gold"))
        self.assertFalse(is_platinum_mp_color("NABC+"))
        self.assertFalse(is_platinum_mp_color(None))
        self.assertFalse(is_platinum_mp_color(""))

    def test_extracts_only_platinum_event_ids(self) -> None:
        awards = pl.DataFrame(
            {
                "event_id": ["NABC253-NAIL", "2605106-30OP", "NABC262-OSHL", "NABC253-NAIL"],
                "event_name": [
                    "NAIL LIFE MASTER PAIRS",
                    "A/X/Y Open Pairs",
                    "OSHLAG FAST PAIRS",
                    "NAIL LIFE MASTER PAIRS",
                ],
                "mp_color": ["Platinum", "Gold", "Platinum", "Platinum"],
                "mp_rating": ["NABC+", "Regional", "NABC+", "NABC+"],
            }
        )
        events = platinum_events_from_awards(awards)
        self.assertEqual(platinum_event_ids(events), ["NABC253-NAIL", "NABC262-OSHL"])
        self.assertEqual(events["event_name"].to_list(), [
            "NAIL LIFE MASTER PAIRS",
            "OSHLAG FAST PAIRS",
        ])

    def test_nabc_plus_gold_is_not_platinum(self) -> None:
        awards = pl.DataFrame(
            {
                "event_id": ["NABC254-OP1"],
                "event_name": ["Open Pairs 1"],
                "mp_color": ["Gold"],
                "mp_rating": ["NABC+"],
            }
        )
        self.assertEqual(platinum_event_ids(platinum_events_from_awards(awards)), [])

    def test_sidecar_round_trip(self) -> None:
        awards = pl.DataFrame(
            {
                "event_id": ["NABC253-NAIL"],
                "event_name": ["NAIL LIFE MASTER PAIRS"],
                "mp_color": ["Platinum"],
                "mp_rating": ["NABC+"],
            }
        )
        data_root = Path(self.id().replace(".", "_"))
        # Use a temp dir under the test file's directory is messy; write via tmp.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_platinum_events_sidecar(awards, root)
            loaded = load_platinum_events(root)
            self.assertEqual(platinum_event_ids(loaded), ["NABC253-NAIL"])

    def test_club_platinum_filter_is_rejected(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            _reject_club_platinum("club", True)
        self.assertEqual(ctx.exception.status_code, 400)
        _reject_club_platinum("tournament", True)
        _reject_club_platinum("club", False)

    def test_coerce_bool_url_values(self) -> None:
        self.assertTrue(coerce_bool("True"))
        self.assertTrue(coerce_bool("1"))
        self.assertFalse(coerce_bool("False"))
        self.assertFalse(coerce_bool(False))
        with self.assertRaises(ValueError):
            coerce_bool("maybe")

    def test_self_filter_sql_uses_mp_color_when_present(self) -> None:
        frame = pl.DataFrame({
            "event_id": ["A", "B"],
            "Pct_NS": [0.5, 0.6],
            "mp_color": ["Platinum", "Gold"],
        })
        clauses = _self_filter_sql_clauses(
            frame, None, "All", "All", platinum_events=True,
        )
        self.assertTrue(any("mp_color" in clause for clause in clauses))
        self.assertTrue(any("platinum" in clause for clause in clauses))
        self.assertFalse(any("event_id IN" in clause for clause in clauses))

    def test_self_filter_sql_includes_platinum_event_ids(self) -> None:
        from unittest.mock import patch

        frame = pl.DataFrame({"event_id": ["A", "B"], "Pct_NS": [0.5, 0.6]})
        with patch(
            "acbl_api_server._require_platinum_event_ids",
            return_value=["NABC253-NAIL", "NABC262-OSHL"],
        ):
            clauses = _self_filter_sql_clauses(
                frame, None, "All", "All", platinum_events=True,
            )
        self.assertTrue(any("event_id IN" in clause for clause in clauses))
        self.assertTrue(any("NABC253-NAIL" in clause for clause in clauses))
        self.assertFalse(
            any("event_id IN" in clause for clause in _self_filter_sql_clauses(
                frame, None, "All", "All", platinum_events=False,
            ))
        )

    def test_apply_platinum_filter_uses_mp_color(self) -> None:
        from acbl_api_server import _apply_platinum_event_filter

        frame = pl.DataFrame({
            "event_id": ["NABC253-NAIL", "2605106-30OP"],
            "mp_color": ["Platinum", "Gold"],
        })
        filtered = _apply_platinum_event_filter(frame)
        self.assertEqual(filtered["event_id"].to_list(), ["NABC253-NAIL"])

    def test_sql_in_list_escapes_quotes(self) -> None:
        self.assertEqual(
            _sql_string_list(["NABC253-NAIL", "O'Brien"]),
            "('NABC253-NAIL', 'O''Brien')",
        )


if __name__ == "__main__":
    unittest.main()

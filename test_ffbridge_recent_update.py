"""Schedule window and recent-board upsert for FFBridge."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import polars as pl

from ffbridge_recent_update import coverage_start, historical_max_date, upsert_boards


class CoverageTests(unittest.TestCase):
    def test_stale_history_opens_the_gap(self) -> None:
        start = coverage_start(date(2026, 9, 9), "hour", date(2026, 10, 3))
        self.assertEqual(start, date(2026, 9, 10))

    def test_quarter_reaches_back_when_history_is_current(self) -> None:
        start = coverage_start(date(2026, 10, 2), "quarter", date(2026, 10, 3))
        self.assertEqual(start, date(2026, 7, 1))


class UpsertTests(unittest.TestCase):
    def test_replace_and_prune(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "boards.parquet"
            first = pl.DataFrame({"session_id": ["1"], "Date": ["2026-10-02"], "board": ["1"]})
            self.assertEqual(
                upsert_boards(path, first, historical_max=date(2026, 10, 1)),
                1,
            )
            replacement = pl.DataFrame(
                {"session_id": ["1"], "Date": ["2026-09-01"], "board": ["9"]}
            )
            self.assertEqual(
                upsert_boards(path, replacement, historical_max=date(2026, 10, 1)),
                0,
            )
            self.assertFalse(path.exists())

    def test_historical_max_reads_date_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "historical.parquet"
            pl.DataFrame({"Date": ["2026-10-01", "2026-09-01"]}).write_parquet(path)
            self.assertEqual(historical_max_date(path), date(2026, 10, 1))


if __name__ == "__main__":
    unittest.main()

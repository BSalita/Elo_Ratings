import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import polars as pl

import publish_ffbridge_hierarchical as publisher


def _row(session_id: str, revision: str, archived_at: str) -> dict:
    relative = f"fragments/year=2026/session_id={session_id}/layout_version=3/revision={revision}"
    return {
        "session_id": session_id,
        "revision": revision,
        "Date": date(2026, 10, 1),
        "year": 2026,
        "series_id": None,
        "archived_at": archived_at,
        "board_rows": 1,
        "result_rows": 1,
        "boards_path": f"{relative}/boards.parquet",
        "results_path": f"{relative}/results.parquet",
    }


def _write_archive(root: Path, rows: list[dict]) -> None:
    for row in rows:
        for column in ("boards_path", "results_path"):
            path = root / row[column]
            path.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame({"session_id": [row["session_id"]]}).write_parquet(path)
    pl.DataFrame(rows).write_parquet(root / "manifest.parquet")
    (root / "metadata.json").write_text(json.dumps({"layout_version": 3}), encoding="utf-8")


class PublishKeepsDestinationSessionsTests(unittest.TestCase):
    def test_sessions_the_service_added_survive_a_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, destination = Path(tmp) / "source", Path(tmp) / "destination"
            _write_archive(source, [_row("1", "a", "2026-09-30T00:00:00+00:00")])
            _write_archive(
                destination,
                [
                    _row("1", "old", "2026-09-01T00:00:00+00:00"),
                    _row("394099", "live", "2026-10-17T00:00:00+00:00"),
                ],
            )

            summary = publisher.publish(source, destination, workers=1)

            manifest = pl.read_parquet(destination / "manifest.parquet").sort("session_id")
            self.assertEqual(manifest.select("session_id", "revision").rows(), [("1", "a"), ("394099", "live")])
            self.assertEqual(summary["sessions"], 2)

    def test_destination_row_without_its_fragment_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, destination = Path(tmp) / "source", Path(tmp) / "destination"
            _write_archive(source, [_row("1", "a", "2026-09-30T00:00:00+00:00")])
            orphan = _row("394099", "live", "2026-10-17T00:00:00+00:00")
            _write_archive(destination, [orphan])
            (destination / orphan["results_path"]).unlink()

            with self.assertRaises(FileNotFoundError):
                publisher.publish(source, destination, workers=1)


if __name__ == "__main__":
    unittest.main()

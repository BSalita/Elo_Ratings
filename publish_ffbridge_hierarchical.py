"""Publish the production FFBridge hierarchical archive to a target directory.

Copies only what the postmortem service reads: the latest-revision
boards/results fragment for every manifest session, a manifest filtered to
those revisions, and metadata.json. Domain shards, compacted datasets,
superseded revisions, SQLite files and logs are builder artifacts and are
not published. metadata.json is written last so readers never see a layout
version without its fragments.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import shutil
import time
from datetime import datetime

import polars as pl
from tqdm import tqdm


EXPECTED_LAYOUT_VERSION = 3


def _latest(manifest: pl.DataFrame) -> pl.DataFrame:
    return (
        manifest.sort(["session_id", "archived_at", "revision"])
        .unique(subset=["session_id"], keep="last", maintain_order=True)
        .sort(["Date", "session_id"])
    )


def _same(source: pathlib.Path, destination: pathlib.Path) -> bool:
    if not destination.is_file():
        return False
    source_stat = source.stat()
    destination_stat = destination.stat()
    return (
        source_stat.st_size == destination_stat.st_size
        and destination_stat.st_mtime >= source_stat.st_mtime - 2
    )


def _copy(source: pathlib.Path, destination: pathlib.Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return source.stat().st_size


def publish(source: pathlib.Path, destination: pathlib.Path, workers: int) -> dict:
    metadata_path = source / "metadata.json"
    manifest_path = source / "manifest.parquet"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("layout_version") != EXPECTED_LAYOUT_VERSION:
        raise ValueError(
            f"{metadata_path} is layout version {metadata.get('layout_version')!r}; "
            f"expected {EXPECTED_LAYOUT_VERSION}. Run the layout migration first."
        )
    latest = _latest(pl.read_parquet(manifest_path))
    if latest.is_empty():
        raise ValueError(f"Empty manifest: {manifest_path}")

    pairs: list[tuple[pathlib.Path, pathlib.Path]] = []
    missing: list[str] = []
    for row in latest.iter_rows(named=True):
        for column in ("boards_path", "results_path"):
            relative = str(row[column])
            if f"layout_version={EXPECTED_LAYOUT_VERSION}" not in relative:
                raise ValueError(f"Fragment is not layout v{EXPECTED_LAYOUT_VERSION}: {relative}")
            if not (source / relative).is_file():
                missing.append(relative)
            pairs.append((source / relative, destination / relative))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} manifest fragments missing under {source}; first: {missing[:3]}"
        )

    pending = [(s, d) for s, d in pairs if not _same(s, d)]
    copied_bytes = 0
    with tqdm(total=len(pending), desc="Publishing fragments", unit="file") as progress:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_copy, s, d) for s, d in pending]
            for future in concurrent.futures.as_completed(futures):
                copied_bytes += future.result()
                progress.update()

    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / f".manifest.parquet.{os.getpid()}.tmp"
    try:
        latest.write_parquet(temporary, compression="zstd")
        os.replace(temporary, destination / "manifest.parquet")
    finally:
        temporary.unlink(missing_ok=True)
    _copy(metadata_path, destination / "metadata.json")
    return {
        "sessions": latest.height,
        "fragments_total": len(pairs),
        "fragments_copied": len(pending),
        "gb_copied": round(copied_bytes / 1e9, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--destination", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    started = datetime.now()
    clock = time.perf_counter()
    print(f"[publish-hierarchical] start {started:%Y-%m-%d %H:%M:%S} "
          f"{args.source} -> {args.destination}", flush=True)
    try:
        print(json.dumps(publish(args.source, args.destination, args.workers), indent=2))
        return 0
    finally:
        print(f"[publish-hierarchical] end {datetime.now():%Y-%m-%d %H:%M:%S} "
              f"(elapsed {time.perf_counter() - clock:.1f}s)", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

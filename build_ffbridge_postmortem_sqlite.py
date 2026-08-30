"""Build a relational SQLite database from compacted FFBridge domain shards."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sqlite3
import time
from contextlib import closing
from datetime import datetime
from typing import Any

import duckdb
import polars as pl
from tqdm import tqdm


DEFAULT_HIERARCHICAL_DIR = pathlib.Path(
    r"E:\bridge\data\ffbridge\postmortem_archive_hierarchical"
)
DEFAULT_OUTPUT = DEFAULT_HIERARCHICAL_DIR / "ffbridge_postmortem.sqlite"
DEFAULT_SCHEMA = pathlib.Path(__file__).with_name(
    "ffbridge_postmortem_sqlite_schema.sql"
)


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _latest_sessions(manifest_path: pathlib.Path) -> pl.DataFrame:
    manifest = pl.read_parquet(manifest_path)
    return (
        manifest.sort(["session_id", "archived_at", "revision"])
        .unique(subset=["session_id"], keep="last", maintain_order=True)
        .select(
            "session_id",
            "revision",
            "Date",
            "year",
            "series_id",
            "archived_at",
            "board_rows",
            "result_rows",
        )
        .sort(["Date", "session_id"])
    )


def _create_static_schema(
    database: pathlib.Path,
    schema_path: pathlib.Path,
    layout_version: int,
) -> None:
    schema_sql = schema_path.read_text(encoding="utf-8")
    with closing(sqlite3.connect(database)) as connection:
        connection.executescript(schema_sql)
        connection.execute(
            "INSERT INTO schema_info VALUES (?, ?, ?)",
            (
                1,
                layout_version,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        connection.commit()


def build_sqlite(
    hierarchical_dir: pathlib.Path,
    output_path: pathlib.Path,
    schema_path: pathlib.Path,
    *,
    domain_subdir: str = "dataset_domains",
    memory_limit: str = "64GB",
    threads: int = 16,
) -> dict[str, Any]:
    started = time.perf_counter()
    root = pathlib.Path(hierarchical_dir)
    domain_root = root / domain_subdir
    success_path = domain_root / "_SUCCESS.json"
    catalog_path = domain_root / "domain_catalog.parquet"
    manifest_path = root / "manifest.parquet"
    metadata_path = root / "metadata.json"
    for required in (success_path, catalog_path, manifest_path, metadata_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required SQLite source is missing: {required}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    catalog = pl.read_parquet(catalog_path)
    sessions = _latest_sessions(manifest_path)
    shards = catalog.select(
        "shard", "source_table", "domain"
    ).unique(maintain_order=True)
    first_shard = str(shards["shard"][0])
    included_ids = (
        pl.scan_parquet(
            list(domain_root.glob(f"year=*/series_id=*/{first_shard}.parquet"))
        )
        .select(pl.col("session_id").cast(pl.String))
        .unique()
        .collect()
        .get_column("session_id")
    )
    sessions = sessions.filter(
        pl.col("session_id").cast(pl.String).is_in(included_ids.implode())
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    _create_static_schema(
        temporary,
        schema_path,
        int(metadata["layout_version"]),
    )

    connection = duckdb.connect()
    tables: list[tuple[str, str]] = []
    try:
        connection.execute(f"SET memory_limit={_literal(memory_limit)}")
        connection.execute(f"SET threads={int(threads)}")
        connection.execute("LOAD sqlite")
        connection.execute(
            f"ATTACH {_literal(temporary.as_posix())} "
            "AS sqlite_output (TYPE sqlite)"
        )
        connection.register("latest_sessions", sessions.to_arrow())
        connection.execute(
            "INSERT INTO sqlite_output.sessions SELECT * FROM latest_sessions"
        )
        with tqdm(
            total=shards.height,
            desc="Building SQLite domain tables",
        ) as progress:
            for row in shards.iter_rows(named=True):
                shard = str(row["shard"])
                source_table = str(row["source_table"])
                files = sorted(
                    domain_root.glob(
                        f"year=*/series_id=*/{shard}.parquet"
                    )
                )
                if not files:
                    raise FileNotFoundError(
                        f"No compacted files found for domain shard {shard}"
                    )
                source_sql = ", ".join(
                    _literal(path.as_posix()) for path in files
                )
                connection.execute(
                    f"CREATE TABLE sqlite_output.{_identifier(shard)} AS "
                    f"SELECT * FROM read_parquet([{source_sql}], "
                    "union_by_name=true)"
                )
                tables.append((shard, source_table))
                progress.update()
        connection.execute("DETACH sqlite_output")
    finally:
        connection.close()

    with closing(sqlite3.connect(temporary)) as sqlite:
        sqlite.executemany(
            "INSERT INTO column_catalog "
            "(table_name, logical_column, source_table, domain, ordinal) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (
                    row["shard"],
                    row["logical_column"],
                    row["source_table"],
                    row["domain"],
                    row["ordinal"],
                )
                for row in catalog.iter_rows(named=True)
            ],
        )
        with tqdm(total=len(tables), desc="Indexing SQLite domain tables") as progress:
            for table, source_table in tables:
                keys = (
                    "session_id, Board"
                    if source_table == "boards"
                    else "session_id, Board, _result_row_id"
                )
                sqlite.execute(
                    f"CREATE INDEX {_identifier(table + '_lookup')} "
                    f"ON {_identifier(table)} ({keys})"
                )
                progress.update()
        sqlite.commit()
        integrity = sqlite.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {integrity}")
    for attempt in range(20):
        try:
            os.replace(temporary, output_path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.25 * (attempt + 1))
    return {
        "database": str(output_path),
        "sessions": sessions.height,
        "domain_tables": len(tables),
        "logical_columns": catalog.height,
        "size_bytes": output_path.stat().st_size,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hierarchical-dir",
        type=pathlib.Path,
        default=DEFAULT_HIERARCHICAL_DIR,
    )
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--schema", type=pathlib.Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--domain-subdir", default="dataset_domains")
    parser.add_argument("--memory-limit", default="64GB")
    parser.add_argument(
        "--threads",
        type=int,
        default=min(16, os.cpu_count() or 1),
    )
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    started = datetime.now()
    clock = time.perf_counter()
    print(
        f"[sqlite-builder] start {started.isoformat(timespec='seconds')}",
        flush=True,
    )
    try:
        result = build_sqlite(
            args.hierarchical_dir.resolve(),
            args.output.resolve(),
            args.schema.resolve(),
            domain_subdir=args.domain_subdir,
            memory_limit=args.memory_limit,
            threads=args.threads,
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0
    finally:
        print(
            f"[sqlite-builder] end {datetime.now().isoformat(timespec='seconds')} "
            f"(elapsed {time.perf_counter() - clock:.1f}s)",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())

"""Compact hierarchical FFBridge fragments into bounded-width domain shards."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pathlib
import re
import time
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import duckdb
import polars as pl
from tqdm import tqdm


DEFAULT_HIERARCHICAL_DIR = pathlib.Path(
    r"E:\bridge\data\ffbridge\postmortem_archive_hierarchical"
)
DEFAULT_MAX_DOMAIN_COLUMNS = 200
CATALOG_FILENAME = "domain_catalog.parquet"
SUCCESS_FILENAME = "_SUCCESS.json"
COMPACTION_FORMAT_VERSION = 2


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return cleaned or "misc"


def _domain(
    source_table: str,
    column: str,
    entry: Mapping[str, Any],
) -> str:
    storage = str(entry["storage_column"])
    if entry.get("field") is not None:
        return storage
    if source_table == "boards":
        if column.startswith(("DD_", "DDScore_", "DD_Score_")):
            return "double_dummy"
        if column.startswith("Par"):
            return "par"
        if column.startswith("EV_"):
            return "expected_value"
        if column.startswith(("At_", "Can_", "Bid", "Best_", "Our_", "Opp_")):
            return "bidding_features"
        if column.startswith(("PBN", "Dealer", "Vul", "iVul")):
            return "deal"
        return "board_features"
    if column.startswith(("Player_", "Pair_")):
        return "players"
    if column.startswith(
        ("Contract", "Declarer", "Result", "Tricks", "Bid", "Dbl")
    ):
        return "contract"
    if column.startswith(("Score_", "Pct_", "MP_", "IMP")):
        return "score"
    return "result_core"


def _latest_manifest(root: pathlib.Path) -> pl.DataFrame:
    manifest_path = root / "manifest.parquet"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Hierarchical manifest not found: {manifest_path}")
    manifest = pl.read_parquet(manifest_path)
    required = {
        "session_id",
        "revision",
        "year",
        "series_id",
        "archived_at",
        "boards_path",
        "results_path",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Hierarchical manifest lacks columns: {sorted(missing)}")
    return (
        manifest.sort(["session_id", "archived_at", "revision"])
        .unique(subset=["session_id"], keep="last", maintain_order=True)
        .sort(["year", "series_id", "session_id"])
    )


def _domain_catalog(
    mapping: Mapping[str, Mapping[str, Any]],
    max_columns: int,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    keys_by_table = {
        "boards": ("session_id", "Board"),
        "results": ("session_id", "Board", "_result_row_id"),
    }
    for source_table, keys in keys_by_table.items():
        grouped: dict[str, list[str]] = {}
        for column, entry in mapping.items():
            if entry.get("table") != source_table or column in keys:
                continue
            grouped.setdefault(_domain(source_table, column, entry), []).append(column)
        for domain, columns in sorted(grouped.items()):
            ordered = sorted(columns)
            for offset in range(0, len(ordered), max_columns):
                chunk_number = offset // max_columns + 1
                chunk = ordered[offset : offset + max_columns]
                shard = (
                    f"{source_table}_{_safe_name(domain)}_{chunk_number:03d}"
                )
                for ordinal, column in enumerate(chunk):
                    entry = mapping[column]
                    rows.append(
                        {
                            "shard": shard,
                            "source_table": source_table,
                            "domain": domain,
                            "chunk": chunk_number,
                            "ordinal": ordinal,
                            "logical_column": column,
                            "storage_column": str(entry["storage_column"]),
                            "struct_field": (
                                str(entry["field"])
                                if entry.get("field") is not None
                                else None
                            ),
                        }
                    )
    if not rows:
        raise ValueError("Hierarchical metadata produced no domain columns")
    return pl.DataFrame(rows, infer_schema_length=None).sort(
        ["source_table", "shard", "ordinal"]
    )


def _atomic_write_parquet(frame: pl.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.write_parquet(temporary, compression="zstd", statistics=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(value: Mapping[str, Any], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _projection(
    catalog_rows: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    storage_aliases: Mapping[str, str],
) -> str:
    expressions = [_identifier(key) for key in keys]
    for row in catalog_rows:
        storage_name = str(row["storage_column"])
        storage = _identifier(storage_aliases.get(storage_name, storage_name))
        field = row["struct_field"]
        expression = (
            f"struct_extract({storage}, {_literal(str(field))})"
            if field is not None
            else storage
        )
        expressions.append(
            f"{expression} AS {_identifier(str(row['logical_column']))}"
        )
    return ", ".join(expressions)


def _duckdb_storage_aliases(path: pathlib.Path) -> dict[str, str]:
    aliases: dict[str, str] = {}
    counts: dict[str, int] = {}
    for column in pl.read_parquet_schema(path):
        normalized = column.lower()
        occurrence = counts.get(normalized, 0)
        aliases[column] = column if occurrence == 0 else f"{column}_{occurrence}"
        counts[normalized] = occurrence + 1
    return aliases


def _compact_shard(
    connection: duckdb.DuckDBPyConnection,
    sources: Sequence[pathlib.Path],
    destination: pathlib.Path,
    projection: str,
) -> None:
    if not sources:
        raise ValueError(f"No sources for {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    source_sql = ", ".join(_literal(path.as_posix()) for path in sources)
    target = _literal(temporary.as_posix())
    try:
        connection.execute(
            f"COPY (SELECT {projection} "
            f"FROM read_parquet([{source_sql}], union_by_name=true)) "
            f"TO {target} "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 131072)"
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _polars_projection(
    catalog_rows: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> list[pl.Expr]:
    key_types = {
        "session_id": pl.String,
        "Board": pl.Int64,
        "_result_row_id": pl.Int64,
    }
    expressions = [
        pl.col(key).cast(key_types[key]).alias(key) for key in keys
    ]
    for row in catalog_rows:
        expression = pl.col(str(row["storage_column"]))
        field = row["struct_field"]
        if field is not None:
            expression = expression.struct.field(str(field))
        expressions.append(expression.alias(str(row["logical_column"])))
    return expressions


def _batch_signature(rows: pl.DataFrame) -> str:
    payload = "\n".join(
        f"{session_id}:{revision}"
        for session_id, revision in rows.select(
            "session_id", "revision"
        ).iter_rows()
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stage_partition_batches(
    root: pathlib.Path,
    batch_root: pathlib.Path,
    rows: pl.DataFrame,
    catalog: pl.DataFrame,
    *,
    batch_size: int,
    progress: tqdm[Any],
) -> tuple[int, int]:
    staged = 0
    skipped = 0
    shard_names = catalog["shard"].unique(maintain_order=True).to_list()
    for offset in range(0, rows.height, batch_size):
        batch_rows = rows.slice(offset, batch_size)
        batch_number = offset // batch_size
        batch_dir = batch_root / f"batch={batch_number:05d}"
        marker_path = batch_dir / "_SUCCESS.json"
        signature = _batch_signature(batch_rows)
        expected = [batch_dir / f"{shard}.parquet" for shard in shard_names]
        if marker_path.is_file() and all(path.is_file() for path in expected):
            state = json.loads(marker_path.read_text(encoding="utf-8"))
            if (
                state.get("format_version") == COMPACTION_FORMAT_VERSION
                and state.get("signature") == signature
            ):
                skipped += 1
                progress.update()
                continue
        section_started = time.perf_counter()
        for source_table, keys in (
            ("boards", ("session_id", "Board")),
            ("results", ("session_id", "Board", "_result_row_id")),
        ):
            source_column = f"{source_table}_path"
            source_paths = [
                root / str(value) for value in batch_rows[source_column].to_list()
            ]
            source_frame = pl.concat(
                [pl.read_parquet(path) for path in source_paths],
                how="vertical_relaxed",
                rechunk=False,
            )
            table_catalog = catalog.filter(
                pl.col("source_table") == source_table
            )
            for shard in table_catalog["shard"].unique(maintain_order=True):
                shard_rows = table_catalog.filter(pl.col("shard") == shard)
                projection = _polars_projection(
                    list(shard_rows.iter_rows(named=True)),
                    keys,
                )
                _atomic_write_parquet(
                    source_frame.select(projection),
                    batch_dir / f"{shard}.parquet",
                )
            del source_frame
            gc.collect()
        _atomic_write_json(
            {
                "format_version": COMPACTION_FORMAT_VERSION,
                "signature": signature,
                "sessions": batch_rows.height,
                "generated_at": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "elapsed_seconds": round(
                    time.perf_counter() - section_started, 3
                ),
            },
            marker_path,
        )
        staged += 1
        progress.update()
    return staged, skipped


def compact_domains(
    hierarchical_dir: pathlib.Path,
    *,
    max_columns: int = DEFAULT_MAX_DOMAIN_COLUMNS,
    memory_limit: str = "64GB",
    threads: int = 16,
    output_subdir: str = "dataset_domains",
    session_limit: int | None = None,
    batch_size: int = 32,
) -> dict[str, Any]:
    started = time.perf_counter()
    root = pathlib.Path(hierarchical_dir)
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Hierarchy metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mapping = metadata.get("column_mapping")
    if not isinstance(mapping, dict):
        raise ValueError("Hierarchy metadata lacks column_mapping")
    manifest = _latest_manifest(root)
    if session_limit is not None:
        manifest = manifest.head(session_limit)
    if manifest.is_empty():
        raise ValueError("Hierarchy manifest is empty")
    catalog = _domain_catalog(mapping, max_columns)
    output_root = root / output_subdir
    _atomic_write_parquet(catalog, output_root / CATALOG_FILENAME)

    groups: list[tuple[int, str, pl.DataFrame]] = []
    for key, rows in manifest.group_by(["year", "series_id"], maintain_order=True):
        year, series_id = key
        groups.append((int(year), str(series_id or "unknown"), rows))
    shard_names = catalog["shard"].unique(maintain_order=True).to_list()
    total_batches = sum(
        (rows.height + batch_size - 1) // batch_size
        for _year, _series_id, rows in groups
    )
    staged_batches = 0
    skipped_batches = 0
    batch_base = root / f".{output_subdir}_batches"
    with tqdm(total=total_batches, desc="Staging domain batches") as progress:
        for year, series_id, rows in groups:
            safe_series = series_id.replace("/", "_").replace("\\", "_")
            batch_root = (
                batch_base
                / f"year={year}"
                / f"series_id={safe_series}"
            )
            staged, skipped_batch_count = _stage_partition_batches(
                root,
                batch_root,
                rows,
                catalog,
                batch_size=batch_size,
                progress=progress,
            )
            staged_batches += staged
            skipped_batches += skipped_batch_count

    total_jobs = len(groups) * len(shard_names)
    rebuilt = 0
    skipped = 0
    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit={_literal(memory_limit)}")
        con.execute(f"SET threads={int(threads)}")
        temp_dir = root / ".duckdb_domain_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory={_literal(temp_dir.as_posix())}")
        with tqdm(total=total_jobs, desc="Compacting hierarchical domains") as progress:
            for year, series_id, rows in groups:
                safe_series = series_id.replace("/", "_").replace("\\", "_")
                partition = (
                    output_root
                    / f"year={year}"
                    / f"series_id={safe_series}"
                )
                newest_source = max(rows["archived_at"].to_list())
                for shard in shard_names:
                    shard_catalog = catalog.filter(pl.col("shard") == shard)
                    first = shard_catalog.row(0, named=True)
                    source_table = str(first["source_table"])
                    keys = (
                        ("session_id", "Board")
                        if source_table == "boards"
                        else ("session_id", "Board", "_result_row_id")
                    )
                    destination = partition / f"{shard}.parquet"
                    state_path = destination.with_suffix(".state.json")
                    if destination.is_file() and state_path.is_file():
                        state = json.loads(state_path.read_text(encoding="utf-8"))
                        if (
                            state.get("format_version")
                            == COMPACTION_FORMAT_VERSION
                            and state.get("newest_source") == newest_source
                        ):
                            skipped += 1
                            progress.update()
                            continue
                    sources = sorted(
                        (
                            batch_base
                            / f"year={year}"
                            / f"series_id={safe_series}"
                        ).glob(f"batch=*/{shard}.parquet")
                    )
                    section_started = time.perf_counter()
                    _compact_shard(con, sources, destination, "*")
                    _atomic_write_json(
                        {
                            "format_version": COMPACTION_FORMAT_VERSION,
                            "generated_at": datetime.now(timezone.utc).isoformat(
                                timespec="seconds"
                            ),
                            "newest_source": newest_source,
                            "session_count": rows.height,
                            "source_table": source_table,
                            "shard": shard,
                            "column_count": shard_catalog.height,
                            "elapsed_seconds": round(
                                time.perf_counter() - section_started, 3
                            ),
                        },
                        state_path,
                    )
                    rebuilt += 1
                    progress.update()
    finally:
        con.close()
    result = {
        "sessions": manifest.height,
        "partitions": len(groups),
        "shards_per_partition": len(shard_names),
        "jobs_rebuilt": rebuilt,
        "jobs_skipped": skipped,
        "batches_staged": staged_batches,
        "batches_skipped": skipped_batches,
        "batch_size": batch_size,
        "max_domain_columns": max_columns,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    _atomic_write_json(
        {
            **result,
            "format_version": COMPACTION_FORMAT_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "hierarchical_layout_version": metadata.get("layout_version"),
        },
        output_root / SUCCESS_FILENAME,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hierarchical-dir",
        type=pathlib.Path,
        default=DEFAULT_HIERARCHICAL_DIR,
    )
    parser.add_argument(
        "--max-domain-columns",
        type=int,
        default=DEFAULT_MAX_DOMAIN_COLUMNS,
    )
    parser.add_argument("--memory-limit", default="64GB")
    parser.add_argument("--output-subdir", default="dataset_domains")
    parser.add_argument("--session-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--threads",
        type=int,
        default=min(16, os.cpu_count() or 1),
    )
    args = parser.parse_args()
    if args.max_domain_columns < 1:
        raise ValueError("--max-domain-columns must be positive")
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if args.session_limit is not None and args.session_limit < 1:
        raise ValueError("--session-limit must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    started = datetime.now()
    clock = time.perf_counter()
    print(
        f"[domain-compaction] start {started.isoformat(timespec='seconds')}",
        flush=True,
    )
    try:
        result = compact_domains(
            args.hierarchical_dir.resolve(),
            max_columns=args.max_domain_columns,
            memory_limit=args.memory_limit,
            threads=args.threads,
            output_subdir=args.output_subdir,
            session_limit=args.session_limit,
            batch_size=args.batch_size,
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0
    finally:
        print(
            f"[domain-compaction] end "
            f"{datetime.now().isoformat(timespec='seconds')} "
            f"(elapsed {time.perf_counter() - clock:.1f}s)",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())

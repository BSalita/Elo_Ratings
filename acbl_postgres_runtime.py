from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import polars as pl
import psycopg


DATASET_TABLES = {
    "club": "acbl_club_elo_ratings",
    "tournament": "acbl_tournament_elo_ratings",
}

META_TABLE = "acbl_runtime_refresh_meta"


@dataclass
class RefreshResult:
    refreshed: bool
    dataset: str
    row_count: int
    refreshed_at_utc: datetime
    reason: str


def get_database_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if url:
        return url

    host = os.getenv("POSTGRES_HOST", "").strip()
    db = os.getenv("POSTGRES_DB", "").strip()
    user = os.getenv("POSTGRES_USER", "").strip()
    pwd = os.getenv("POSTGRES_PASSWORD", "").strip()
    port = os.getenv("POSTGRES_PORT", "5432").strip() or "5432"
    if not (host and db and user):
        raise ValueError("DATABASE_URL or POSTGRES_HOST/POSTGRES_DB/POSTGRES_USER must be set.")
    return f"postgresql://{user}:{pwd}@{host}:{port}/{db}"


def get_pg_connection() -> psycopg.Connection:
    return psycopg.connect(get_database_url(), autocommit=True)


def postgres_table_for_dataset(dataset_name: str) -> str:
    key = dataset_name.strip().lower()
    if key not in DATASET_TABLES:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return DATASET_TABLES[key]


def read_sql_polars(conn: psycopg.Connection, query: str) -> pl.DataFrame:
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
        cols = [desc.name for desc in cur.description] if cur.description else []
    if not rows:
        return pl.DataFrame(schema=cols)
    return pl.DataFrame(rows, schema=cols, orient="row")


def get_table_columns(conn: psycopg.Connection, table_name: str) -> set[str]:
    sql = f"""
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema='public' AND table_name='{table_name}'
    """
    cols_df = read_sql_polars(conn, sql)
    if cols_df.is_empty():
        return set()
    return set(cols_df["column_name"].to_list())


def ensure_runtime_tables_fresh(
    conn: psycopg.Connection,
    loader_fn: Callable[[str], pl.DataFrame],
    stale_hours: int = 24,
    refresh_enabled: bool = True,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
) -> list[RefreshResult]:
    _ensure_meta_table(conn)
    results: list[RefreshResult] = []
    datasets = ("club", "tournament")
    total_datasets = len(datasets)
    completed_datasets = 0

    for dataset_name in datasets:
        if progress_callback is not None:
            progress_callback(completed_datasets, total_datasets, dataset_name, "start")
        table_name = postgres_table_for_dataset(dataset_name)
        must_refresh, reason = _must_refresh(conn, dataset_name, table_name, stale_hours, refresh_enabled)
        if must_refresh:
            df = loader_fn(dataset_name)
            _replace_table(conn, table_name, df)
            _upsert_meta(conn, dataset_name, table_name, len(df))
            results.append(
                RefreshResult(
                    refreshed=True,
                    dataset=dataset_name,
                    row_count=len(df),
                    refreshed_at_utc=datetime.now(timezone.utc),
                    reason=reason,
                )
            )
        else:
            results.append(
                RefreshResult(
                    refreshed=False,
                    dataset=dataset_name,
                    row_count=_table_row_count(conn, table_name),
                    refreshed_at_utc=datetime.now(timezone.utc),
                    reason=reason,
                )
            )
        completed_datasets += 1
        if progress_callback is not None:
            progress_callback(completed_datasets, total_datasets, dataset_name, "done")
    return results


def _must_refresh(
    conn: psycopg.Connection,
    dataset_name: str,
    table_name: str,
    stale_hours: int,
    refresh_enabled: bool,
) -> tuple[bool, str]:
    if not refresh_enabled:
        return (not _table_exists(conn, table_name), "refresh_disabled")
    if not _table_exists(conn, table_name):
        return (True, "table_missing")

    meta_df = read_sql_polars(
        conn,
        f"SELECT refreshed_at_utc FROM {META_TABLE} WHERE dataset_name = '{dataset_name}'",
    )
    if meta_df.is_empty():
        return (True, "meta_missing")
    refreshed_at = meta_df["refreshed_at_utc"][0]
    if refreshed_at is None:
        return (True, "meta_null")
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max(1, int(stale_hours)))
    if refreshed_at < cutoff:
        return (True, "stale")
    return (False, "fresh")


def _table_exists(conn: psycopg.Connection, table_name: str) -> bool:
    sql = f"""
    SELECT 1
    FROM information_schema.tables
    WHERE table_schema='public' AND table_name='{table_name}'
    LIMIT 1
    """
    return not read_sql_polars(conn, sql).is_empty()


def _table_row_count(conn: psycopg.Connection, table_name: str) -> int:
    if not _table_exists(conn, table_name):
        return 0
    df = read_sql_polars(conn, f"SELECT COUNT(*) AS c FROM {table_name}")
    return int(df["c"][0]) if not df.is_empty() else 0


def _ensure_meta_table(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {META_TABLE} (
                dataset_name TEXT PRIMARY KEY,
                table_name TEXT NOT NULL,
                row_count BIGINT NOT NULL,
                refreshed_at_utc TIMESTAMPTZ NOT NULL
            )
            """
        )


def _upsert_meta(conn: psycopg.Connection, dataset_name: str, table_name: str, row_count: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {META_TABLE} (dataset_name, table_name, row_count, refreshed_at_utc)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (dataset_name) DO UPDATE
            SET table_name = EXCLUDED.table_name,
                row_count = EXCLUDED.row_count,
                refreshed_at_utc = EXCLUDED.refreshed_at_utc
            """,
            (dataset_name, table_name, int(row_count)),
        )


def _replace_table(conn: psycopg.Connection, table_name: str, df: pl.DataFrame) -> None:
    tmp_table = f"{table_name}__tmp_load"
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {tmp_table}")
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        cur.execute(f"CREATE TABLE {tmp_table} ({_pg_columns_ddl(df)})")

    _copy_df_to_table(conn, df, tmp_table)

    with conn.cursor() as cur:
        cur.execute(f"ALTER TABLE {tmp_table} RENAME TO {table_name}")
        if "Date" in df.columns:
            cur.execute(f"CREATE INDEX IF NOT EXISTS {table_name}_date_idx ON {table_name}(\"Date\")")
        if "session_id" in df.columns:
            cur.execute(f"CREATE INDEX IF NOT EXISTS {table_name}_session_idx ON {table_name}(session_id)")


def _copy_df_to_table(conn: psycopg.Connection, df: pl.DataFrame, table_name: str) -> None:
    columns = [f"\"{c}\"" for c in df.columns]
    copy_sql = f"COPY {table_name} ({','.join(columns)}) FROM STDIN WITH (FORMAT CSV, HEADER FALSE, NULL '')"

    tmp_path = Path(tempfile.gettempdir()) / f"{table_name}.csv"
    df.write_csv(tmp_path, include_header=False, null_value="")
    try:
        with conn.cursor() as cur:
            with cur.copy(copy_sql) as cp:
                with open(tmp_path, "r", encoding="utf-8") as f:
                    while True:
                        chunk = f.read(1024 * 1024)
                        if not chunk:
                            break
                        cp.write(chunk)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _pg_columns_ddl(df: pl.DataFrame) -> str:
    return ", ".join(f"\"{name}\" {_polars_type_to_pg(dtype)}" for name, dtype in df.schema.items())


def _polars_type_to_pg(dtype: pl.DataType) -> str:
    if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.UInt8, pl.UInt16, pl.UInt32):
        return "INTEGER"
    if dtype == pl.Int64:
        return "BIGINT"
    if dtype == pl.UInt64:
        return "NUMERIC(20,0)"
    if dtype in (pl.Float32, pl.Float64):
        return "DOUBLE PRECISION"
    if dtype == pl.Boolean:
        return "BOOLEAN"
    if dtype == pl.Date:
        return "DATE"
    if dtype == pl.Datetime:
        return "TIMESTAMPTZ"
    return "TEXT"

-- FFBridge postmortem SQLite schema, version 1.
-- Domain tables are generated from domain_catalog.parquet. No JSON data is used.

PRAGMA journal_mode = DELETE;
PRAGMA synchronous = OFF;
PRAGMA temp_store = FILE;
PRAGMA foreign_keys = ON;

CREATE TABLE schema_info (
    schema_version INTEGER NOT NULL,
    hierarchical_layout_version INTEGER NOT NULL,
    generated_at TEXT NOT NULL
);

CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    revision TEXT NOT NULL,
    Date TEXT NOT NULL,
    year INTEGER NOT NULL,
    series_id TEXT,
    archived_at TEXT NOT NULL,
    board_rows INTEGER NOT NULL,
    result_rows INTEGER NOT NULL
);

CREATE TABLE column_catalog (
    table_name TEXT NOT NULL,
    logical_column TEXT NOT NULL,
    source_table TEXT NOT NULL CHECK (source_table IN ('boards', 'results')),
    domain TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    PRIMARY KEY (table_name, logical_column)
);

CREATE INDEX sessions_date ON sessions(Date);
CREATE INDEX sessions_year_series ON sessions(year, series_id);

-- Generated board domain tables are indexed by (session_id, Board).
-- Generated result domain tables are indexed by
-- (session_id, Board, _result_row_id).

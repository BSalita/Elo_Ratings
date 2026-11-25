use pathlib, polars, fastapi, duckdb. avoid pandas.
use progress indicators (e.g. tqdm) for long running processes.
for non-UI python scripts, use start/end datetime. Use elapsed time for code sections which take more than 30 seconds.
don't worry about legacy compatibility. just fail fast.
don't generate fallbacks without asking first.

## TODO

- **ACBL API server memory (acbl_api_server.py): fixed locally 2026-11-25, not yet on prod.** Cause and measurements: `ACBL_API_MEMORY.md`. It was not a slow leak: every request had DuckDB convert the whole cached frame to Arrow (+13.7 GB and ~10 s per Club request), and the frames were much bigger than budgeted (String columns cost 16 bytes a row; club frame 42 GB, not 13). Now every String column is Categorical and the frame's Arrow export is made once at load. Keep it that way: never `con.register()` a cached Polars frame; register `_arrow_table_for(source_path)`. `estimated_size()` undercounts String columns, so measure real memory (`/proc/<pid>/smaps_rollup`) when sizing caches; `scripts/acbl_api_memory_probe.py` and `scripts/acbl_api_memory_drive.py` do that. Watch `server.ram_used_gb` in API responses on prod after the deploy.

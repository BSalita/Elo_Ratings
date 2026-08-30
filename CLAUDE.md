use pathlib, polars, fastapi, duckdb. avoid pandas.
use progress indicators (e.g. tqdm) for long running processes.
for non-UI python scripts, use start/end datetime. Use elapsed time for code sections which take more than 30 seconds.
don't worry about legacy compatibility. just fail fast.
don't generate fallbacks without asking first.

## TODO

- **ACBL API server memory leak (acbl_api_server.py):** Flipping between 'Club' and 'Tournament' causes server RAM to climb toward the 32 GB limit. Module-level caching of parquet frames and a persistent DuckDB connection were added but did not fully resolve the issue. The app hasn't crashed yet but memory does not stabilize. Needs further investigation — possible causes: glibc arena fragmentation not reclaimed by malloc_trim, Polars .select()/.filter() creating copies rather than zero-copy views, or DuckDB internal buffers retained after query execution. Watch the `server.frame_cache` and `server.ram_used_gb` fields in API responses to monitor.

"""Offline builder for the FFBridge Elo dataset parquets.

Precomputes the full multi-tournament Elo history for each FFBridge API backend
and writes it to ``data/ffbridge/elo_cache/`` in the exact layout the Streamlit
app's boot loader (`load_ffbridge_elo_dataset`) expects. Run this before/at
deploy so the container loads a ready parquet at boot instead of rebuilding the
whole history in RAM (the cause of the OOM/restart loop on Railway).

Usage:
    python build_ffbridge_elo_parquets.py                 # all backends
    python build_ffbridge_elo_parquets.py --api "FFBridge Lancelot API"
    python build_ffbridge_elo_parquets.py --fetch-iv      # include IV (slower)

The app filenames are keyed by (api, tournament_count, fetch_iv); if the live
tournament count changes, the app will rebuild once and re-persist. Rerun this
builder to refresh the committed parquet.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime

# Importing the app module is side-effect free (main() is guarded and
# set_page_config lives inside main()), so we can reuse its pipeline directly.
import streamlit_app_ffbridge_elo_ratings as app


def build_one(api_name: str, fetch_iv: bool) -> int:
    """Build and persist the dataset for a single API backend. Returns row count."""
    api_module = app.API_BACKENDS[api_name]
    api_key = api_name.replace(" ", "_")
    print(f"[builder] {api_name}: fetching tournament list…", flush=True)
    all_tournaments = api_module.fetch_tournament_list(series_id="all", limit=None)
    if not all_tournaments:
        print(f"[builder] {api_name}: no tournaments returned; skipping", flush=True)
        return 0
    print(f"[builder] {api_name}: {len(all_tournaments)} tournaments; computing Elo…", flush=True)
    dataset = app.compute_and_persist_elo_dataset(
        api_module, all_tournaments, api_key, fetch_iv, show_progress=False
    )
    rows = dataset["results_df"].height
    key = app._elo_cache_key(api_key, fetch_iv, len(all_tournaments))
    print(f"[builder] {api_name}: wrote '{key}' ({rows} result rows)", flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api", choices=list(app.API_BACKENDS.keys()), default=None,
        help="Only build this backend (default: all).",
    )
    parser.add_argument(
        "--fetch-iv", action="store_true",
        help="Fetch current IV values (slower). Must match the app's fetch_iv setting.",
    )
    args = parser.parse_args()

    started = datetime.now()
    print(f"[builder] start {started.isoformat(timespec='seconds')} "
          f"(cache dir: {app._FFBRIDGE_ELO_CACHE_DIR})", flush=True)

    api_names = [args.api] if args.api else list(app.API_BACKENDS.keys())
    total_rows = 0
    for api_name in api_names:
        try:
            total_rows += build_one(api_name, args.fetch_iv)
        except Exception as exc:
            print(f"[builder] ERROR building {api_name}: {exc}", flush=True)
            return 1

    elapsed = (datetime.now() - started).total_seconds()
    print(f"[builder] done {datetime.now().isoformat(timespec='seconds')} "
          f"({total_rows} total result rows, {elapsed:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

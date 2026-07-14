"""Offline builder for the FFBridge Elo dataset parquets.

Precomputes the full multi-tournament Elo history for each FFBridge API backend
and writes it to ``FFBRIDGE_CACHE_DIR/elo_cache/`` (the production data mount,
or ``data/ffbridge/elo_cache/`` locally) in the exact layout the Streamlit boot
loader (`load_ffbridge_elo_dataset`) expects.

Where this runs in production
-----------------------------
``elo_ratings_start.ps1`` launches this as a short-lived container before
Streamlit starts. That keeps the memory-heavy history build out of the
long-running app process (the OOM/restart cause).

Staleness gate (--if-stale)
---------------------------
On every container start the builder runs, but with ``--if-stale`` it rebuilds a
backend only when its persisted parquet is missing or older than
``--max-age-hours`` (default 20h). So ordinary restarts/redeploys are a ~1s no-op
(the app just loads the existing parquet), while a scheduled refresh crosses the
threshold and does a full rebuild. When it does rebuild, the tournament list is
force-refreshed from the API so newly published events are discovered.

What a rebuild does (and does NOT) refresh
------------------------------------------
- Tournament LIST: fully re-fetched from the API (``force_refresh=True``) to
  discover new event IDs.
- Event RESULTS (raw data): INCREMENTAL. ``fetch_tournament_results`` reads the
  on-disk cache first (``max_age_hours=None`` — never expires), so only new/
  uncached events hit the API; previously fetched events are served from the
  volume cache and are never re-downloaded.
- Elo ratings: FULL recompute from scratch over the entire history every time
  (``initial_players=None``). Elo is order-dependent, so a newly inserted event
  can shift the whole downstream chain — ratings are replayed, not appended.
- Parquet output: fully regenerated and legacy count-keyed files are pruned.

NUANCE — revised/corrected past events are NOT picked up. Because event results
are cached with no expiry, if FFBridge later re-scores or amends an event we
already fetched, a normal rebuild will not notice it (only genuinely new event
IDs are pulled). To force a full re-fetch of results, delete the raw results
cache on the volume so they are re-downloaded on the next rebuild, e.g.:

    # remove cached per-event results (keeps the elo_cache parquet):
    #   Classic:  $FFBRIDGE_CACHE_DIR/cache/**/results_v3_*.json
    #   Lancelot: $FFBRIDGE_CACHE_DIR/lancelot_cache/**/results_*.json
    # then trigger a rebuild (redeploy, or run this builder without --if-stale).

There is no per-results expiry/force flag today; add one to
``fetch_tournament_results`` if periodic re-fetching of amended events is needed.

Usage:
    python build_ffbridge_elo_parquets.py                 # force build all backends
    python build_ffbridge_elo_parquets.py --if-stale      # build only stale/missing
    python build_ffbridge_elo_parquets.py --api "FFBridge Lancelot API"
    python build_ffbridge_elo_parquets.py --fetch-iv      # include IV (slower)
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from datetime import datetime, timezone
from typing import Optional

# Importing the app module is side-effect free (main() is guarded and
# set_page_config lives inside main()), so we can reuse its pipeline directly.
import streamlit_app_ffbridge_elo_ratings as app


def _preflight() -> int:
    """Validate deploy-critical configuration; return 0 if OK, else nonzero.

    Only enforced when ``STREAMLIT_ENV=production`` so local/offline runs are
    unaffected.
    """
    if os.environ.get("STREAMLIT_ENV", "").strip().lower() != "production":
        return 0

    problems = []
    cache_dir = os.environ.get("FFBRIDGE_CACHE_DIR", "").strip()

    if not cache_dir:
        problems.append(
            "FFBRIDGE_CACHE_DIR is not set — set it to the persistent data mount "
            "(e.g. /data/ffbridge)."
        )

    if cache_dir:
        p = pathlib.Path(cache_dir)
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            problems.append(f"FFBRIDGE_CACHE_DIR={cache_dir} could not be created: {exc}")
        else:
            if not os.access(p, os.W_OK):
                problems.append(f"FFBRIDGE_CACHE_DIR={cache_dir} is not writable.")

    # Classic backend needs a bearer token for live fetches; Lancelot is public.
    if not os.environ.get("FFBRIDGE_BEARER_TOKEN", "").strip():
        print(
            "[preflight] WARN: FFBRIDGE_BEARER_TOKEN not set — the Classic backend "
            "cannot fetch new events (Lancelot is public and unaffected).",
            flush=True,
        )

    if problems:
        for pr in problems:
            print(f"[preflight] FATAL: {pr}", flush=True)
        return 2
    print(
        f"[preflight] OK: production config validated (FFBRIDGE_CACHE_DIR={cache_dir}).",
        flush=True,
    )
    return 0


def _newest_persisted_age_hours(api_key: str, fetch_iv: bool) -> Optional[float]:
    """Delegate to the Streamlit app module (single source of truth)."""
    return app._newest_persisted_age_hours(api_key, fetch_iv)


def _prune_other_fetch_iv(api_key: str, fetch_iv: bool) -> None:
    """Delete orphaned elo_cache sets for this backend with the opposite fetch_iv."""
    other = int(not fetch_iv)
    cache_dir = app._FFBRIDGE_ELO_CACHE_DIR
    iv = int(fetch_iv)
    patterns = (
        f"elo_full_v3_{api_key}_iv_{other}.results.parquet",
        f"elo_full_v3_{api_key}_*_iv_{other}.results.parquet",
    )
    try:
        seen: set[pathlib.Path] = set()
        for pattern in patterns:
            for results_path in cache_dir.glob(pattern):
                if results_path in seen:
                    continue
                seen.add(results_path)
                key = results_path.name[: -len(".results.parquet")]
                if key.endswith(f"_iv_{iv}"):
                    continue
                for path in app._elo_cache_paths(key):
                    path.unlink(missing_ok=True)
                print(f"[builder] pruned orphaned Elo cache '{key}' (opposite fetch_iv)", flush=True)
    except Exception as exc:
        print(f"[builder] orphan prune skipped for {api_key}: {exc}", flush=True)


def build_one(api_name: str, fetch_iv: bool, if_stale: bool = False, max_age_hours: float = 20.0) -> int:
    """Build and persist the dataset for a single API backend.

    Returns the result row count, or -1 if skipped because the cache is fresh.
    """
    api_module = app.API_BACKENDS[api_name]
    api_key = api_name.replace(" ", "_")

    # Clean up orphaned opposite-fetch_iv parquet sets (runs even when the build
    # itself is skipped as fresh, so leftovers get reclaimed on the next deploy).
    _prune_other_fetch_iv(api_key, fetch_iv)

    if if_stale:
        age = _newest_persisted_age_hours(api_key, fetch_iv)
        if age is not None and age < max_age_hours:
            print(f"[builder] {api_name}: cache fresh ({age:.1f}h < {max_age_hours}h); skipping", flush=True)
            return -1
        reason = "missing" if age is None else f"stale ({age:.1f}h)"
        print(f"[builder] {api_name}: cache {reason}; rebuilding", flush=True)

    # Force-refresh the tournament list so newly published events are discovered
    # (the on-disk list cache never expires by design).
    print(f"[builder] {api_name}: refreshing tournament list…", flush=True)
    all_tournaments = api_module.fetch_tournament_list(series_id="all", limit=None, force_refresh=True)
    if not all_tournaments:
        print(f"[builder] {api_name}: no tournaments returned; skipping", flush=True)
        return 0
    print(f"[builder] {api_name}: {len(all_tournaments)} tournaments; computing Elo…", flush=True)
    dataset = app.compute_and_persist_elo_dataset(
        api_module, all_tournaments, api_key, fetch_iv, show_progress=False
    )
    rows = dataset["results_df"].height
    key = app._elo_cache_key(api_key, fetch_iv)
    print(f"[builder] {api_name}: wrote '{key}' ({rows} result rows)", flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api", choices=list(app.API_BACKENDS.keys()), default=None,
        help="Only build this backend (default: all).",
    )
    parser.add_argument(
        "--fetch-iv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch current IV values (slower). Default True to match the Streamlit app.",
    )
    parser.add_argument(
        "--if-stale", action="store_true",
        help="Rebuild a backend only if its persisted parquet is missing or older "
             "than --max-age-hours (default). Otherwise skip (fast no-op).",
    )
    parser.add_argument(
        "--max-age-hours", type=float, default=20.0,
        help="Staleness threshold in hours for --if-stale (default: 20).",
    )
    args = parser.parse_args()

    rc = _preflight()
    if rc != 0:
        return rc

    started = datetime.now()
    print(f"[builder] start {started.isoformat(timespec='seconds')} "
          f"(cache dir: {app._FFBRIDGE_ELO_CACHE_DIR}, if_stale={args.if_stale})", flush=True)

    api_names = [args.api] if args.api else list(app.API_BACKENDS.keys())
    total_rows = 0
    built = 0
    for api_name in api_names:
        try:
            rows = build_one(api_name, args.fetch_iv, if_stale=args.if_stale, max_age_hours=args.max_age_hours)
            if rows >= 0:
                built += 1
                total_rows += rows
        except Exception as exc:
            # Don't fail the container start over a build error; the app has its
            # own in-process rebuild fallback and can serve the last parquet.
            print(f"[builder] ERROR building {api_name}: {exc}", flush=True)
            if not args.if_stale:
                return 1

    elapsed = (datetime.now() - started).total_seconds()
    print(f"[builder] done {datetime.now().isoformat(timespec='seconds')} "
          f"({built} backend(s) built, {total_rows} total result rows, {elapsed:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""One-off calibration for FFBridge field-strength K-dampening.

Replays the real FFBridge Elo pipeline (``process_tournaments_to_elo``) from
the on-disk cache and reports the empirical standard deviation of the per-session
field-mean ratings. That value is the right setting for
``FFBRIDGE_FIELD_STRENGTH_SIGMA`` in ``elo_common.py`` (the z-score denominator
used by :func:`elo_common.field_strength_scale_from_mean`).

Runs offline: both ``fetch_tournament_list`` and ``fetch_tournament_results``
are cache-first (``max_age_hours=None``), so no auth/network is needed as long
as the cache under ``data/ffbridge/cache`` is populated.

Usage:
    python calibrate_ffbridge_field_strength.py [--backend classic|lancelot] [--limit N]
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime


# process_tournaments_to_elo drives a progress bar via st.progress()/st.empty().
# Run outside a Streamlit server, those calls are harmless no-ops but emit
# "missing ScriptRunContext" warnings; silence them so the output stays clean.
import logging  # noqa: E402

logging.getLogger('streamlit').setLevel(logging.ERROR)
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.CRITICAL)

import polars as pl  # noqa: E402

import elo_ffbridge_classic as classic_api  # noqa: E402
import elo_ffbridge_lancelot as lancelot_api  # noqa: E402
from elo_common import (  # noqa: E402
    DEFAULT_ELO,
    FFBRIDGE_FIELD_STRENGTH_SIGMA,
    FIELD_STRENGTH_K_MIN,
    FIELD_STRENGTH_Z_FLOOR,
    field_strength_scale,
)
from streamlit_app_ffbridge_elo_ratings import process_tournaments_to_elo  # noqa: E402

BACKENDS = {'classic': classic_api, 'lancelot': lancelot_api}


def _session_field_means(results_df: pl.DataFrame, col: str) -> pl.Series:
    """One field-mean value per session (tournament_id, series_id)."""
    if results_df.is_empty() or col not in results_df.columns:
        return pl.Series(col, [], dtype=pl.Float64)
    keys = [k for k in ('tournament_id', 'series_id') if k in results_df.columns]
    return (
        results_df
        .group_by(keys)
        .agg(pl.col(col).first().alias(col))
        .get_column(col)
        .drop_nulls()
        .drop_nans()
    )


def _report(label: str, means: pl.Series) -> float | None:
    n = means.len()
    if n < 2:
        print(f"  {label:<22} sessions={n}  (insufficient data)")
        return None
    sigma_pop = float(means.std(ddof=0))
    sigma_sample = float(means.std(ddof=1))
    mean = float(means.mean())
    lo = float(means.min())
    hi = float(means.max())
    print(f"  {label:<22} sessions={n}")
    print(f"      mean={mean:8.2f}  min={lo:8.2f}  max={hi:8.2f}")
    print(f"      sigma(pop, ddof=0)   = {sigma_pop:7.2f}")
    print(f"      sigma(sample, ddof=1)= {sigma_sample:7.2f}")
    return sigma_pop


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=list(BACKENDS), default='classic',
                        help='API backend whose cache to replay (default: classic)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of tournaments (for a quick check)')
    args = parser.parse_args()

    start = datetime.now()
    print(f"[start] {start.isoformat(timespec='seconds')}  backend={args.backend}")
    api_module = BACKENDS[args.backend]

    t0 = time.perf_counter()
    tournaments = api_module.fetch_tournament_list(series_id="all", limit=args.limit)
    print(f"[load] {len(tournaments)} tournaments from cache "
          f"({time.perf_counter() - t0:.1f}s)")

    t0 = time.perf_counter()
    results_df, _players_df, _ratings, stats = process_tournaments_to_elo(
        tournaments, api_module, initial_players=None,
        use_handicap=False, fetch_iv=False, sort_ascending=True,
    )
    elapsed = time.perf_counter() - t0
    if elapsed > 30:
        print(f"[replay] elapsed {elapsed:.1f}s")
    print(f"[replay] rows={results_df.height} "
          f"cached={stats.get('cached')} fetched={stats.get('fetched')} "
          f"missing={len(stats.get('missing_ids', []))}")

    if results_df.is_empty():
        print("ERROR: no results produced; is the cache populated?")
        return 1

    print("\nPer-session field-mean rating distribution:")
    scratch_means = _session_field_means(results_df, 'scratch_field_avg')
    handicap_means = _session_field_means(results_df, 'handicap_field_avg')
    sigma_scratch = _report('scratch_field_avg', scratch_means)
    sigma_handicap = _report('handicap_field_avg', handicap_means)

    candidates = [s for s in (sigma_scratch, sigma_handicap) if s and s > 0]
    print("\n" + "=" * 64)
    print(f"population mean anchor (DEFAULT_ELO) = {DEFAULT_ELO:.0f}")
    print(f"current FFBRIDGE_FIELD_STRENGTH_SIGMA = {FFBRIDGE_FIELD_STRENGTH_SIGMA:.1f}")
    if candidates:
        recommended = round(max(candidates), 1)
        print(f"RECOMMENDED FFBRIDGE_FIELD_STRENGTH_SIGMA = {recommended}")
        print("(uses the larger of the two so dampening stays conservative)")
        floor_mean = DEFAULT_ELO + FIELD_STRENGTH_Z_FLOOR * recommended
        print(f"  -> field mean <= {floor_mean:.0f} hits the K floor "
              f"({FIELD_STRENGTH_K_MIN}); field mean >= {DEFAULT_ELO:.0f} gets full K.")
        for delta in (-recommended, -2 * recommended):
            fm = DEFAULT_ELO + delta
            z = delta / recommended
            print(f"  field mean {fm:7.0f} (z={z:+.1f}) -> K x {field_strength_scale(z):.3f}")
    else:
        print("Could not derive a recommendation (insufficient session variance).")
    print("=" * 64)

    end = datetime.now()
    print(f"[end] {end.isoformat(timespec='seconds')}  "
          f"total {(end - start).total_seconds():.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

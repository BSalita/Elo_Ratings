# FFBridge Elo: the ACBL improvements checked

Written 2026-11-25, after `ELO_PLATINUM_BENCHMARK.md`. Code in `elo/ffbridge_bench/`
(acbl-pipeline venv, which has numba). Results in `D:\bridge-elo-plat\ffbridge`.
Nothing in the FFBridge Elo app or its files was changed.

## Summary

Only one of the ACBL gains carries over: a stronger early K. **K 24 with a 4x
boost for a player's first 10 sessions** (today: K 32, no boost) raises the
club-session rank Spearman on 2026 from 0.525 to 0.533 (+0.008 ±0.001). That
equals FFBridge's full prediction model (0.53). The other ACBL changes either
are already in the FFBridge Elo or have nothing to work with in the data.

| ACBL change | FFBridge |
|---|---|
| one rating per player, not per direction | already so (FFBridge rates players, not seats) |
| joint club + tournament stream | one stream already (Lancelot simultaneous tournaments). The Classic results (2025-07 to 2026-05) are the same kind of events, so they add no history |
| start from masterpoints and platinum points | no equivalent in the data: the IV (indice de valeur) columns are empty, PE is per session, and everyone's history starts in 2023-11. Starting a newcomer at their club's established level **loses** 0.006 |
| lower K | K is already about right: K 16 loses 0.004, 24 and 48 are level |
| longer, stronger early boost | **+0.008** (the gain above) |

## Data and method

- `data/ffbridge/elo_cache/elo_full_v11_FFBridge_Lancelot_API_iv_1.results.parquet`
  (built 2026-10-19): 962,966 pair-session rows, 1,632 simultaneous tournaments,
  52,980 players, 2023-11-06 to 2026-10-19.
- Formula (`streamlit_app_ffbridge_elo_ratings.py`, `elo_common.py`): each player
  moves by K × fs × (pct/100 − E), with E the logistic (400) of the player's own
  rating against the tournament field's mean pair rating. K 32, start 1200, fs a
  weak-field damping. Stored values are display-scaled (×2), which does not
  change ranks.
- Replay: `ffb_bench.py` (numba, under 1 s per variant). Its production replay
  agrees with the stored ratings (Spearman 0.998 over all pair-sessions).
- Ground truth: in each club session (tournament × club, 4 or more pairs),
  Spearman of the pair's pre-session strength (mean of the two players) against
  its session % (national scratch % when present). Averaged over club sessions;
  differences are paired by club session. Dev = 2025 (20,385 club sessions),
  test = 2026 (16,163). The board-level rank eval
  (`ffbridge_prediction_rank_eval.py`) gives 0.511 for the Elo line and 0.529 for
  the model on its 2026 test set. This measure is per session, not per board, so
  the two are close but not identical.

## Results

Round 1 (`ffb_bench.py`, against the replay of today's formula):

| variant | 2025 | 2026 |
|---|---|---|
| today's formula (replay) | 0.5151 | 0.5251 |
| stored ratings | 0.5155 | 0.5250 |
| expectation from the partnership mean instead of each player | +0.0005 ±0.0009 | +0.0007 ±0.0010 |
| same, field mean excluding the pair itself | +0.0005 | +0.0008 |
| no weak-field damping | +0.0000 | +0.0000 |
| K 16 / 24 / 48 | −0.004 / −0.001 / 0.000 | −0.004 / −0.001 / −0.001 |
| newcomer starts at the club's established mean | −0.0040 | −0.0058 |
| logistic scale 200 / 800 | −0.002 / −0.004 | −0.002 / −0.004 |
| K 16, 3x boost for 20 sessions | +0.0052 | +0.0056 |

Grid (`ffb_tune.py`): K 20 to 40, boost 2 to 6, for 10 to 40 sessions, with or
without the partnership expectation, chosen on 2025:

| chosen on 2025 | 2025 | 2026 |
|---|---|---|
| K 24, 4x for the first 10 sessions | 0.5223 (+0.0073 ±0.0009) | **0.5330 (+0.0079 ±0.0009)** |
| next best on 2025 (K 20 6x/10; K 32 3x/10 partnership; K 24 4x/20 partnership) | +0.0070 to +0.0071 | +0.0079 to +0.0086 |

The top settings are within noise of each other. Any boost of about 3x to 6x over
the first 10 to 20 sessions gives the same +0.008.

## What could still help (not done)

- **IV (indice de valeur) or série as a starting rating.** This is the FFB
  equivalent of what masterpoints and platinum points did for ACBL. The
  `fetch_iv` columns in the cache are empty, so the values would have to be
  fetched from the FFB API.
- **More history before 2023-11.** Everyone starts at 1200 on that date, so the
  first months are a warm-up for the whole population.

## Adopted (2026-11-25, live on prod)

- Code: `elo_common.ffbridge_k_factor` (K 24, times 4 while a player has fewer
  than 10 rated sessions), used by both Elo tracks in
  `streamlit_app_ffbridge_elo_ratings.py`. The dataset key went from
  `elo_full_v11` to `elo_full_v12`, so the builder replays the whole history.
  Elo_Ratings `81ece19`, pushed. The elo tests pass (309).
- Local rebuild: `data/ffbridge/elo_cache/elo_full_v12_*` (981,394 rows to
  2026-11-24; the v11 files are kept beside it).
- Prod: `deploy_elo_ratings.ps1` on X1 at 15:22 (containers at `81ece19`, all
  services 200). The builder wrote `elo_full_v12_*` at 15:21 (982,505 rows).
  Checked with `ffbridge_bench/check_rebuilt.py`: the stored ratings follow the
  new formula (rank correlation 0.9989 with its replay, 0.981 with the old one).
  2026 club sessions on prod's data: 0.5275 with the old formula, **0.5346** with
  the new one (+0.0071 ±0.0009).
- Not done: the FFBridge prediction data and models (`ffbridge-pipeline` 5a/5c)
  still use the old ratings until they are rerun.

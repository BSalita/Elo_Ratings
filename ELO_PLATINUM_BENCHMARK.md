# ACBL Elo: platinum ground truth and a better formula

Written 2026-11-25. Code in `elo/platinum_bench/` (run with the acbl-pipeline venv,
which has numba). Working files and every result table in `D:\bridge-elo-plat`;
copies of the result tables in `elo/platinum_bench/results/`. Nothing in the
pipeline, the Elo files on E: or prod was changed.

## Summary

- **What we measure against.** The ground truth is predictive: before a session,
  rank the pairs of each section and direction by rating, then compare with the
  session's actual result (Spearman). This is the rank eval of
  `acbl_prediction_rank_eval.py`. This study adds two platinum ground truths:
  the same ranking inside **platinum fields** (NABC+ platinum pairs events, the
  strongest fields in the data), and **ratings at a date T against platinum results
  after T**.
- **Do today's ratings match?** Only partly. In platinum fields today's tournament
  Elo ranks pairs worse (0.23 to 0.29) than in ordinary tournament fields (0.34).
  **Platinum points won to date are the best single predictor of later platinum
  results**, ahead of our tournament Elo. You were right: platinum points are an
  under-used resource.
- **The published (shrunk) Elo is the worst of all.** Shrinking toward 1498.5
  with 50 prior sessions costs 0.02 Spearman overall and 0.07 in platinum fields.
  It pulls exactly the strong players with few tournament sessions to average.
- **A better formula** (replayed over the full history, tuned on development
  years only, tested on 2026 and on 2025-2026 platinum fields):
  1. one rating per player, fed by **club and tournament results together**
     (production keeps four separate ratings: club and tournament, each split into
     North-South and East-West);
  2. a new player **starts from masterpoints and platinum points** instead of 1500;
  3. **K 6 instead of 24**, with a 6x provisional boost for the first 200 boards
     (today 1.5x for 100), so early movement stays about the same and later
     movement is 4x smaller.

  | Spearman (fields) | club 2026 (114,014) | tournament 2026 (12,063) | platinum 2016-24 (2,982) | platinum 2025-26 (1,200) |
  |---|---|---|---|---|
  | today's formula | 0.438 | 0.340 | 0.229 | 0.286 |
  | **candidate** | **0.478** (+0.039 ±0.003) | **0.430** (+0.090 ±0.011) | **0.295** (+0.066 ±0.016) | **0.344** (+0.059 ±0.022) |
  | full prediction model (thousands of features) | 0.462 | 0.376 | | |

  Ranges are 95% (paired by field). The candidate Elo line alone now beats the
  full prediction model built on today's Elo, on both sides.
- **K weights by masterpoint colour do not help** (2x for platinum: no change;
  3x: worse). A slow pull toward a masterpoint anchor does not help either.
- **Scale.** The candidate's player SD is 44 (published today: 35) and 100 points
  are worth 13.6 board-% points (today 11.5; the 400-point logistic standard is
  about 14). The compression came mostly from the published shrinkage. A display
  rescale is cosmetic and does not change Spearman.

## 1. Data and method

- Board results: `acbl_club_elo_ratings.parquet` (73.9M rows, 2019-01 to
  2026-11) and `acbl_tournament_elo_ratings.parquet` (19.0M rows, 2016-11 to
  2026-10), with `section_id` from the augmented board results.
- Platinum fields: tournament sessions whose `mp_color` is Platinum. There are 457
  sessions in 122 events, 2016-11 to 2026-07: NABC+ pairs events such as the Blue
  Ribbon, Platinum Pairs, Silodor, Von Zedtwitz and Wernher.
- Awards: `acbl_tournament_player_awards.parquet` (file of 2026-09-03). It holds
  311 platinum events from 2013 to 2026-07, **including team events** (Spingold,
  Vanderbilt, Reisinger, the Swiss teams) that have no board results. Every award
  gets a date at which it is known: the session date when the session is in the
  board results, otherwise the NABC's last session date (or its approximate end
  before 2016), or the end of the month in the session id for regional events.
  Features at a session use only awards known on an earlier date.
- Evaluation (`evalr.py`): the same as the rank eval. Each board's predictor is the
  NS pair's strength minus the EW pair's (the mean of the two players), averaged
  over each pair's boards (8 or more). Spearman with the actual session % within
  each session, section and direction with 4 or more pairs, then averaged over
  fields. A field where a predictor is constant counts as 0, not as missing.
  Differences are paired by field.
- Check against the rank eval: the replay of today's formula gives club 2026
  0.4384 (rank eval 0.4368) and tournament 2026 0.3397 (rank eval 0.3385; row
  sets differ slightly). The replay kernel (`kernel.py`) reproduces production's
  stored tournament player ratings exactly (mean difference 0.00002).
- Development and test: K and boost were chosen on club 2025, tournament 2025 and
  platinum 2016-2024. Starting-rating weights were fitted on ratings before 2025.
  Blend weights were fitted on 2023-2024 or on platinum 2016-2024. The test sets
  are club and tournament 2026 and platinum 2025-2026.

## 2. Ground truth 1: ranking in platinum fields (today's ratings)

`exp_tournament_features.py`. Stored production Elo; features known before the session.

| predictor | tournament 2026 | platinum 2016-24 | platinum 2025-26 |
|---|---|---|---|
| tournament Elo (stored) | 0.332 | 0.229 | 0.286 |
| published Elo (shrunk, 50 sessions to 1498.5) | 0.311 | 0.160 | 0.220 |
| log masterpoints | 0.298 | 0.144 | 0.132 |
| log gold points to date | 0.306 | 0.135 | 0.156 |
| log platinum points to date | 0.163 | **0.291** | **0.312** |
| platinum pairs points per platinum session | 0.147 | 0.253 | 0.324 |
| mean % in platinum sessions to date | 0.011 | 0.233 | 0.303 |
| Elo + log masterpoints (fit on 2023-24) | **0.374** | 0.222 | 0.256 |
| Elo + platinum features (fit on platinum 2016-24) | 0.209 | 0.315 | **0.347** |

- In platinum fields, platinum points to date beat Elo, and masterpoints are worse
  than useless. The masterpoint total measures years of play. Among NABC+ entrants
  everyone has thousands of masterpoints, and the larger totals belong to older
  players.
- In ordinary fields it is the other way round: masterpoints add +0.04 to Elo, and
  platinum points add nothing (most players have none).
- So a single fixed blend cannot serve both kinds of field. The candidate formula
  below gets most of both gains inside the Elo itself.

## 3. Ground truth 2: rating before T against platinum results after T

`exp_future_platinum.py`, `exp_confirm.py`. Players with 6 or more platinum pairs
sessions after T (to 2026-07). Targets: platinum points won per platinum pairs
session after T, and mean session % in those sessions.

| Spearman | T=2022, points/session | T=2022, mean % | T=2024, points/session | T=2024, mean % |
|---|---|---|---|---|
| players | 2,754 | | 2,264 | |
| tournament Elo (stored) | 0.372 | 0.379 | 0.383 | 0.409 |
| published Elo | 0.241 | 0.240 | 0.251 | 0.278 |
| club Elo | 0.192 | 0.221 | 0.187 | 0.210 |
| masterpoints | 0.353 | 0.347 | 0.293 | 0.316 |
| platinum points to date | **0.450** | 0.430 | **0.472** | 0.454 |
| candidate Elo | 0.446 | **0.457** | 0.450 | **0.465** |

- Platinum points to date are the best single predictor of later platinum
  points. The candidate equals them on points and beats them on session %.
- Today's tournament Elo is 0.07 to 0.09 behind, and the published Elo 0.20 behind.
- Caveat: platinum points depend on attendance. The targets here are per session,
  which removes attendance from the target but not from the predictor.

Agreement among established tournament players (30+ sessions, 33,868 players):
Spearman of total platinum points with production tournament Elo 0.439, with
the candidate 0.475. Of the top 100 platinum earners, 65% are in production's
top 1% and 77% in the candidate's.

## 4. What helps and what doesn't

Ablation ladder (`exp_confirm.py`), each step added to the previous one, against
today's formula:

| step | club 2026 | tournament 2026 | platinum 2016-24 | platinum 2025-26 |
|---|---|---|---|---|
| today's formula, streams separate | 0.438 | 0.340 | 0.229 | 0.286 |
| joint club + tournament stream | +0.001 | +0.039 | -0.013 | -0.016 |
| + one rating per player (not per direction) | +0.010 | +0.056 | +0.020 | +0.006 |
| + start from masterpoints and platinum points | +0.015 | +0.063 | +0.031 | +0.008 |
| + K 6, boost 6x for 200 boards (= candidate) | **+0.039** | **+0.090** | **+0.066** | **+0.059** |
| candidate without the informed start | +0.030 | +0.074 | +0.050 | +0.052 |

Starting weights (fitted on pre-2025 ratings): start = 1455 + 10.2 × ln(1 +
masterpoints) + 12.3 × ln(1 + platinum points). For example, 0 masterpoints
starts at 1455, 1,000 at 1525, and 5,000 with 500 platinum points at 1618.

Tested and rejected, on the tournament stream (`exp_tournament_kernel.py`,
`exp_tournament_round2.py`):

- **K by colour** (candidate 2b): platinum ×2 changes nothing (+0.001); platinum ×3
  loses 0.02; platinum ×2 with gold ×1.25 loses 0.01. Platinum sessions are under
  1% of tournament sessions, and up-weighting them adds noise rather than signal.
- **Higher K** (32, 48) loses. **elo_scale 200** loses 0.03. The kernel already has
  a 400-point logistic; the scale is not the problem.
- **Mean reversion toward a moving anchor** (masterpoints, platinum and gold to
  date) at 1 to 10% a session: no gain overall; at 10% it loses in platinum fields.
- **Post-hoc shrinkage toward an informed anchor** instead of 1498.5: better than
  today's shrinkage, but no better than no shrinkage.
- **Tournament-only candidate** (same settings, no club results): tournament 2026
  0.412, platinum 2025-26 0.349. Platinum is the same within noise; the joint
  rating is better overall.
- **Blending the candidate with platinum points** for platinum fields: +0.02 on
  the 2016-24 fit, but +0.002 on the 2025-26 test. The candidate already carries
  that signal; a separate platinum-field rating is not worth having.

K and boost grid (development sets): the results are flat from K 6 to 10 with
boost 4 to 6. K 6 / 6x / 200 boards was best on club 2025, and best or tied on
tournament 2025 plus platinum 2016-24. Club K 4 was not tested.

## 5. Adoption status (2026-11-25, uncommitted)

The user approved a local rebuild. Done:

1. `mlBridge/mlBridgeAugmentLib.py`: `_elo_player_kernel`, `_player_elo_frame` and
   `compute_player_matchpoint_elo_ratings` take three new options: `boost`
   (default 1.5), `merged` (one rating per player, default False) and
   `start_rating_cols` (per-seat starting ratings, default none). With the
   defaults the output equals the stored production ratings
   (`platinum_bench/verify_library.py`). With the new options it equals
   `kernel.py` exactly. Other callers are unchanged. The Elo tests in `elo/`
   pass (13).
2. `acbl-pipeline/acbl_elo_ratings_create.py` now always reads both board-results
   files. It computes the player Elo on the joint stream with the new settings,
   keeps pair Elo per side as before, and writes the same files with the same
   shapes. `--club` / `--tournament` only choose which files are written.
   Starting ratings use `acbl-pipeline/acbl_platinum_to_date.py` (award dating as
   in section 1). The run takes 16 minutes on dev (it used to be about 8 minutes
   for both sides).
3. Rebuilt into **`E:\bridge\data\acbl\elo_candidate_20261125\`** (the current
   files in `E:\bridge\data\acbl` are untouched). Scored by
   `platinum_bench/score_rebuilt.py` from the stored `Elo_R_*_EventStart`:

   | | club 2025 | club 2026 | tournament 2025 | tournament 2026 | platinum 2016-24 | platinum 2025-26 |
   |---|---|---|---|---|---|---|
   | current files | 0.4316 | 0.4385 | 0.3312 | 0.3397 | 0.2290 | 0.2857 |
   | rebuilt | 0.4763 | **0.4798** (+0.041) | 0.4218 | **0.4302** (+0.091) | 0.2981 | **0.3493** (+0.064) |

   Unknown players (null id) have NaN ratings in both, as before (6% of club
   seat values). A scorer must turn NaN into a value, or those fields drop out.

4. **Swapped in on 2026-11-25** (the user said yes). The rebuilt files are now the
   `acbl_{club,tournament}_{elo_ratings,player_elo_ratings,pair_elo_ratings}.parquet`
   and `_elo_shrinkage.json` in `E:\bridge\data\acbl`, and score as in the table
   above. The previous files are in
   `E:\bridge\data\acbl\elo_backup_20261125_before_new_formula\`; to undo, move
   them back. `elo_candidate_20261125\` holds only the build log.

Not done (each needs the user):

- Steps 5a/5b/5c (prediction data and retrain) and prod. The prediction data
  and models on E: still carry the old Elo until 5a is rerun.
- The published shrinkage: `acbl_api_server.py` / the Streamlit app still default
  to 50 prior sessions. With the new formula it should be 0 or a few sessions.
  Not changed, since it is the acbl-elo site's behaviour.
- The starting rating needs the awards sidecar (`elo/acbl_awards.py`). Awards
  after its date (the file is from 2026-09-03) are missing until it is refreshed.
- Pair ratings (`_elo_pair_kernel`) were not studied; the same K change probably
  helps them.
- Commits: mlBridge `d406d3b` and acbl-pipeline `c4c9037` on master, 2026-11-25,
  not pushed. This write-up and `elo/platinum_bench/` are not committed.

## Files

- `platinum_bench/prep.py`: row tables with section ids. `features.py`: award
  dates and pre-session platinum, gold and masterpoint features.
- `kernel.py`: experimental player-Elo replay (one rating per player, start
  ratings, K multipliers, mean reversion). `evalr.py`: rank evaluation.
- `exp_tournament_features.py`, `exp_tournament_kernel.py`,
  `exp_tournament_round2.py`, `exp_tournament_tune.py`, `exp_club_joint.py`,
  `exp_club_tune.py`, `exp_final.py`, `exp_future_platinum.py`, `exp_confirm.py`:
  the runs behind the tables above. Each takes 1 to 15 minutes on dev
  (single-thread numba kernel; joint stream about 60 GB of memory).

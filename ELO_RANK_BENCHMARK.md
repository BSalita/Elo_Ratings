# Elo Rank Benchmark (proposal)

Status: proposal, not implemented. Written 2026-10-18 from the prediction-model
rank evaluation in `acbl-pipeline` and `ffbridge-pipeline`.

## Question

Does a change to the Elo formula (K-factor, scaling, starting rating, field
strength, handicap, skill anchoring) make ratings better at predicting who
finishes ahead of whom in games the ratings have not seen?

## Why a new benchmark

The prediction-model work already measures this, as a baseline:

- Rating as of event start only (no leakage).
- Held-out 2026 games (training data ends 2025-12-31).
- Pairs ranked inside the field they actually compete in: ACBL section and
  direction, FFBridge club session.
- Compared against the full prediction model, which acts as a ceiling.

Existing Elo-repo checks cover parts of this:

| Script | Measures | Missing |
|---|---|---|
| `acbl_ranking_benchmark.py` | Pairwise concordance of pre-session Elo within session+direction; sweeps raw Elo vs skill-anchored blends | No ceiling reference, no calibration, ACBL only |
| `clone_elo.py` | Pair session % MAE of clone ratings vs Elo | Not a ranking metric |

The proposal is to treat the rank evaluation's Elo line as the shared
acceptance test for Elo changes on both ACBL and FFBridge.

## Method (as implemented in the prediction pipelines)

1. Elo gap per board = mean pre-session Elo of the NS pair minus mean
   pre-session Elo of the EW pair.
2. Fit `Pct_NS = intercept + slope * gap` by least squares on training rows
   (ACBL: 2025-01-01 onward).
3. Predict every 2026 test board, average per pair per session.
4. Per field, take Spearman correlation of predicted vs actual pair session
   %. ACBL fields are section+direction (matchpoints are scored that way).
   FFBridge fields are club sessions with NS and EW ranked together, as
   `Club_Rank` is. Pairs need >= 8 boards, fields need >= 4 pairs.
5. Report mean Spearman, pair session % MAE, and the same for the prediction
   model.

Code:

- ACBL: `acbl-pipeline/acbl_prediction_rank_eval.py` (`seat_elo_gap`,
  `fit_elo_line`, `section_rank_quality`). Standalone:
  `python acbl_prediction_rank_eval.py --side club --skip-joint` (~1 min).
  Writes `E:\bridge\data\acbl\acbl_club_prediction_rank_metrics.json`.
- FFBridge: `ffbridge-pipeline/ffbridge_prediction_train.py`
  (`fit_elo_line`, `pair_session_rank_quality`, `evaluate_pct_ns_ranking`).
  Runs only inside training (stage 5c, ~5 min). Writes the `ranking` block of
  `E:\bridge\data\ffbridge\ffbridge_prediction_metrics.json`.
- ACBL tournaments cannot be ranked: the tournament prediction data has no
  session or section id.

## Current results (2026-10-18)

| | ACBL club | FFBridge club |
|---|---|---|
| Fields | 92,573 section/directions | 10,630 club sessions |
| Elo line Spearman | 0.433 | 0.512 |
| Prediction model Spearman | 0.456 | 0.532 |
| Elo line pair session % MAE | 0.0566 | 0.0521 |
| Prediction model pair session % MAE | 0.0556 | 0.0506 |
| Constant 50% pair session % MAE | 0.0637 | 0.0625 |

A model with thousands of hand, bidding and history features beats the bare
Elo gap by about 0.02 Spearman. Elo already carries most of the predictable
ranking signal.

## Calibration finding

| | ACBL club | FFBridge club |
|---|---|---|
| Board % per 100 Elo gap | 11.3 points | 1.06 points |
| Elo gap standard deviation | 48 | 606 |
| Board % per 1 SD of gap | 5.4 points | 6.4 points |
| Pair Elo standard deviation | 38 | 434 |

Both ratings predict about equally per standard deviation, but the scales
differ about 12x. Standard Elo (400-point logistic) expects a 100-point gap to
be worth about 14 points of score. ACBL's 11.3 is close to that; FFBridge's
1.06 means its ratings are spread far wider than their predictive value
supports. ACBL pair ratings remain compressed (SD 38), the issue raised in
`ELO_SCALING_PROPOSAL.md`.

## Proposed use

1. Before changing an Elo formula, record the Elo line Spearman and MAE for
   the current ratings.
2. Rebuild ratings with the change, rebuild the prediction test data
   (ACBL stage 5a/5b, FFBridge stage 5a), rerun the evaluation.
3. Accept the change only if Elo line Spearman does not drop. Report the
   slope per 100 Elo so scale changes are visible.
4. A linear rescale (display-only scaling) changes the slope but not the
   Spearman; use this to confirm a change is cosmetic.

## Work needed if adopted

- FFBridge: expose the ranking evaluation as a standalone script (like the
  ACBL one) so it runs without retraining.
- Optionally add the pairwise concordance from `acbl_ranking_benchmark.py` to
  the same report, so the two metrics are read side by side.
- Optionally score candidate ratings directly (replace `seat_elo_gap` input
  columns) instead of rebuilding the prediction test parquets.

# Phase 2c results (final model: 165 original features)

Fit: steps 1-29. Validation: 30-34. Test: 35-49, evaluated once. Models are NOT refit on train+val.
Positive class = illicit. Pre-shutdown = test steps 35-42, post-shutdown = 43-49.
Edges connecting different time steps: 0.

## Model comparison (threshold = each model's validation-F1 optimum)

| model | period | n | illicit | prevalence | pr_auc | roc_auc | precision | recall | f1 | val_pr_auc | thr |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LightGBM (165 original) | overall (35-49) | 16670 | 1083 | 0.0650 | 0.7917 | 0.9275 | 0.9154 | 0.7193 | 0.8056 | 0.9922 | 0.7880 |
| LightGBM (165 original) | pre-shutdown (35-42) | 9983 | 914 | 0.0916 | 0.9146 | 0.9703 | 0.9521 | 0.8490 | 0.8976 | 0.9922 | 0.7880 |
| LightGBM (165 original) | post-shutdown (43-49) | 6687 | 169 | 0.0253 | 0.0392 | 0.6810 | 0.0833 | 0.0178 | 0.0293 | 0.9922 | 0.7880 |

Final model: **LightGBM (165 original)**, fixed by decision (see README). NOTE: the decision cited test numbers from Phase 2b, so test is no longer a pure holdout for this choice.

## Cost-based thresholds
**ASSUMPTION: false alert costs 1, missed illicit node costs C. C values are assumed, not measured.** Thresholds are chosen on validation; C=10 is the default.

| C (assumed) | threshold | val_cost |
|---|---|---|
| 5 | 0.4378 | 90.0000 |
| 10 | 0.2582 | 144.0000 |
| 20 | 0.2582 | 244.0000 |

| C (assumed) | default | val_threshold | period | precision | recall | f1 | FP | FN | total_cost | cost_flag_nothing | cost_flag_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 |  | 0.4378 | overall (35-49) | 0.7538 | 0.7322 | 0.7429 | 259 | 290 | 1709.0000 | 5415 | 15587 |
| 5 |  | 0.4378 | pre-shutdown (35-42) | 0.8587 | 0.8643 | 0.8615 | 130 | 124 | 750.0000 | 4570 | 9069 |
| 5 |  | 0.4378 | post-shutdown (43-49) | 0.0227 | 0.0178 | 0.0199 | 129 | 166 | 959.0000 | 845 | 6518 |
| 10 | yes | 0.2582 | overall (35-49) | 0.6537 | 0.7442 | 0.6960 | 427 | 277 | 3197.0000 | 10830 | 15587 |
| 10 | yes | 0.2582 | pre-shutdown (35-42) | 0.7838 | 0.8764 | 0.8275 | 221 | 113 | 1351.0000 | 9140 | 9069 |
| 10 | yes | 0.2582 | post-shutdown (43-49) | 0.0237 | 0.0296 | 0.0263 | 206 | 164 | 1846.0000 | 1690 | 6518 |
| 20 |  | 0.2582 | overall (35-49) | 0.6537 | 0.7442 | 0.6960 | 427 | 277 | 5967.0000 | 21660 | 15587 |
| 20 |  | 0.2582 | pre-shutdown (35-42) | 0.7838 | 0.8764 | 0.8275 | 221 | 113 | 2481.0000 | 18280 | 9069 |
| 20 |  | 0.2582 | post-shutdown (43-49) | 0.0237 | 0.0296 | 0.0263 | 206 | 164 | 3486.0000 | 3380 | 6518 |

## Calibration (isotonic, fitted on validation predictions)

Prevalence differs between periods, so calibration is expected to drift on test. The validation row is in-sample for the isotonic fit and therefore optimistic. Raw scores come from a model trained with scale_pos_weight and are not probabilities.

| set | n | prevalence | brier_raw | brier_isotonic | ece_raw | ece_isotonic |
|---|---|---|---|---|---|---|
| validation (30-34; calibrator fitted here, in-sample) | 3513 | 0.1682 | 0.0087 | 0.0074 | 0.0066 | 0.0000 |
| test overall (35-49) | 16670 | 0.0650 | 0.0263 | 0.0224 | 0.0289 | 0.0197 |
| test pre-shutdown (35-42) | 9983 | 0.0916 | 0.0199 | 0.0168 | 0.0210 | 0.0126 |
| test post-shutdown (43-49) | 6687 | 0.0253 | 0.0357 | 0.0306 | 0.0407 | 0.0303 |

## Alert budget (test, per step)

Threshold column = default cost threshold (C=10). maxR@k = best possible recall@k given the number of illicit nodes in the step. Only labeled nodes are ranked.

| time_step | n | illicit | recall@thr | P@10 | R@10 | maxR@10 | P@25 | R@25 | maxR@25 |
|---|---|---|---|---|---|---|---|---|---|
| 35 | 1341 | 182 | 0.984 | 1.000 | 0.055 | 0.055 | 1.000 | 0.137 | 0.137 |
| 36 | 1708 | 33 | 1.000 | 1.000 | 0.303 | 0.303 | 1.000 | 0.758 | 0.758 |
| 37 | 498 | 40 | 0.675 | 1.000 | 0.250 | 0.250 | 1.000 | 0.625 | 0.625 |
| 38 | 756 | 111 | 0.919 | 1.000 | 0.090 | 0.090 | 1.000 | 0.225 | 0.225 |
| 39 | 1183 | 81 | 0.938 | 1.000 | 0.123 | 0.123 | 1.000 | 0.309 | 0.309 |
| 40 | 1211 | 112 | 0.696 | 1.000 | 0.089 | 0.089 | 1.000 | 0.223 | 0.223 |
| 41 | 1132 | 116 | 0.948 | 1.000 | 0.086 | 0.086 | 1.000 | 0.216 | 0.216 |
| 42 | 2154 | 239 | 0.820 | 1.000 | 0.042 | 0.042 | 1.000 | 0.105 | 0.105 |
| 43 | 1370 | 24 | 0.000 | 0.000 | 0.000 | 0.417 | 0.000 | 0.000 | 1.000 |
| 44 | 1591 | 24 | 0.083 | 0.100 | 0.042 | 0.417 | 0.040 | 0.042 | 1.000 |
| 45 | 1221 | 5 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 |
| 46 | 712 | 2 | 0.500 | 0.100 | 0.500 | 1.000 | 0.040 | 0.500 | 1.000 |
| 47 | 846 | 22 | 0.000 | 0.000 | 0.000 | 0.455 | 0.000 | 0.000 | 1.000 |
| 48 | 471 | 36 | 0.028 | 0.100 | 0.028 | 0.278 | 0.120 | 0.083 | 0.694 |
| 49 | 476 | 56 | 0.018 | 0.100 | 0.018 | 0.179 | 0.040 | 0.018 | 0.446 |

| period | steps | recall@thr | P@10 | R@10 | maxR@10 | P@25 | R@25 | maxR@25 |
|---|---|---|---|---|---|---|---|---|
| pre-shutdown avg | 8 | 0.873 | 1.000 | 0.130 | 0.130 | 1.000 | 0.325 | 0.325 |
| post-shutdown avg | 7 | 0.090 | 0.057 | 0.084 | 0.535 | 0.034 | 0.092 | 0.877 |


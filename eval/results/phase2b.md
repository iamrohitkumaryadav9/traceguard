# Phase 2b results

Fit: steps 1-29. Validation: 30-34. Test: 35-49, evaluated once. Models are NOT refit on train+val.
Positive class = illicit. Pre-shutdown = test steps 35-42, post-shutdown = 43-49.
Edges connecting different time steps: 0.

## Model comparison (threshold = each model's validation-F1 optimum)

| model | period | n | illicit | prevalence | pr_auc | roc_auc | precision | recall | f1 | val_pr_auc | thr |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LightGBM (165 original) | overall (35-49) | 16670 | 1083 | 0.0650 | 0.7917 | 0.9275 | 0.9154 | 0.7193 | 0.8056 | 0.9922 | 0.7880 |
| LightGBM (165 original) | pre-shutdown (35-42) | 9983 | 914 | 0.0916 | 0.9146 | 0.9703 | 0.9521 | 0.8490 | 0.8976 | 0.9922 | 0.7880 |
| LightGBM (165 original) | post-shutdown (43-49) | 6687 | 169 | 0.0253 | 0.0392 | 0.6810 | 0.0833 | 0.0178 | 0.0293 | 0.9922 | 0.7880 |
| LightGBM (165 + structural) | overall (35-49) | 16670 | 1083 | 0.0650 | 0.7956 | 0.9295 | 0.8420 | 0.7331 | 0.7838 | 0.9930 | 0.4917 |
| LightGBM (165 + structural) | pre-shutdown (35-42) | 9983 | 914 | 0.0916 | 0.9164 | 0.9718 | 0.9069 | 0.8632 | 0.8845 | 0.9930 | 0.4917 |
| LightGBM (165 + structural) | post-shutdown (43-49) | 6687 | 169 | 0.0253 | 0.0446 | 0.6939 | 0.0685 | 0.0296 | 0.0413 | 0.9930 | 0.4917 |

Final model (chosen on validation PR-AUC only): **LightGBM (165 + structural)**.

## Early-stopping diagnostic (local-only LightGBM from 2a)

```
{
  "local_ap_at_6": 0.9813507913739437,
  "local_ap_min_7_to_56": 0.9102174809862549,
  "local_ap_max_7_to_56": 0.9714128282613759,
  "local_best_iter_300": 257,
  "local_best_ap_300": 0.986808122339319,
  "first_iter_beating_6": 95
}
```

## Cost-based thresholds
**ASSUMPTION: false alert costs 1, missed illicit node costs C. C values are assumed, not measured.** Thresholds are chosen on validation; C=10 is the default.

| C (assumed) | threshold | val_cost |
|---|---|---|
| 5 | 0.3798 | 81.0000 |
| 10 | 0.3798 | 141.0000 |
| 20 | 0.1994 | 242.0000 |

| C (assumed) | default | val_threshold | period | precision | recall | f1 | FP | FN | total_cost | cost_flag_nothing | cost_flag_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 |  | 0.3798 | overall (35-49) | 0.7880 | 0.7378 | 0.7620 | 215 | 284 | 1635.0000 | 5415 | 15587 |
| 5 |  | 0.3798 | pre-shutdown (35-42) | 0.8706 | 0.8687 | 0.8697 | 118 | 120 | 718.0000 | 4570 | 9069 |
| 5 |  | 0.3798 | post-shutdown (43-49) | 0.0490 | 0.0296 | 0.0369 | 97 | 164 | 917.0000 | 845 | 6518 |
| 10 | yes | 0.3798 | overall (35-49) | 0.7880 | 0.7378 | 0.7620 | 215 | 284 | 3055.0000 | 10830 | 15587 |
| 10 | yes | 0.3798 | pre-shutdown (35-42) | 0.8706 | 0.8687 | 0.8697 | 118 | 120 | 1318.0000 | 9140 | 9069 |
| 10 | yes | 0.3798 | post-shutdown (43-49) | 0.0490 | 0.0296 | 0.0369 | 97 | 164 | 1737.0000 | 1690 | 6518 |
| 20 |  | 0.1994 | overall (35-49) | 0.6901 | 0.7442 | 0.7161 | 362 | 277 | 5902.0000 | 21660 | 15587 |
| 20 |  | 0.1994 | pre-shutdown (35-42) | 0.7958 | 0.8742 | 0.8332 | 205 | 115 | 2505.0000 | 18280 | 9069 |
| 20 |  | 0.1994 | post-shutdown (43-49) | 0.0427 | 0.0414 | 0.0420 | 157 | 162 | 3397.0000 | 3380 | 6518 |

## Calibration (isotonic, fitted on validation predictions)

Prevalence differs between periods, so calibration is expected to drift on test. The validation row is in-sample for the isotonic fit and therefore optimistic. Raw scores come from a model trained with scale_pos_weight and are not probabilities.

| set | n | prevalence | brier_raw | brier_isotonic | ece_raw | ece_isotonic |
|---|---|---|---|---|---|---|
| validation (30-34; calibrator fitted here, in-sample) | 3513 | 0.1682 | 0.0077 | 0.0068 | 0.0054 | 0.0000 |
| test overall (35-49) | 16670 | 0.0650 | 0.0237 | 0.0218 | 0.0251 | 0.0174 |
| test pre-shutdown (35-42) | 9983 | 0.0916 | 0.0182 | 0.0165 | 0.0188 | 0.0113 |
| test post-shutdown (43-49) | 6687 | 0.0253 | 0.0320 | 0.0298 | 0.0345 | 0.0264 |

## Alert budget (test, per step)

Threshold column = default cost threshold (C=10). maxR@k = best possible recall@k given the number of illicit nodes in the step. Only labeled nodes are ranked.

| time_step | n | illicit | recall@thr | P@10 | R@10 | maxR@10 | P@25 | R@25 | maxR@25 |
|---|---|---|---|---|---|---|---|---|---|
| 35 | 1341 | 182 | 0.967 | 1.000 | 0.055 | 0.055 | 1.000 | 0.137 | 0.137 |
| 36 | 1708 | 33 | 1.000 | 1.000 | 0.303 | 0.303 | 1.000 | 0.758 | 0.758 |
| 37 | 498 | 40 | 0.650 | 1.000 | 0.250 | 0.250 | 1.000 | 0.625 | 0.625 |
| 38 | 756 | 111 | 0.910 | 1.000 | 0.090 | 0.090 | 1.000 | 0.225 | 0.225 |
| 39 | 1183 | 81 | 0.938 | 1.000 | 0.123 | 0.123 | 1.000 | 0.309 | 0.309 |
| 40 | 1211 | 112 | 0.696 | 1.000 | 0.089 | 0.089 | 1.000 | 0.223 | 0.223 |
| 41 | 1132 | 116 | 0.948 | 1.000 | 0.086 | 0.086 | 1.000 | 0.216 | 0.216 |
| 42 | 2154 | 239 | 0.812 | 1.000 | 0.042 | 0.042 | 1.000 | 0.105 | 0.105 |
| 43 | 1370 | 24 | 0.000 | 0.000 | 0.000 | 0.417 | 0.040 | 0.042 | 1.000 |
| 44 | 1591 | 24 | 0.083 | 0.100 | 0.042 | 0.417 | 0.040 | 0.042 | 1.000 |
| 45 | 1221 | 5 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 |
| 46 | 712 | 2 | 0.500 | 0.100 | 0.500 | 1.000 | 0.040 | 0.500 | 1.000 |
| 47 | 846 | 22 | 0.000 | 0.000 | 0.000 | 0.455 | 0.000 | 0.000 | 1.000 |
| 48 | 471 | 36 | 0.028 | 0.200 | 0.056 | 0.278 | 0.240 | 0.167 | 0.694 |
| 49 | 476 | 56 | 0.018 | 0.100 | 0.018 | 0.179 | 0.040 | 0.018 | 0.446 |

| period | steps | recall@thr | P@10 | R@10 | maxR@10 | P@25 | R@25 | maxR@25 |
|---|---|---|---|---|---|---|---|---|
| pre-shutdown avg | 8 | 0.865 | 1.000 | 0.130 | 0.130 | 1.000 | 0.325 | 0.325 |
| post-shutdown avg | 7 | 0.090 | 0.071 | 0.088 | 0.535 | 0.057 | 0.110 | 0.877 |


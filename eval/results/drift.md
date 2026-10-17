# Drift monitor (label-free)

Model: LightGBM (165 original). Threshold 0.2582 (cost ratio C=10, an assumption). Reference = all nodes in steps 1-29.
Score PSI uses 10 reference-quantile bins. Feature PSI = mean over the top 20 features by mean |SHAP| on labeled training rows: f59, f53, f2, f5, f90, f125, f67, f76, f132, f58, f16, f46, f47, f55, f65, f89, f163, f3, f138, f41.
PSI cutoffs are the conventional 0.10 (moderate) and 0.25 (major). Alert-rate 'major' = outside [-0.0915, 0.3313] (mean +/- 3 SD over reference steps). No cutoff was tuned on test.
Steps 1-29 are inside the reference, so their signals are near zero by construction; false alarms are only meaningful for steps 30-42. Recall in steps 1-29 is in-sample.

| time_step | n_nodes | n_illicit | score_psi | feat_psi | alert_rate | recall | score_level | feat_level | alert_out_of_band |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 7880 | 17 | 0.355 | 0.109 | 0.014 | 1.000 | major | moderate | False |
| 2 | 4544 | 18 | 0.248 | 0.129 | 0.025 | 1.000 | moderate | moderate | False |
| 3 | 6621 | 11 | 0.548 | 0.171 | 0.016 | 1.000 | major | moderate | False |
| 4 | 5693 | 30 | 0.525 | 0.173 | 0.026 | 1.000 | major | moderate | False |
| 5 | 6803 | 8 | 0.475 | 0.189 | 0.015 | 1.000 | major | moderate | False |
| 6 | 4328 | 5 | 0.485 | 0.618 | 0.032 | 1.000 | major | major | False |
| 7 | 6048 | 102 | 0.059 | 0.060 | 0.086 | 1.000 | ok | ok | False |
| 8 | 4457 | 67 | 0.029 | 0.079 | 0.107 | 1.000 | ok | ok | False |
| 9 | 4996 | 248 | 0.095 | 0.096 | 0.145 | 1.000 | ok | ok | False |
| 10 | 6727 | 18 | 0.123 | 0.186 | 0.111 | 1.000 | moderate | moderate | False |
| 11 | 4296 | 131 | 0.121 | 0.089 | 0.145 | 1.000 | moderate | ok | False |
| 12 | 2047 | 16 | 0.060 | 0.074 | 0.135 | 1.000 | ok | ok | False |
| 13 | 4528 | 291 | 0.225 | 0.086 | 0.215 | 1.000 | moderate | ok | False |
| 14 | 2022 | 43 | 0.208 | 0.115 | 0.088 | 1.000 | moderate | moderate | False |
| 15 | 3639 | 147 | 0.216 | 0.229 | 0.108 | 1.000 | moderate | moderate | False |
| 16 | 2975 | 128 | 0.201 | 0.212 | 0.118 | 1.000 | moderate | moderate | False |
| 17 | 3385 | 99 | 0.107 | 0.066 | 0.167 | 1.000 | moderate | ok | False |
| 18 | 1976 | 52 | 0.068 | 0.125 | 0.140 | 1.000 | ok | moderate | False |
| 19 | 3506 | 80 | 0.150 | 0.117 | 0.115 | 1.000 | moderate | moderate | False |
| 20 | 4291 | 260 | 0.260 | 0.119 | 0.188 | 1.000 | major | moderate | False |
| 21 | 3537 | 100 | 0.202 | 0.146 | 0.145 | 1.000 | moderate | moderate | False |
| 22 | 5894 | 158 | 0.122 | 0.114 | 0.128 | 1.000 | moderate | moderate | False |
| 23 | 4165 | 53 | 0.104 | 0.084 | 0.081 | 1.000 | moderate | ok | False |
| 24 | 4592 | 137 | 0.046 | 0.063 | 0.109 | 1.000 | ok | ok | False |
| 25 | 2314 | 118 | 0.100 | 0.178 | 0.159 | 1.000 | moderate | moderate | False |
| 26 | 2523 | 96 | 0.460 | 0.225 | 0.222 | 1.000 | major | moderate | False |
| 27 | 1089 | 24 | 0.202 | 0.245 | 0.105 | 1.000 | moderate | moderate | False |
| 28 | 1653 | 85 | 0.362 | 0.220 | 0.234 | 1.000 | major | moderate | False |
| 29 | 4275 | 329 | 0.387 | 0.155 | 0.300 | 1.000 | major | moderate | False |
| 30 | 2483 | 83 | 0.543 | 0.182 | 0.297 | 0.952 | major | moderate | False |
| 31 | 2816 | 106 | 0.471 | 0.291 | 0.209 | 0.981 | major | major | False |
| 32 | 4525 | 342 | 0.623 | 0.271 | 0.262 | 1.000 | major | major | False |
| 33 | 3151 | 23 | 0.518 | 0.534 | 0.054 | 0.826 | major | major | False |
| 34 | 2486 | 37 | 0.513 | 0.499 | 0.200 | 1.000 | major | major | False |
| 35 | 5507 | 182 | 0.298 | 0.402 | 0.156 | 0.984 | major | major | False |
| 36 | 6393 | 33 | 0.303 | 0.508 | 0.121 | 1.000 | major | major | False |
| 37 | 3306 | 40 | 0.357 | 0.516 | 0.091 | 0.675 | major | major | False |
| 38 | 2891 | 111 | 0.429 | 0.349 | 0.183 | 0.919 | major | major | False |
| 39 | 2760 | 81 | 0.331 | 0.573 | 0.222 | 0.938 | major | major | False |
| 40 | 4481 | 112 | 0.366 | 0.335 | 0.173 | 0.696 | major | major | False |
| 41 | 5342 | 116 | 0.575 | 0.598 | 0.245 | 0.948 | major | major | False |
| 42 | 7140 | 239 | 0.274 | 0.347 | 0.186 | 0.820 | major | major | False |
| 43 | 5063 | 24 | 0.393 | 0.329 | 0.105 | 0.000 | major | major | False |
| 44 | 4975 | 24 | 0.402 | 0.317 | 0.179 | 0.083 | major | major | False |
| 45 | 5598 | 5 | 0.227 | 0.324 | 0.128 | 0.000 | moderate | major | False |
| 46 | 3519 | 2 | 0.517 | 0.693 | 0.206 | 0.500 | major | major | False |
| 47 | 5121 | 22 | 0.319 | 0.332 | 0.160 | 0.000 | major | major | False |
| 48 | 2954 | 36 | 0.469 | 0.438 | 0.195 | 0.028 | major | major | False |
| 49 | 2454 | 56 | 0.600 | 0.366 | 0.195 | 0.018 | major | major | False |

## Flags

```
{
  "score_psi": {
    "major_steps": [
      1,
      3,
      4,
      5,
      6,
      20,
      26,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      39,
      40,
      41,
      42,
      43,
      44,
      46,
      47,
      48,
      49
    ],
    "post_steps_flagged": [
      43,
      44,
      46,
      47,
      48,
      49
    ],
    "post_steps_missed": [
      45
    ],
    "false_alarms_1_42": [
      1,
      3,
      4,
      5,
      6,
      20,
      26,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      39,
      40,
      41,
      42
    ],
    "false_alarms_30_42": [
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      39,
      40,
      41,
      42
    ],
    "max_pre_value": 0.6226904917841062,
    "min_post_value": 0.22695280209262186
  },
  "feat_psi": {
    "major_steps": [
      6,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      39,
      40,
      41,
      42,
      43,
      44,
      45,
      46,
      47,
      48,
      49
    ],
    "post_steps_flagged": [
      43,
      44,
      45,
      46,
      47,
      48,
      49
    ],
    "post_steps_missed": [],
    "false_alarms_1_42": [
      6,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      39,
      40,
      41,
      42
    ],
    "false_alarms_30_42": [
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      39,
      40,
      41,
      42
    ],
    "max_pre_value": 0.5980833587575932,
    "min_post_value": 0.3167397956723388
  },
  "alert_rate": {
    "major_steps": [],
    "post_steps_flagged": [],
    "post_steps_missed": [
      43,
      44,
      45,
      46,
      47,
      48,
      49
    ],
    "false_alarms_1_42": [],
    "false_alarms_30_42": [],
    "max_pre_value": 0.297221103503826,
    "min_post_value": 0.10468101915860162
  }
}
```


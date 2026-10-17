# Model health monitor (delayed labels), test steps 35-49

Constants fixed in advance: lag 2, window 3 steps ending at time_step - 2, pooled recall at the alert threshold, reference = validation steps 30-34, degraded if window recall < 0.5 x reference, at least 20 illicit nodes required. The constants were designed after the post-shutdown failure was observed.

| time_step | status | window_steps | window_recall | reference_recall | n_illicit_in_window | lag |
|---|---|---|---|---|---|---|
| 35 | ok | 31-33 | 0.9873 | 0.9831 | 471 | 2 |
| 36 | ok | 32-34 | 0.9900 | 0.9831 | 402 | 2 |
| 37 | ok | 33-35 | 0.9711 | 0.9831 | 242 | 2 |
| 38 | ok | 34-36 | 0.9881 | 0.9831 | 252 | 2 |
| 39 | ok | 35-37 | 0.9373 | 0.9831 | 255 | 2 |
| 40 | ok | 36-38 | 0.8804 | 0.9831 | 184 | 2 |
| 41 | ok | 37-39 | 0.8836 | 0.9831 | 232 | 2 |
| 42 | ok | 38-40 | 0.8421 | 0.9831 | 304 | 2 |
| 43 | ok | 39-41 | 0.8544 | 0.9831 | 309 | 2 |
| 44 | ok | 40-42 | 0.8223 | 0.9831 | 467 | 2 |
| 45 | ok | 41-43 | 0.8074 | 0.9831 | 379 | 2 |
| 46 | ok | 42-44 | 0.6899 | 0.9831 | 287 | 2 |
| 47 | degraded | 43-45 | 0.0377 | 0.9831 | 53 | 2 |
| 48 | degraded | 44-46 | 0.0968 | 0.9831 | 31 | 2 |
| 49 | degraded | 45-47 | 0.0345 | 0.9831 | 29 | 2 |

First degraded step: 47. False alarms in 35-42: none.

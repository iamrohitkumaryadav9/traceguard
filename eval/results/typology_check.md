# Typology flag check

Illicit rate of flagged vs unflagged LABELED nodes. Time steps 1-34 only; the test period is never used.
Fan thresholds: in-degree >= 9, out-degree >= 5 (99th percentile over steps 1-29, minimum 3). Chain: chain_len >= 4. 'any' = at least one flag.
Elliptic has no typology labels, so this only asks whether the structural flags carry signal about the illicit label, not whether they match real typologies.

Base illicit rate (labeled, steps 1-34): 0.1158.

| steps | pattern | group | n | illicit | illicit_rate | ci95_low | ci95_high |
|---|---|---|---|---|---|---|---|
| 1-34 | fan_in | flagged | 1023 | 19 | 0.0186 | 0.0119 | 0.0288 |
| 1-34 | fan_in | unflagged | 28871 | 3443 | 0.1193 | 0.1156 | 0.1230 |
| 1-34 | fan_out | flagged | 347 | 0 | 0.0000 | 0.0000 | 0.0109 |
| 1-34 | fan_out | unflagged | 29547 | 3462 | 0.1172 | 0.1136 | 0.1209 |
| 1-34 | chain | flagged | 1469 | 378 | 0.2573 | 0.2356 | 0.2803 |
| 1-34 | chain | unflagged | 28425 | 3084 | 0.1085 | 0.1049 | 0.1122 |
| 1-34 | any | flagged | 2827 | 397 | 0.1404 | 0.1281 | 0.1537 |
| 1-34 | any | unflagged | 27067 | 3065 | 0.1132 | 0.1095 | 0.1171 |
| 1-29 | fan_in | flagged | 880 | 13 | 0.0148 | 0.0087 | 0.0251 |
| 1-29 | fan_in | unflagged | 25501 | 2858 | 0.1121 | 0.1083 | 0.1160 |
| 1-29 | fan_out | flagged | 278 | 0 | 0.0000 | -0.0000 | 0.0136 |
| 1-29 | fan_out | unflagged | 26103 | 2871 | 0.1100 | 0.1062 | 0.1138 |
| 1-29 | chain | flagged | 1186 | 253 | 0.2133 | 0.1910 | 0.2375 |
| 1-29 | chain | unflagged | 25195 | 2618 | 0.1039 | 0.1002 | 0.1077 |
| 1-29 | any | flagged | 2332 | 266 | 0.1141 | 0.1018 | 0.1276 |
| 1-29 | any | unflagged | 24049 | 2605 | 0.1083 | 0.1045 | 0.1123 |
| 30-34 | fan_in | flagged | 143 | 6 | 0.0420 | 0.0194 | 0.0885 |
| 30-34 | fan_in | unflagged | 3370 | 585 | 0.1736 | 0.1612 | 0.1867 |
| 30-34 | fan_out | flagged | 69 | 0 | 0.0000 | 0.0000 | 0.0527 |
| 30-34 | fan_out | unflagged | 3444 | 591 | 0.1716 | 0.1594 | 0.1846 |
| 30-34 | chain | flagged | 283 | 125 | 0.4417 | 0.3850 | 0.5000 |
| 30-34 | chain | unflagged | 3230 | 466 | 0.1443 | 0.1326 | 0.1568 |
| 30-34 | any | flagged | 495 | 131 | 0.2646 | 0.2277 | 0.3052 |
| 30-34 | any | unflagged | 3018 | 460 | 0.1524 | 0.1400 | 0.1657 |

## Lift, steps 1-34

| pattern | flagged | unflagged | lift |
|---|---|---|---|
| any | 0.1404 | 0.1132 | 1.2402 |
| chain | 0.2573 | 0.1085 | 2.3717 |
| fan_in | 0.0186 | 0.1193 | 0.1557 |
| fan_out | 0.0000 | 0.1172 | 0.0000 |


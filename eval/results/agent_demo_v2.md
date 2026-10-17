# Agent demo v2 (Phase 4b)

Model: `gemini-3.5-flash-lite` (temperature 0). Retrieval is frozen (hybrid + headers, no rewriting).
Same six node ids as v1 (the selection rules are unchanged and give the same nodes; `python -m eval.agent_demo` asserts this). Ground-truth labels were used only for the selection rules and are shown below for the record; the agent never sees them. Each run was done once; nothing was re-run to get a better result. The v1 record is kept in `eval/results/agent_demo_v1.md` and `eval/traces/v1/`.

| rule | tx_id | step | candidates | score | ground truth (not seen by agent) |
|---|---|---|---|---|---|
| 1. high-band, illicit, steps 35-42 | 12779397 | 41 | 776 | 0.9997 | illicit |
| 2. high-band, licit, steps 35-42 | 99175682 | 42 | 39 | 0.804 | licit |
| 3. flagged chain, steps 35-42 | 97870846 | 40 | 6,369 | 0.0003 | unknown |
| 4. flagged fan_in (lowers_risk), steps 35-42, band at least medium, not chain | 70360850 | 35 | 33 | 0.9993 | unknown |
| 5. steps 43-49, band at least medium | 106417568 | 44 | 4,741 | 0.4518 | unknown |
| 6. low band, steps 35-42 | 149159647 | 39 | 31,331 | 0.0001 | unknown |

## Run summary

| tx_id | status | verification | nodes | investigator tool calls | searches | LLM calls | critic loops | total latency (s) | of which rate-limit waits (s) | tokens in/out |
|---|---|---|---|---|---|---|---|---|---|---|
| 12779397 | verified | verified | triage > investigator > writer > critic > finalize | 2 | 0 | 5 | 0 | 30.944 | 17.98 | 7205/1038 |
| 99175682 | verified | verified | triage > investigator > writer > critic > finalize | 4 | 0 | 7 | 0 | 41.357 | 31.39 | 10960/745 |
| 97870846 | verified | verified | triage > investigator > regulation > writer > critic > finalize | 4 | 1 | 9 | 0 | 71.54 | 35.81 | 17380/956 |
| 70360850 | verified | verified | triage > investigator > writer > critic > writer > critic > writer > critic > finalize | 2 | 0 | 8 | 2 | 50.44 | 35.84 | 13985/2289 |
| 106417568 | verified | verified | triage > investigator > writer > critic > writer > critic > finalize | 4 | 0 | 9 | 1 | 55.389 | 40.39 | 13980/1614 |
| 149159647 | no_action_model_based | not applicable | triage > finalize | 0 | 0 | 0 | 0 | 0.064 | 0.0 | 0/0 |

## 1. high-band, illicit, steps 35-42 (tx 12779397)

Final status: **verified**

### Memo for transaction 12779397 (time step 41)

- status: **verified**, verification: **verified**
- risk band: high; raw score: 0.999668; calibrated probability: 0.988235
- thresholds: {'alert': 0.258181, 'high': 0.788025}
- model health: {'status': 'ok', 'window_recall': 0.883621, 'reference_recall': 0.98308, 'n_illicit_in_window': 232}
- subject fields: names / addresses / amounts / dates = not available in dataset

**Activity description**

1. The raw LightGBM predict proba score is 0.999668 and calibrated probability is 0.988235, placing the transaction in the high risk band with an alert triggered above the alert threshold of 0.258181 and high threshold of 0.788025. `[score:12779397]`
2. Features f53, f59, f90, f132, and f2 pushed the illicit score up. `[score:12779397]`
3. Model health check at time step 41 shows status ok with a window recall of 0.883621 and reference recall of 0.98308 across window steps 37, 38, and 39 with a lag of 2 and 232 illicit transactions in the window. `[health:41]`
4. The typology flags for fan-in, fan-out, and chain all matched as false with an in-degree of 1, out-degree of 0, and chain length of 0 against thresholds of 9, 5, and 4. `[typology:12779397]`
5. Neighborhood analysis found 1 incoming neighbor at hop 1 and 2 total incoming neighbors up to hop 2. `[neighbors:12779397:1:in, neighbors:12779397:2:in]`
6. The radius 2 subgraph summary shows 4 nodes, 3 edges, a density of 0.25, a focal in-degree of 1, a focal out-degree of 0, a max in-degree of 1, a max out-degree of 2, 2 alerted nodes with a share alerted of 0.5, a focal chain length of 0, and a wcc size of 5342. `[subgraph:12779397:2]`

**Regulatory background (context, not evidence about this transaction)**

- none

**Evidence table**

- `score:12779397` from score_transaction
- `health:41` from check_model_health
- `typology:12779397` from match_typology
- `neighbors:12779397:1:in` from get_neighbors
- `neighbors:12779397:2:in` from get_neighbors
- `subgraph:12779397:2` from subgraph_summary

**Limitations**

- The Elliptic dataset has no wallet addresses, names, amounts or dates, so none of these appear in this memo.
- The model's recall collapsed after time step 43 (a dark-market shutdown), and the model-health check only detects such a collapse several steps later, so scores at later time steps may be unreliable.
- Regulation retrieval is weak (hybrid hit@3 about 0.6 on a small test set): retrieved passages are leads for an analyst, not verified authorities, and text from the India document comes from OCR.
- Scores are raw gradient-boosting outputs, not calibrated probabilities; the calibrated value shifts with the prevalence of illicit activity.
- Structural patterns are resemblances only: Elliptic has no typology labels.
- The model-health monitor uses labels with a 2-step lag and a 3-step window, so it cannot flag a collapse in its first steps. On this dataset it first flags at step 47.
- This is a draft for a human analyst. It does not decide whether a report should be filed.

**Recommendation:** refer for human review

**Critic verdicts**

- round 1, claim 1: OK - reviewer: supported (All facts in the claim match the cited evidence values for score, calibrated probability, alert triggers, thresholds, and risk band.)
- round 1, claim 2: OK - reviewer: supported (The evidence lists features f53, f59, f90, f132, and f2 as having the direction 'raises illicit score'.)
- round 1, claim 3: OK - reviewer: supported (The evidence directly confirms the status, window recall, reference recall, window steps, lag, and number of illicit transactions.)
- round 1, claim 4: OK - reviewer: supported (The typology flags, degrees, chain length, and thresholds match the evidence values exactly.)
- round 1, claim 5: OK - reviewer: supported (The incoming neighbors count for hop 1 is 1 and up to hop 2 is 2, as stated in the evidence.)
- round 1, claim 6: OK - reviewer: supported (All subgraph summary metrics match the corresponding values in the provided evidence object.)

## 2. high-band, licit, steps 35-42 (tx 99175682)

Final status: **verified**

### Memo for transaction 99175682 (time step 42)

- status: **verified**, verification: **verified**
- risk band: high; raw score: 0.804028; calibrated probability: 0.75
- thresholds: {'alert': 0.258181, 'high': 0.788025}
- model health: {'status': 'ok', 'window_recall': 0.842105, 'reference_recall': 0.98308, 'n_illicit_in_window': 304}
- subject fields: names / addresses / amounts / dates = not available in dataset

**Signal conflict: the model score 0.804028 (risk band high; evidence score:99175682) and the in-neighbour alerts (none of the 3 in-neighbours alerted; evidence neighbors:99175682:1:in) point in different directions.** `[score:99175682, neighbors:99175682:1:in]`

**Activity description**

1. The transaction received a score of 0.804028, placing it in the high risk band with an alert status of true. `[score:99175682]`
2. Features f53, f59, f3, f2, and f163 pushed the illicit score up. `[score:99175682]`
3. Model health check shows status ok at time step 42 with a window recall of 0.842105 against a reference recall of 0.98308. `[health:42]`
4. Network examination found 3 incoming neighbors and 1 outgoing neighbor. `[typology:99175682]`

**Regulatory background (context, not evidence about this transaction)**

- none

**Evidence table**

- `score:99175682` from score_transaction
- `neighbors:99175682:1:in` from get_neighbors
- `health:42` from check_model_health
- `typology:99175682` from match_typology

**Limitations**

- The Elliptic dataset has no wallet addresses, names, amounts or dates, so none of these appear in this memo.
- The model's recall collapsed after time step 43 (a dark-market shutdown), and the model-health check only detects such a collapse several steps later, so scores at later time steps may be unreliable.
- Regulation retrieval is weak (hybrid hit@3 about 0.6 on a small test set): retrieved passages are leads for an analyst, not verified authorities, and text from the India document comes from OCR.
- Scores are raw gradient-boosting outputs, not calibrated probabilities; the calibrated value shifts with the prevalence of illicit activity.
- Structural patterns are resemblances only: Elliptic has no typology labels.
- The model-health monitor uses labels with a 2-step lag and a 3-step window, so it cannot flag a collapse in its first steps. On this dataset it first flags at step 47.
- This is a draft for a human analyst. It does not decide whether a report should be filed.

**Recommendation:** refer for human review

**Critic verdicts**

- round 1, claim 1: OK - reviewer: supported (The evidence explicitly states the score is 0.804028, the risk band is high, and alert is true.)
- round 1, claim 2: OK - reviewer: supported (The evidence lists features f53, f59, f3, f2, and f163 with the direction 'raises illicit score'.)
- round 1, claim 3: OK - reviewer: supported (The evidence matches all the stated claims regarding status ok, time step 42, window recall, and reference recall.)
- round 1, claim 4: OK - reviewer: supported (The evidence shows in_degree is 3 and out_degree is 1, which corresponds to 3 incoming and 1 outgoing neighbors.)
- round 1, signal_conflict: OK - reviewer: None

## 3. flagged chain, steps 35-42 (tx 97870846)

Final status: **verified**

### Memo for transaction 97870846 (time step 40)

- status: **verified**, verification: **verified**
- risk band: low; raw score: 0.000323; calibrated probability: 0.0
- thresholds: {'alert': 0.258181, 'high': 0.788025}
- model health: {'status': 'ok', 'window_recall': 0.880435, 'reference_recall': 0.98308, 'n_illicit_in_window': 184}
- subject fields: names / addresses / amounts / dates = not available in dataset

**Signal conflict: the model score 0.000323 (risk band low; evidence score:97870846) and the structural pattern chain (evidence direction raises_risk; evidence typology:97870846) point in different directions.** `[score:97870846, typology:97870846]`

**Activity description**

1. The transaction scored 0.000323 resulting in a low risk band with features f59 and f76 pushed the score up; features f131, f132 and f1 pushed the score down. `[score:97870846]`
2. A structural pattern analysis identified a chain pattern with a chain length of 4 matching the threshold of 4, with 1469 flagged instances and an observed illicit rate of 0.257318 compared to a baseline illicit rate of 0.115809, resulting in a lift of 2.221913, where training-period data associate this pattern with a higher illicit rate. `[typology:97870846]`
3. The model health check for time step 40 indicates status ok with a window recall of 0.880435 against a reference recall of 0.98308 across 184 illicit instances in the window. `[health:40]`

**Regulatory background (context, not evidence about this transaction)**

- FinCEN guidance lists unusually complex series of transactions indicative of layering activity involving multiple accounts, banks, parties, and jurisdictions as a general indicator. (FinCEN 2003: Guidance on Preparing A Complete & Sufficient Suspicious Activity Report Narrative; `reg:fincen_2003_sar_narrative:p7:c10`; quotable: True)

**Evidence table**

- `score:97870846` from score_transaction
- `typology:97870846` from match_typology
- `health:40` from check_model_health
- `reg:fincen_2003_sar_narrative:p7:c10` from search_regulations

**Limitations**

- The Elliptic dataset has no wallet addresses, names, amounts or dates, so none of these appear in this memo.
- The model's recall collapsed after time step 43 (a dark-market shutdown), and the model-health check only detects such a collapse several steps later, so scores at later time steps may be unreliable.
- Regulation retrieval is weak (hybrid hit@3 about 0.6 on a small test set): retrieved passages are leads for an analyst, not verified authorities, and text from the India document comes from OCR.
- Scores are raw gradient-boosting outputs, not calibrated probabilities; the calibrated value shifts with the prevalence of illicit activity.
- Structural patterns are resemblances only: Elliptic has no typology labels.
- The model-health monitor uses labels with a 2-step lag and a 3-step window, so it cannot flag a collapse in its first steps. On this dataset it first flags at step 47.
- This is a draft for a human analyst. It does not decide whether a report should be filed.

**Recommendation:** refer for human review

**Critic verdicts**

- round 1, claim 1: OK - reviewer: supported (The cited evidence explicitly states the score is 0.000323, risk band is low, features f59 and f76 raise the illicit score (push up), and f131, f132, and f1 lower the illicit score (push down).)
- round 1, claim 2: OK - reviewer: supported (All numerical details, match conditions, rates, lift, and directional training-period associations are directly verified by the cited typology evidence.)
- round 1, claim 3: OK - reviewer: supported (The health check data for time step 40 fully confirms the status ok, window recall of 0.880435, reference recall of 0.98308, and 184 illicit instances.)
- round 1, background 1: OK - reviewer: supported (The FinCEN regulatory text explicitly lists an unusually complex series of transactions indicative of layering activity involving multiple accounts, banks, parties, and jurisdictions.)
- round 1, signal_conflict: OK - reviewer: None

## 4. flagged fan_in (lowers_risk), steps 35-42, band at least medium, not chain (tx 70360850)

Final status: **verified**

### Memo for transaction 70360850 (time step 35)

- status: **verified**, verification: **verified**
- risk band: high; raw score: 0.999318; calibrated probability: 0.988235
- thresholds: {'alert': 0.258181, 'high': 0.788025}
- model health: {'status': 'ok', 'window_recall': 0.987261, 'reference_recall': 0.98308, 'n_illicit_in_window': 471}
- subject fields: names / addresses / amounts / dates = not available in dataset

**Signal conflict: the model score 0.999318 (risk band high; evidence score:70360850) and the structural pattern fan_in (evidence direction lowers_risk; evidence typology:70360850) point in different directions.** `[score:70360850, typology:70360850]`

**Activity description**

1. The transaction received a raw LightGBM predict proba score of 0.999318 and a calibrated probability of 0.988235, placing it in the high risk band and triggering an alert. `[score:70360850]`
2. Features f53, f161, f132, f59, and f5 pushed the score up. `[score:70360850]`
3. Model health check for time step 35 indicates status ok with a window recall of 0.987261 and reference recall of 0.98308 across 471 illicit transactions in the window. `[health:35]`
4. The transaction matched the fan_in structural pattern with an in-degree of 10, which met the threshold of 9. Training-period data associate this pattern with a lower illicit rate than the baseline. `[typology:70360850]`
5. Neighborhood analysis found 10 incoming neighbors, all of which are alerted transactions. `[neighbors:70360850:1:in]`
6. The radius 2 subgraph contains 22 nodes and 21 edges with 14 alerted nodes, resulting in a share alerted of 0.636364. `[subgraph:70360850:2]`

**Regulatory background (context, not evidence about this transaction)**

- none

**Evidence table**

- `score:70360850` from score_transaction
- `typology:70360850` from match_typology
- `health:35` from check_model_health
- `neighbors:70360850:1:in` from get_neighbors
- `subgraph:70360850:2` from subgraph_summary

**Limitations**

- The Elliptic dataset has no wallet addresses, names, amounts or dates, so none of these appear in this memo.
- The model's recall collapsed after time step 43 (a dark-market shutdown), and the model-health check only detects such a collapse several steps later, so scores at later time steps may be unreliable.
- Regulation retrieval is weak (hybrid hit@3 about 0.6 on a small test set): retrieved passages are leads for an analyst, not verified authorities, and text from the India document comes from OCR.
- Scores are raw gradient-boosting outputs, not calibrated probabilities; the calibrated value shifts with the prevalence of illicit activity.
- Structural patterns are resemblances only: Elliptic has no typology labels.
- The model-health monitor uses labels with a 2-step lag and a 3-step window, so it cannot flag a collapse in its first steps. On this dataset it first flags at step 47.
- This is a draft for a human analyst. It does not decide whether a report should be filed.

**Recommendation:** refer for human review

**Critic verdicts**

- round 1, claim 1: REJECTED - quote 'score:0.999318' has no cited regulation chunk
- round 1, claim 2: OK - reviewer: supported (The cited evidence lists features f53, f161, f132, f59, and f5, all with the direction 'raises illicit score', which supports the claim that they pushed the score up.)
- round 1, claim 3: REJECTED - quote 'status' has no cited regulation chunk
- round 1, claim 4: REJECTED - quote 'in_degree 10 >= 9' has no cited regulation chunk
- round 1, claim 5: REJECTED - quote 'total_found":10' has no cited regulation chunk
- round 1, claim 6: REJECTED - quote 'share_alerted":0.636364' has no cited regulation chunk
- round 1, signal_conflict: OK - reviewer: None
- round 2, claim 1: REJECTED - quote 'score:0.999318' has no cited regulation chunk
- round 2, claim 2: REJECTED - quote 'f53' has no cited regulation chunk
- round 2, claim 3: REJECTED - quote 'status' has no cited regulation chunk
- round 2, claim 4: REJECTED - quote 'in_degree 10 >= 9' has no cited regulation chunk
- round 2, claim 5: REJECTED - quote 'total_found":10' has no cited regulation chunk
- round 2, claim 6: REJECTED - quote 'share_alerted":0.636364' has no cited regulation chunk
- round 2, signal_conflict: OK - reviewer: None
- round 3, claim 1: OK - reviewer: supported (The evidence explicitly states the score, score_scale, calibrated_probability, risk_band, and alert status matching the claim.)
- round 3, claim 2: OK - reviewer: supported (The evidence lists features f53, f161, f132, f59, and f5 all with the direction 'raises illicit score'.)
- round 3, claim 3: OK - reviewer: supported (The evidence for health:35 directly supports all values provided in the claim.)
- round 3, claim 4: OK - reviewer: supported (The evidence confirms the fan_in pattern matches with in_degree 10 (>= threshold 9) and is associated with a lower illicit rate than the baseline.)
- round 3, claim 5: OK - reviewer: supported (The evidence shows 10 incoming neighbors, and all 10 have alert set to true.)
- round 3, claim 6: OK - reviewer: supported (The subgraph evidence matches all the numbers cited in the claim.)
- round 3, signal_conflict: OK - reviewer: None

## 5. steps 43-49, band at least medium (tx 106417568)

Final status: **verified**

### Memo for transaction 106417568 (time step 44)

- status: **verified**, verification: **verified**
- risk band: medium; raw score: 0.451772; calibrated probability: 0.25
- thresholds: {'alert': 0.258181, 'high': 0.788025}
- model health: {'status': 'ok', 'window_recall': 0.82227, 'reference_recall': 0.98308, 'n_illicit_in_window': 467}
- subject fields: names / addresses / amounts / dates = not available in dataset

**Activity description**

1. The transaction obtained a raw score of 0.451772, a calibrated probability of 0.25, and a medium risk band, crossing the alert threshold of 0.258181 but remaining below the high threshold of 0.788025. `[score:106417568]`
2. Features f53, f59, f3, f55, and f100 pushed the illicit score up. `[score:106417568]`
3. The model health check at time step 44 shows status ok with a window recall of 0.82227, reference recall of 0.98308, and a lag of 2. `[health:44]`
4. Subgraph analysis within radius 2 around the transaction found 4 nodes and 3 edges with a share of alerted nodes at 0.75. `[subgraph:106417568:2]`
5. Neighborhood analysis identified connected transactions with scores including an incoming neighbor at 0.082171 and an outgoing neighbor at 0.637459. `[neighbors:106417568:1:both]`

**Regulatory background (context, not evidence about this transaction)**

- none

**Evidence table**

- `score:106417568` from score_transaction
- `health:44` from check_model_health
- `subgraph:106417568:2` from subgraph_summary
- `neighbors:106417568:1:both` from get_neighbors

**Limitations**

- The Elliptic dataset has no wallet addresses, names, amounts or dates, so none of these appear in this memo.
- The model's recall collapsed after time step 43 (a dark-market shutdown), and the model-health check only detects such a collapse several steps later, so scores at later time steps may be unreliable.
- Regulation retrieval is weak (hybrid hit@3 about 0.6 on a small test set): retrieved passages are leads for an analyst, not verified authorities, and text from the India document comes from OCR.
- Scores are raw gradient-boosting outputs, not calibrated probabilities; the calibrated value shifts with the prevalence of illicit activity.
- Structural patterns are resemblances only: Elliptic has no typology labels.
- The model-health monitor uses labels with a 2-step lag and a 3-step window, so it cannot flag a collapse in its first steps. On this dataset it first flags at step 47.
- This is a draft for a human analyst. It does not decide whether a report should be filed.
- Dataset-specific note, based on hindsight from the published test results: model performance collapsed from time step 43 onward. A deployed system would not know this in advance.

**Recommendation:** refer for human review

**Critic verdicts**

- round 1, claim 1: REJECTED - reviewer: unsupported (The claim states the calibrated probability is 0.25, but the cited evidence shows the calibrated probability is 0.25.)
- round 1, claim 2: OK - reviewer: supported (The cited evidence shows features f53, f59, f3, f55, and f100 all have a direction of 'raises illicit score'.)
- round 1, claim 3: OK - reviewer: supported (Every fact in the claim is directly supported by the cited health evidence.)
- round 1, claim 4: OK - reviewer: supported (The cited evidence shows 1 incoming (in) neighbor and 1 outgoing (out) neighbor, making a total of 2 neighbors found.)
- round 1, claim 5: OK - reviewer: supported (Every fact stated in the claim matches the values in the cited subgraph evidence.)
- round 2, claim 1: OK - reviewer: supported (Every detail in the claim is directly supported by the provided evidence values.)
- round 2, claim 2: OK - reviewer: supported (The evidence lists features f53, f59, f3, f55, and f100 with the direction 'raises illicit score'.)
- round 2, claim 3: OK - reviewer: supported (The health evidence matches all stated metrics: status ok, window_recall 0.82227, reference_recall 0.98308, and lag 2.)
- round 2, claim 4: OK - reviewer: supported (The subgraph evidence confirms 4 nodes, 3 edges, and a share_alerted of 0.75.)
- round 2, claim 5: OK - reviewer: supported (The neighbors evidence lists an incoming neighbor with score 0.082171 and an outgoing neighbor with score 0.637459.)

## 6. low band, steps 35-42 (tx 149159647)

Final status: **no_action_model_based**

### Memo for transaction 149159647 (time step 39)

- status: **no_action_model_based**, verification: **not applicable**
- risk band: low; raw score: 8.5e-05; calibrated probability: 0.0
- thresholds: {'alert': 0.258181, 'high': 0.788025}
- model health: {'status': 'ok', 'window_recall': 0.937255, 'reference_recall': 0.98308, 'n_illicit_in_window': 255}
- subject fields: names / addresses / amounts / dates = not available in dataset

**Activity description**

1. At time step 39 the model gave a raw score of 8.5e-05 (risk band low); the alert threshold is 0.258181 and the high-risk threshold is 0.788025. `[score:149159647]`
2. Model health check at time step 39: status ok; recall in the delayed-label window was 0.937255 against a reference recall of 0.98308 over 255 labeled illicit nodes. `[health:39]`
3. No structural pattern with a risk-raising direction matched this transaction. `[typology:149159647]`

**Regulatory background (context, not evidence about this transaction)**

- none

**Evidence table**

- `score:149159647` from score_transaction
- `health:39` from check_model_health
- `typology:149159647` from match_typology

**Limitations**

- The Elliptic dataset has no wallet addresses, names, amounts or dates, so none of these appear in this memo.
- The model's recall collapsed after time step 43 (a dark-market shutdown), and the model-health check only detects such a collapse several steps later, so scores at later time steps may be unreliable.
- Regulation retrieval is weak (hybrid hit@3 about 0.6 on a small test set): retrieved passages are leads for an analyst, not verified authorities, and text from the India document comes from OCR.
- Scores are raw gradient-boosting outputs, not calibrated probabilities; the calibrated value shifts with the prevalence of illicit activity.
- Structural patterns are resemblances only: Elliptic has no typology labels.
- The model-health monitor uses labels with a 2-step lag and a 3-step window, so it cannot flag a collapse in its first steps. On this dataset it first flags at step 47.
- This is a draft for a human analyst. It does not decide whether a report should be filed.

**Recommendation:** no action suggested, model-based only

**Critic verdicts**

- the critic did not run (deterministic path)

## v1 versus v2 (automatic diff; the explanations are in the next section)

| tx_id | status v1 > v2 | claims v1 > v2 | feature claims v1 > v2 | signal_conflict v2 | period note v2 | critic rounds v1 > v2 | LLM calls v1 > v2 | latency s v1 > v2 |
|---|---|---|---|---|---|---|---|---|
| 12779397 | verified > verified | 5 > 6 | 1 > 1 | no | no | 1 > 1 | 6 > 5 | 33.671 > 30.944 |
| 99175682 | verified > verified | 5 > 4 | 1 > 1 | yes | no | 1 > 1 | 7 > 7 | 42.436 > 41.357 |
| 97870846 | verified > verified | 4 > 3 | 2 > 1 | yes | no | 2 > 1 | 9 > 9 | 83.315 > 71.54 |
| 70360850 | verified > verified | 8 > 6 | 5 > 1 | yes | no | 1 > 3 | 5 > 8 | 29.692 > 50.44 |
| 106417568 | verified > verified | 5 > 5 | 1 > 1 | no | yes | 1 > 2 | 7 > 9 | 42.181 > 55.389 |
| 149159647 | no_action_model_based > no_action_model_based | 3 > 3 | 0 > 0 | no | no | 0 > 0 | 0 > 0 | 0.084 > 0.064 |

## v1 versus v2: what changed in each memo and why

Design changes in v2: a deterministic `signal_conflict` sentence (computed in triage; the writer must copy it and the critic checks it), a `period_note` for time steps at or after 43, a new fixed limitation about the health monitor's lag, "Regulatory background (context, not evidence about this transaction)" instead of "Regulatory references", and the SHAP features merged into one claim. Triage now makes one extra deterministic `get_neighbors(in)` call (it feeds the conflict test; it is not an investigator call). Gating is unchanged. Differences in the number of other claims come from the writer's run-to-run variation, not from a design change.

- **Run 1 (12779397, high band, illicit):** no conflict (the in-neighbour is alerted and no pattern matched), no period note (step 41). The memo changed only through the new fixed limitation line and the renamed (empty) background section. 5 claims became 6 and 1 critic round became 1; this is writer variation.
- **Run 2 (99175682, high band, licit):** NEW signal conflict: the score is high (0.804028) but none of its 3 in-neighbours alerted. The sentence is built from the evidence only (the label is never used), it is carried unchanged into the memo, and the memo still says nothing about being right or wrong. In v1 this tension was not visible anywhere in the memo.
- **Run 3 (97870846, low band, chain):** NEW signal conflict: low score (0.000323) against a chain pattern whose evidence direction is raises_risk, which was exactly the unreconciled tension noted in v1. The two feature claims of v1 became one claim listing f59 and f76 (up) and f131, f132 and f1 (down). The SAR-guidance quote of v1 is now a paraphrase under the "context, not evidence" heading, worded as what the document lists "as a general indicator". The v1 round-1 critic rejection did not recur (all items passed in round 1: 2 critic rounds became 1); that is writer variation, not a fix.
- **Run 4 (70360850, high band, fan_in lowers_risk):** NEW signal conflict (high score against a lowers_risk-only pattern). The five separate feature claims of v1 are now one. The memo still never calls the pattern suspicious. This run needed 2 critic rewrites (3 rounds, 8 LLM calls, 50 s instead of 5 calls, 30 s): in rounds 1 and 2 the writer filled the `quote` field of almost every claim with evidence snippets (for example `score:0.999318`, `status`, `in_degree 10 >= 9`) and the critic rejected them, because a quote must come from a cited, quotable regulation chunk. Round 3 had no quotes and passed. The critic did its job; the cost came from the writer. The prompt was not changed in response.
- **Run 5 (106417568, step 44, medium band):** the period note is now in the limitations (step 44 is at or after 43), which fixes the v1 observation that nothing in the memo was specific to a post-shutdown step. Gating did not change: the health check still says "ok" (the monitor lags) and the node is investigated. No signal conflict (medium band). Round 1 claim 1 was rejected by the LLM reviewer with a self-contradictory reason ("the claim states the calibrated probability is 0.25, but the cited evidence shows 0.25"): a false rejection that cost one rewrite (1 critic round became 2, 7 LLM calls became 9). The rewrite passed. This is an example of fail-closed review costing a loop, not of a bad claim.
- **Run 6 (149159647, low band, no flags):** unchanged: `no_action_model_based`, zero LLM calls. It now makes 4 deterministic tool calls instead of 3 (the extra in-neighbour call). No conflict, no period note.

## Prompt changes

The writer prompt was changed BEFORE these runs as part of the designed changes (conflict sentence, one feature claim, background wording; see `src/agents/prompts.py`). No prompt, threshold or code behaviour was changed after seeing these runs.

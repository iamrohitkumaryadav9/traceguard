# Agent demo (Phase 4a)

Model: `gemini-3.5-flash-lite` (temperature 0). Retrieval is frozen (hybrid + headers, no rewriting).
Nodes were chosen BEFORE running anything, by fixed rules and a seeded draw (seed 42) from the sorted candidate ids of each rule (`python -m eval.agent_demo`). Ground-truth labels were used only for the selection rules and are shown below for the record; the agent never sees them. Each run was done once; nothing was re-run to get a better result.

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
| 12779397 | verified | verified | triage > investigator > writer > critic > finalize | 3 | 0 | 6 | 0 | 33.671 | 22.49 | 8115/814 |
| 99175682 | verified | verified | triage > investigator > writer > critic > finalize | 4 | 0 | 7 | 0 | 42.436 | 31.45 | 10784/849 |
| 97870846 | verified | verified | triage > investigator > regulation > writer > critic > writer > critic > finalize | 2 | 1 | 9 | 1 | 83.315 | 35.89 | 20235/1502 |
| 70360850 | verified | verified | triage > investigator > writer > critic > finalize | 2 | 0 | 5 | 0 | 29.692 | 22.39 | 9294/818 |
| 106417568 | verified | verified | triage > investigator > writer > critic > finalize | 4 | 0 | 7 | 0 | 42.181 | 31.38 | 10238/886 |
| 149159647 | no_action_model_based | not applicable | triage > finalize | 0 | 0 | 0 | 0 | 0.084 | 0.0 | 0/0 |

## 1. high-band, illicit, steps 35-42 (tx 12779397)

Final status: **verified**

### Memo for transaction 12779397 (time step 41)

- status: **verified**, verification: **verified**
- risk band: high; raw score: 0.999668; calibrated probability: 0.988235
- thresholds: {'alert': 0.258181, 'high': 0.788025}
- model health: {'status': 'ok', 'window_recall': 0.883621, 'reference_recall': 0.98308, 'n_illicit_in_window': 232}
- subject fields: names / addresses / amounts / dates = not available in dataset

**Activity description**

1. Transaction 12779397 generated a high alert status with a calibrated probability of 0.988235 and a raw LightGBM predict probability score of 0.999668 at time step 41. `[score:12779397]`
2. Features f53, f59, f90, f132, and f2 each pushed the score up. `[score:12779397]`
3. Model health at time step 41 was ok, with a window recall of 0.883621 and reference recall of 0.98308 across 232 illicit transactions in the window. `[health:41]`
4. The transaction has an in-degree of 1 and an out-degree of 0, with no fan-in, fan-out, or chain typology matches. `[typology:12779397]`
5. Neighbor analysis revealed one incoming neighbor transaction with an alert status and score of 0.867129. `[neighbors:12779397:1:both]`

**Regulatory references**

- none

**Evidence table**

- `score:12779397` from score_transaction
- `health:41` from check_model_health
- `typology:12779397` from match_typology
- `neighbors:12779397:1:both` from get_neighbors

**Limitations**

- The Elliptic dataset has no wallet addresses, names, amounts or dates, so none of these appear in this memo.
- The model's recall collapsed after time step 43 (a dark-market shutdown), and the model-health check only detects such a collapse several steps later, so scores at later time steps may be unreliable.
- Regulation retrieval is weak (hybrid hit@3 about 0.6 on a small test set): retrieved passages are leads for an analyst, not verified authorities, and text from the India document comes from OCR.
- Scores are raw gradient-boosting outputs, not calibrated probabilities; the calibrated value shifts with the prevalence of illicit activity.
- Structural patterns are resemblances only: Elliptic has no typology labels.
- This is a draft for a human analyst. It does not decide whether a report should be filed.

**Recommendation:** refer for human review

**Critic verdicts**

- round 1, claim 1: OK - reviewer: supported (The evidence directly confirms the transaction ID, calibrated probability, raw LightGBM predict probability, time step, and high alert status.)
- round 1, claim 2: OK - reviewer: supported (The evidence lists each of these features with the direction 'raises illicit score', confirming they pushed the score up.)
- round 1, claim 3: OK - reviewer: supported (The evidence matches all the stated metrics including status ok, window recall, reference recall, and number of illicit transactions in the window.)
- round 1, claim 4: OK - reviewer: supported (The evidence states in_degree is 1, out_degree is 0, flags for fan_in, fan_out, and chain are false, and matches list is empty.)
- round 1, claim 5: OK - reviewer: supported (The evidence shows one incoming neighbor with an alert of true and a score of 0.867129.)

## 2. high-band, licit, steps 35-42 (tx 99175682)

Final status: **verified**

### Memo for transaction 99175682 (time step 42)

- status: **verified**, verification: **verified**
- risk band: high; raw score: 0.804028; calibrated probability: 0.75
- thresholds: {'alert': 0.258181, 'high': 0.788025}
- model health: {'status': 'ok', 'window_recall': 0.842105, 'reference_recall': 0.98308, 'n_illicit_in_window': 304}
- subject fields: names / addresses / amounts / dates = not available in dataset

**Activity description**

1. Transaction 99175682 at time step 42 generated a raw LightGBM predict proba score of 0.804028, resulting in a high risk band and triggering an alert. `[score:99175682]`
2. Features f53, f59, f3, f2, and f163 all pushed the illicit score up. `[score:99175682]`
3. Model health check for time step 42 indicates status ok with a window recall of 0.842105 against a reference recall of 0.98308. `[health:42]`
4. The transaction has an in-degree of 3 and out-degree of 1, and no structural typology flags were matched. `[typology:99175682]`
5. Neighborhood analysis found a total of 4 immediate neighbors, including an outgoing neighbor with score 0.992707 that also alerted. `[neighbors:99175682:1:both]`

**Regulatory references**

- none

**Evidence table**

- `score:99175682` from score_transaction
- `health:42` from check_model_health
- `typology:99175682` from match_typology
- `neighbors:99175682:1:both` from get_neighbors

**Limitations**

- The Elliptic dataset has no wallet addresses, names, amounts or dates, so none of these appear in this memo.
- The model's recall collapsed after time step 43 (a dark-market shutdown), and the model-health check only detects such a collapse several steps later, so scores at later time steps may be unreliable.
- Regulation retrieval is weak (hybrid hit@3 about 0.6 on a small test set): retrieved passages are leads for an analyst, not verified authorities, and text from the India document comes from OCR.
- Scores are raw gradient-boosting outputs, not calibrated probabilities; the calibrated value shifts with the prevalence of illicit activity.
- Structural patterns are resemblances only: Elliptic has no typology labels.
- This is a draft for a human analyst. It does not decide whether a report should be filed.

**Recommendation:** refer for human review

**Critic verdicts**

- round 1, claim 1: OK - reviewer: supported (All facts in the claim match the provided evidence exactly.)
- round 1, claim 2: OK - reviewer: supported (The evidence lists features f53, f59, f3, f2, and f163 with the direction 'raises illicit score'.)
- round 1, claim 3: OK - reviewer: supported (The evidence confirms status ok, window recall 0.842105, and reference recall 0.98308 for time step 42.)
- round 1, claim 4: OK - reviewer: supported (The evidence shows in_degree of 3, out_degree of 1, and an empty matches list indicating no structural typology flags were matched.)
- round 1, claim 5: OK - reviewer: supported (The evidence shows total_found of 4, with one outgoing neighbor having a score of 0.992707 and alert set to true.)

## 3. flagged chain, steps 35-42 (tx 97870846)

Final status: **verified**

### Memo for transaction 97870846 (time step 40)

- status: **verified**, verification: **verified**
- risk band: low; raw score: 0.000323; calibrated probability: 0.0
- thresholds: {'alert': 0.258181, 'high': 0.788025}
- model health: {'status': 'ok', 'window_recall': 0.880435, 'reference_recall': 0.98308, 'n_illicit_in_window': 184}
- subject fields: names / addresses / amounts / dates = not available in dataset

**Activity description**

1. Model health check status was ok with a window recall of 0.880435 and reference recall of 0.98308. `[health:40]`
2. Feature f59 pushed the illicit score up. `[score:97870846]`
3. Feature f131 pushed the illicit score down. `[score:97870846]`
4. A chain pattern matched with an observed illicit rate of 0.257318 and a lift of 2.221913, which training-period data associate with a higher illicit rate. `[typology:97870846]`

**Regulatory references**

- FinCEN guidance mentions unusually complex series of transactions indicative of layering activity involving multiple accounts, banks, parties, and jurisdictions. Quote: "• unusually complex series of transactions indicative of layering activity involving multiple accounts, banks, parties, jurisdictions;" (FinCEN 2003: Guidance on Preparing A Complete & Sufficient Suspicious Activity Report Narrative; `reg:fincen_2003_sar_narrative:p7:c10`; quotable: True)

**Evidence table**

- `health:40` from check_model_health
- `score:97870846` from score_transaction
- `typology:97870846` from match_typology
- `reg:fincen_2003_sar_narrative:p7:c10` from search_regulations

**Limitations**

- The Elliptic dataset has no wallet addresses, names, amounts or dates, so none of these appear in this memo.
- The model's recall collapsed after time step 43 (a dark-market shutdown), and the model-health check only detects such a collapse several steps later, so scores at later time steps may be unreliable.
- Regulation retrieval is weak (hybrid hit@3 about 0.6 on a small test set): retrieved passages are leads for an analyst, not verified authorities, and text from the India document comes from OCR.
- Scores are raw gradient-boosting outputs, not calibrated probabilities; the calibrated value shifts with the prevalence of illicit activity.
- Structural patterns are resemblances only: Elliptic has no typology labels.
- This is a draft for a human analyst. It does not decide whether a report should be filed.

**Recommendation:** refer for human review

**Critic verdicts**

- round 1, claim 1: OK - reviewer: supported (The claim accurately reflects the score, score scale, and risk band found in the cited evidence.)
- round 1, claim 2: REJECTED - numbers not found in the cited evidence: 3
- round 1, claim 3: OK - reviewer: supported (The cited evidence matches the feature name, group, direction, and shap value stated in the claim.)
- round 1, claim 4: OK - reviewer: supported (The cited evidence confirms the local feature f76, direction, and shap value mentioned in the claim.)
- round 1, claim 5: OK - reviewer: supported (The claim is fully supported by the entry for feature f131 in the cited evidence.)
- round 1, claim 6: OK - reviewer: supported (The claim matches the chain length, threshold, and training-period association with the higher illicit rate specified in the evidence.)
- round 1, reference 1: REJECTED - reviewer: unsupported (the reviewer returned no verdict for this item)
- round 2, claim 1: OK - reviewer: supported (The evidence states status is ok, window_recall is 0.880435, and reference_recall is 0.98308.)
- round 2, claim 2: OK - reviewer: supported (The evidence lists feature f59 with direction raises illicit score.)
- round 2, claim 3: OK - reviewer: supported (The evidence lists feature f131 with direction lowers illicit score.)
- round 2, claim 4: OK - reviewer: supported (The evidence shows a chain pattern match with observed_illicit_rate 0.257318, lift 2.221913, and notes that training-period data provide the evidence direction.)
- round 2, reference 1: OK - reviewer: supported (The evidence text specifically includes the phrase 'unusually complex series of transactions indicative of layering activity involving multiple accounts, banks, parties, jurisdictions'.)

## 4. flagged fan_in (lowers_risk), steps 35-42, band at least medium, not chain (tx 70360850)

Final status: **verified**

### Memo for transaction 70360850 (time step 35)

- status: **verified**, verification: **verified**
- risk band: high; raw score: 0.999318; calibrated probability: 0.988235
- thresholds: {'alert': 0.258181, 'high': 0.788025}
- model health: {'status': 'ok', 'window_recall': 0.987261, 'reference_recall': 0.98308, 'n_illicit_in_window': 471}
- subject fields: names / addresses / amounts / dates = not available in dataset

**Activity description**

1. Transaction score is 0.999318, placing it in the high risk band and exceeding the alert threshold of 0.258181. `[score:70360850]`
2. Model health check shows status ok with a window recall of 0.987261. `[health:35]`
3. Feature f53 pushed the score up. `[score:70360850]`
4. Feature f161 pushed the score up. `[score:70360850]`
5. Feature f132 pushed the score up. `[score:70360850]`
6. Feature f59 pushed the score up. `[score:70360850]`
7. Feature f5 pushed the score up. `[score:70360850]`
8. The transaction matched the fan_in pattern with an in degree of 10, which matched training-period data associated with a lower illicit rate. `[typology:70360850]`

**Regulatory references**

- none

**Evidence table**

- `score:70360850` from score_transaction
- `health:35` from check_model_health
- `typology:70360850` from match_typology

**Limitations**

- The Elliptic dataset has no wallet addresses, names, amounts or dates, so none of these appear in this memo.
- The model's recall collapsed after time step 43 (a dark-market shutdown), and the model-health check only detects such a collapse several steps later, so scores at later time steps may be unreliable.
- Regulation retrieval is weak (hybrid hit@3 about 0.6 on a small test set): retrieved passages are leads for an analyst, not verified authorities, and text from the India document comes from OCR.
- Scores are raw gradient-boosting outputs, not calibrated probabilities; the calibrated value shifts with the prevalence of illicit activity.
- Structural patterns are resemblances only: Elliptic has no typology labels.
- This is a draft for a human analyst. It does not decide whether a report should be filed.

**Recommendation:** refer for human review

**Critic verdicts**

- round 1, claim 1: OK - reviewer: supported (The evidence explicitly confirms the score, risk band, and alert threshold.)
- round 1, claim 2: OK - reviewer: supported (The evidence shows the status is ok and window recall is 0.987261.)
- round 1, claim 3: OK - reviewer: supported (The evidence shows feature f53 raises illicit score.)
- round 1, claim 4: OK - reviewer: supported (The evidence shows feature f161 raises illicit score.)
- round 1, claim 5: OK - reviewer: supported (The evidence shows feature f132 raises illicit score.)
- round 1, claim 6: OK - reviewer: supported (The evidence shows feature f59 raises illicit score.)
- round 1, claim 7: OK - reviewer: supported (The evidence shows feature f5 raises illicit score.)
- round 1, claim 8: OK - reviewer: supported (The evidence matches fan_in with in_degree 10 and associates it with a lower illicit rate in training data.)

## 5. steps 43-49, band at least medium (tx 106417568)

Final status: **verified**

### Memo for transaction 106417568 (time step 44)

- status: **verified**, verification: **verified**
- risk band: medium; raw score: 0.451772; calibrated probability: 0.25
- thresholds: {'alert': 0.258181, 'high': 0.788025}
- model health: {'status': 'ok', 'window_recall': 0.82227, 'reference_recall': 0.98308, 'n_illicit_in_window': 467}
- subject fields: names / addresses / amounts / dates = not available in dataset

**Activity description**

1. Transaction score is 0.451772 with a calibrated probability of 0.25, placing it in the medium risk band and triggering an alert at time_step 44. `[score:106417568]`
2. Features f53, f59, f3, f55, and f100 pushed the score up. `[score:106417568]`
3. Model health check for time_step 44 reports status ok with a window recall of 0.82227, reference recall of 0.98308, 467 illicit instances in the window, and a lag of 2. `[health:44]`
4. Typology matching found no fan-in, fan-out, or chain flags, with an in-degree of 1, out-degree of 1, and chain length of 2. `[typology:106417568]`
5. Graph neighborhood analysis within one hop found a total of 2 neighbors, consisting of one outgoing alerted transaction and one incoming non-alerted transaction. `[neighbors:106417568:1:both]`

**Regulatory references**

- none

**Evidence table**

- `score:106417568` from score_transaction
- `health:44` from check_model_health
- `typology:106417568` from match_typology
- `neighbors:106417568:1:both` from get_neighbors

**Limitations**

- The Elliptic dataset has no wallet addresses, names, amounts or dates, so none of these appear in this memo.
- The model's recall collapsed after time step 43 (a dark-market shutdown), and the model-health check only detects such a collapse several steps later, so scores at later time steps may be unreliable.
- Regulation retrieval is weak (hybrid hit@3 about 0.6 on a small test set): retrieved passages are leads for an analyst, not verified authorities, and text from the India document comes from OCR.
- Scores are raw gradient-boosting outputs, not calibrated probabilities; the calibrated value shifts with the prevalence of illicit activity.
- Structural patterns are resemblances only: Elliptic has no typology labels.
- This is a draft for a human analyst. It does not decide whether a report should be filed.

**Recommendation:** refer for human review

**Critic verdicts**

- round 1, claim 1: OK - reviewer: supported (The score, calibrated probability, risk band, alert status, and time step are all explicitly stated in the evidence.)
- round 1, claim 2: OK - reviewer: supported (The evidence lists features f53, f59, f3, f55, and f100 with the direction 'raises illicit score' for each, which matches the claim.)
- round 1, claim 3: OK - reviewer: supported (Every detail in the claim regarding status, window recall, reference recall, n_illicit_in_window, and lag is directly supported by the evidence.)
- round 1, claim 4: OK - reviewer: supported (The evidence shows fan_in, fan_out, and chain are false, in-degree is 1, out-degree is 1, and chain_len is 2.)
- round 1, claim 5: OK - reviewer: supported (The evidence shows total_found is 2, with one outgoing neighbor that has alert=true and one incoming neighbor that has alert=false.)

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

**Regulatory references**

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
- This is a draft for a human analyst. It does not decide whether a report should be filed.

**Recommendation:** no action suggested, model-based only

**Critic verdicts**

- the critic did not run (deterministic path)

## Observations (written after reading the runs; nothing below was fixed)

- **No run hit a cap or failed.** 5 runs used the LLM (5 to 9 LLM calls, 2 to 4 investigator tool calls out of 8, 0 or 1 searches out of 3); the sixth took the deterministic no-action path with zero LLM calls (0.08 s). Latency was 30 to 83 s per LLM run, of which 22 to 36 s were deliberate waits for the free-tier rate limit (a call at most every 4.5 s), not model time.
- **Tx 97870846 (chain, run 3) needed one critic rewrite for two reasons.** In round 1 a claim was rejected by the deterministic number check (it contained a number, 3, that is not in the cited evidence), and the LLM reviewer returned no verdict for the last item, which the critic treats as unsupported (fail closed). The rewrite then passed. So one rewrite was caused by the reviewer's omission and not by a bad claim.
- **The critic checks support, not relevance.** The regulation reference in run 3 is a verbatim quote from the 2003 SAR guidance about "unusually complex series of transactions indicative of layering". It is a valid quote of a quotable chunk and the reviewer judged it supported, but it is only loosely related to a chain pattern: retrieval is weak and nothing in the critic asks whether the passage is the right one for this transaction.
- **Tx 97870846 has a low model score (0.0003) and a chain flag.** The memo reports both facts but does not discuss that they point in different directions; it still recommends human review because the chain pattern raised risk in training-period data.
- **Tx 106417568 is at time step 44, after the shutdown at 43, and the model-health check still says "ok"** (window recall 0.82), because the check lags the collapse by several steps (it first flags step 47). The memo contains no warning specific to this time step beyond the fixed limitations text, and the agent has no knowledge of the shutdown. A deterministic note for time steps at or after 43 would be a simple fix; it was NOT added (this is a design decision, not a prompt tweak).
- **Tx 99175682 is a licit node in the high band** (a false alarm by the model). The memo states the evidence correctly and recommends review; nothing in the evidence can show it is a false alarm, and none is claimed.
- **Ground truth is shown only for runs 1 and 2.** Nodes 3 to 6 are unlabeled ("unknown"), so these memos cannot be scored for correctness, only for grounding.
- **Style:** the fan_in memo (run 4) lists five features as five separate claims and words the fan_in claim awkwardly ("matched training-period data associated with a lower illicit rate"), but it never calls the pattern suspicious. Prompts were not edited to improve style.
- **Judge numbering:** the critic's verdict reasons are written by the LLM reviewer and sometimes refer to the wrong claim number in round 1 of run 3; the OK/REJECTED outcomes come from code.

## Prompt changes

None. No prompt, threshold or code behaviour was changed after seeing these runs.

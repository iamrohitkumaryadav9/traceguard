# Agent evaluation report (Phase 5a)

Pre-registered in `eval/agent_eval_spec.md` (committed before any case was selected or run). Arms: A0 = ML only (templated), A1 = the writer's first draft, A2 = the full system after the critic loops (A1 and A2 are paired, from the same run). Counts are shown next to percentages; intervals are 95% case-level bootstrap intervals (1,000 resamples, seed 42). This evaluation says nothing about whether the agents improve detection.

Note added in Phase 5b: the metric first called "groundedness" is now called "citation and number consistency". It shows a ceiling effect (100% in every arm) and shares a blind spot with the critic (the scientific-notation bug found in Phase 5a affected both), so it must not be read as evidence that the claims are faithful to the evidence; see `report_5b.md` for the independent judge and the fault-injection results.

Cases selected: 40; included in the aggregates: 40; failed or skipped: 0.

Of the 40 included cases, 27 reached the writer and 13 ended on the deterministic path (cannot_assess or no_action_model_based), where A1 and A2 are the same memo by construction.

## Headline (all included cases)

### Overall (40 cases)

| arm | citation and number consistency, all cases | citation and number consistency, writer cases | completeness (mean over cases) | violations a/b/c/d (cases) |
|---|---|---|---|---|
| A0 | 75/75 = 100% 100% [100%, 100%] | 54/54 = 100% 100% [100%, 100%] (n=27 cases) | 79% [75%, 83%] | 0 / 0 / 0 / 0 |
| A1 | 176/176 = 100% 100% [100%, 100%] | 142/142 = 100% 100% [100%, 100%] (n=27 cases) | 96% [93%, 98%] | 0 / 0 / 0 / 0 |
| A2 | 175/175 = 100% 100% [100%, 100%] | 141/141 = 100% 100% [100%, 100%] (n=27 cases) | 96% [93%, 98%] | 0 / 0 / 0 / 0 |

Paired A2 minus A1 (cases with both arms), case-level bootstrap:

- citation and number consistency (pooled share): 0% [0%, 0%]
- completeness (mean): 0% [-2%, 1%]

### Completeness by element

| element | A0 present/applicable | A1 present/applicable | A2 present/applicable |
|---|---|---|---|
| score | 35/35 = 100% | 35/35 = 100% | 35/35 = 100% |
| band | 35/35 = 100% | 35/35 = 100% | 35/35 = 100% |
| health status | 40/40 = 100% | 40/40 = 100% | 40/40 = 100% |
| typology evidence with direction | 0/17 = 0% | 11/17 = 65% | 11/17 = 65% |
| neighbour summary | 0/27 = 0% | 21/27 = 78% | 21/27 = 78% |
| signal conflict | 0/13 = 0% | 13/13 = 100% | 13/13 = 100% |
| period note | 5/8 = 62% | 8/8 = 100% | 8/8 = 100% |
| limitations | 40/40 = 100% | 40/40 = 100% | 40/40 = 100% |
| subject fields | 40/40 = 100% | 40/40 = 100% | 40/40 = 100% |

### Tool-call correctness

- R1 (a matched pattern implies match_typology was called): 17/17 = 100% of cases with a match.
- R2 as pre-registered (a regulation search only when a raises_risk pattern has a retrieval_query): 0 of 40 runs searched without a raises_risk pattern, of which 0 had only uninformative patterns (the declared mismatch: they also have a retrieval_query) and 0 had only lowers_risk patterns (must be 0).
- Within all caps: 40/40 = 100%; runs ended by a budget cap (budget_exceeded): 0.
- Invalid-argument rate: 0/82 = 0% of investigator and regulation tool calls (82 calls).

### Cost and latency

| arm | LLM calls (mean / total) | prompt tokens (total) | output tokens (total) | wall time s, mean (with waits) | wall time s, mean (without waits) |
|---|---|---|---|---|---|
| A0 | 0.0 / 0 | 0 | 0 | 0.0 | 0.0 |
| A1 | 3.3 / 132 | 224,360 | 16,993 | 20.1 | n/a (not recorded per event) |
| A2 | 4.3 / 173 | 304,036 | 31,289 | 26.8 | 8.0 |

## Policy violations (counted; case ids)

- **A0**: a_lowers_risk_called_suspicious: 0; b_low_risk_while_degraded: 0; c_quote_from_non_quotable: 0; d_label_leak: 0
- **A1**: a_lowers_risk_called_suspicious: 0; b_low_risk_while_degraded: 0; c_quote_from_non_quotable: 0; d_label_leak: 0
- **A2**: a_lowers_risk_called_suspicious: 0; b_low_risk_while_degraded: 0; c_quote_from_non_quotable: 0; d_label_leak: 0

Information only (not violations): A0 {'quotes_not_in_stored_text': 0, 'claims_with_quote_field': 0}; A1 {'quotes_not_in_stored_text': 0, 'claims_with_quote_field': 0}; A2 {'quotes_not_in_stored_text': 0, 'claims_with_quote_field': 0}

Violation details:

- none

## Critic statistics (A2 runs)

Runs that reached the writer and the critic: 27.

Critic rounds needed: 1 round(s): 21 runs, 2 round(s): 5 runs, 3 round(s): 1 runs

Runs with at least one rejected item: 6/27 = 22%.

Deterministic rejection reasons (counted per rejected item and reason):

- quote problem: 4

LLM-reviewer rejections: unsupported 2, overstated 5.

Reviewer false-rejection candidates (reason states the same numbers before and after 'but'; to be hand-verified in 5b): 0

## Per stratum

### 1 high band, illicit (6 cases)

| arm | citation and number consistency, all cases | citation and number consistency, writer cases | completeness (mean over cases) | violations a/b/c/d (cases) |
|---|---|---|---|---|
| A0 | 12/12 = 100% 100% [100%, 100%] | 12/12 = 100% 100% [100%, 100%] (n=6 cases) | 79% [75%, 83%] | 0 / 0 / 0 / 0 |
| A1 | 30/30 = 100% 100% [100%, 100%] | 30/30 = 100% 100% [100%, 100%] (n=6 cases) | 100% [100%, 100%] | 0 / 0 / 0 / 0 |
| A2 | 30/30 = 100% 100% [100%, 100%] | 30/30 = 100% 100% [100%, 100%] (n=6 cases) | 100% [100%, 100%] | 0 / 0 / 0 / 0 |

Cases: [73002915, 97132844, 70361272, 115924670, 70365734, 12779384]; statuses: {'verified': 6}

### 2 high band, licit (6 cases)

| arm | citation and number consistency, all cases | citation and number consistency, writer cases | completeness (mean over cases) | violations a/b/c/d (cases) |
|---|---|---|---|---|
| A0 | 12/12 = 100% 100% [100%, 100%] | 12/12 = 100% 100% [100%, 100%] (n=6 cases) | 79% [75%, 83%] | 0 / 0 / 0 / 0 |
| A1 | 30/30 = 100% 100% [100%, 100%] | 30/30 = 100% 100% [100%, 100%] (n=6 cases) | 100% [100%, 100%] | 0 / 0 / 0 / 0 |
| A2 | 30/30 = 100% 100% [100%, 100%] | 30/30 = 100% 100% [100%, 100%] (n=6 cases) | 100% [100%, 100%] | 0 / 0 / 0 / 0 |

Cases: [97930880, 78079432, 148546737, 97930510, 98544621, 116107565]; statuses: {'verified': 6}

### 3 medium band (4 cases)

| arm | citation and number consistency, all cases | citation and number consistency, writer cases | completeness (mean over cases) | violations a/b/c/d (cases) |
|---|---|---|---|---|
| A0 | 8/8 = 100% 100% [100%, 100%] | 8/8 = 100% 100% [100%, 100%] (n=4 cases) | 80% [74%, 83%] | 0 / 0 / 0 / 0 |
| A1 | 22/22 = 100% 100% [100%, 100%] | 22/22 = 100% 100% [100%, 100%] (n=4 cases) | 100% [100%, 100%] | 0 / 0 / 0 / 0 |
| A2 | 20/20 = 100% 100% [100%, 100%] | 20/20 = 100% 100% [100%, 100%] (n=4 cases) | 100% [100%, 100%] | 0 / 0 / 0 / 0 |

Cases: [13383872, 74206107, 114502915, 199949376]; statuses: {'verified': 4}

### 4 chain-flagged (6 cases)

| arm | citation and number consistency, all cases | citation and number consistency, writer cases | completeness (mean over cases) | violations a/b/c/d (cases) |
|---|---|---|---|---|
| A0 | 12/12 = 100% 100% [100%, 100%] | 12/12 = 100% 100% [100%, 100%] (n=6 cases) | 62% [62%, 62%] | 0 / 0 / 0 / 0 |
| A1 | 34/34 = 100% 100% [100%, 100%] | 34/34 = 100% 100% [100%, 100%] (n=6 cases) | 85% [77%, 96%] | 0 / 0 / 0 / 0 |
| A2 | 34/34 = 100% 100% [100%, 100%] | 34/34 = 100% 100% [100%, 100%] (n=6 cases) | 83% [77%, 92%] | 0 / 0 / 0 / 0 |

Cases: [29935560, 69777051, 198741596, 15452133, 73274568, 69763472]; statuses: {'verified': 6}

### 5 fan_in or fan_out flagged (6 cases)

| arm | citation and number consistency, all cases | citation and number consistency, writer cases | completeness (mean over cases) | violations a/b/c/d (cases) |
|---|---|---|---|---|
| A0 | 12/12 = 100% 100% [100%, 100%] | 2/2 = 100% 100% [100%, 100%] (n=1 cases) | 80% [73%, 83%] | 0 / 0 / 0 / 0 |
| A1 | 25/25 = 100% 100% [100%, 100%] | 5/5 = 100% 100% [100%, 100%] (n=1 cases) | 100% [100%, 100%] | 0 / 0 / 0 / 0 |
| A2 | 25/25 = 100% 100% [100%, 100%] | 5/5 = 100% 100% [100%, 100%] (n=1 cases) | 100% [100%, 100%] | 0 / 0 / 0 / 0 |

Cases: [7292103, 70781568, 12843031, 200528363, 2692055, 66666000]; statuses: {'no_action_model_based': 5, 'verified': 1}

### 6 low band (4 cases)

| arm | citation and number consistency, all cases | citation and number consistency, writer cases | completeness (mean over cases) | violations a/b/c/d (cases) |
|---|---|---|---|---|
| A0 | 8/8 = 100% 100% [100%, 100%] | 2/2 = 100% 100% [100%, 100%] (n=1 cases) | 91% [72%, 100%] | 0 / 0 / 0 / 0 |
| A1 | 14/14 = 100% 100% [100%, 100%] | 5/5 = 100% 100% [100%, 100%] (n=1 cases) | 94% [81%, 100%] | 0 / 0 / 0 / 0 |
| A2 | 15/15 = 100% 100% [100%, 100%] | 6/6 = 100% 100% [100%, 100%] (n=1 cases) | 97% [91%, 100%] | 0 / 0 / 0 / 0 |

Cases: [70002760, 71242667, 54925062, 28875554]; statuses: {'no_action_model_based': 3, 'verified': 1}

### 7 late steps (43-49), band at least medium (8 cases)

| arm | citation and number consistency, all cases | citation and number consistency, writer cases | completeness (mean over cases) | violations a/b/c/d (cases) |
|---|---|---|---|---|
| A0 | 11/11 = 100% 100% [100%, 100%] | 6/6 = 100% 100% [100%, 100%] (n=3 cases) | 82% [72%, 93%] | 0 / 0 / 0 / 0 |
| A1 | 21/21 = 100% 100% [100%, 100%] | 16/16 = 100% 100% [100%, 100%] (n=3 cases) | 93% [87%, 98%] | 0 / 0 / 0 / 0 |
| A2 | 21/21 = 100% 100% [100%, 100%] | 16/16 = 100% 100% [100%, 100%] (n=3 cases) | 93% [87%, 98%] | 0 / 0 / 0 / 0 |

Cases: [158612797, 105858009, 219546233, 190135471, 106335719, 188965393, 218299511, 105590801]; statuses: {'cannot_assess': 5, 'verified': 3}

## Failed or skipped cases

- none

## Selected cases

| stratum | tx_id | step | score | ground truth (not seen by the system) | status |
|---|---|---|---|---|---|
| 1 high band, illicit | 73002915 | 42 | 0.999662 | illicit | verified |
| 1 high band, illicit | 97132844 | 40 | 0.999968 | illicit | verified |
| 1 high band, illicit | 70361272 | 35 | 0.999992 | illicit | verified |
| 1 high band, illicit | 115924670 | 38 | 0.994961 | illicit | verified |
| 1 high band, illicit | 70365734 | 35 | 0.999982 | illicit | verified |
| 1 high band, illicit | 12779384 | 41 | 0.999729 | illicit | verified |
| 2 high band, licit | 97930880 | 40 | 0.979061 | licit | verified |
| 2 high band, licit | 78079432 | 39 | 0.926495 | licit | verified |
| 2 high band, licit | 148546737 | 39 | 0.965522 | licit | verified |
| 2 high band, licit | 97930510 | 40 | 0.832896 | licit | verified |
| 2 high band, licit | 98544621 | 40 | 0.993764 | licit | verified |
| 2 high band, licit | 116107565 | 38 | 0.908188 | licit | verified |
| 3 medium band | 13383872 | 41 | 0.563082 | unknown | verified |
| 3 medium band | 74206107 | 39 | 0.345277 | unknown | verified |
| 3 medium band | 114502915 | 38 | 0.371723 | unknown | verified |
| 3 medium band | 199949376 | 36 | 0.33322 | unknown | verified |
| 4 chain-flagged | 29935560 | 37 | 4.5e-05 | unknown | verified |
| 4 chain-flagged | 69777051 | 35 | 1.2e-05 | unknown | verified |
| 4 chain-flagged | 198741596 | 36 | 5.1e-05 | unknown | verified |
| 4 chain-flagged | 15452133 | 36 | 0.001587 | unknown | verified |
| 4 chain-flagged | 73274568 | 42 | 0.019788 | unknown | verified |
| 4 chain-flagged | 69763472 | 35 | 0.000318 | licit | verified |
| 5 fan_in or fan_out flagged | 7292103 | 40 | 0.000327 | unknown | no_action_model_based |
| 5 fan_in or fan_out flagged | 70781568 | 42 | 0.000431 | unknown | no_action_model_based |
| 5 fan_in or fan_out flagged | 12843031 | 40 | 9e-06 | licit | no_action_model_based |
| 5 fan_in or fan_out flagged | 200528363 | 36 | 5.2e-05 | licit | no_action_model_based |
| 5 fan_in or fan_out flagged | 2692055 | 36 | 1e-06 | licit | no_action_model_based |
| 5 fan_in or fan_out flagged | 66666000 | 38 | 0.854182 | unknown | verified |
| 6 low band | 70002760 | 35 | 0.000876 | unknown | no_action_model_based |
| 6 low band | 71242667 | 42 | 6e-06 | licit | no_action_model_based |
| 6 low band | 54925062 | 35 | 1.5e-05 | unknown | verified |
| 6 low band | 28875554 | 37 | 0.001844 | unknown | no_action_model_based |
| 7 late steps (43-49), band at least medium | 158612797 | 49 | 0.956072 | unknown | cannot_assess |
| 7 late steps (43-49), band at least medium | 105858009 | 44 | 0.296912 | unknown | verified |
| 7 late steps (43-49), band at least medium | 219546233 | 47 | 0.901456 | unknown | cannot_assess |
| 7 late steps (43-49), band at least medium | 190135471 | 49 | 0.934958 | illicit | cannot_assess |
| 7 late steps (43-49), band at least medium | 106335719 | 44 | 0.820817 | unknown | verified |
| 7 late steps (43-49), band at least medium | 188965393 | 49 | 0.558207 | unknown | cannot_assess |
| 7 late steps (43-49), band at least medium | 218299511 | 47 | 0.429414 | unknown | cannot_assess |
| 7 late steps (43-49), band at least medium | 105590801 | 44 | 0.400017 | unknown | verified |

## Limitations

Small n (40 cases, 4 to 8 per stratum), so intervals are wide and per-stratum results are anecdotal. One LLM (gemini-3.5-flash-lite) acts as investigator, writer and reviewer. Free-tier throttling inflates wall time. The test steps informed an earlier model-selection decision. The metrics are rules written by the system's author and check form and grounding, not relevance. A1 is the writer's first draft inside the same run, not a separately configured system. Completeness includes header, limitations and subject fields that code fills in for every arm.

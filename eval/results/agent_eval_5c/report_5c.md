# Phase 5c report: code-rendered facts (v3) versus LLM-paraphrased facts (v2)

Pre-registered in `eval/agent_eval_5c_spec.md` (spec and amendment committed before any run). Fresh set: seed 43, 30 nodes disjoint from the demo nodes and the Phase 5a set. Same model, frozen retrieval, v2 = tag `agent-v2`. Intervals: 95% case-level bootstrap (1,000 resamples, seed 42) and Wilson intervals for hand-check proportions. Nothing here says anything about detection quality.

Cases selected: 30; both designs finished and included in the paired comparison: 30.

## Headline (paired cases)

| design | citation and number consistency | completeness (mean) | violations a/b/c/d (cases) | code-built / LLM-authored statements | LLM calls (mean / total) | prompt tokens | output tokens | wall s, mean without waits |
|---|---|---|---|---|---|---|---|---|
| v2 | 128/128 = 100% 100% [100%, 100%] | 97% [94%, 99%] | 0 / 0 / 0 / 0 | 28 / 100 | 4.4 / 131 | 214,908 | 22,842 | 12.6 |
| v3 | 235/235 = 100% 100% [100%, 100%] | 100% [100%, 100%] | 0 / 0 / 0 / 0 | 174 / 61 | 4.8 / 143 | 179,121 | 20,703 | 7.3 |

Paired completeness difference, v3 minus v2 (mean): 0.03 [0.01, 0.06]

**Caveat that governs the completeness number (amendment, written before any run).** For v3 the completeness metric is near-tautological, because code inserts the required items. It measures the design guarantee, not LLM quality. The keep rule below is applied as pre-registered, and this caveat belongs to its conclusion.

### Completeness by element

| element | v2 present/applicable | v3 present/applicable |
|---|---|---|
| score | 28/28 = 100% | 28/28 = 100% |
| band | 28/28 = 100% | 28/28 = 100% |
| health status | 30/30 = 100% | 30/30 = 100% |
| typology evidence with direction | 5/8 = 62% | 8/8 = 100% |
| neighbour summary | 17/21 = 81% | 21/21 = 100% |
| signal conflict | 9/9 = 100% | 9/9 = 100% |
| period note | 8/8 = 100% | 8/8 = 100% |
| limitations | 30/30 = 100% | 30/30 = 100% |
| subject fields | 30/30 = 100% | 30/30 = 100% |

## Keep rule (pre-registered: adopt v3 if mean completeness improves by at least 0.05 and there are no new policy violations)

- completeness gain +0.031 (threshold 0.05): not met
- new policy violations (present in v3, absent in v2 for the same case and rule): 0 
- **rule outcome: do not adopt v3 on this rule**, with the completeness caveat above: the gain reflects what code guarantees, not an improvement in LLM quality.

## Policy violations (case ids)

- **v2**: a_lowers_risk_called_suspicious: 0; b_low_risk_while_degraded: 0; c_quote_from_non_quotable: 0; d_label_leak: 0
- **v3**: a_lowers_risk_called_suspicious: 0; b_low_risk_while_degraded: 0; c_quote_from_non_quotable: 0; d_label_leak: 0

Violation details:

- none

## Critic statistics

- **v2**: 21 runs reached the writer; runs with a rejected item: 4; deterministic reasons {'quote problem': 4, 'feature claims not merged': 2}; reviewer rejections unsupported 1, overstated 3
- **v3**: 21 runs reached the writer; runs with a rejected item: 8; deterministic reasons {'other deterministic': 9, 'unknown evidence id': 1}; reviewer rejections unsupported 2, overstated 9

## Hand-check (pre-registered checks (a) and (b); sheet `handcheck_5c.csv`)

Not yet scored: fill in the `human_verdict` column of `handcheck_5c.csv`, then run `python -m eval.agent_eval_5c.handcheck_5c score`.

## Selected cases

| stratum | tx_id | step | score | ground truth (not seen by the system) | v2 | v3 |
|---|---|---|---|---|---|---|
| 1 high band, illicit | 73002538 | 42 | 0.999979 | illicit | verified | verified |
| 1 high band, illicit | 12664247 | 41 | 0.999157 | illicit | verified | verified |
| 1 high band, illicit | 71987847 | 42 | 0.999825 | illicit | verified | verified |
| 1 high band, illicit | 69662492 | 35 | 0.999978 | illicit | verified | verified |
| 2 high band, licit | 149570741 | 39 | 0.88761 | licit | verified | verified |
| 2 high band, licit | 71769835 | 42 | 0.927049 | licit | verified | verified |
| 2 high band, licit | 73333415 | 42 | 0.911319 | licit | verified | verified |
| 2 high band, licit | 97933782 | 40 | 0.885416 | licit | verified | verified |
| 3 medium band | 61745735 | 42 | 0.400043 | unknown | verified | partially_verified |
| 3 medium band | 30122458 | 37 | 0.550728 | unknown | verified | verified |
| 3 medium band | 29584495 | 37 | 0.259718 | unknown | verified | partially_verified |
| 4 chain-flagged | 29055204 | 37 | 0.000426 | unknown | verified | verified |
| 4 chain-flagged | 200950509 | 36 | 0.127512 | unknown | partially_verified | verified |
| 4 chain-flagged | 69779650 | 35 | 0.000146 | unknown | verified | verified |
| 4 chain-flagged | 64413379 | 35 | 0.000126 | licit | verified | verified |
| 5 fan_in or fan_out flagged | 18477792 | 42 | 7e-06 | unknown | no_action_model_based | no_action_model_based |
| 5 fan_in or fan_out flagged | 114656654 | 38 | 6.2e-05 | licit | no_action_model_based | no_action_model_based |
| 5 fan_in or fan_out flagged | 72097077 | 42 | 4e-06 | licit | no_action_model_based | no_action_model_based |
| 5 fan_in or fan_out flagged | 12669334 | 41 | 2e-06 | licit | no_action_model_based | no_action_model_based |
| 6 low band | 200638883 | 36 | 0.15882 | unknown | no_action_model_based | no_action_model_based |
| 6 low band | 72624896 | 42 | 0.000843 | unknown | no_action_model_based | no_action_model_based |
| 6 low band | 44282463 | 39 | 0.129222 | unknown | no_action_model_based | no_action_model_based |
| 7 late steps (43-49), band at least medium | 109871125 | 44 | 0.873789 | unknown | verified | verified |
| 7 late steps (43-49), band at least medium | 155603453 | 45 | 0.609283 | unknown | verified | verified |
| 7 late steps (43-49), band at least medium | 125406909 | 47 | 0.838595 | unknown | cannot_assess | cannot_assess |
| 7 late steps (43-49), band at least medium | 106300125 | 44 | 0.999862 | unknown | verified | verified |
| 7 late steps (43-49), band at least medium | 196092852 | 46 | 0.998845 | unknown | verified | verified |
| 7 late steps (43-49), band at least medium | 156058072 | 45 | 0.999748 | unknown | verified | verified |
| 7 late steps (43-49), band at least medium | 261387954 | 44 | 0.766879 | unknown | verified | verified |
| 7 late steps (43-49), band at least medium | 218683036 | 47 | 0.981565 | unknown | cannot_assess | cannot_assess |

## Limitations

Small n (30 nodes, 3 to 8 per stratum), so intervals are wide. One LLM acts as investigator, writer and reviewer. Metrics are rules written by the system's author. Completeness for v3 is guaranteed by construction (see the caveat). The deterministic consistency metric shares a blind spot with the critic; the hand-check is the independent measure.

## Post-hoc decision (written after the runs; NOT part of the pre-registered rule outcome above)

The rule outcome above stands unchanged: the pre-registered keep rule was **not met** (completeness gain +0.031 against a threshold of 0.05). This section lists evidence for and against shipping v3 as the default design anyway. It is a judgement made after seeing the results, on 30 nodes, and it is not a result of the pre-registered test. Details of every rejection are in `rejection_diagnosis.md`. The hand-check (pre-registered checks (a) and (b)) is not yet filled in, so it is not evidence either way.

**For shipping v3**
- No policy violations in either design (0 in all four rules), so v3 introduces none.
- The element completeness gains are in the elements v2 got wrong: typology evidence with direction 8/8 versus 5/8, neighbour summary 21/21 versus 17/21. These are the wording problems found in Phase 5b, now made impossible by construction (with the caveat that this is a design guarantee, not LLM quality).
- 174 of 235 memo statements are rendered by code from the tool outputs (v2: 28 of 128), so fewer statements depend on the LLM being right.
- Of the critic rejections, v2's were mostly genuine LLM errors (8 of 10: a fabricated pattern match, unscoped neighbour counts, whole-preview quotes ending in "...", an overgeneralized regulatory claim); v3's were mostly not (2 of 21 genuine).
- Prompt tokens 179k versus 215k (-17%), output tokens 20.7k versus 22.8k, mean wall time without waits 7.3 s versus 12.6 s.

**Against shipping v3 (or for waiting)**
- The pre-registered rule was not met, and the one metric it uses is near-tautological for v3, so the shortfall is not evidence about LLM quality either way.
- Citation and number consistency is 100% in both designs, so it cannot separate them.
- v3 had more runs with a rejected item (8 versus 4) and two partially_verified runs against one for v2. In v3 the rejections are mostly from the critic itself being too strict: reviewer objections to correct sentences (14 of 21 rejected items), a next-check rule that fires on "hop 1" and on feature identifiers like f53 (9 items), and one reviewer verdict that contradicts its own reason. Both v3 partially_verified outcomes cost a summary sentence, one of them (61745735) to an unsupported objection. These were left unfixed on purpose.
- v3 made slightly more LLM calls (143 versus 131 in total) because of the extra rewrite rounds those rejections trigger.
- The summary sentences are still LLM-authored (61 of them); their error rate (check (b)) and the code-built text's disagreement with its evidence (check (a)) are unmeasured until the hand-check is filled in.
- n = 30 with 9 deterministic-path nodes where the two designs are identical by construction, so the comparison rests on 21 nodes; intervals are wide and the interval on the completeness gain, [0.01, 0.06], straddles the threshold.

**Not decided here.** Whether to ship v3 as the default is left to the author after the hand-check; nothing in the system was changed in response to these results.

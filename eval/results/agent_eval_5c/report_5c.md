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

- (a) code-built claims whose text disagrees with their cited evidence: 0/20 = 0.0%, 95% Wilson interval [0.0%, 16.1%]
- (b) LLM-authored executive-summary sentences with an error: 1/61 = 1.6%, 95% Wilson interval [0.3%, 8.7%]
- The deterministic consistency metric called 1 of the sentences the human did not mark supported grounded.
- H042 (llm-authored, overstated): "Feature f59 pushed the score up and features f5, f52 pushed the score down, presenting a signal conflict with the structural pattern chain which raises risk with an illicit rate of 0.257318 against a baseline of 0.115809." Reader's note: Directions of f59/f5/f52 are correct, but the claim attributes the 'signal conflict' to the mixed feature contributions; the evidence only supports a conflict between the low score (0.000146, not stated) and the chain pattern. f53 and f100 also raised the score.

These rates are not comparable to the Phase 5b development-set rate (3 of 79). No claim is made that v3 is more accurate than v2; the finding is only that two error classes (neighbour and subgraph counts merged or missing hop and direction, and an LLM paraphrase of regulation text) cannot occur in v3 by construction. One human read the sheet, blinded to the claim source.

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

The rule outcome above stands unchanged: the pre-registered keep rule was **not met** (completeness gain +0.031 against a threshold of 0.05). This section lists evidence for and against shipping v3 as the default design anyway. It is a judgement made after seeing the results, on 30 nodes, and it is not a result of the pre-registered test. Details of every critic rejection are in `rejection_diagnosis.md`; the next-check rule bugfix (agent-v3.1) and its verification are in `../agent_eval_5c_bugfix/bugfix_report.md`. The headline numbers in this report are the agent-v3 run.

**For shipping v3**
- No policy violations in either design (0 in all four rules), so v3 introduces none.
- The element completeness gains are in the elements v2 got wrong: typology evidence with direction 8/8 versus 5/8, neighbour summary 21/21 versus 17/21. These are the wording problems found in Phase 5b, now impossible by construction (a design guarantee, not LLM quality).
- 174 of 235 memo statements are rendered by code from the tool outputs (v2: 28 of 128), so fewer statements depend on the LLM being right.
- Hand-check: 0 of 20 code-built claims wrong, and 1 of 61 LLM-authored summary sentences overstated (see the hand-check section above).
- Of the critic rejections, v2's were mostly genuine LLM errors (8 of 10); v3's were mostly not (2 of 21 genuine).
- Prompt tokens 179k versus 215k (-17%), output tokens 20.7k versus 22.8k, mean wall time without waits 7.3 s versus 12.6 s.

**Against shipping v3 (or for waiting)**
- The pre-registered rule was not met, and its one metric is near-tautological for v3, so the shortfall is not evidence about LLM quality either way.
- Citation and number consistency is 100% in both designs, so it cannot separate them. It also called the one wrong summary sentence (H042) grounded.
- v3 had more runs with a rejected item (8 versus 4) and two partially_verified runs against one for v2. Most v3 rejections come from the critic being too strict (14 of 21 items) or from a next-check rule that fired on "hop 1" and feature ids (9 items); the rule has since been fixed (agent-v3.1), the reviewer has not been touched.
- v3 made slightly more LLM calls (143 versus 131 in total) because of the extra rewrite rounds those rejections trigger.
- The hand-check has one human reader, 20 code-built claims and 61 summary sentences; its rates are not comparable to the Phase 5b rate (3 of 79, development set), and it cannot show that v3 is more accurate than v2.
- n = 30, of which 9 nodes take the deterministic path where the two designs are identical by construction, so the comparison rests on 21 nodes; the interval on the completeness gain, [0.01, 0.06], straddles the threshold.

**Decision.** Phase 6 builds on agent-v3.1. This is a judgement, not a test result: the reasons are the by-construction removal of two error classes (neighbour and subgraph counts merged or missing hop and direction, and LLM paraphrase of regulation text) and the lower cost, not a measured accuracy gain.

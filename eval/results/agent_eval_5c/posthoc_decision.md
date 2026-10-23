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

**Decision.** Phase 6 builds on agent-v3.1. This is a judgement, not a test result: the reasons are the by-construction removal of two error classes and the lower cost, not a measured accuracy gain.

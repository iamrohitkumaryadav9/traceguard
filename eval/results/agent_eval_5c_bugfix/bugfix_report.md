# Phase 5c bugfix verification (agent-v3.1): the next-check digit rule

**This is a bugfix verification, not an evaluation.** The 8 nodes below are the nodes whose agent-v3 runs had critic rejections; they were used for the diagnosis (`../agent_eval_5c/rejection_diagnosis.md`), so no claim about generalisation is made. The headline Phase 5c numbers remain the agent-v3 run (`../agent_eval_5c/report_5c.md`). One run per node, same model and frozen retrieval; the LLM is not deterministic, so a node's second run can differ for reasons other than the fix.

**Change (only this).** Next checks must contain no number, as before, except that feature ids (`f53`), hop references (`hop 1`, `2 hops`) and time step references (`time step 44`) are allowed. Any other number is still rejected (`score 0.9`, `5 neighbours`, `0.25`, `hop 1.5`). Code tagged `agent-v3` (commit `10c6711`) produced the Phase 5c runs; the fix is tagged `agent-v3.1` (commit `55ff160`). The reviewer, prompts and thresholds are unchanged.

## Rejections by class, same 8 nodes, before (agent-v3) and after (agent-v3.1)

Classes as defined in `rejection_diagnosis.md`; the 10 new rows were classified by the same reader after reading the new ledgers and drafts.

| | rejected items | genuine LLM error | check too strict | possible check bug | next-check rejections | runs with a rejection |
|---|---|---|---|---|---|---|
| before (agent-v3) | 21 | 2 | 14 | 5 | 9 | 8 of 8 |
| after (agent-v3.1) | 10 | 2 | 8 | 0 | 2 | 3 of 8 |

The 2 next-check rejections that remain are correct: 155603453 wrote "outgoing neighbour at hop 1 with score 0.783844" and "component of size 5598" in next checks, which are numbers about the transaction. The 8 'too strict' rows are all reviewer objections to correct sentences (106300125: two health/subgraph sentences, 155603453: one health sentence, 61745735: five rows, e.g. omitting f2 and f138 from a list of the top 3 features, "undirected" and "delayed-label" wording the code itself uses). The reviewer was not changed.

## Final statuses, same 8 nodes

| tx_id | agent-v3 | agent-v3.1 |
|---|---|---|
| 106300125 | verified | verified |
| 149570741 | verified | verified |
| 155603453 | verified | verified |
| 200950509 | verified | verified |
| 29584495 | partially_verified | verified |
| 61745735 | partially_verified | partially_verified (2 reviewer-rejected summary sentences removed) |
| 71987847 | verified | verified |
| 97933782 | verified | verified |

Before: 6 verified, 2 partially_verified. After: 7 verified, 1 partially_verified. Total LLM calls on the 8 nodes: 71 before, 52 after (writer rounds 20 before, 13 after). The change on 29584495 may be partly the fix and partly run-to-run LLM variation: its v3 run had a "hop 1" next-check rejection in round 1 that used up a rewrite round, and its sentence was finally removed for a genuine mixed-scope error; the new run did not repeat that sentence. One run per node cannot separate the two.

## What this does and does not show

It shows that the rule no longer rejects feature ids and hop references (0 'possible check bug' rows remain, and the 9 earlier next-check rejections fell to 2, both correct). It does not show that the critic is better overall: reviewer over-strictness is untouched and accounts for 8 of the 10 remaining rejections, and 61745735 is still partially_verified for that reason.

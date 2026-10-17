# Phase 5b report

Pre-registered in `eval/agent_eval_5b_spec.md`. Measurement only: no system code, prompt or threshold was changed.

## A. LLM judge (faithfulness, independent of the critic)

Writer runs: 27. Judge model(s): `gemini-3.5-flash-lite` (93 unique claims), `gemini-3.8-flash` (16 unique claims). Models tried in order: ['gemini-3.8-flash', 'gemini-3.5-flash-lite']. Daily-quota switches: ['gemini-3.8-flash', 'gemini-3.5-flash-lite']. STOPPED: all candidate models exhausted.

The judge saw only each claim's text and the evidence it cites (one call per unique claim and evidence pair), with its own rubric; it never saw critic verdicts, other claims or labels. Counts are over the claims of each arm (identical claims in A1 and A2 are counted in both).

| arm | claims in the arm | claims judged | supported (of judged) | unsupported (of judged) | overstated (of judged) | not judged |
|---|---|---|---|---|---|---|
| A1 | 142 | 89 | 88/89 = 99% | 1/89 = 1% | 0/89 = 0% | 53 |
| A2 | 141 | 90 | 89/90 = 99% | 1/90 = 1% | 0/90 = 0% | 51 |

**Judge independence and coverage:** 93 of the 109 judged unique claims were judged by `gemini-3.5-flash-lite`, which is also the writer's model and the critic's reviewer (the pre-registered fallback, because the stronger candidates were not usable: the pro model's quota, `gemini-2.5-pro` removed, `gemini-3.8-flash` 503 at the first probe). The other 16 were judged by a different model (`gemini-3.8-flash`) after the first model's daily quota ran out. The judge is therefore NOT independent of the writer for most claims. 53 claims of A1 and 51 of A2 were not judged because the daily quotas of both models were exhausted; the run can be resumed after the quota resets (see `BUGLOG_5b.md`).

Claims the judge did not mark supported:

- tx 73002915 (A1+A2): **unsupported** - Neighborhood analysis found 1 incoming neighbor and 0 outgoing neighbors within 1 hop. (The evidence only provides information for incoming neighbors and does not contain any data regarding outgoing neighbors.)

## B. Hand-check sheet (to be filled in by a human)

`eval/results/agent_eval/handcheck.md` (and `handcheck.csv`): 15 runs (the writer runs in which the critic changed the memo plus 9 random others, seed 42). The sheet is blinded: claim text and cited evidence only. Fill in the `human_verdict` column of the CSV (`supported`, `unsupported` or `overstated`), do not open `handcheck_key.json` first, then run:

```
python -m eval.agent_eval_5b.handcheck score
```

It prints human-versus-judge agreement (percent and Cohen's kappa), human-versus-deterministic agreement (the check of the 100% ceiling of the Phase 5a citation and number consistency metric) and the critic's rejection precision.

## C. Critic fault injection (critic only, no writer calls)

**Coverage: 129 of 140 fault copies and 19 of 20 control copies were run. The run STOPPED: daily quota exhausted: 429 RESOURCE_EXHAUSTED (the free tier allows 500 requests per day per model); it is resumable (`python -m eval.run_5b faults`) once the quota resets, so the counts below are partial.**

20 verified A2 memos (seed 42): [12779384, 13383872, 29935560, 54925062, 66666000, 69763472, 70365734, 73002915, 73274568, 74206107, 97132844, 97930510, 97930880, 98544621, 105590801, 114502915, 115924670, 148546737, 198741596, 199949376]. One programmatic fault per copy; the critic node alone (with the system's own reviewer, gemini-3.5-flash-lite) judged each tampered draft, round 1. The injected faults are synthetic and probably easier to catch than natural errors; the reviewer is the same model family as the writer; counts per fault type are small.

| fault | applicable | caught by the deterministic layer | caught by the LLM reviewer only | missed | caught by either (95% Wilson interval) |
|---|---|---|---|---|---|
| F1 numeric tamper | 19 applicable of 19 run (planned 20) | 19 | 0 | 0 | 19/19 = 100% [83%, 100%] |
| F2 fabricated evidence_id | 19 applicable of 19 run (planned 20) | 19 | 0 | 0 | 19/19 = 100% [83%, 100%] |
| F3 polarity flip | 3 applicable of 19 run (planned 20) | 0 | 3 | 0 | 3/3 = 100% [44%, 100%] |
| F4 unsupported claim, valid id | 18 applicable of 18 run (planned 20) | 0 | 18 | 0 | 18/18 = 100% [82%, 100%] |
| F5 overstated | 18 applicable of 18 run (planned 20) | 0 | 18 | 0 | 18/18 = 100% [82%, 100%] |
| F6 fabricated quote | 4 applicable of 18 run (planned 20) | 4 | 0 | 0 | 4/4 = 100% [51%, 100%] |
| F7 dropped signal_conflict | 8 applicable of 18 run (planned 20) | 8 | 0 | 0 | 8/8 = 100% [68%, 100%] |

**Control** (the 19 untampered originals through the same critic): 0 of 19 copies had at least one item rejected (false-rejection rate 0%, 95% interval [0%, 17%]); item level: 0 of 108 items rejected. Copies with a rejection: [].

Missed faults:

- none

## Limitations

About 27 writer runs and about 150 unique claims; one author wrote the system, the metrics and this evaluation. The judge is a Gemini model, like the writer and the critic's reviewer (a stronger one if the log above says so), and sees chunk text that can be truncated to 1,200 characters. The human check covers 15 runs and one person. Injected faults are templates.

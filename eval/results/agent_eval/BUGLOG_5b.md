# Phase 5b incident log

## 1. Judge model: the stronger candidates were not usable

Probes at the start (the pre-registered order): `gemini-3.1-pro-preview` returned 429 (quota), `gemini-2.5-pro` returned 404 (no longer available to new users), `gemini-3.8-flash` returned 503 (high demand, apparently transient). By the rule, the judge started on the writer's own model `gemini-3.5-flash-lite` with the separate rubric (the spec's fallback and a stated limitation).

## 2. Free-tier daily quota exhausted (500 requests per day per model)

- `gemini-3.5-flash-lite` reached its daily limit (500 requests per project per day, free tier; used by the 5a evaluation, the 5b judge fallback and the 5b fault injection together). The judge stopped with 93 of 171 unique claims judged. The fault injection (the critic's reviewer is the same model) crashed on the same error after 129 of 140 fault copies and 19 of 20 control copies, because the runner did not catch the quota error. Fix, in evaluation code only: the runner now stops cleanly, records why, and is resumable.
- The judge was resumed after re-probing: `gemini-3.8-flash` now answered, so, as the spec says, the remaining claims moved to the next candidate. It judged 16 more claims and then hit its own daily quota. Result: 109 of 171 unique claims judged (93 by `gemini-3.5-flash-lite`, 16 by `gemini-3.8-flash`), 62 not judged. The judge verdict counts in the report are therefore partial, and mix two models.
- Nothing was retried in a loop. To finish: after the daily quotas reset, run `python -m eval.run_5b judge` and `python -m eval.run_5b faults` (both resume from their saved results), then `python -m eval.run_5b report` and `python -m eval.agent_eval_5b.handcheck build` (this refreshes only the key file if the hand-check CSV already has human entries).

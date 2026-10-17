# Agent evaluation: pre-registered specification (Phase 5a)

This file is committed BEFORE any case is selected or any agent run is made for this evaluation. Nothing below is changed in response to results; if something here turns out to be wrong, the correction is written in `eval/results/agent_eval/BUGLOG.md` (what, which case ids, why), not edited here.

## What is and is not being evaluated

- Evaluated: whether the investigation system produces memos that are grounded in tool evidence, complete against a fixed checklist, free of a fixed list of policy violations, and cheap enough to run, and how much the critic changes the writer's first draft.
- NOT evaluated and NOT claimed: that the agents improve detection of illicit transactions. The ML model decides the score and band; the agents only investigate and write. No metric below measures whether a memo's recommendation is right about the transaction.
- Not evaluated here: memo quality as judged by a human (planned for a later phase).

## Cases

Forty transaction nodes from test time steps 35-49, drawn with `numpy.random.default_rng(42)` (one generator for the whole selection, strata taken in the order below). For each stratum the candidates are the node ids that qualify, sorted ascending, minus the six demo nodes from Phases 4a/4b (12779397, 99175682, 97870846, 70360850, 106417568, 149159647) and minus nodes already chosen for an earlier stratum; then `rng.choice(len(candidates), size=k, replace=False)`. If a stratum has fewer than k candidates, all of them are used and the report says so.

| # | Stratum | Definition | k |
|---|---|---|---|
| 1 | high band, illicit | steps 35-42, raw score >= the high threshold (0.788, validation-F1), ground-truth label illicit | 6 |
| 2 | high band, licit | steps 35-42, raw score >= the high threshold, ground-truth label licit | 6 |
| 3 | medium band | steps 35-42, alert threshold (0.2582) <= raw score < high threshold, any label including unknown | 4 |
| 4 | chain-flagged | steps 35-42, chain_len >= 4, any band and label | 6 |
| 5 | fan_in or fan_out flagged | steps 35-42, in-degree >= the fan-in threshold or out-degree >= the fan-out threshold (99th percentile over steps 1-29, minimum 3), any band and label | 6 |
| 6 | low band | steps 35-42, raw score < alert threshold, any label | 4 |
| 7 | late steps | steps 43-49, raw score >= alert threshold | 8 |

Ground truth is used only to apply strata 1 and 2 and is stored for the report; it is never shown to the system (no tool returns a label, and a test checks that flipping a node's label changes no prompt, trace or memo). Stratum 7 mixes steps whose model-health check is still "ok" (43-46) with steps where it says degraded (47-49): the second kind ends as `cannot_assess` without any LLM call, by design.

## Arms

- **A0, ML only.** A deterministic templated memo from the triage output, with no agent and no LLM: the header (raw score, risk band, thresholds, model health), one score claim and one health claim, the fixed limitations, the "not available in dataset" subject fields, and a recommendation by rule (low band with healthy model: "no action suggested, model-based only"; otherwise "refer for human review"). If the health check says degraded, A0 is the same `cannot_assess` memo as the full system (no rating). A0 has no typology claims, no neighbour summary, no signal conflict and no period note, by definition.
- **A1, agents without critic.** The writer's first draft (round 0, before any critic feedback) turned into a memo by the same assembly code, without any filtering. The writer's own `signal_conflict` text is used as written (it is not replaced by the required sentence). Header, limitations, period note and subject fields are code-built and identical in all arms.
- **A2, full system.** The final memo after the critic loops (at most two rewrites; claims that still fail are removed and the memo is marked "partially verified"). If the writer never carried the required `signal_conflict` sentence, code inserts it and the memo says so.

A1 and A2 come from the SAME run for each case, so they are paired. A1 is the writer's first draft, not a separately configured system (it ran with the same investigator and regulation steps and the same prompt, and it did not see critic feedback). For cases that end without the writer (`cannot_assess`, `no_action_model_based`), A1 and A2 are the same deterministic memo and are identical by construction; the report states how many cases these are and also gives numbers for the writer cases only.

Cases whose run fails for a non-budget reason (`agent_error`, `unknown_transaction`, a quota failure) are excluded from all aggregates (all arms, to keep the pairing) and are listed with their error. `budget_exceeded` runs are included with their partial memo and counted.

## Metrics (all deterministic)

They are computed by `eval/agent_eval/metrics.py`, which does NOT import the critic code (`src/agents/checks.py`, `src/agents/nodes/critic.py`) and has its own number and phrase matching. The same person wrote both, so the independence is of code, not of design.

1. **Groundedness.** Over the claims of an arm's memo (the activity claims and the regulatory-background items): the share of claims that (a) cite at least one evidence_id, (b) every cited id exists in the run's evidence ledger and is not a tool error, and (c) every number in the claim text (feature names such as f57 excluded; percentages compared to value x 100; compared after rounding to the claim's own number of decimals; absolute values) appears among the numeric values of the cited outputs. Reported as claims-grounded / claims-total, pooled over cases, for all cases and for writer cases only.
2. **Completeness.** Per case, the share of applicable required elements that are present; then the mean over cases. Elements and applicability: score (header score present; not applicable when cannot_assess), band (same), health status (header; always), typology evidence with direction (applicable when the triage typology output has at least one match; present when a claim citing the typology evidence id names every matched pattern together with a direction cue: higher/raises for raises_risk, lower/lowers for lowers_risk, uninformative/no risk signal for uninformative), neighbour summary (applicable when the case reached the writer; present when a claim cites a `neighbors:` or `subgraph:` evidence id), signal conflict (applicable when triage computed one; present when the memo's signal_conflict equals the required sentence exactly), period note (applicable when step >= 43; present in the limitations), limitations (all fixed limitation items present), subject fields (all four equal "not available in dataset"). Element-level counts are reported too.
3. **Policy violations (counted, not averaged; case ids listed).**
   - (a) a lowers_risk pattern called suspicious or risk-raising: a claim names a lowers_risk pattern of that case and contains a suspicion cue (suspicious, risk-raising, raises risk, red flag, indicative of, concerning, layering, laundering, warrants, elevated risk, higher risk) without a negation or lowering cue (not, no, never, without, lower, lowers) in the same claim;
   - (b) a "low risk" statement or equivalent while health is degraded: for degraded cases, a populated risk band or score in the header, the recommendation "no action suggested, model-based only", or a claim matching low risk, risk band low, unlikely, benign, not suspicious or nothing suspicious;
   - (c) a quote from a non-quotable document: any quote (a background item's quote field, or a quotation of 12 or more characters in any claim) attributed to a chunk whose `quotable` flag is false in the ledger;
   - (d) a label leak: a ledger output, prompt or trace event containing a key named label, labels, class, ground_truth or y_true; or a claim stating the transaction itself is illicit or licit ("is/was (an) illicit/licit", "labeled", "confirmed illicit").
   Information-only counts (not violations): quotes that are not verbatim substrings, and claims that carry a quote field.
4. **Tool-call correctness** (per run, then rates and counts): (i) a matched pattern implies match_typology was called (true by construction in triage; checked in the trace); (ii) rule R2 as written: a regulation search happens only when a raises_risk pattern has a retrieval_query. KNOWN MISMATCH, declared in advance: the system runs the regulation step whenever any matched pattern has a non-null retrieval_query, and "uninformative" patterns have one too, so searches triggered by an uninformative pattern only will count as R2 violations; they are reported separately from searches with only lowers_risk patterns (which must be 0); (iii) counters within the caps (8 investigator tool calls, 3 searches, 3 chunk fetches, 2 critic loops, 30 LLM calls), and the number of budget_exceeded runs; (iv) invalid-argument rate: the share of investigator and regulation tool calls whose output is an invalid_argument error.
5. **Cost and latency**: LLM calls, prompt and output tokens, and wall time with and without the deliberate rate-limit waits, per arm. A0 has zero. A1 cost is everything before the first critic event; its wall time is the time of the writer's first draft (with waits included, since the waits are not recorded per event). A2 is the whole run (wall time with and without waits).
6. **Critic statistics** (A2 runs that reached the critic): critic rounds needed (distribution), rejection reasons by type, split into deterministic reasons (uncited; unknown evidence id; tool-error evidence; numbers not in evidence; lowers_risk described as suspicious; quote problems; feature meaning or direction; feature claims not merged; claims carrying quotes; signal_conflict missing, altered or invented; background not from a pattern-triggered search) and LLM-reviewer verdicts (unsupported, overstated), and the reviewer false-rejection candidates: reviewer rejections whose reason contradicts its own evidence, flagged by a simple check (the numbers stated before "but" in the reason equal the numbers stated after it). Candidates are only candidates; they are to be hand-verified in Phase 5b.

## Reporting rules

- Counts next to percentages everywhere (for example 41/46 claims = 89%).
- Bootstrap 95% confidence intervals: 1,000 resamples, seed 42, resampling CASES with replacement (so that claims within a case are not treated as independent), recomputing the pooled ratio or the mean each time; paired A2-minus-A1 differences use the same resampled cases.
- A per-stratum breakdown for every metric. With 4 to 8 cases per stratum the intervals are wide and are shown as they are.
- No claim about detection accuracy of the agents, no significance tests beyond the intervals, and no metric added or dropped after seeing the data. Any additional analysis is labelled exploratory.

## Run protocol

- `eval/run_agent_eval.py` selects the cases (writes `eval/results/agent_eval/cases.json`), runs each case once with `gemini-3.5-flash-lite` at temperature 0 and the frozen retrieval, and writes one result file per (case, arm) under `eval/results/agent_eval/`. It is resumable (finished cases are skipped). Free-tier per-minute limits are waited out; a daily-quota error is recorded for the case and stops the run cleanly, saying where it stopped. A case that failed for another reason is retried once on a later invocation.
- No prompt, threshold or code path is changed in response to results. A bug in the evaluation or the system is fixed, logged in `eval/results/agent_eval/BUGLOG.md` with the affected case ids, and only those cases are re-run (or re-scored if only the metrics code was wrong).
- Step 0 before this evaluation (already done, committed with the runs): claims have no `quote` field; only regulatory background may carry a quote.

## Limitations (known in advance)

- Small n: 40 cases, 4 to 8 per stratum, so intervals are wide and per-stratum results are anecdotal.
- One LLM (gemini-3.5-flash-lite) acts as investigator, writer and reviewer, so the critic's LLM check and the writer share a model family and probably blind spots.
- Free-tier throttling inflates wall time; the "without waits" figure removes only the deliberate waits.
- The test time steps (35-49) informed an earlier model-selection decision, so they are not a pristine holdout for the ML model; the nodes are only used here to exercise the memo pipeline.
- Metrics are regular-expression and number-matching rules written by the same author as the system; they check form and grounding, not whether a claim is relevant or insightful.
- A1 is the writer's first draft within the same run, not an independently configured no-critic system.
- Regulation retrieval is weak (see `eval/results/retrieval_final.md`); background items are leads, not proof.
- Completeness counts header, limitations and subject fields that code fills in for every arm; the arms differ mainly in typology, neighbour and conflict content.

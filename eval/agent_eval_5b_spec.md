# Phase 5b: pre-registered specification (judge, hand-check sheet, critic fault injection)

Committed BEFORE anything below is run. This phase is measurement only: no system code, prompt or threshold is changed. The only edit to existing text is the label rename in the Phase 5a report described in section D. If this file turns out to be wrong, the correction goes in a log (`eval/results/agent_eval/BUGLOG_5b.md`), not into this file.

Data: the 40 Phase 5a cases (`eval/results/agent_eval/`). "Writer runs" are the 27 runs that reached the writer. "A1" is the writer's first draft and "A2" the final memo after the critic, as defined in `eval/agent_eval_spec.md`.

## A. LLM judge (faithfulness, independent of the critic)

**Model choice (rule fixed now).** The models available to this account were listed with the API (`models.list`, generateContent-capable Gemini models). The writer and the critic's reviewer use `gemini-3.5-flash-lite`. The judge must be a different and, if possible, stronger model. Candidates in this order, each tried with a one-word probe call, the first that answers is used: (1) `gemini-3.1-pro-preview` (pro tier; a preview model), (2) `gemini-2.5-pro`, (3) `gemini-3.8-flash` (the newest flash model, newer and larger than flash-lite), (4) the writer's own model with the separate rubric below, which would be reported as a limitation. If the chosen model's daily quota runs out before all claims are judged, the remaining claims use the next candidate and the report states which model judged which claims (counts per model).

**What the judge sees.** For each claim of A1 and of A2 in each of the 27 writer runs (activity claims, and regulatory-background items as paraphrase plus the cited chunk): the claim text and ONLY the evidence it cites, taken from the run's evidence ledger (tool outputs as compact JSON; for regulation chunks, the stored chunk text, which can be truncated to 1,200 characters in the ledger). One call per claim. The judge never sees the critic's verdicts, other claims, the memo, or any ground-truth label. The code-built `signal_conflict` sentence is not judged (it is a fixed template). Identical (claim, cited evidence) pairs are judged once and reused for both arms.

**Rubric (its own prompt, not the critic's).** `supported`: every fact in the statement is explicitly in the cited evidence and the numbers are equal; `unsupported`: the statement contains a fact that is not in the cited evidence or contradicts it, including a reversed direction; `overstated`: the evidence is related but the statement is more certain, broader or more causal than the evidence allows. Output: verdict plus a one-line reason. Temperature 0.

**Reported:** verdict counts (not just percentages) per arm and in total, the number of claims judged, and which model judged them.

## B. Hand-check sheet (to be filled in by a human)

- **Runs (15):** the writer runs in which the critic changed the memo (at least one rejected item in any critic round; 6 in Phase 5a), plus 9 others drawn at random with seed 42 (`numpy.random.default_rng(42).choice` over the sorted ids of the other writer runs, without replacement).
- **Blinded:** each row shows the claim text and the cited evidence text only. No judge verdict, no critic reasoning and no deterministic result is shown in `handcheck.md` or `handcheck.csv`. They are kept in a separate key file (`handcheck_key.json`) that the scoring script reads.
- **Critic-changed runs (6):** the sheet lists every distinct claim that appeared in any critic round (first draft, intermediate drafts, final), shows the first draft and the final claims side by side, and marks which claims the critic rejected (only that fact, without its reason), so the human can mark whether each rejection was correct.
- **Boxes:** every claim gets one verdict column to fill in with `supported`, `unsupported` or `overstated`, and a notes column.
- **Scoring script** (`python -m eval.agent_eval_5b.handcheck score`), run once the CSV is filled in: human-versus-judge agreement (percent, and Cohen's kappa, on the three classes and on supported-versus-not), human-versus-deterministic agreement (the Phase 5a "groundedness" check reduced to grounded / not grounded, against human supported / not supported; this directly tests the 100% ceiling), and critic rejection precision (the share of critic-rejected claims that the human marked unsupported or overstated).

## C. Critic fault injection (critic only; no writer calls)

- **Memos:** 20 verified A2 memos drawn with seed 42 (`default_rng(42).choice` over the sorted ids of verified writer runs, without replacement; all of them if fewer than 20). The reviewer is the same one the system uses (`gemini-3.5-flash-lite`).
- **Faults, injected programmatically (no LLM), one fault per copy of the original final draft:**
  - F1 numeric tamper: in the first claim that contains a number, change its first number to a different value that is not in the cited evidence (integers +7, repeated +1 until absent; decimals x1.37 at the same precision).
  - F2 fabricated evidence_id: the first claim cites `score:0`, which is not in the ledger.
  - F3 polarity flip: in the first claim that names fan_in or fan_out (otherwise the chain claim) and contains "lower illicit rate", "higher illicit rate", "lowers_risk" or "raises_risk", flip it to the opposite phrase. Not applicable if no such claim exists.
  - F4 unsupported claim with a valid id: append the claim "This transaction is linked to a known darknet market." citing `score:<tx_id>`.
  - F5 overstated: replace the first claim's text with "The model score confirms and proves that this transaction is part of illicit activity." keeping its evidence ids.
  - F6 fabricated quote: for the first regulatory-background item on a quotable chunk, set its quote to its original quote (or the first 25 characters of the chunk text if it had none) followed by " fabricated words", so it is no longer a substring. Not applicable if there is no such item.
  - F7 dropped signal_conflict: set it to null, only on cases where one exists.
- **Run:** the critic node alone (`src/agents/nodes/critic.py`, unchanged) on each tampered draft with the real evidence ledger and the real regulation tool; round 1 only. For F1 to F6 the injected item is the item that counts; for F7 the signal_conflict check.
- **Outcome per applicable copy:** `caught by the deterministic layer` (the injected item has deterministic rejection reasons), `caught by the LLM reviewer only` (no deterministic reason, reviewer verdict unsupported or overstated), or `missed` (the item passes). Counts per fault type with 95% Wilson intervals for "caught by either layer".
- **Control:** the 20 untampered originals through the same critic. False-rejection rate = copies with at least one rejected item / 20, plus the item-level rate.
- **State plainly in the report:** the faults are synthetic and probably easier to catch than natural errors; the critic's reviewer is the same model family as the writer; applicable counts are small.

## D. Report

`eval/results/agent_eval/report_5b.md` holds sections A and C and the instructions for the hand-check script. The Phase 5a report text is left as it is, with two exceptions: the metric label "groundedness" is renamed "citation and number consistency" everywhere in the 5a report (generator and output), and a note is added that the metric shows a ceiling effect (100% in every arm) and shares a blind spot with the critic (the scientific-notation bug of Phase 5a affected both).

## Limitations (known in advance)

- 27 writer runs and about 150 unique claims: small, and one author wrote the system, the metrics and this evaluation.
- Judge and writer are both Gemini models; a stronger model reduces but does not remove shared blind spots, and the judge sees possibly truncated chunk text.
- The human check covers 15 runs, filled in by one person.
- Injected faults are templates; real errors are subtler.
- Free-tier rate limits and daily quotas may force the judge model fallback described in section A.

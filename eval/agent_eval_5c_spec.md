# Phase 5c: facts rendered by code, evaluated on fresh nodes (pre-registered specification)

Committed BEFORE any code change for Phase 5c and before any evaluation run. The code as it stands when this file is committed is tagged `agent-v2` (see "Reproducing v2"). If something below turns out to be wrong, the correction goes in `eval/results/agent_eval_5c/BUGLOG_5c.md`, not into this file.

## Why (what the Phase 5b hand check found)

A human checked 110 memo claims (15 runs) and flagged 7: three neighbour claims that dropped a hop or direction qualifier (13383872-05, 114502915-06, 73002915-05), two regulation paraphrases that over-generalised or presented a retrieval query as a finding (54925062-04 / 54925062-06 / 69777051-04, of which 54925062-04 and 69777051-04 are paraphrase errors and 54925062-06 is the query shown as a finding), one invented "multiple jurisdictions" (54925062-04), and one claim with a missing citation (114502915-11). Every one was an LLM paraphrase of a tool fact. The critic caught 4 of the 7, the LLM judge 2, the deterministic metric 1. The design response in this phase is to stop letting an LLM paraphrase tool facts: facts are rendered by code from the tool outputs.

## Step 1: why `get_neighbors` said 3 and `subgraph_summary` said 6 (run 106335719)

The two tools define the neighbourhood differently, by design. The saved ledger of the run holds `get_neighbors(direction both, hops 2)` with `total_found` 3 (nodes 106017670 incoming at hop 1, 106335716 outgoing at hop 1, 106748149 outgoing at hop 2) and `subgraph_summary(radius 2)` with `n_nodes` 6. Recomputing both sets on the real graph gives the difference exactly:

- **`get_neighbors` (`src/tools/neighbors.py`, `expand` and `get_neighbors`) follows directed chains only.** For direction `both` it computes `in_map = expand(ctx.AT, ...)` (predecessor-of-predecessor chains) and `out_map = expand(ctx.A, ...)` (successor-of-successor chains) separately and returns their union; a hop-2 neighbour is reached by in-in or out-out, never by in-then-out or out-then-in. It excludes the focal node.
- **`subgraph_summary` (`src/tools/subgraph.py`, line `nodes = sorted([i, *expand(ctx.S, i, args.radius)])`) uses the undirected adjacency `ctx.S = A + A.T` (`src/tools/context.py`).** Radius 2 means every node within two undirected steps, so mixed paths count, and `n_nodes` includes the focal node itself.
- **The numbers:** undirected radius 2 reaches 5 other nodes: the 3 above plus 106335717 and 106335718, which are two other predecessors (co-parents) of the focal node's successor 106335716 (path: focal, then out to 106335716, then back in from 106335717 or 106335718: a mixed path). Counting the focal node gives 6. So 6 = 3 (directed chains) + 2 (mixed paths) + 1 (the focal node).

Consequence for the templates (step 2): each tool's claim states its own parameters ("N incoming neighbours at 1 hop", "subgraph of radius 2, undirected, includes the focal node: N nodes") and the counts of the two tools are never merged or compared.

## Step 2: design changes (exactly these; v3)

The design is selected by `config.DESIGN` ("v2" keeps the previous behaviour so that the old tests still run; "v3" is the new one and the default). Retrieval, thresholds, gating and every prompt except what these changes require are untouched.

a) **Neighbour summary, built by code.** One claim per `get_neighbors` output in the evidence ledger and one per `subgraph_summary` output, citing the output's evidence id. A `get_neighbors` claim states the node, the direction, the hop limit, the total found, and lists the highest-scoring neighbours each with its own direction, hop, score and alert status (using only values in the output; counts are never derived). A `subgraph_summary` claim states the radius, that the subgraph is undirected and includes the focal node and mixed paths, and the output's own numbers.
b) **Typology evidence, built by code.** For every matched pattern one claim with the pattern, the evidence direction (raises_risk, lowers_risk or uninformative, in words), n, the observed illicit rate, the baseline and the lift from the output, citing the typology evidence id. Required whenever a pattern matched. The retrieval query is never shown.
c) **Regulatory background, rendered by code, no LLM paraphrase.** For every matched pattern with a retrieval query, over all regulation chunks present in the ledger (the regulation step is unchanged), code builds one BM25 index over the sentences of those chunks (tokenisation of `src/rag/retriever.py`, sentence split of `src/rag/ingest.py`) and takes the sentence with the highest BM25 score for the query (ties: first chunk in ledger order, then first sentence; a best score of 0 renders nothing). For a quotable chunk it renders a verbatim excerpt (at most 40 words, cut at a word boundary, "..." appended outside the quotation when cut) with document title, publisher, year, page, chunk id and the quotable flag. For a non-quotable chunk it renders only "Retrieved guidance: <document title>, <page>, <chunk id>" with no text. Exact chunk text is fetched with `get_regulation_chunk` when the ledger copy is truncated. Identical excerpts for two patterns are rendered once.
d) **Score, band, health, period note, signal conflict:** unchanged (code-built or templated). The score, SHAP-feature and health claims use the existing templates.
e) **The writer LLM** now writes only (i) an executive summary of at most 3 sentences restating the code-built facts, each sentence with evidence ids, and (ii) a list "what a human analyst could check next" with no factual claims. The schema offered to the LLM contains only these two fields; the code-built claims and the rendered background are attached by code after the call. The critic checks (i) with the existing deterministic checks (citations, numbers, quotes not allowed, lowers_risk never suspicious, anonymous features) plus its LLM support check; checks that (ii) contains no numbers and no claims about lowers_risk patterns being suspicious; and recomputes the code-built claims and the rendered background from the ledger and checks that the draft's copies are present and unaltered (a mismatch is recorded as a rejected item, the code-built version is restored in the memo, the memo is marked "partially verified" and a note says so; the writer is not asked to fix it, since it did not write them).

## Step 4: fresh evaluation set

Thirty nodes from test steps 35-49, seed 43 (`numpy.random.default_rng(43)`, one generator, strata in order; candidates sorted ascending, minus every excluded node, minus nodes already chosen; `rng.choice(len, size=k, replace=False)`; all candidates if a stratum is short, and the report says so). Excluded: the six demo nodes (12779397, 99175682, 97870846, 70360850, 106417568, 149159647) and the 40 Phase 5a nodes (`eval/results/agent_eval/cases.json`). Strata and definitions are those of `eval/agent_eval_spec.md`:

| # | Stratum | k |
|---|---|---|
| 1 | high band, illicit, steps 35-42 | 4 |
| 2 | high band, licit, steps 35-42 | 4 |
| 3 | medium band, steps 35-42 | 3 |
| 4 | chain-flagged, steps 35-42 | 4 |
| 5 | fan_in or fan_out flagged, steps 35-42 | 4 |
| 6 | low band, steps 35-42 | 3 |
| 7 | steps 43-49, band at least medium | 8 |

Ground truth is used only for selecting strata 1 and 2 and is never shown to the system. The earlier 40 nodes were DEVELOPMENT data (the design changes were made after looking at them); these 30 are the held-out check. The test steps also informed an earlier model-selection decision, so they are not a pristine holdout for the ML model; they only exercise the memo pipeline.

## Step 5: arms and metrics

Both designs are run once on the same 30 nodes with the same model (`gemini-3.5-flash-lite`, temperature 0), the same frozen retrieval and the same data: **v2** = the code at the git tag `agent-v2`, **v3** = the code with `DESIGN = "v3"`. The compared memo is the final memo (after the critic loops). Pairing is by node. Cases that end without the writer (cannot_assess, no_action_model_based) are identical in both designs by construction and are reported, together with numbers for the writer cases only. Cases whose run fails for a non-budget reason are excluded from all aggregates and listed.

1. **Completeness** (checklist of Phase 5a: score, band, health status, typology evidence with direction, neighbour summary, signal conflict, period note, limitations, subject fields), paired v2 versus v3. Pre-registered caveat: in v3 several elements are present by construction (the code renders them whenever the facts exist), so the completeness difference is partly built in; that is why the keep rule below is combined with the checks on errors.
2. **Critic rounds, rewrites, LLM calls, tokens, latency** (wall time without the deliberate throttle waits), paired.
3. **Policy violations** (the four rules of Phase 5a), counted.
4. **Citation and number consistency** (the Phase 5a metric, formerly "groundedness"), reported with its ceiling caveat (it is near 100% by construction and shares a blind spot with the critic).
5. **Share of memo claims that are code-built versus LLM-authored** (per memo: activity claims, regulatory-background items and executive-summary sentences; code-built = rendered by code, LLM-authored = written by the writer).

Keep rule (fixed now): **v3 is adopted as the final design if the mean completeness improves by at least 0.05 over v2 and v3 has no new policy violations** (no violation category with more violating cases under v3 than under v2). Otherwise v3 is reported as not adopted and the default design reverts to v2. Intervals are 95% case-level bootstrap intervals (1,000 resamples, seed 42) as in Phase 5a; paired differences use the same resampled cases. Counts are shown next to percentages. No claim is made about detection accuracy.

## Step 6: hand-check sheet for the LLM-authored claims

`eval/results/agent_eval_5c/handcheck_5c.csv` (and a markdown copy) with ONLY the LLM-authored claims of the v3 runs: the executive-summary sentences (each with its cited evidence text). Blinded like Phase 5b: claim text and cited evidence only; no critic, judge or deterministic verdicts (kept in a separate key file); one verdict column (`supported`, `unsupported`, `overstated`) and a notes column, to be filled in by a human. A scoring script mirrors the Phase 5b one. The next-checks list is not a set of claims and is not in the sheet.

## Run protocol

- A resumable runner writes one result file per (case, design) under `eval/results/agent_eval_5c/`, records quota errors for the case without retrying forever (the free tier allows 500 requests per day per model), and stops cleanly saying where it stopped. A full run may take two days.
- No prompt, threshold or code path is changed in response to results. A bug is fixed, logged in `BUGLOG_5c.md` with the affected node ids, and only those nodes are re-run (or re-scored if only the metrics code was wrong).
- The runs are NOT started until the project owner says so (the day's quota is mostly spent).

## Reproducing v2

The tag `agent-v2` points at the commit that contained the final Phase 5b code (commit `c7a412e`). To run v2: `git worktree add ../traceguard-v2 agent-v2`, then, because `data/`, `models/` and the regulation PDFs are not in git, make the worktree see the same files (for example directory junctions for `data` and `models`), and run the evaluation runner from that directory.

## Limitations (known in advance)

- 30 nodes, 3 to 8 per stratum: intervals are wide and per-stratum results are anecdotal.
- One LLM family acts as investigator, writer and reviewer; the support check shares its blind spots.
- Completeness is partly built in for v3 (see metric 1); the code-rendered claims are checked by construction, so the remaining risk sits in the LLM-written summary and next-checks, which is what the hand-check sheet targets.
- Retrieval is weak (see `eval/results/retrieval_final.md`): a code-selected excerpt is verbatim but may still be only loosely relevant; the memo labels it as background, not as evidence about the transaction.
- Metrics are rules written by the system's author; the earlier bug found in them (scientific notation) shows they can share blind spots with the critic.
- Free-tier throttling inflates wall time; "without waits" removes only the deliberate waits.

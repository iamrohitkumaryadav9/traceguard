# Phase 5b report

Pre-registered in `eval/agent_eval_5b_spec.md`. Measurement only: no system code, prompt or threshold was changed.

## A. LLM judge (faithfulness, independent of the critic)

Writer runs: 27; unique claims (a claim and its cited evidence, shared by A1 and A2 when identical): 171, all judged (171). Judge model(s): `gemini-3.5-flash-lite` (142 unique claims), `gemini-3.8-flash` (29 unique claims). Models tried in order: ['gemini-3.8-flash', 'gemini-3.5-flash-lite']; the first one's daily quota ran out and the next candidate took over (switches: ['gemini-3.8-flash']). 

The judge saw only each claim's text and the evidence it cites (one call per unique claim and evidence pair), with its own rubric; it never saw critic verdicts, other claims or labels. Counts are over the claims of each arm (identical claims in A1 and A2 are counted in both).

| arm | claims in the arm | judged | supported | unsupported | overstated |
|---|---|---|---|---|---|
| A1 | 142 | 142 | 139/142 = 98% | 3/142 = 2% | 0/142 = 0% |
| A2 | 141 | 141 | 139/141 = 99% | 2/141 = 1% | 0/141 = 0% |

**Judge independence:** 142 of the 171 unique claims (83%) were judged by `gemini-3.5-flash-lite`, which is also the writer's model and the critic's reviewer (the pre-registered fallback, because the stronger candidates were not usable at the start: the pro model's quota, `gemini-2.5-pro` removed, `gemini-3.8-flash` returned 503 at the first probe). The other 29 were judged by `gemini-3.8-flash`. For most claims the judge is therefore NOT independent of the writer.

Claims the judge did not mark supported:

- tx 73002915 (A1+A2): **unsupported** (gemini-3.5-flash-lite) - Neighborhood analysis found 1 incoming neighbor and 0 outgoing neighbors within 1 hop. (judge's reason: The evidence only provides information for incoming neighbors and does not contain any data regarding outgoing neighbors.)
- tx 114502915 (A1): **unsupported** (gemini-3.5-flash-lite) - At hop 2 outbound, 2 neighbors were found, including one with a score of 0.873609 that triggered an alert and another with a score of 0.2489 (judge's reason: The statement claims that both neighbors were found at hop 2 outbound, but the evidence shows that the second neighbor was found at hop 1 outbound.)
- tx 115924670 (A1+A2): **unsupported** (gemini-3.5-flash-lite) - Network neighborhood analysis found connected transactions including an incoming neighbor with an alert status and an outgoing neighbor with (judge's reason: The evidence shows that the incoming neighbor has an alert status and the outgoing neighbor has a score of 0.999983, but the statement reverses the directions for these attributes.)

## B. Hand-check results (human)

A human filled in the blinded sheet for 15 runs (110 claims): 6 runs in which the critic changed the memo and 9 random others (seed 42). Output of `python -m eval.agent_eval_5b.handcheck score`:

```json
{
  "rows_filled": 110,
  "judge": {
    "n": 109,
    "agreement": 0.9541284403669725,
    "kappa_3class": 0.42691903259726616,
    "kappa_supported_vs_not": 0.42328042328042326
  },
  "deterministic": {
    "n": 109,
    "agreement": 0.944954128440367,
    "kappa": 0.22877358490566035,
    "deterministic_grounded_but_human_not_supported": 5,
    "deterministic_flagged_but_human_supported": 1
  },
  "critic_rejection_precision": {
    "rejections_marked": 11,
    "human_agrees_it_was_not_supported": 4,
    "precision": 0.36363636363636365,
    "with_recorded_evidence_only": [
      3,
      10
    ]
  }
}
```

- **Human versus judge:** 95% agreement on 109 claims that had a judge verdict (Cohen's kappa 0.43 on three classes, 0.42 supported-versus-not). The high agreement comes from almost every claim being supported; the kappa is moderate.
- **Human versus the deterministic citation and number consistency check:** 94% agreement on 109 claims, kappa 0.23; the check called 5 claims grounded that the human did not mark supported, and flagged 1 that the human marked supported. This is the direct test of the 100% ceiling of the Phase 5a metric.
- **Critic rejection precision:** 11 claims were rejected by the critic in the runs it changed, the human agrees that 4 of them were not supported (precision 36%; 3 of 10 when only claims with recorded evidence are counted).

**Summary (computed from the key file, the hand-check sheet and `judge/verdicts.json`).** The human flagged 7 of 110 claims (5 unsupported, 2 overstated). The critic rejected 11 claims (the human agrees with 4) and missed 3 of the 7. The judge caught 2 of the 7 (1 had no recorded evidence and could not be judged) with 1 false positive. The deterministic metric caught 1 of the 7 (1 had no recorded evidence) with 1 false positive.

### The human-flagged claims

| claim | run | human verdict | human's note | critic rejected | judge flagged | deterministic flagged |
|---|---|---|---|---|---|---|
| 54925062-04: FATF guidance discusses indicators involving multiple layering of fiat funds prior to vir... | 54925062 | unsupported | Evidence says accounts at FIs 'across the jurisdiction' (one); 'multiple jurisdictions' is not stat... | yes | no | yes |
| 54925062-06: The transaction matched a chain structural pattern indicating sequential transfers throug... | 54925062 | overstated | 'sequential transfers ... layering of funds' is the tool's retrieval_query, not a finding; caveat s... | no | no | no |
| 69777051-04: FATF guidance notes that multiple layering of fiat funds prior to virtual asset purchases... | 69777051 | overstated | Evidence is one case study; claim generalises it into a rule ('can indicate ML') and adds chains of... | yes | no | no |
| 114502915-06: At hop 2 outbound, 2 neighbors were found, including one with a score of 0.873609 that tr... | 114502915 | unsupported | Only one of the 2 neighbours is at hop 2 (0.873609); the 0.248901 one is at hop 1. Rejection correc... | yes | yes (unsupported) | no |
| 114502915-11: Model health status was ok at time step 38 with a window recall of 0.988095. | 114502915 | unsupported | No cited evidence recorded; content would match health:38 if cited. Rejection correct (missing cita... | yes | n/a (no evidence recorded) | n/a (no evidence recorded) |
| 13383872-05: Neighborhood analysis found 3 incoming neighbors and 2 outgoing neighbors. | 13383872 | unsupported | The 2 outgoing are from a 2-hop query (only 1 is at hop 1); claim drops the hop qualifier, so 'outg... | no | no | no |
| 73002915-05: Neighborhood analysis found 1 incoming neighbor and 0 outgoing neighbors within 1 hop. | 73002915 | unsupported | Cited evidence only covers incoming neighbours (direction 'in'); '0 outgoing' is not in it. | no | yes (unsupported) | no |

**Judge false positive: 115924670-05 (tx 115924670).** Claim: "Network neighborhood analysis found connected transactions including an incoming neighbor with an alert status and an outgoing neighbor with a score of 0.999983." The judge marked it **unsupported**; the human marked it **supported**. The judge's reason: "The evidence shows that the incoming neighbor has an alert status and the outgoing neighbor has a score of 0.999983, but the statement reverses the directions for these attributes." First cited evidence entry: `[neighbors:115924670:1:both] (get_neighbors) {"tx_id":115924670,"time_step":38,"hops":1,"direction":"both","total_found":2,"truncated":false,"neighbors":[{"tx_id":116217174,"hop":1,"direction":"out","score":0.999983,"alert":true},{"tx_id":115133562,"hop":1,"direction":"in","score":0.972439,"alert":true}],"evidence_id":"neighbors:115924670:1:both"}` Read against that entry, the claim is stated as the evidence has it (compare the direction and the alert and score values of the in and out neighbours), so the judge's reason does not match the cited evidence: it is an error by the judge, not by the memo.

**Deterministic false positive: 15452133-06 (tx 15452133).** The metric flagged this regulatory-background item (7 numbers such as 150, 108, 352 "not in cited outputs"), the human marked it supported. The numbers are inside the quoted regulation passage; the metric only compares numbers with the numeric values of tool outputs and does not search numbers inside the chunk text.

### Contrast with the synthetic fault injection

In the injected-fault test (section C) the critic caught 98 of 98 applicable faulty copies (140 copies were run, 42 faults were not applicable to their memo) and rejected 0 of 20 untampered controls. On the 7 real errors flagged by the human it rejected 4 and missed 3, and across 110 real claims it made 7 rejections the human did not agree with. Synthetic faults are much easier to catch than the natural errors, which were mostly subtle: a claim that goes beyond the cited evidence, or evidence of a different scope than the claim.

**Disclosure.** The human saw which run's claim had been flagged (tx 73002915) in a status message before scoring, so that run was not blinded.

**Limitations of the hand check.** One human did the check. The 15 runs were not a random sample: 6 were chosen because the critic changed them, and only 9 were drawn at random (seed 42). Only 7 claims were flagged by the human, so every rate here (agreement, kappa, precision, recall) has a very wide interval and should be read as an anecdote.

## C. Critic fault injection (critic only, no writer calls)

**Coverage: 140 of 140 fault copies and 20 of 20 control copies were run (98 fault copies were applicable to their memo, 42 were not).**

20 verified A2 memos (seed 42): [12779384, 13383872, 29935560, 54925062, 66666000, 69763472, 70365734, 73002915, 73274568, 74206107, 97132844, 97930510, 97930880, 98544621, 105590801, 114502915, 115924670, 148546737, 198741596, 199949376]. One programmatic fault per copy; the critic node alone (with the system's own reviewer, gemini-3.5-flash-lite) judged each tampered draft, round 1. The injected faults are synthetic and probably easier to catch than natural errors; the reviewer is the same model family as the writer; counts per fault type are small.

| fault | applicable | caught by the deterministic layer | caught by the LLM reviewer only | missed | caught by either (95% Wilson interval) |
|---|---|---|---|---|---|
| F1 numeric tamper | 20 of 20 | 20 | 0 | 0 | 20/20 = 100% [84%, 100%] |
| F2 fabricated evidence_id | 20 of 20 | 20 | 0 | 0 | 20/20 = 100% [84%, 100%] |
| F3 polarity flip | 4 of 20 | 0 | 4 | 0 | 4/4 = 100% [51%, 100%] |
| F4 unsupported claim, valid id | 20 of 20 | 0 | 20 | 0 | 20/20 = 100% [84%, 100%] |
| F5 overstated | 20 of 20 | 0 | 20 | 0 | 20/20 = 100% [84%, 100%] |
| F6 fabricated quote | 5 of 20 | 5 | 0 | 0 | 5/5 = 100% [57%, 100%] |
| F7 dropped signal_conflict | 9 of 20 | 9 | 0 | 0 | 9/9 = 100% [70%, 100%] |

**Control** (the 20 untampered originals through the same critic): 0 of 20 copies had at least one item rejected (false-rejection rate 0%, 95% interval [0%, 16%]); item level: 0 of 114 items rejected. Copies with a rejection: [].

Missed faults:

- none

## Limitations

27 writer runs and 171 unique claims; one author wrote the system, the metrics and this evaluation. For most claims the judge is the same model as the writer and the critic's reviewer (see section A). The judge sees chunk text that can be truncated to 1,200 characters. The human check covers 15 runs, one person, and only 7 flagged claims (section B). Injected faults are templates and easier than natural errors.

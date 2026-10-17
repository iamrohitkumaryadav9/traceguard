# TraceGuard

An ML model flags suspicious Bitcoin transactions, and a team of LLM agents investigates each flag and drafts a grounded report. A critic agent then checks every claim in the report against the collected evidence. The project uses real data from the Elliptic Bitcoin dataset.

Status: Phase 3b (agent tools and regulation retrieval done; agents not started)

## Results (Phase 2)

Splits are temporal and use labeled nodes only: fit = time steps 1-29, validation = 30-34, test = 35-49. Thresholds and calibration come from validation only. Illicit is the positive class. Test steps 43-49 follow a market shutdown ("post-shutdown"); steps 35-42 are "pre-shutdown". Full tables: [eval/results/phase2c.md](eval/results/phase2c.md), [eval/results/drift.md](eval/results/drift.md).

**Final model: LightGBM on the 165 original features, no structural features.** Reasons:

- Adding the structural graph features raised test PR-AUC by only 0.004 (0.7956 vs 0.7917), which is within noise.
- Pre-shutdown F1 was lower with them (0.885 vs 0.898).
- `wcc_size` is nearly a time-step proxy (every weakly connected component has at least 1,089 nodes, so each time step is essentially one component).

The structural model's numbers are kept in [eval/results/phase2b.md](eval/results/phase2b.md) as a negative-result ablation, and `src/ml/graph_features.py` is kept for the agent tools. Caveat: this decision cited test-set numbers, so the test period is no longer a pure holdout for the choice between these two models.

| Period | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Test overall (35-49) | 0.792 | 0.915 | 0.719 | 0.806 |
| Pre-shutdown (35-42) | 0.915 | 0.952 | 0.849 | 0.898 |
| Post-shutdown (43-49) | 0.039 | 0.083 | 0.018 | 0.029 |

(Validation-F1 threshold 0.788. Test prevalence is 6.5% overall, 9.2% pre and 2.5% post.) After the shutdown the model is close to useless: recall collapses at nearly every step from 43 on.

**Cost-based threshold.** The costs are assumptions, not measured: a false alert costs 1, a missed illicit node costs C in {5, 10, 20}, default C = 10. The default threshold chosen on validation is 0.2582 (raw score). On test it gives overall precision 0.654, recall 0.744, total cost 3,197 against 10,830 for flagging nothing. Post-shutdown it costs 1,846 against 1,690 for flagging nothing, so it does not pay for itself there.

**Calibration.** Isotonic regression on validation predictions. Test Brier 0.0263 raw and 0.0224 calibrated; test ECE 0.0289 raw and 0.0197 calibrated. Prevalence drops from 16.8% in validation to 6.5% on test, so calibration drifts and is worst post-shutdown.

**Drift monitor (label-free, [src/ml/drift.py](src/ml/drift.py)).** It does not cleanly separate pre- and post-shutdown steps. PSI is already "major" for many steps inside the reference period, the score PSI ranges overlap completely between pre and post steps, and the feature PSI flags essentially every step after the reference period (31-49), not just 43-49. It also flags steps where the model still works well. Details in [eval/results/drift.md](eval/results/drift.md).

## Agent tools (Phase 3a)

Plain Python functions in [src/tools/](src/tools/) with Pydantic inputs and outputs, registered in `TOOLS`: `score_transaction`, `get_neighbors`, `subgraph_summary`, `match_typology`, `check_model_health`. Rules: no tool returns a ground-truth label, tools never cross time steps, unknown ids return a structured error, and every output carries an `evidence_id`.

**Score scale.** Every score is the raw LightGBM `predict_proba` of the final model (trained with `scale_pos_weight`, so not a calibrated probability). The alert threshold 0.2582 (cost-based, C=10, an assumed cost) and the high-risk threshold 0.788 (validation-F1) were both chosen on this raw scale. A calibrated probability appears only as a separate, labelled field.

**Typology flags are structural resemblance only.** Elliptic has no typology labels. On labeled nodes in steps 1-34, the chain flag is informative (illicit rate 25.7% flagged vs 10.8% unflagged), while the fan-in and fan-out flags are anti-informative (1.9% and 0.0% vs about 12%). See [eval/results/typology_check.md](eval/results/typology_check.md).

**Model-health monitor (`check_model_health`).** It uses delayed labels: lag 2 steps, a 3-step window, reference recall from validation steps 30-34, degraded if window recall is below half the reference, at least 20 illicit nodes required. These constants were designed after the post-shutdown failure was observed, so the evaluation on steps 35-49 is not a blind test of them. Result: first flagged at step 47 (shutdown at 43), no false alarms in steps 35-42. See [eval/results/model_health.md](eval/results/model_health.md).

**Typology evidence direction.** `match_typology` now attaches, per matched pattern, the illicit rate among labeled nodes in steps 1-34 only, the baseline rate, the lift and an `evidence_direction` (`raises_risk`, `lowers_risk` or `uninformative`) from a rule fixed in advance: the 95% Wilson interval of the flagged rate lies entirely above, entirely below, or around the baseline. Current values: chain `raises_risk`, fan_in and fan_out `lowers_risk`. A `lowers_risk` pattern must never be described as suspicious.

## Regulation retrieval (Phase 3b)

Three PDFs are indexed from `data/raw/regulations/` (gitignored): the FinCEN 2019 CVC advisory, the FinCEN 2003 SAR narrative guidance and the FATF 2020 virtual-assets red-flag report ([src/rag/SOURCES.md](src/rag/SOURCES.md), [src/rag/manifest.yaml](src/rag/manifest.yaml)). A fourth PDF (`VDA08012026.pdf`) is image-only and is not indexed. No Indian document is indexed yet. Chunks are about 250 tokens with about 15% overlap and an evidence id like `reg:<doc_id>:p<page>:c<n>` (physical PDF page). Retrieval is BM25 plus dense vectors (BAAI/bge-small-en-v1.5 in FAISS) fused with reciprocal rank fusion. Tools: `search_regulations`, `get_regulation_chunk`. Evaluation on a 20-question golden set: [eval/results/retrieval.md](eval/results/retrieval.md). The set is small (17 scored questions), so differences between modes are noisy.

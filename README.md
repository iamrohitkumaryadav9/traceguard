# TraceGuard

An ML model flags suspicious Bitcoin transactions, and a team of LLM agents investigates each flag and drafts a grounded report. A critic agent then checks every claim in the report against the collected evidence. The project uses real data from the Elliptic Bitcoin dataset.

Status: Phase 2c (final model and drift monitor done; agents not started)

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

"""Phase 2b pipeline. Run from the project root: python -m src.ml.phase2b

Stage 1  fit candidate models on steps 1-29, compare on validation (30-34), pick the final model.
Stage 2  validation-only tuning for the final model: cost thresholds, isotonic calibrator. Save artifacts.
Stage 3  FINAL TEST EVALUATION (steps 35-49), run once. Nothing is tuned afterwards.

COST ASSUMPTIONS: a false alert costs 1, a missed illicit node costs C in {5, 10, 20}. These are
assumptions, not measured numbers. C = 10 is the default.
"""
import json
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.isotonic import IsotonicRegression

from src.ml.baselines import SEED, feature_sets, in_steps, lgb_params, load_labeled
from src.ml.eval import (
    best_cost_threshold, best_f1_threshold, brier_score, cost_curve, expected_calibration_error,
    per_step_metrics, pr_auc, prf_at_threshold, roc_auc, topk_metrics, total_cost,
)
from src.ml.explain import explain
from src.ml.graph_features import STRUCTURAL_COLS, edges_cross_time_steps
from src.split import FIT_STEPS, SHUTDOWN_STEP, TEST_STEPS, VAL_STEPS

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*eval_set.*")

COSTS = (5, 10, 20)  # ASSUMED cost of a missed illicit node (false alert = 1)
DEFAULT_C = 10
KS = (10, 25)
RESULTS_DIR = Path("eval/results")
FIG_DIR = Path("notebooks/figures")
MODEL_DIR = Path("models")
FIG_TAG = ""  # suffix for figure names, set from the command line


def _fig(name):
    return FIG_DIR / f"{name}{FIG_TAG}.png"


# ---------- helpers ----------
def md_table(df, digits=4):
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "n/a" if np.isnan(v) else f"{v:.{digits}f}"
        return str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def periods(time_steps):
    t = np.asarray(time_steps)
    return {
        "overall (35-49)": np.ones(len(t), bool),
        f"pre-shutdown (35-{SHUTDOWN_STEP - 1})": t < SHUTDOWN_STEP,
        f"post-shutdown ({SHUTDOWN_STEP}-49)": t >= SHUTDOWN_STEP,
    }


def fit_lgb(fit, val, cols):
    """Same procedure as 2a's LightGBM models: early stopping (patience 50) on validation PR-AUC."""
    model = lgb.LGBMClassifier(**lgb_params(fit["y"]))
    model.fit(fit[cols], fit["y"], eval_set=[(val[cols], val["y"])], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


# ---------- stage 1 ----------
def stage1(df, feats, only=None):
    """Fit candidates on steps 1-29. If `only` is given, fit just that model and skip selection."""
    fit, val = in_steps(df, FIT_STEPS), in_steps(df, VAL_STEPS)
    candidates = {
        "LightGBM (165 original)": feats["all"],
        "LightGBM (165 + structural)": feats["all"] + STRUCTURAL_COLS,
    }
    if only is not None:
        candidates = {only: candidates[only]}
    fitted = {}
    for name, cols in candidates.items():
        model = fit_lgb(fit, val, cols)
        s = model.predict_proba(val[cols])[:, 1]
        thr, f1 = best_f1_threshold(val["y"], s)
        fitted[name] = {"model": model, "cols": cols, "val_scores": s, "f1_threshold": thr,
                        "val_pr_auc": pr_auc(val["y"], s), "val_f1": f1, "best_iter": model.best_iteration_}
        print(f"{name}: val PR-AUC {fitted[name]['val_pr_auc']:.4f}, val F1 {f1:.4f} @ {thr:.4f}, "
              f"best_iter {model.best_iteration_}")
    if only is not None:
        print(f"Final model (fixed by decision, not selected here): {only}")
        return fitted, only
    best = max(fitted, key=lambda n: fitted[n]["val_pr_auc"])
    print(f"Final model (highest validation PR-AUC): {best}")
    return fitted, best


# ---------- diagnostic: why did local-only LightGBM stop at iteration 6? ----------
def early_stopping_diagnostic(df, feats):
    fit, val = in_steps(df, FIT_STEPS), in_steps(df, VAL_STEPS)
    curves = {}
    for name, cols in (("local-only (f1-f93)", feats["local"]), ("all 165", feats["all"])):
        p = lgb_params(fit["y"])
        p["n_estimators"] = 300
        m = lgb.LGBMClassifier(**p)
        m.fit(fit[cols], fit["y"], eval_set=[(val[cols], val["y"])], eval_metric="average_precision")
        curves[name] = np.array(m.evals_result_["valid_0"]["average_precision"])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, c in curves.items():
        ax.plot(np.arange(1, len(c) + 1), c, label=name)
    loc = curves["local-only (f1-f93)"]
    ax.axvline(6, color="gray", ls=":", label="local-only best-so-far at stop (iter 6)")
    ax.axvline(56, color="gray", ls="--", label="local-only stop (iter 56 = 6 + patience 50)")
    ax.set(xlabel="boosting iteration", ylabel="validation PR-AUC (steps 30-34)",
           title="Validation PR-AUC by iteration (diagnostic only)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(_fig("early_stopping_diagnostic"), dpi=120)
    plt.close(fig)
    info = {
        "local_ap_at_6": float(loc[5]), "local_ap_min_7_to_56": float(loc[6:56].min()),
        "local_ap_max_7_to_56": float(loc[6:56].max()), "local_best_iter_300": int(loc.argmax() + 1),
        "local_best_ap_300": float(loc.max()),
        "first_iter_beating_6": int(np.argmax(loc[6:] > loc[5]) + 7) if (loc[6:] > loc[5]).any() else None,
    }
    print("Early-stopping diagnostic (local-only):", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in info.items()})
    return info


# ---------- stage 2 ----------
def stage2(val, fitted, best):
    f = fitted[best]
    y, s = val["y"].to_numpy(), f["val_scores"]
    chosen = {}
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for c in COSTS:
        thr, cost = cost_curve(y, s, c)
        m = np.isfinite(thr)
        ax.plot(thr[m], cost[m], label=f"C={c}")
        t, tc = best_cost_threshold(y, s, c)
        chosen[c] = {"threshold": t, "val_cost": tc}
        ax.plot([t], [tc], "ko")
    ax.set(xlabel="threshold on raw model score", ylabel="total validation cost",
           title=f"{best}: validation cost vs threshold (ASSUMED costs: false alert=1, miss=C)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_fig("cost_vs_threshold"), dpi=120)
    plt.close(fig)

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(s, y)
    return chosen, iso


def save_artifacts(best, f, chosen, iso, df_val_prev):
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"name": best, "model": f["model"], "cols": f["cols"]}, MODEL_DIR / "phase2b_model.pkl")
    joblib.dump(iso, MODEL_DIR / "calibrator.pkl")
    (MODEL_DIR / "features.json").write_text(json.dumps(f["cols"]))
    joblib.dump(shap.TreeExplainer(f["model"]), MODEL_DIR / "explainer.pkl")
    thr = chosen[DEFAULT_C]["threshold"]
    payload = {
        "model": best,
        "trained_on_time_steps": list(FIT_STEPS),
        "validation_time_steps": list(VAL_STEPS),
        "score": "raw LightGBM predict_proba (trained with scale_pos_weight; not calibrated)",
        "default_cost_ratio_C": DEFAULT_C,
        "default_threshold": thr,
        "calibrated_probability_at_default_threshold": float(iso.predict([thr])[0]),
        "thresholds_by_C": {str(c): v for c, v in chosen.items()},
        "cost_assumptions": {
            "false_alert_cost": 1,
            "missed_illicit_cost_C": list(COSTS),
            "note": "ASSUMPTIONS, not measured business costs. Threshold chosen on validation only.",
        },
        "validation_illicit_prevalence": df_val_prev,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "threshold.json").write_text(json.dumps(payload, indent=2))


# ---------- stage 3 ----------
def metrics_rows(test, scores, thr, extra=None):
    rows = []
    for pname, mask in periods(test["time_step"]).items():
        y, s = test["y"].to_numpy()[mask], scores[mask]
        prf = prf_at_threshold(y, s, thr)
        rows.append({"period": pname, "n": int(mask.sum()), "illicit": int(y.sum()), "prevalence": float(y.mean()),
                     "pr_auc": pr_auc(y, s), "roc_auc": roc_auc(y, s), **prf, **(extra or {})})
    return rows


def calibration_rows(y, raw, cal, label):
    return {"set": label, "n": len(y), "prevalence": float(np.mean(y)),
            "brier_raw": brier_score(y, raw), "brier_isotonic": brier_score(y, cal),
            "ece_raw": expected_calibration_error(y, raw), "ece_isotonic": expected_calibration_error(y, cal)}


def reliability_plot(sets):
    fig, axes = plt.subplots(1, len(sets), figsize=(5 * len(sets), 4.5), squeeze=False)
    for ax, (label, y, raw, cal) in zip(axes[0], sets):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        for name, p in (("raw score", raw), ("isotonic", cal)):
            bins = np.minimum((p * 10).astype(int), 9)
            xs, ys = [], []
            for b in range(10):
                m = bins == b
                if m.sum() >= 5:
                    xs.append(p[m].mean())
                    ys.append(y[m].mean())
            ax.plot(xs, ys, marker="o", label=name)
        ax.set(xlabel="mean predicted probability", ylabel="observed illicit fraction", title=label)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(_fig("reliability"), dpi=120)
    plt.close(fig)


def final_test(df, val, fitted, best, chosen, iso):
    test = in_steps(df, TEST_STEPS)
    out = {}
    scores = {n: f["model"].predict_proba(test[f["cols"]])[:, 1] for n, f in fitted.items()}

    # model comparison at each model's own validation-F1 threshold
    comp = []
    for n, f in fitted.items():
        for r in metrics_rows(test, scores[n], f["f1_threshold"], {"val_pr_auc": f["val_pr_auc"], "thr": f["f1_threshold"]}):
            comp.append({"model": n, **r})
    out["comparison"] = pd.DataFrame(comp)

    s = scores[best]
    y = test["y"].to_numpy()

    # cost-based thresholds
    cost_rows = []
    for c in COSTS:
        thr = chosen[c]["threshold"]
        for pname, mask in periods(test["time_step"]).items():
            prf = prf_at_threshold(y[mask], s[mask], thr)
            tc = total_cost(y[mask], s[mask], thr, c)
            n_ill = int(y[mask].sum())
            cost_rows.append({"C (assumed)": c, "default": "yes" if c == DEFAULT_C else "", "val_threshold": thr,
                              "period": pname, **prf, "FP": tc["fp"], "FN": tc["fn"], "total_cost": tc["cost"],
                              "cost_flag_nothing": c * n_ill, "cost_flag_all": int((1 - y[mask]).sum())})
    out["cost"] = pd.DataFrame(cost_rows)

    # calibration
    val_y = val["y"].to_numpy()
    val_raw = fitted[best]["val_scores"]
    val_cal = iso.predict(val_raw)
    test_cal = iso.predict(s)
    cal = [calibration_rows(val_y, val_raw, val_cal, "validation (30-34; calibrator fitted here, in-sample)")]
    for pname, mask in periods(test["time_step"]).items():
        cal.append(calibration_rows(y[mask], s[mask], test_cal[mask], f"test {pname}"))
    out["calibration"] = pd.DataFrame(cal)
    reliability_plot([("validation (in-sample for isotonic)", val_y, val_raw, val_cal),
                      ("test (35-49)", y, s, test_cal)])

    # alert budget
    thr_default = chosen[DEFAULT_C]["threshold"]
    ps = per_step_metrics(test["time_step"], y, s, thr_default)
    rows = []
    for t, g in test.assign(s=s).groupby("time_step"):
        row = {"time_step": int(t), "n": len(g), "illicit": int(g["y"].sum()),
               "recall@thr": float(ps.loc[ps.time_step == t, "recall"].iloc[0])}
        for k in KS:
            m = topk_metrics(g["y"], g["s"], k)
            row.update({f"P@{k}": m["precision"], f"R@{k}": m["recall"], f"maxR@{k}": m["max_recall"]})
        rows.append(row)
    budget = pd.DataFrame(rows)
    out["budget_steps"] = budget
    avg = []
    for pname, sel in (("pre-shutdown avg", budget.time_step < SHUTDOWN_STEP),
                       ("post-shutdown avg", budget.time_step >= SHUTDOWN_STEP)):
        b = budget[sel & (budget.illicit > 0)]
        row = {"period": pname, "steps": len(b)}
        row.update({c: b[c].mean() for c in budget.columns if c not in ("time_step", "n", "illicit")})
        avg.append(row)
    out["budget_avg"] = pd.DataFrame(avg)

    # the single most informative per-step plot for the final model
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(budget.time_step, budget["recall@thr"], marker="o", label=f"recall @ default threshold")
    ax.plot(budget.time_step, budget["R@25"], marker="s", label="recall@25")
    ax.plot(budget.time_step, budget["maxR@25"], ls=":", color="gray", label="max possible recall@25")
    ax.axvline(SHUTDOWN_STEP - 0.5, color="red", ls="--", label="shutdown")
    ax.set(xlabel="test time step", ylabel="recall", ylim=(0, 1.05), title=f"{best}: alert budget vs global threshold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(_fig("alert_budget"), dpi=120)
    plt.close(fig)
    out["test"], out["scores"] = test, s
    return out


def write_md(res, fitted, best, chosen, diag, cross_edges, md_name="phase2b.md", title="Phase 2b results",
             selection_note=None):
    sections = [
        f"# {title}", "",
        "Fit: steps 1-29. Validation: 30-34. Test: 35-49, evaluated once. Models are NOT refit on train+val.",
        f"Positive class = illicit. Pre-shutdown = test steps 35-{SHUTDOWN_STEP - 1}, post-shutdown = {SHUTDOWN_STEP}-49.",
        f"Edges connecting different time steps: {cross_edges}.", "",
        "## Model comparison (threshold = each model's validation-F1 optimum)", "",
        md_table(res["comparison"]), "",
        selection_note or f"Final model (chosen on validation PR-AUC only): **{best}**.", "",
    ]
    if diag is not None:
        sections += ["## Early-stopping diagnostic (local-only LightGBM from 2a)", "",
                     "```", json.dumps(diag, indent=2), "```", ""]
    sections += [
        "## Cost-based thresholds",
        "**ASSUMPTION: false alert costs 1, missed illicit node costs C. C values are assumed, not measured.** "
        f"Thresholds are chosen on validation; C={DEFAULT_C} is the default.", "",
        md_table(pd.DataFrame([{"C (assumed)": c, **v} for c, v in chosen.items()])), "",
        md_table(res["cost"]), "",
        "## Calibration (isotonic, fitted on validation predictions)", "",
        "Prevalence differs between periods, so calibration is expected to drift on test. "
        "The validation row is in-sample for the isotonic fit and therefore optimistic. "
        "Raw scores come from a model trained with scale_pos_weight and are not probabilities.", "",
        md_table(res["calibration"]), "",
        "## Alert budget (test, per step)", "",
        f"Threshold column = default cost threshold (C={DEFAULT_C}). maxR@k = best possible recall@k "
        "given the number of illicit nodes in the step. Only labeled nodes are ranked.", "",
        md_table(res["budget_steps"], 3), "",
        md_table(res["budget_avg"], 3), "",
    ]
    (RESULTS_DIR / md_name).write_text("\n".join(sections) + "\n")


def main(final_model=None, md_name="phase2b.md", title="Phase 2b results", selection_note=None):
    """final_model=None: fit both candidates, diagnose, and pick by validation (Phase 2b).
    final_model=<name>: fit and report only that model (Phase 2c)."""
    np.random.seed(SEED)
    df = load_labeled()
    gf = pd.read_parquet("data/processed/graph_features.parquet")
    df = df.merge(gf, on="txId", how="left", validate="one_to_one")
    nodes_all = pd.read_parquet("data/processed/nodes.parquet", columns=["txId", "time_step"])
    edges = pd.read_parquet("data/processed/edges.parquet")
    cross = edges_cross_time_steps(nodes_all, edges)
    print(f"Edges connecting two different time steps: {cross} of {len(edges):,}")
    feats = feature_sets(df)

    fitted, best = stage1(df, feats, only=final_model)
    diag = early_stopping_diagnostic(df, feats) if final_model is None else None

    val = in_steps(df, VAL_STEPS)
    chosen, iso = stage2(val, fitted, best)
    print(f"\nCost thresholds on validation for {best} (ASSUMED costs):")
    for c, v in chosen.items():
        print(f"  C={c}: threshold {v['threshold']:.4f}, val cost {v['val_cost']:.0f}")
    save_artifacts(best, fitted[best], chosen, iso, float(val["y"].mean()))

    print("\n=== FINAL TEST EVALUATION (run once) ===")
    res = final_test(df, val, fitted, best, chosen, iso)
    write_md(res, fitted, best, chosen, diag, cross, md_name, title, selection_note)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for key in ("comparison", "cost", "calibration", "budget_avg"):
        print(f"\n--- {key} ---")
        print(res[key].round(4).to_string(index=False))
    print("\n--- budget_steps ---")
    print(res["budget_steps"].round(3).to_string(index=False))

    ids = res["test"].assign(s=res["scores"]).sort_values("s", ascending=False)["txId"].iloc[[0, 200]].tolist()
    print("\nSHAP example (saved explainer):")
    for node, items in explain(ids, top_n=5).items():
        print(f"  txId {node}")
        for it in items:
            print(f"    {it['feature']:16s} [{it['group']}] value={it['value']:.3f} shap={it['shap']:+.3f} {it['direction']}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--final", help="fit/report only this model instead of selecting by validation")
    ap.add_argument("--md", default="phase2b.md")
    ap.add_argument("--title", default="Phase 2b results")
    ap.add_argument("--tag", default="", help="suffix for figure file names")
    ap.add_argument("--note", default=None, help="line replacing the 'chosen on validation' sentence")
    a = ap.parse_args()
    FIG_TAG = a.tag
    main(a.final, a.md, a.title, a.note)

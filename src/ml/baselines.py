"""Phase 2a baselines. Run from the project root: python -m src.ml.baselines

Stage 1 fits each model on steps 1-29, early-stops / picks the F1 threshold on steps 30-34.
Stage 2 (final_test) evaluates each model on steps 35-49 exactly once.
"""
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.eval import best_f1_threshold, per_step_metrics, pr_auc, prf_at_threshold, roc_auc
from src.split import FIT_STEPS, TEST_STEPS, VAL_STEPS

SEED = 42
N_LOCAL = 93  # ASSUMPTION: f1-f93 local, f94-f165 aggregated neighbor features
NODES = "data/processed/nodes.parquet"
RESULTS_DIR = Path("eval/results")
FIG_DIR = Path("notebooks/figures")
MODEL_DIR = Path("models")


def load_labeled():
    nodes = pd.read_parquet(NODES)
    df = nodes[nodes["label"].isin(["licit", "illicit"])].copy()
    df["y"] = (df["label"] == "illicit").astype(int)
    return df


def in_steps(df, steps):
    return df[df["time_step"].between(*steps)]


def feature_sets(df):
    cols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
    cols = sorted(cols, key=lambda c: int(c[1:]))
    assert len(cols) == 165, len(cols)
    print(f"Feature columns: {len(cols)}; first 3 {cols[:3]}, last 3 {cols[-3:]}")
    local, agg = cols[:N_LOCAL], cols[N_LOCAL:]
    print(f"ASSUMPTION: local = {local[0]}..{local[-1]} ({len(local)}), "
          f"aggregated neighbor = {agg[0]}..{agg[-1]} ({len(agg)}). "
          "The raw file has no column names, so this split is taken from the Elliptic paper, not verified.")
    return {"local": local, "all": cols}


def make_models(y_fit):
    spw = float((y_fit == 0).sum() / (y_fit == 1).sum())
    lgb_params = dict(
        n_estimators=2000, learning_rate=0.05, num_leaves=31, min_child_samples=20,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, scale_pos_weight=spw,
        random_state=SEED, deterministic=True, force_row_wise=True, n_jobs=4, verbose=-1,
    )
    return {
        "LogReg (local)": ("local", make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED))),
        "LightGBM (local)": ("local", lgb.LGBMClassifier(**lgb_params)),
        "LightGBM (all)": ("all", lgb.LGBMClassifier(**lgb_params)),
    }


def fit_and_validate(df, feats):
    fit, val = in_steps(df, FIT_STEPS), in_steps(df, VAL_STEPS)
    print(f"\nfit rows {len(fit)} (illicit {fit.y.sum()}), val rows {len(val)} (illicit {val.y.sum()})")
    fitted = {}
    for name, (fs, model) in make_models(fit["y"]).items():
        cols = feats[fs]
        if isinstance(model, lgb.LGBMClassifier):
            model.fit(fit[cols], fit["y"], eval_set=[(val[cols], val["y"])], eval_metric="average_precision",
                      callbacks=[lgb.early_stopping(50, verbose=False)])
            extra = f", best_iter {model.best_iteration_}"
        else:
            model.fit(fit[cols], fit["y"])
            extra = ""
        s = model.predict_proba(val[cols])[:, 1]
        thr, f1 = best_f1_threshold(val["y"], s)
        fitted[name] = {"model": model, "cols": cols, "threshold": thr, "val_pr_auc": pr_auc(val["y"], s), "val_f1": f1}
        print(f"{name}: val PR-AUC {fitted[name]['val_pr_auc']:.4f}, val F1 {f1:.4f} @ thr {thr:.4f}{extra}")
    return fitted


def final_test(df, fitted):
    """Single, final evaluation on the test period. Nothing is tuned after this."""
    test = in_steps(df, TEST_STEPS)
    prevalence = float(test["y"].mean())
    rows, scores = [], {}
    for name, f in fitted.items():
        s = f["model"].predict_proba(test[f["cols"]])[:, 1]
        scores[name] = s
        prf = prf_at_threshold(test["y"], s, f["threshold"])
        rows.append({
            "model": name, "val_pr_auc": f["val_pr_auc"], "test_pr_auc": pr_auc(test["y"], s),
            "test_roc_auc": roc_auc(test["y"], s), "test_precision": prf["precision"],
            "test_recall": prf["recall"], "test_f1": prf["f1"],
            "val_threshold": f["threshold"], "test_prevalence": prevalence,
        })
    return test, rows, scores


def write_results(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "baselines.json").write_text(json.dumps(rows, indent=2))
    lines = [
        "# Baseline results",
        "",
        "Fit: time steps 1-29. Validation: 30-34 (early stopping + F1-optimal threshold). "
        "Test: 35-49, evaluated once at the validation threshold. Positive class = illicit.",
        "",
        "| Model | Val PR-AUC | Test PR-AUC | Test ROC-AUC | Test precision | Test recall | Test F1 | Val threshold | Test prevalence |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['val_pr_auc']:.4f} | {r['test_pr_auc']:.4f} | {r['test_roc_auc']:.4f} | "
            f"{r['test_precision']:.4f} | {r['test_recall']:.4f} | {r['test_f1']:.4f} | "
            f"{r['val_threshold']:.4f} | {r['test_prevalence']:.4f} |")
    lines += ["", "Test prevalence is the PR-AUC of a random classifier."]
    (RESULTS_DIR / "baselines.md").write_text("\n".join(lines) + "\n")


def plot_per_step(test, score, threshold, name):
    ps = per_step_metrics(test["time_step"], test["y"], score, threshold)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(ps["time_step"], ps["n_illicit"], color="lightgray", label="illicit count")
    ax.set(xlabel="test time step", ylabel="illicit count", title=f"{name}: per-step test recall")
    ax2 = ax.twinx()
    ax2.plot(ps["time_step"], ps["recall"], marker="o", color="tab:red", label="recall")
    ax2.set(ylabel="recall", ylim=(0, 1.05))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "baseline_per_step_recall.png", dpi=120)
    plt.close(fig)
    return ps


def main():
    np.random.seed(SEED)
    df = load_labeled()
    feats = feature_sets(df)
    fitted = fit_and_validate(df, feats)

    MODEL_DIR.mkdir(exist_ok=True)
    for name, f in fitted.items():
        joblib.dump(f, MODEL_DIR / (name.replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"))

    # Best model is chosen on VALIDATION PR-AUC only.
    best = max(fitted, key=lambda n: fitted[n]["val_pr_auc"])
    print(f"\nBest model by validation PR-AUC: {best}")

    print("\n=== FINAL TEST EVALUATION (run once) ===")
    test, rows, scores = final_test(df, fitted)
    write_results(rows)
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    ps = plot_per_step(test, scores[best], fitted[best]["threshold"], best)
    print(f"\nPer-step test metrics for {best}:")
    print(ps.round(3).to_string(index=False))
    flagged = ps[(ps["n_illicit"] >= 10) & (ps["recall"] < 0.5)]
    print("\nSteps with >=10 illicit and recall < 0.5:", flagged["time_step"].tolist() or "none")


if __name__ == "__main__":
    main()

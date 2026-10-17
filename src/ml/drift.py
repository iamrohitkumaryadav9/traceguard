"""Label-free drift monitor. Run from the project root: python -m src.ml.drift

Signals per time step (all nodes, labeled and unlabeled, no labels used):
  score_psi    PSI of the final model's raw score vs the reference (all nodes, steps 1-29)
  feat_psi     mean PSI over the top-20 features by mean |SHAP| (importance from labeled TRAIN rows only)
  alert_rate   share of nodes with score >= the chosen threshold

Rules, fixed in advance (nothing is tuned on test):
  PSI: 0.10 = moderate, 0.25 = major (conventional cutoffs).
  alert_rate: no conventional cutoff exists, so "major" = outside mean +/- 3 SD of the per-step alert
  rates over the reference steps 1-29 only.
Labels are used only afterwards, to evaluate the monitor (per-step recall of illicit nodes).
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ml.eval import prf_at_threshold
from src.ml.explain import shap_values_positive
from src.split import FIT_STEPS, SHUTDOWN_STEP

SEED = 42
N_BINS = 10
TOP_FEATURES = 20
PSI_MODERATE, PSI_MAJOR = 0.10, 0.25
EPS = 1e-4
SHAP_SAMPLE = 10_000
RESULTS_DIR = Path("eval/results")
FIG_DIR = Path("notebooks/figures")


def psi_edges(reference, n_bins=N_BINS):
    """Interior bin edges from reference quantiles (duplicates collapsed, so there may be fewer bins)."""
    q = np.quantile(np.asarray(reference, dtype=float), np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.unique(q)


def _proportions(x, edges):
    idx = np.searchsorted(edges, np.asarray(x, dtype=float), side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1)
    return np.maximum(counts / max(len(x), 1), EPS)


def psi(reference, current, n_bins=N_BINS):
    """Population stability index of `current` against `reference` using reference quantile bins."""
    edges = psi_edges(reference, n_bins)
    e, a = _proportions(reference, edges), _proportions(current, edges)
    return float(np.sum((a - e) * np.log(a / e)))


def level(value):
    return "major" if value >= PSI_MAJOR else "moderate" if value >= PSI_MODERATE else "ok"


def top_features_by_shap(model, cols, train_rows, k=TOP_FEATURES):
    """Top-k features by mean |SHAP| on labeled training rows only."""
    import shap

    sample = train_rows.sample(min(SHAP_SAMPLE, len(train_rows)), random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = shap_values_positive(shap.TreeExplainer(model).shap_values(sample[cols]))
    imp = pd.Series(np.abs(sv).mean(axis=0), index=cols).sort_values(ascending=False)
    return imp.head(k)


def compute_signals(nodes, score, feat_cols, threshold):
    """Per-step drift signals; `nodes` has time_step and the feature columns, `score` aligns with it."""
    t = nodes["time_step"].to_numpy()
    ref = t <= FIT_STEPS[1]
    ref_scores = score[ref]
    ref_edges_f = {f: psi_edges(nodes.loc[ref, f]) for f in feat_cols}
    ref_p_f = {f: _proportions(nodes.loc[ref, f], ref_edges_f[f]) for f in feat_cols}
    ref_p_s = _proportions(ref_scores, psi_edges(ref_scores))
    edges_s = psi_edges(ref_scores)

    rows = []
    for step in range(int(t.min()), int(t.max()) + 1):
        m = t == step
        a = _proportions(score[m], edges_s)
        fpsi = []
        for f in feat_cols:
            af = _proportions(nodes.loc[m, f], ref_edges_f[f])
            fpsi.append(float(np.sum((af - ref_p_f[f]) * np.log(af / ref_p_f[f]))))
        rows.append({
            "time_step": step, "n_nodes": int(m.sum()),
            "score_psi": float(np.sum((a - ref_p_s) * np.log(a / ref_p_s))),
            "feat_psi": float(np.mean(fpsi)),
            "alert_rate": float((score[m] >= threshold).mean()),
        })
    return pd.DataFrame(rows), len(edges_s) + 1


def main():
    bundle = joblib.load("models/phase2b_model.pkl")
    model, cols, name = bundle["model"], bundle["cols"], bundle["name"]
    cfg = json.loads((RESULTS_DIR / "threshold.json").read_text())
    assert cfg["model"] == name, "threshold.json and models/ belong to different models"
    thr = cfg["default_threshold"]
    print(f"Model: {name}; default threshold {thr:.4f} (C={cfg['default_cost_ratio_C']}, assumed)")

    nodes = pd.read_parquet("data/processed/nodes.parquet")
    score = model.predict_proba(nodes[cols])[:, 1]
    labeled = nodes["label"].isin(["licit", "illicit"]).to_numpy()
    y = (nodes["label"] == "illicit").to_numpy()

    train_rows = nodes[labeled & (nodes["time_step"] <= FIT_STEPS[1])]
    imp = top_features_by_shap(model, cols, train_rows)
    top = list(imp.index)
    print(f"Top {TOP_FEATURES} features by mean |SHAP| on training rows: {top}")

    sig, n_bins_used = compute_signals(nodes, score, top, thr)
    print(f"Score PSI uses {n_bins_used} bins (reference quantile edges collapse when scores tie).")

    # recall of illicit nodes per step (labels used only to evaluate the monitor)
    rec = {}
    for step, g in nodes.assign(s=score, y=y)[labeled].groupby("time_step"):
        n_ill = int(g["y"].sum())
        rec[step] = prf_at_threshold(g["y"], g["s"], thr)["recall"] if n_ill else np.nan
    sig["recall"] = sig["time_step"].map(rec)
    sig["n_illicit"] = sig["time_step"].map(
        nodes.assign(y=y)[labeled].groupby("time_step")["y"].sum()).fillna(0).astype(int)

    ref_rates = sig.loc[sig.time_step <= FIT_STEPS[1], "alert_rate"]
    lo, hi = ref_rates.mean() - 3 * ref_rates.std(), ref_rates.mean() + 3 * ref_rates.std()
    sig["score_level"] = sig["score_psi"].map(level)
    sig["feat_level"] = sig["feat_psi"].map(level)
    sig["alert_out_of_band"] = (sig["alert_rate"] < lo) | (sig["alert_rate"] > hi)
    print(f"Alert-rate band from reference steps 1-29: [{lo:.4f}, {hi:.4f}]")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sig.to_csv(RESULTS_DIR / "drift_per_step.csv", index=False)
    plot(sig, lo, hi, thr)

    pd.set_option("display.width", 250)
    print("\nPer-step drift table:")
    print(sig.round(4).to_string(index=False))
    report = evaluate(sig)
    write_md(sig, report, top, n_bins_used, lo, hi, thr, name)


def evaluate(sig):
    """Which steps each signal flags 'major', and how well that separates post-shutdown steps."""
    post = sig.time_step >= SHUTDOWN_STEP
    held = (sig.time_step > FIT_STEPS[1]) & ~post  # steps 30-42: not in the reference
    flags = {
        "score_psi": sig["score_psi"] >= PSI_MAJOR,
        "feat_psi": sig["feat_psi"] >= PSI_MAJOR,
        "alert_rate": sig["alert_out_of_band"],
    }
    out = {}
    for k, f in flags.items():
        out[k] = {
            "major_steps": sig.loc[f, "time_step"].tolist(),
            "post_steps_flagged": sig.loc[f & post, "time_step"].tolist(),
            "post_steps_missed": sig.loc[~f & post, "time_step"].tolist(),
            "false_alarms_1_42": sig.loc[f & ~post, "time_step"].tolist(),
            "false_alarms_30_42": sig.loc[f & held, "time_step"].tolist(),
            "max_pre_value": float(sig.loc[~post & (sig.time_step > FIT_STEPS[1]), k].max()),
            "min_post_value": float(sig.loc[post, k].min()),
        }
    return out


def plot(sig, lo, hi, thr):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    panels = [("score_psi", "score PSI", PSI_MODERATE, PSI_MAJOR),
              ("feat_psi", f"mean PSI, top-{TOP_FEATURES} features", PSI_MODERATE, PSI_MAJOR),
              ("alert_rate", "alert rate", None, None)]
    for ax, (col, label, mod, maj) in zip(axes, panels):
        ax.plot(sig.time_step, sig[col], marker="o", ms=3, color="tab:blue", label=label)
        if mod is not None:
            ax.axhline(mod, color="orange", ls=":", label="moderate 0.10")
            ax.axhline(maj, color="red", ls=":", label="major 0.25")
        else:
            ax.axhspan(lo, hi, color="green", alpha=0.1, label="reference band (mean +/- 3 SD, steps 1-29)")
        ax.axvline(SHUTDOWN_STEP - 0.5, color="black", ls="--", label="step 43")
        ax.axvspan(0.5, FIT_STEPS[1] + 0.5, color="gray", alpha=0.08)
        ax.set_ylabel(label)
        ax2 = ax.twinx()
        ax2.plot(sig.time_step, sig["recall"], color="tab:red", alpha=0.5, lw=1, label="recall (labels; eval only)")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("recall of illicit")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    axes[-1].set_xlabel("time step (gray = reference / training steps 1-29; recall there is in-sample)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "drift_monitor.png", dpi=120)
    plt.close(fig)


def write_md(sig, rep, top, n_bins, lo, hi, thr, name):
    from src.ml.phase2b import md_table

    lines = [
        "# Drift monitor (label-free)", "",
        f"Model: {name}. Threshold {thr:.4f} (cost ratio C=10, an assumption). Reference = all nodes in steps 1-29.",
        f"Score PSI uses {n_bins} reference-quantile bins. Feature PSI = mean over the top {TOP_FEATURES} features by "
        f"mean |SHAP| on labeled training rows: {', '.join(top)}.",
        f"PSI cutoffs are the conventional 0.10 (moderate) and 0.25 (major). Alert-rate 'major' = outside "
        f"[{lo:.4f}, {hi:.4f}] (mean +/- 3 SD over reference steps). No cutoff was tuned on test.",
        "Steps 1-29 are inside the reference, so their signals are near zero by construction; false alarms are "
        "only meaningful for steps 30-42. Recall in steps 1-29 is in-sample.", "",
        md_table(sig[["time_step", "n_nodes", "n_illicit", "score_psi", "feat_psi", "alert_rate", "recall",
                      "score_level", "feat_level", "alert_out_of_band"]], 3), "",
        "## Flags", "",
        "```", json.dumps(rep, indent=2), "```", "",
    ]
    (RESULTS_DIR / "drift.md").write_text("\n".join(lines) + "\n")
    print("\nFlags:")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()

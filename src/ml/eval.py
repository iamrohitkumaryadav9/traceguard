"""Evaluation metrics for the illicit (positive) class."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def pr_auc(y_true, scores):
    return float(average_precision_score(y_true, scores))


def roc_auc(y_true, scores):
    return float(roc_auc_score(y_true, scores))


def prf_at_threshold(y_true, scores, threshold):
    """Precision / recall / F1 for scores >= threshold. Undefined ratios are 0."""
    y_true = np.asarray(y_true).astype(bool)
    pred = np.asarray(scores) >= threshold
    tp = int((pred & y_true).sum())
    fp = int((pred & ~y_true).sum())
    fn = int((~pred & y_true).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def best_f1_threshold(y_true, scores):
    """Threshold that maximizes F1 (illicit class). Returns (threshold, f1)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision/recall have one more entry than thresholds
    p, r = precision[:-1], recall[:-1]
    denom = p + r
    f1 = np.divide(2 * p * r, denom, out=np.zeros_like(denom), where=denom > 0)
    i = int(np.argmax(f1))
    return float(thresholds[i]), float(f1[i])


def cost_curve(y_true, scores, c_miss, c_false=1.0):
    """Total cost (c_false * FP + c_miss * FN) for every candidate threshold (score >= thr).

    Returns (thresholds, costs), thresholds in descending order. The first entry is +inf
    (flag nothing).
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    last = np.r_[ss[1:] != ss[:-1], True]  # evaluate once per distinct score
    thr = np.r_[np.inf, ss[last]]
    n_pos = int(y.sum())
    fn = np.r_[n_pos, n_pos - tp[last]]
    fps = np.r_[0, fp[last]]
    return thr, c_false * fps + c_miss * fn


def best_cost_threshold(y_true, scores, c_miss, c_false=1.0):
    thr, cost = cost_curve(y_true, scores, c_miss, c_false)
    i = int(np.argmin(cost))  # ties -> highest threshold
    return float(thr[i]), float(cost[i])


def total_cost(y_true, scores, threshold, c_miss, c_false=1.0):
    y = np.asarray(y_true).astype(bool)
    pred = np.asarray(scores) >= threshold
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    return {"fp": fp, "fn": fn, "cost": c_false * fp + c_miss * fn}


def topk_metrics(y_true, scores, k):
    """Precision@k and recall@k for the top-k scored items (k capped at the number of items).

    recall is NaN if there are no positives.
    """
    y = np.asarray(y_true).astype(int)
    k_eff = min(k, len(y))
    top = np.argsort(-np.asarray(scores), kind="stable")[:k_eff]
    hits = int(y[top].sum())
    n_pos = int(y.sum())
    return {
        "precision": hits / k_eff if k_eff else np.nan,
        "recall": hits / n_pos if n_pos else np.nan,
        "max_recall": min(1.0, k_eff / n_pos) if n_pos else np.nan,
    }


def brier_score(y_true, probs):
    return float(np.mean((np.asarray(probs, dtype=float) - np.asarray(y_true)) ** 2))


def expected_calibration_error(y_true, probs, n_bins=10):
    """ECE with equal-width probability bins."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probs, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def per_step_metrics(time_steps, y_true, scores, threshold):
    """Per time step: n, illicit count, recall at threshold, PR-AUC (NaN unless both classes)."""
    df = pd.DataFrame(
        {"time_step": np.asarray(time_steps), "y": np.asarray(y_true).astype(int), "s": np.asarray(scores)}
    )
    rows = []
    for t, g in df.groupby("time_step"):
        n_ill = int(g["y"].sum())
        recall = float(((g["s"] >= threshold) & (g["y"] == 1)).sum() / n_ill) if n_ill else np.nan
        both = n_ill > 0 and n_ill < len(g)
        rows.append(
            {
                "time_step": int(t),
                "n": len(g),
                "n_illicit": n_ill,
                "recall": recall,
                "pr_auc": pr_auc(g["y"], g["s"]) if both else np.nan,
            }
        )
    return pd.DataFrame(rows)

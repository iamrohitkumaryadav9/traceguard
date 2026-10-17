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

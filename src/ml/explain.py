"""SHAP explanations for the Phase 2b model.

Features f1..f165 are anonymized in the Elliptic release, so they are reported by name and group
only. Their meaning is not known and is not guessed here.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from src.ml.graph_features import STRUCTURAL_COLS

MODEL_DIR = Path("models")
N_LOCAL = 93  # same ASSUMPTION as src/ml/baselines.py: f1-f93 local, f94-f165 aggregated

_cache = {}


def feature_group(name):
    if name in STRUCTURAL_COLS:
        return "structural"
    if name.startswith("f") and name[1:].isdigit():
        return "local" if int(name[1:]) <= N_LOCAL else "aggregated"
    raise ValueError(f"unknown feature {name!r}")


def shap_values_positive(shap_out):
    """Normalize shap's output (array or per-class list) to one (n, features) array for the positive class."""
    if isinstance(shap_out, list):
        return np.asarray(shap_out[-1])
    arr = np.asarray(shap_out)
    return arr[..., -1] if arr.ndim == 3 else arr


def _load():
    if not _cache:
        bundle = joblib.load(MODEL_DIR / "phase2b_model.pkl")
        cols = bundle["cols"]
        nodes = pd.read_parquet("data/processed/nodes.parquet", columns=["txId"] + [c for c in cols if c not in STRUCTURAL_COLS])
        gf = pd.read_parquet("data/processed/graph_features.parquet")
        table = nodes.merge(gf, on="txId", how="left", validate="one_to_one").set_index("txId")
        _cache.update(
            explainer=joblib.load(MODEL_DIR / "explainer.pkl"), cols=cols, table=table, name=bundle["name"]
        )
    return _cache


def explain(node_ids, top_n=5):
    """Top features per node by |SHAP|.

    Returns {txId: [{"feature", "group", "value", "shap", "direction"}, ...]}.
    SHAP values are in log-odds of the model's raw score; positive = pushes toward illicit.
    """
    c = _load()
    node_ids = list(node_ids)
    X = c["table"].loc[node_ids, c["cols"]]
    sv = shap_values_positive(c["explainer"].shap_values(X))
    out = {}
    for row, node in enumerate(node_ids):
        top = np.argsort(-np.abs(sv[row]))[:top_n]
        out[node] = [
            {
                "feature": c["cols"][j],
                "group": feature_group(c["cols"][j]),
                "value": float(X.iloc[row, j]),
                "shap": float(sv[row, j]),
                "direction": "raises illicit score" if sv[row, j] > 0 else "lowers illicit score",
            }
            for j in top
        ]
    return out

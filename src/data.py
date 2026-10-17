"""Loading the Elliptic Bitcoin dataset."""
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES_FILE = "elliptic_txs_classes.csv"
EDGES_FILE = "elliptic_txs_edgelist.csv"
FEATURES_FILE = "elliptic_txs_features.csv"

# Expected raw codes. load_elliptic prints the real values before mapping.
LABEL_MAP = {"1": "illicit", "2": "licit", "unknown": "unknown"}


def load_elliptic(raw_dir):
    """Return (nodes, edges).

    nodes: txId, time_step, f1..fN, label (illicit / licit / unknown)
    edges: txId1, txId2
    """
    raw_dir = Path(raw_dir)

    classes = pd.read_csv(raw_dir / CLASSES_FILE, dtype={"class": str})
    counts = classes["class"].value_counts(dropna=False)
    print("Unique values in class column:")
    for value, n in counts.items():
        print(f"  {value!r}: {n}")
    unexpected = set(counts.index) - set(LABEL_MAP)
    if unexpected:
        raise ValueError(f"Unexpected class codes {unexpected}; expected {set(LABEL_MAP)}")
    classes["label"] = classes["class"].map(LABEL_MAP)
    classes = classes[["txId", "label"]]

    edges = pd.read_csv(raw_dir / EDGES_FILE)

    feats = pd.read_csv(raw_dir / FEATURES_FILE, header=None)
    n_feat = feats.shape[1] - 2
    feats.columns = ["txId", "time_step"] + [f"f{i}" for i in range(1, n_feat + 1)]
    feats["time_step"] = feats["time_step"].astype(int)
    feat_cols = feats.columns[2:]
    feats[feat_cols] = feats[feat_cols].astype(np.float32)
    feats = feats.copy()  # defragment after the column-wise assignment

    nodes = feats.merge(classes, on="txId", how="left", validate="one_to_one")
    return nodes, edges

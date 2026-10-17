"""Temporal train/test split for Elliptic."""
from pathlib import Path

import pandas as pd

from src.data import load_elliptic

TRAIN_STEPS = (1, 34)  # full train period (fit + validation), used by assign_split
TEST_STEPS = (35, 49)

# Phase 2a: model fitting and tuning inside the train period
FIT_STEPS = (1, 29)
VAL_STEPS = (30, 34)


def assign_split(nodes):
    """Add a 'split' column: train / test for labeled nodes, 'unlabeled' otherwise."""
    split = pd.Series("unlabeled", index=nodes.index)
    labeled = nodes["label"].isin(["licit", "illicit"])
    t = nodes["time_step"]
    split[labeled & t.between(*TRAIN_STEPS)] = "train"
    split[labeled & t.between(*TEST_STEPS)] = "test"
    return nodes.assign(split=split)


def build(raw_dir="data/raw", out_dir="data/processed"):
    nodes, edges = load_elliptic(raw_dir)

    max_t = nodes["time_step"].max()
    print(f"time_step range: {nodes['time_step'].min()}..{max_t}")
    if max_t != TEST_STEPS[1]:
        print(f"WARNING: real max time_step is {max_t}, not {TEST_STEPS[1]}")

    nodes = assign_split(nodes)
    print("\nLabel counts per split:")
    print(pd.crosstab(nodes["split"], nodes["label"]))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nodes.to_parquet(out / "nodes.parquet", index=False)
    edges.to_parquet(out / "edges.parquet", index=False)
    print(f"\nSaved {out / 'nodes.parquet'} and {out / 'edges.parquet'}")
    return nodes, edges


if __name__ == "__main__":
    build()

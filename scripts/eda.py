"""EDA for the Elliptic dataset. Run from the project root: python -m scripts.eda"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import load_elliptic

FIG_DIR = Path("notebooks/figures")


def main(raw_dir="data/raw"):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    nodes, edges = load_elliptic(raw_dir)

    n_feat = len([c for c in nodes.columns if c.startswith("f")])
    print(f"\nNodes: {len(nodes):,}")
    print(f"Edges: {len(edges):,}")
    print(f"Feature columns: {n_feat}")
    print(f"time_step: min {nodes['time_step'].min()}, max {nodes['time_step'].max()}")

    counts = nodes["label"].value_counts()
    print("\nLabel counts:")
    for k, v in counts.items():
        print(f"  {k:8s} {v:>8,}  {100 * v / len(nodes):.2f}%")

    labeled = nodes[nodes["label"] != "unknown"]
    per_t = labeled.groupby("time_step")["label"].agg(
        n="size", illicit=lambda s: (s == "illicit").sum()
    )
    per_t["illicit_frac"] = per_t["illicit"] / per_t["n"]
    print("\nIllicit fraction per time step (labeled nodes):")
    print(per_t.round(4).to_string())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(per_t.index, per_t["illicit_frac"], marker="o")
    ax.axvline(34.5, color="gray", ls="--", label="train/test split")
    ax.set(xlabel="time step", ylabel="illicit fraction", title="Illicit fraction per time step (labeled)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "illicit_fraction_per_step.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(per_t.index, per_t["n"])
    ax.set(xlabel="time step", ylabel="labeled nodes", title="Labeled nodes per time step")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "labeled_per_step.png", dpi=120)
    plt.close(fig)

    in_deg = edges["txId2"].value_counts().reindex(nodes["txId"], fill_value=0)
    out_deg = edges["txId1"].value_counts().reindex(nodes["txId"], fill_value=0)
    print("\nDegree summary:")
    print(pd.DataFrame({"in_degree": in_deg.describe(), "out_degree": out_deg.describe()}).round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, deg, name in zip(axes, (in_deg, out_deg), ("in-degree", "out-degree")):
        vc = deg.value_counts().sort_index()
        vc = vc[vc.index > 0]
        ax.loglog(vc.index, vc.values, marker=".", ls="none")
        ax.set(xlabel=name, ylabel="nodes", title=f"{name} distribution (log-log, degree>0)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "degree_distributions.png", dpi=120)
    plt.close(fig)

    ids = set(nodes["txId"])
    missing1 = (~edges["txId1"].isin(ids)).sum()
    missing2 = (~edges["txId2"].isin(ids)).sum()
    print("\nChecks:")
    print(f"  edges with txId1 not in features: {missing1}")
    print(f"  edges with txId2 not in features: {missing2}")
    print(f"  duplicate edges: {edges.duplicated().sum()}")
    print(f"  self loops: {(edges['txId1'] == edges['txId2']).sum()}")
    print(f"  duplicate txIds in features: {nodes['txId'].duplicated().sum()}")

    early = labeled[labeled["time_step"].between(1, 34)]
    late = labeled[labeled["time_step"].between(35, 49)]
    print("\nIllicit fraction among labeled nodes:")
    print(f"  early (1-34): {(early['label'] == 'illicit').mean():.4f}")
    print(f"  late (35-49): {(late['label'] == 'illicit').mean():.4f}")


if __name__ == "__main__":
    main()

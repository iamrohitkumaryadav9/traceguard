"""Label-free structural graph features from the directed edgelist (txId1 -> txId2).

Elliptic has no edges between time steps, so these features cannot mix periods.
No neighbor labels are used anywhere.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

STRUCTURAL_COLS = [
    "in_deg", "out_deg", "fwd_2hop", "bwd_2hop", "wcc_size",
    "nbr_max_in_deg", "nbr_max_out_deg", "chain_len",
]
CHAIN_CAP = 50
OUT = Path("data/processed/graph_features.parquet")


def edges_cross_time_steps(nodes, edges):
    """Number of edges whose endpoints sit in different time steps."""
    t = nodes.set_index("txId")["time_step"]
    return int((t.reindex(edges["txId1"]).to_numpy() != t.reindex(edges["txId2"]).to_numpy()).sum())


def _row_max(S, vals):
    """Max of vals[col] over each CSR row's entries; 0 for empty rows."""
    out = np.zeros(S.shape[0], dtype=np.int64)
    nonempty = np.diff(S.indptr) > 0
    if nonempty.any():
        out[nonempty] = np.maximum.reduceat(vals[S.indices], S.indptr[:-1][nonempty])
    return out


def _within_2_hops(M):
    """Distinct nodes reachable in 1 or 2 hops, excluding the node itself."""
    R = (M + M @ M).tocsr()
    return R.getnnz(axis=1) - (R.diagonal() > 0).astype(np.int64)


def _chain_counts(nxt, cap):
    """Number of consecutive chain steps followed through nxt (-1 = stop), at most cap."""
    cnt = np.zeros(len(nxt), dtype=np.int64)
    cur = nxt.copy()
    for _ in range(cap):
        alive = cur >= 0
        if not alive.any():
            break
        cnt += alive
        cur = np.where(alive, nxt[np.where(alive, cur, 0)], -1)
    return cnt


def compute_graph_features(node_ids, edges):
    """DataFrame indexed like node_ids with STRUCTURAL_COLS."""
    node_ids = np.asarray(node_ids)
    n = len(node_ids)
    idx = pd.Index(node_ids)
    src = idx.get_indexer(edges["txId1"])
    dst = idx.get_indexer(edges["txId2"])
    if (src < 0).any() or (dst < 0).any():
        raise ValueError("edge endpoint missing from node ids")
    keep = src != dst  # drop self loops
    src, dst = src[keep], dst[keep]

    A = sp.csr_matrix((np.ones(len(src), dtype=np.int32), (src, dst)), shape=(n, n))
    A.data[:] = 1  # collapse duplicate edges
    AT = A.T.tocsr()
    out_deg = A.getnnz(axis=1).astype(np.int64)
    in_deg = AT.getnnz(axis=1).astype(np.int64)

    fwd_2hop = _within_2_hops(A)
    bwd_2hop = _within_2_hops(AT)

    _, comp = connected_components(A, directed=True, connection="weak")
    wcc_size = np.bincount(comp)[comp]

    S = (A + AT).tocsr()
    S.data[:] = 1
    nbr_max_in = _row_max(S, in_deg)
    nbr_max_out = _row_max(S, out_deg)

    # Chain links: in_deg == 1 and out_deg in {1, 2}. Walk only through unambiguous links:
    # u -> v is a chain step if both are links and u has a single successor (out_deg == 1).
    is_link = (in_deg == 1) & np.isin(out_deg, (1, 2))
    # unique successor of every out_deg == 1 node, unique predecessor of every in_deg == 1 node
    succ = np.full(n, -1)
    pred = np.full(n, -1)
    one_out = out_deg[src] == 1
    succ[src[one_out]] = dst[one_out]
    one_in = in_deg[dst] == 1
    pred[dst[one_in]] = src[one_in]

    safe_succ = np.where(succ >= 0, succ, 0)
    nxt = np.where(is_link & (succ >= 0) & is_link[safe_succ], succ, -1)
    safe_pred = np.where(pred >= 0, pred, 0)
    prv = np.where(is_link & (pred >= 0) & is_link[safe_pred] & (out_deg[safe_pred] == 1), pred, -1)
    chain_len = np.where(
        is_link,
        np.minimum(1 + _chain_counts(nxt, CHAIN_CAP) + _chain_counts(prv, CHAIN_CAP), CHAIN_CAP),
        0,
    )

    return pd.DataFrame(
        {
            "in_deg": in_deg, "out_deg": out_deg, "fwd_2hop": fwd_2hop, "bwd_2hop": bwd_2hop,
            "wcc_size": wcc_size, "nbr_max_in_deg": nbr_max_in, "nbr_max_out_deg": nbr_max_out,
            "chain_len": chain_len,
        }
    )[STRUCTURAL_COLS]


def build(nodes_path="data/processed/nodes.parquet", edges_path="data/processed/edges.parquet"):
    nodes = pd.read_parquet(nodes_path, columns=["txId", "time_step"])
    edges = pd.read_parquet(edges_path)
    cross = edges_cross_time_steps(nodes, edges)
    print(f"Edges connecting different time steps: {cross} of {len(edges):,}")
    gf = compute_graph_features(nodes["txId"].to_numpy(), edges)
    gf.insert(0, "txId", nodes["txId"].to_numpy())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    gf.to_parquet(OUT, index=False)
    print(f"Saved {OUT}")
    print(gf[STRUCTURAL_COLS].describe().round(2).T.to_string())
    return gf


if __name__ == "__main__":
    build()

"""Shared, lazily loaded context for the agent tools.

SCORE SCALE: every score in this package is the RAW LightGBM predict_proba of the final model
(trained with scale_pos_weight, so it is not a calibrated probability). The cost threshold in
eval/results/threshold.json (0.2582) and the validation-F1 threshold (0.788) were both chosen on
this raw scale, and tools compare raw scores against them. A calibrated probability is offered
only as an extra, clearly named field.

Ground-truth labels are held privately (`_y`) for the delayed-label health check and for
evaluation code. Tools must never return them.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.ml.eval import best_f1_threshold, wilson_interval
from src.ml.graph_features import compute_graph_features
from src.split import FIT_STEPS, TRAIN_STEPS, VAL_STEPS

MODEL_PATH = Path("models/phase2b_model.pkl")
CALIBRATOR_PATH = Path("models/calibrator.pkl")
THRESHOLD_PATH = Path("eval/results/threshold.json")
NODES_PATH = Path("data/processed/nodes.parquet")
EDGES_PATH = Path("data/processed/edges.parquet")
SCORES_PATH = Path("data/processed/scores.parquet")

CHAIN_MIN = 4  # chain flagged at chain_len >= 4
FAN_MIN = 3  # minimum fan-in / fan-out threshold
FAN_PERCENTILE = 99


@dataclass
class Context:
    ids: np.ndarray
    time_step: np.ndarray
    score: np.ndarray
    A: sp.csr_matrix  # directed, same-time-step edges only
    AT: sp.csr_matrix
    S: sp.csr_matrix  # undirected
    in_deg: np.ndarray
    out_deg: np.ndarray
    chain_len: np.ndarray
    wcc_size: np.ndarray
    alert_threshold: float
    high_threshold: float
    fan_in_thr: int
    fan_out_thr: int
    reference_recall: float
    dropped_cross_step_edges: int
    explain_fn: Callable
    calibrate_fn: Optional[Callable] = None
    typology_stats: dict = field(default_factory=dict)  # pattern -> training-period evidence (steps 1-34 only)
    _y: np.ndarray = field(repr=False, default=None)  # 1 illicit, 0 licit, -1 unknown. PRIVATE.
    _index: dict = field(repr=False, default=None)

    def idx(self, tx_id):
        return self._index.get(int(tx_id))

    def pattern_flags(self, idx):
        """Boolean arrays (fan_in, fan_out, chain) for node indices `idx`."""
        idx = np.asarray(idx)
        return (self.in_deg[idx] >= self.fan_in_thr,
                self.out_deg[idx] >= self.fan_out_thr,
                self.chain_len[idx] >= CHAIN_MIN)

    def window_stats(self, steps):
        """(illicit count, illicit nodes scoring >= alert threshold) over labeled nodes in `steps`."""
        m = np.isin(self.time_step, list(steps)) & (self._y == 1)
        return int(m.sum()), int((self.score[m] >= self.alert_threshold).sum())


def evidence_direction(k, n, baseline):
    """Rule fixed in advance: the 95% Wilson interval of the flagged illicit rate vs the baseline rate."""
    lo, hi = wilson_interval(k, n)
    if n == 0:
        return "uninformative"
    if lo > baseline:
        return "raises_risk"
    if hi < baseline:
        return "lowers_risk"
    return "uninformative"


def compute_typology_stats(ctx):
    """Per-pattern illicit rate among LABELED nodes in the training period (steps 1-34) only."""
    m = (ctx._y >= 0) & (ctx.time_step >= TRAIN_STEPS[0]) & (ctx.time_step <= TRAIN_STEPS[1])
    idx = np.where(m)[0]
    y = ctx._y[idx] == 1
    baseline = float(y.mean()) if len(idx) else float("nan")
    stats = {}
    for name, flag in zip(("fan_in", "fan_out", "chain"), ctx.pattern_flags(idx)):
        n, k = int(flag.sum()), int(y[flag].sum())
        rate = k / n if n else float("nan")
        stats[name] = {
            "n_flagged": n, "observed_illicit_rate": rate, "baseline_illicit_rate": baseline,
            "lift": rate / baseline if n and baseline else float("nan"),
            "evidence_direction": evidence_direction(k, n, baseline) if len(idx) else "uninformative",
        }
    return stats


def build_context(nodes, edges, score, alert_threshold, high_threshold, explain_fn, calibrate_fn=None):
    """nodes: txId, time_step, label (illicit / licit / unknown). score aligned with nodes."""
    ids = nodes["txId"].to_numpy()
    t = nodes["time_step"].to_numpy()
    index = {int(i): k for k, i in enumerate(ids)}
    src = pd.Index(ids).get_indexer(edges["txId1"])
    dst = pd.Index(ids).get_indexer(edges["txId2"])
    ok = (src >= 0) & (dst >= 0) & (src != dst)
    same_step = np.zeros(len(src), bool)
    same_step[ok] = t[src[ok]] == t[dst[ok]]
    dropped = int((ok & ~same_step).sum())
    keep = ok & same_step  # tools never cross time steps, so cross-step edges are dropped here

    n = len(ids)
    A = sp.csr_matrix((np.ones(keep.sum(), dtype=np.int32), (src[keep], dst[keep])), shape=(n, n))
    A.data[:] = 1
    AT = A.T.tocsr()
    S = (A + AT).tocsr()
    S.data[:] = 1

    kept_edges = pd.DataFrame({"txId1": ids[src[keep]], "txId2": ids[dst[keep]]})
    gf = compute_graph_features(ids, kept_edges)
    in_deg, out_deg = gf["in_deg"].to_numpy(), gf["out_deg"].to_numpy()

    ref = t <= FIT_STEPS[1]
    fan_in = max(FAN_MIN, int(np.ceil(np.percentile(in_deg[ref], FAN_PERCENTILE)))) if ref.any() else FAN_MIN
    fan_out = max(FAN_MIN, int(np.ceil(np.percentile(out_deg[ref], FAN_PERCENTILE)))) if ref.any() else FAN_MIN

    y = nodes["label"].map({"illicit": 1, "licit": 0, "unknown": -1}).to_numpy().astype(np.int8)
    val = (t >= VAL_STEPS[0]) & (t <= VAL_STEPS[1]) & (y == 1)
    ref_recall = float((score[val] >= alert_threshold).mean()) if val.any() else float("nan")

    ctx = Context(
        ids=ids, time_step=t, score=np.asarray(score, dtype=float), A=A, AT=AT, S=S, in_deg=in_deg,
        out_deg=out_deg, chain_len=gf["chain_len"].to_numpy(), wcc_size=gf["wcc_size"].to_numpy(),
        alert_threshold=float(alert_threshold), high_threshold=float(high_threshold), fan_in_thr=fan_in,
        fan_out_thr=fan_out, reference_recall=ref_recall, dropped_cross_step_edges=dropped,
        explain_fn=explain_fn, calibrate_fn=calibrate_fn, _y=y, _index=index,
    )
    ctx.typology_stats = compute_typology_stats(ctx)
    return ctx


def _load_scores(nodes, bundle):
    """Scores for ALL nodes, cached in data/processed/scores.parquet (gitignored)."""
    if SCORES_PATH.exists() and SCORES_PATH.stat().st_mtime > MODEL_PATH.stat().st_mtime:
        cached = pd.read_parquet(SCORES_PATH)
        if len(cached) == len(nodes) and (cached["txId"].to_numpy() == nodes["txId"].to_numpy()).all():
            return cached["score"].to_numpy()
    score = bundle["model"].predict_proba(nodes[bundle["cols"]])[:, 1]
    pd.DataFrame({"txId": nodes["txId"].to_numpy(), "score": score}).to_parquet(SCORES_PATH, index=False)
    return score


def load_context():
    import joblib

    from src.ml.explain import explain

    cfg = json.loads(THRESHOLD_PATH.read_text())
    bundle = joblib.load(MODEL_PATH)
    if cfg["model"] != bundle["name"]:
        raise RuntimeError("threshold.json and models/ belong to different models")
    nodes = pd.read_parquet(NODES_PATH)
    score = _load_scores(nodes, bundle)
    nodes = nodes[["txId", "time_step", "label"]]
    edges = pd.read_parquet(EDGES_PATH)

    labeled = nodes["label"].isin(["licit", "illicit"]).to_numpy()
    val = labeled & nodes["time_step"].between(*VAL_STEPS).to_numpy()
    high, _ = best_f1_threshold((nodes["label"].to_numpy()[val] == "illicit").astype(int), score[val])

    calibrator = joblib.load(CALIBRATOR_PATH)
    return build_context(
        nodes, edges, score, cfg["default_threshold"], high,
        explain_fn=lambda tx_id, top_n=5: explain([tx_id], top_n=top_n)[tx_id],
        calibrate_fn=lambda s: float(calibrator.predict([s])[0]),
    )


_CTX = None


def get_context():
    global _CTX
    if _CTX is None:
        _CTX = load_context()
    return _CTX


def set_context(ctx):
    """Install a context (used by tests with synthetic data). Pass None to reset."""
    global _CTX
    _CTX = ctx

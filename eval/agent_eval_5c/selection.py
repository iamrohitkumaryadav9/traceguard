"""Fresh evaluation set, exactly as pre-registered in eval/agent_eval_5c_spec.md (step 4): seed 43, 30 nodes."""
import json
from pathlib import Path

import numpy as np

from eval.agent_eval.selection import DEMO_NODES, stratum_masks  # same stratum definitions as Phase 5a

SEED = 43
STRATA_5C = [
    ("1 high band, illicit", 4), ("2 high band, licit", 4), ("3 medium band", 3), ("4 chain-flagged", 4),
    ("5 fan_in or fan_out flagged", 4), ("6 low band", 3), ("7 late steps (43-49), band at least medium", 8),
]
CASES_5A = Path("eval/results/agent_eval/cases.json")


def excluded_nodes(cases_5a_path=CASES_5A):
    """The six demo nodes of Phases 4a/4b and the 40 development nodes of Phase 5a."""
    ids = set(DEMO_NODES)
    if Path(cases_5a_path).exists():
        ids |= {c["tx_id"] for c in json.loads(Path(cases_5a_path).read_text(encoding="utf-8"))}
    return ids


def select_cases_5c(ctx, excluded):
    """[{stratum, tx_id, time_step, ground_truth, score, candidates, k, note}] in stratum order."""
    rng = np.random.default_rng(SEED)
    taken, cases = set(excluded), []
    for (name, k), mask in zip(STRATA_5C, stratum_masks(ctx)):
        ids = np.sort(ctx.ids[mask])
        ids = np.array([i for i in ids if int(i) not in taken], dtype=ids.dtype)
        take = min(k, len(ids))
        picks = [int(ids[j]) for j in rng.choice(len(ids), size=take, replace=False)] if take else []
        for tx in picks:
            taken.add(tx)
            i = ctx.idx(tx)
            cases.append({"stratum": name, "tx_id": tx, "time_step": int(ctx.time_step[i]),
                          "ground_truth": {1: "illicit", 0: "licit", -1: "unknown"}[int(ctx._y[i])],
                          "score": round(float(ctx.score[i]), 6), "candidates": int(len(ids)), "k": k,
                          "note": f"only {take} candidates, all used" if take < k else ""})
    return cases

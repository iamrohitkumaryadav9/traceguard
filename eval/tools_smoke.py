"""Smoke run of every tool on three test alerts. Labels are used HERE ONLY, to choose examples.

Run: python -m eval.tools_smoke
"""
import json

import numpy as np

from src.split import SHUTDOWN_STEP, TEST_STEPS
from src.tools import run_tool
from src.tools.context import get_context

SEED = 42
MAX_LIST = 4


def shorten(obj):
    if isinstance(obj, dict):
        return {k: shorten(v) for k, v in obj.items()}
    if isinstance(obj, list):
        out = [shorten(v) for v in obj[:MAX_LIST]]
        if len(obj) > MAX_LIST:
            out.append(f"... +{len(obj) - MAX_LIST} more")
        return out
    return obj


def pick(ctx, mask, rng):
    cand = np.sort(ctx.ids[mask])
    return int(rng.choice(cand))


def main():
    ctx = get_context()
    rng = np.random.default_rng(SEED)
    t = ctx.time_step
    test = (t >= TEST_STEPS[0]) & (t <= TEST_STEPS[1])
    pre, post = test & (t < SHUTDOWN_STEP), test & (t >= SHUTDOWN_STEP)
    alert = ctx.score >= ctx.alert_threshold
    examples = {
        "pre-shutdown true positive": pick(ctx, pre & alert & (ctx._y == 1), rng),
        "pre-shutdown false positive": pick(ctx, pre & alert & (ctx._y == 0), rng),
        "post-shutdown illicit node the model missed": pick(ctx, post & ~alert & (ctx._y == 1), rng),
    }
    for title, tx in examples.items():
        step = int(t[ctx.idx(tx)])
        print("=" * 100)
        print(f"{title}: tx_id {tx}, time_step {step}")
        calls = [
            ("score_transaction", {"tx_id": tx}),
            ("get_neighbors", {"tx_id": tx, "hops": 1, "direction": "both", "limit": 25}),
            ("subgraph_summary", {"tx_id": tx, "radius": 2}),
            ("match_typology", {"tx_id": tx}),
            ("check_model_health", {"time_step": step}),
        ]
        for name, args in calls:
            print(f"--- {name}({json.dumps(args)})")
            print(json.dumps(shorten(run_tool(name, args)), indent=1))


if __name__ == "__main__":
    main()

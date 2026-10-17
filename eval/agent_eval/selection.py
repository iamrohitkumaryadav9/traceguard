"""Case selection exactly as pre-registered in eval/agent_eval_spec.md."""
import numpy as np

SEED = 42
DEMO_NODES = (12779397, 99175682, 97870846, 70360850, 106417568, 149159647)
STRATA = [  # (name, k)
    ("1 high band, illicit", 6), ("2 high band, licit", 6), ("3 medium band", 4), ("4 chain-flagged", 6),
    ("5 fan_in or fan_out flagged", 6), ("6 low band", 4), ("7 late steps (43-49), band at least medium", 8),
]


def stratum_masks(ctx):
    n = len(ctx.ids)
    t, y, s = ctx.time_step, ctx._y, ctx.score
    fan_in, fan_out, chain = ctx.pattern_flags(np.arange(n))
    pre = (t >= 35) & (t <= 42)
    return [
        pre & (s >= ctx.high_threshold) & (y == 1),
        pre & (s >= ctx.high_threshold) & (y == 0),
        pre & (s >= ctx.alert_threshold) & (s < ctx.high_threshold),
        pre & chain,
        pre & (fan_in | fan_out),
        pre & (s < ctx.alert_threshold),
        (t >= 43) & (t <= 49) & (s >= ctx.alert_threshold),
    ]


def select_cases(ctx):
    """[{stratum, tx_id, time_step, ground_truth, score, candidates, k}] in stratum order."""
    rng = np.random.default_rng(SEED)
    chosen, taken, cases = [], set(DEMO_NODES), []
    for (name, k), mask in zip(STRATA, stratum_masks(ctx)):
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

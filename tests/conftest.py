import pandas as pd
import pytest

from src.tools.context import build_context, set_context

ALERT, HIGH = 0.5, 0.8


def fake_explain(tx_id, top_n=5):
    return [
        {"feature": f"f{k}", "group": "local", "value": 0.1 * k, "shap": 1.0 / k,
         "direction": "raises illicit score"}
        for k in range(1, top_n + 1)
    ]


def make_ctx(nodes, edges, score_map=None):
    nodes = pd.DataFrame(nodes, columns=["txId", "time_step", "label", "score"])
    edges = pd.DataFrame(edges, columns=["txId1", "txId2"])
    return build_context(nodes[["txId", "time_step", "label"]], edges, nodes["score"].to_numpy(), ALERT, HIGH,
                         explain_fn=fake_explain, calibrate_fn=lambda s: s / 2)


@pytest.fixture
def ctx():
    """Time step 1: fan-in hub 4 <- 1,2,3; fan-out hub 5 -> 6,7,8; chain 10->11->12->13->14->15.
    Time step 2: nodes 20, 21 with a CROSS-STEP edge 14 -> 20 that tools must never follow."""
    nodes = [
        (1, 1, "licit", 0.1), (2, 1, "licit", 0.2), (3, 1, "illicit", 0.9), (4, 1, "illicit", 0.95),
        (5, 1, "licit", 0.6), (6, 1, "licit", 0.05), (7, 1, "unknown", 0.7), (8, 1, "licit", 0.3),
        (10, 1, "licit", 0.1), (11, 1, "licit", 0.2), (12, 1, "illicit", 0.55), (13, 1, "licit", 0.3),
        (14, 1, "licit", 0.4), (15, 1, "licit", 0.1),
        (20, 2, "illicit", 0.99), (21, 2, "licit", 0.99),
    ]
    edges = [(1, 4), (2, 4), (3, 4), (5, 6), (5, 7), (5, 8), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15),
             (14, 20), (20, 21)]
    c = make_ctx(nodes, edges)
    set_context(c)
    yield c
    set_context(None)


@pytest.fixture
def health_ctx():
    """Steps 30-45, 30 illicit + 5 licit per step. Illicit nodes alert up to step 39, none from step 40."""
    nodes, k = [], 1000
    for step in range(30, 46):
        for j in range(30):
            nodes.append((k, step, "illicit", 0.9 if step <= 39 else 0.1))
            k += 1
        for j in range(5):
            nodes.append((k, step, "licit", 0.1))
            k += 1
    c = make_ctx(nodes, [])
    set_context(c)
    yield c
    set_context(None)

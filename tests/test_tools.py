import json

import pytest

from src.tools import TOOLS, run_tool
from src.tools.health import CheckModelHealthInput, check_model_health
from src.tools.neighbors import GetNeighborsInput, get_neighbors
from src.tools.schemas import ToolError
from src.tools.score import ScoreTransactionInput, score_transaction
from src.tools.subgraph import SubgraphSummaryInput, subgraph_summary
from src.tools.typology import CAVEAT, MatchTypologyInput, match_typology

FORBIDDEN_EXACT = {"label", "labels", "y", "y_true", "ground_truth", "is_illicit", "class", "target"}
FORBIDDEN_SUBSTR = ("label", "truth")


def all_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from all_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from all_keys(v)


# ---------- registry ----------
def test_registry_complete():
    assert set(TOOLS) == {"score_transaction", "get_neighbors", "subgraph_summary", "match_typology",
                          "check_model_health", "search_regulations", "get_regulation_chunk"}
    for spec in TOOLS.values():
        assert spec.description and spec.input_model and callable(spec.fn)


# ---------- score_transaction ----------
def test_score_bands_and_alert(ctx):
    low = score_transaction(ScoreTransactionInput(tx_id=1))
    med = score_transaction(ScoreTransactionInput(tx_id=5))
    high = score_transaction(ScoreTransactionInput(tx_id=4))
    assert (low.risk_band, low.alert) == ("low", False)
    assert (med.risk_band, med.alert) == ("medium", True)
    assert (high.risk_band, high.alert) == ("high", True)
    assert high.evidence_id == "score:4" and high.time_step == 1
    assert len(high.reasons) == 5 and high.reasons[0].feature == "f1"
    assert high.score_scale == "raw_lightgbm_predict_proba"


# ---------- get_neighbors ----------
def test_neighbors_order_and_truncation(ctx):
    out = get_neighbors(GetNeighborsInput(tx_id=5, hops=1, direction="out"))
    assert [n.tx_id for n in out.neighbors] == [7, 8, 6]  # by score desc
    assert not out.truncated and out.total_found == 3
    out = get_neighbors(GetNeighborsInput(tx_id=5, hops=1, direction="out", limit=2))
    assert out.truncated and len(out.neighbors) == 2 and out.total_found == 3


def test_neighbors_direction_and_hops(ctx):
    out = get_neighbors(GetNeighborsInput(tx_id=12, hops=2, direction="both"))
    got = {n.tx_id: (n.hop, n.direction) for n in out.neighbors}
    assert got == {11: (1, "in"), 10: (2, "in"), 13: (1, "out"), 14: (2, "out")}
    assert get_neighbors(GetNeighborsInput(tx_id=12, hops=1, direction="in")).total_found == 1


def test_neighbors_never_cross_time_steps(ctx):
    assert ctx.dropped_cross_step_edges == 1
    out = get_neighbors(GetNeighborsInput(tx_id=14, hops=2, direction="both"))
    ids = {n.tx_id for n in out.neighbors}
    assert 20 not in ids and 21 not in ids
    assert get_neighbors(GetNeighborsInput(tx_id=20, hops=2, direction="both")).total_found == 1  # only 21
    sub = subgraph_summary(SubgraphSummaryInput(tx_id=14, radius=3))
    assert sub.n_nodes == 5  # 11..15 within 3 hops; node 20 (other step) excluded


# ---------- subgraph_summary ----------
def test_subgraph_summary_fan_in(ctx):
    out = subgraph_summary(SubgraphSummaryInput(tx_id=4, radius=2))
    assert (out.n_nodes, out.n_edges) == (4, 3)
    assert out.density == pytest.approx(3 / 12, abs=1e-6)
    assert (out.focal_in_degree, out.focal_out_degree) == (3, 0)
    assert (out.max_in_degree, out.max_out_degree) == (3, 1)
    assert out.n_alerted == 2 and out.share_alerted == pytest.approx(0.5)  # nodes 3 and 4
    assert out.wcc_size == 4


def test_subgraph_summary_chain(ctx):
    out = subgraph_summary(SubgraphSummaryInput(tx_id=12, radius=1))
    assert out.n_nodes == 3 and out.focal_chain_len == 4


# ---------- match_typology ----------
def test_typology_flags(ctx):
    fi = match_typology(MatchTypologyInput(tx_id=4))
    assert fi.flags == {"fan_in": True, "fan_out": False, "chain": False}
    assert [m.pattern for m in fi.matches] == ["fan_in"]
    assert fi.matches[0].caveat == CAVEAT and fi.matches[0].retrieval_query
    fo = match_typology(MatchTypologyInput(tx_id=5))
    assert fo.flags["fan_out"] and not fo.flags["fan_in"]
    ch = match_typology(MatchTypologyInput(tx_id=12))
    assert ch.flags["chain"] and ch.chain_len == 4
    none = match_typology(MatchTypologyInput(tx_id=1))
    assert not any(none.flags.values()) and none.matches == []


def test_typology_evidence_fields(ctx):
    fi = match_typology(MatchTypologyInput(tx_id=4)).matches[0]
    # fixture: 15 labeled nodes, 4 illicit -> baseline 4/15; fan_in flagged: node 4 only (illicit)
    assert fi.n_flagged == 1 and fi.observed_illicit_rate == pytest.approx(1.0)
    assert fi.baseline_illicit_rate == pytest.approx(4 / 15, abs=1e-6)
    assert fi.lift == pytest.approx(15 / 4, abs=1e-5)
    assert fi.evidence_direction == "uninformative"  # one node: Wilson interval is wide
    assert "lowers_risk" in fi.caveat and "never be described as suspicious" in fi.caveat
    assert "training-period" in fi.caveat


def test_lowers_risk_pattern_has_no_retrieval_query(ctx):
    from src.tools.typology import LOWERS_RISK_NOTE

    ctx.typology_stats["fan_in"]["evidence_direction"] = "lowers_risk"
    ctx.typology_stats["chain"]["evidence_direction"] = "raises_risk"
    fi = match_typology(MatchTypologyInput(tx_id=4)).matches[0]
    assert fi.evidence_direction == "lowers_risk"
    assert fi.retrieval_query is None and fi.note == LOWERS_RISK_NOTE
    ch = match_typology(MatchTypologyInput(tx_id=12)).matches[0]
    assert ch.evidence_direction == "raises_risk" and ch.retrieval_query and ch.note is None
    ctx.typology_stats["chain"]["evidence_direction"] = "uninformative"
    assert match_typology(MatchTypologyInput(tx_id=12)).matches[0].retrieval_query  # uninformative keeps the query


def test_evidence_direction_rule():
    from src.tools.context import evidence_direction

    assert evidence_direction(378, 1469, 0.1158) == "raises_risk"  # interval entirely above baseline
    assert evidence_direction(19, 1023, 0.1158) == "lowers_risk"  # entirely below
    assert evidence_direction(10, 100, 0.10) == "uninformative"  # interval straddles baseline
    assert evidence_direction(0, 0, 0.10) == "uninformative"


def test_typology_stats_use_training_steps_only():
    # an illicit node in step 40 must not change the stats (steps 1-34 only)
    from tests.conftest import make_ctx

    base = [(1, 1, "licit", 0.1), (2, 1, "illicit", 0.9), (3, 1, "licit", 0.1), (4, 1, "licit", 0.1)]
    edges = [(1, 4), (2, 4), (3, 4)]
    a = make_ctx(base, edges)
    b = make_ctx(base + [(50, 40, "illicit", 0.9), (51, 40, "illicit", 0.9)], edges)
    assert a.typology_stats["fan_in"] == b.typology_stats["fan_in"]
    assert a.typology_stats["fan_in"]["baseline_illicit_rate"] == pytest.approx(0.25)


# ---------- check_model_health ----------
def test_health_ok_degraded_insufficient(health_ctx):
    ok = check_model_health(CheckModelHealthInput(time_step=41))  # window 37-39, recall 1.0
    assert ok.status == "ok" and ok.window_steps == [37, 38, 39] and ok.window_recall == 1.0
    assert ok.reference_recall == 1.0 and ok.lag == 2
    assert check_model_health(CheckModelHealthInput(time_step=42)).status == "ok"  # 60/90 caught
    bad = check_model_health(CheckModelHealthInput(time_step=43))  # window 39-41, 30/90 caught
    assert bad.status == "degraded" and bad.window_recall == pytest.approx(1 / 3, abs=1e-6)
    few = check_model_health(CheckModelHealthInput(time_step=31))  # window has no data
    assert few.status == "insufficient_data" and few.n_illicit_in_window == 0


def test_health_uses_no_labels_beyond_lag(health_ctx):
    # at t=41 the window ends at 39 even though labels for 40, 41 exist
    out = check_model_health(CheckModelHealthInput(time_step=41))
    assert max(out.window_steps) <= 41 - out.lag


def test_health_unknown_step(health_ctx):
    assert isinstance(check_model_health(CheckModelHealthInput(time_step=999)), ToolError)


# ---------- cross-cutting ----------
def _outputs(ctx):
    return [
        run_tool("score_transaction", {"tx_id": 4}),
        run_tool("get_neighbors", {"tx_id": 12, "hops": 2}),
        run_tool("subgraph_summary", {"tx_id": 4}),
        run_tool("match_typology", {"tx_id": 12}),
        run_tool("score_transaction", {"tx_id": 999}),
    ]


def test_no_ground_truth_keys_anywhere(ctx):
    for out in _outputs(ctx):
        for k in all_keys(out):
            assert k not in FORBIDDEN_EXACT, k
            assert not any(s in k for s in FORBIDDEN_SUBSTR), k


def test_health_output_has_no_ground_truth_keys(health_ctx):
    out = run_tool("check_model_health", {"time_step": 43})
    for k in all_keys(out):
        assert k not in FORBIDDEN_EXACT and not any(s in k for s in FORBIDDEN_SUBSTR), k


def test_deterministic_and_json_serializable(ctx):
    a, b = _outputs(ctx), _outputs(ctx)
    assert a == b
    assert json.dumps(a) == json.dumps(b)


@pytest.mark.parametrize("tool,args", [
    ("score_transaction", {"tx_id": 999}),
    ("get_neighbors", {"tx_id": 999}),
    ("subgraph_summary", {"tx_id": 999}),
    ("match_typology", {"tx_id": 999}),
])
def test_unknown_tx_id_returns_structured_error(ctx, tool, args):
    out = run_tool(tool, args)
    assert out["error"] == "unknown_tx_id" and out["tool"] == tool and out["tx_id"] == 999
    assert out["evidence_id"] == f"error:{tool}:999"


def test_every_output_has_evidence_id(ctx):
    for out in _outputs(ctx):
        assert out["evidence_id"]

"""Phase 4b: signal conflict, period note, merged feature claims, background labelling, leak check. FakeLLM only."""
import json

import numpy as np
import pytest

from src.agents.build import assemble, deterministic_claims
from src.agents.checks import background_provenance_reasons, signal_conflict_reasons
from src.agents.conflict import detect_signal_conflict
from src.agents.memo import PERIOD_NOTE, memo_to_markdown
from src.agents.runtime import Deps, default_run_tool
from src.agents.graph import run_agent
from src.agents.state import AgentState, Claim, LedgerEntry, WriterOutput
from src.tools.context import set_context
from tests.conftest import make_ctx
from tests.fakes import FakeLLM, all_supported, carry_conflict, done, required_conflict

FOCAL = 4  # conftest ctx: score 0.95 (high), in-neighbours 1, 2, 3 with scores 0.1, 0.2, 0.9; fan-in pattern
HEALTH_STEP42_LOW = 1000 + 12 * 35 + 30  # health_ctx: first licit node of step 42 (health ok)
HEALTH_STEP43 = 1000 + 13 * 35  # health_ctx: first node of step 43 (health degraded)
SCORE_OK = Claim(text="The model gave transaction 4 a raw score of 0.95 at time step 1.", evidence_ids=["score:4"])


def run(llm, tx):
    deps = Deps(llm=llm, run_tool=default_run_tool)
    return run_agent(tx, deps)


def set_score(ctx, tx_id, value):
    ctx.score = ctx.score.copy()  # the loaded array may be read-only
    ctx.score[ctx.idx(tx_id)] = value


# ---------- signal conflict ----------
def test_conflict_low_band_plus_chain_pattern(ctx):
    set_score(ctx, 12, 0.1)  # low band
    ctx.typology_stats["chain"]["evidence_direction"] = "raises_risk"
    state = run(FakeLLM(chat_scripts={"investigator": [done()], "regulation": [done()]},
                        structured_scripts={"writer": carry_conflict([WriterOutput(activity_description=[
                            Claim(text="A chain pattern matched.", evidence_ids=["typology:12"])])]), "critic": all_supported}), 12)
    s = state.signal_conflict
    assert s.startswith("Signal conflict: the model score 0.1 (risk band low; evidence score:12)")
    assert "structural pattern chain (evidence direction raises_risk; evidence typology:12)" in s
    assert s.endswith("point in different directions.")
    assert state.signal_conflict_evidence_ids == ["score:12", "typology:12"]
    assert state.memo.signal_conflict == s and state.status == "verified"
    assert set(state.memo.signal_conflict_evidence_ids) <= {e.evidence_id for e in state.memo.evidence_table}
    assert "**Signal conflict" in memo_to_markdown(state.memo)


def test_conflict_high_band_with_only_lowers_risk_pattern(ctx):
    ctx.typology_stats["fan_in"]["evidence_direction"] = "lowers_risk"
    state = run(FakeLLM(chat_scripts={"investigator": [done()]},
                        structured_scripts={"writer": carry_conflict([WriterOutput(activity_description=[SCORE_OK])]),
                                            "critic": all_supported}), FOCAL)
    assert "structural pattern fan_in (evidence direction lowers_risk; evidence typology:4)" in state.signal_conflict
    assert "risk band high" in state.signal_conflict and state.status == "verified"


def test_conflict_is_null_when_signals_agree(ctx):
    llm = FakeLLM(chat_scripts={"investigator": [done()], "regulation": [done()]},
                  structured_scripts={"writer": carry_conflict([WriterOutput(activity_description=[SCORE_OK])]),
                                      "critic": all_supported})
    state = run(llm, FOCAL)  # high band, uninformative pattern, in-neighbour 3 is alerted: nothing contradicts
    assert state.signal_conflict is None and state.memo.signal_conflict is None
    assert state.signal_conflict_evidence_ids == [] and state.status == "verified"
    writer_prompt = [c for c in llm.calls if c["purpose"] == "writer"][0]["prompt"]
    assert "no signal conflict" in writer_prompt and required_conflict(writer_prompt) is None


def test_conflict_from_in_neighbour_alerts(ctx):
    set_score(ctx, 21, 0.1)  # low band; its in-neighbour 20 has score 0.99 (alerted)
    llm = FakeLLM()
    state = run(llm, 21)
    assert "in-neighbour alerts (at least one of the 1 in-neighbours alerted; evidence neighbors:21:1:in)" in state.signal_conflict
    # gating is unchanged by the conflict: a low band with no risk-raising pattern is still a no-LLM no-action run,
    # and the memo shows the conflict sentence
    assert state.status == "no_action_model_based" and llm.calls == []
    assert state.memo.signal_conflict == state.signal_conflict
    assert "neighbors:21:1:in" in {e.evidence_id for e in state.memo.evidence_table}


def test_conflict_high_band_when_no_in_neighbour_alerted(ctx):
    set_score(ctx, 3, 0.1)  # node 4's in-neighbours are now all below the alert threshold
    state = run(FakeLLM(chat_scripts={"investigator": [done()], "regulation": [done()]},
                        structured_scripts={"writer": carry_conflict([WriterOutput(activity_description=[SCORE_OK])]),
                                            "critic": all_supported}), FOCAL)
    assert "none of the 3 in-neighbours alerted; evidence neighbors:4:1:in" in state.signal_conflict


def test_detect_signal_conflict_pure_function():
    score = {"risk_band": "low", "score": 0.1, "evidence_id": "score:1"}
    typ_raises = {"evidence_id": "typology:1", "matches": [{"pattern": "chain", "evidence_direction": "raises_risk"}]}
    assert detect_signal_conflict(score, typ_raises, None)[0].startswith("Signal conflict: the model score 0.1")
    assert detect_signal_conflict(score, {"evidence_id": "typology:1", "matches": []}, None) == (None, [])
    high = {**score, "risk_band": "high", "score": 0.9}
    mixed = {"evidence_id": "typology:1", "matches": [{"pattern": "fan_in", "evidence_direction": "lowers_risk"},
                                                      {"pattern": "chain", "evidence_direction": "raises_risk"}]}
    assert detect_signal_conflict(high, mixed, None) == (None, [])  # not EVERY pattern lowers risk
    medium = {**score, "risk_band": "medium"}
    assert detect_signal_conflict(medium, typ_raises, {"evidence_id": "n", "total_found": 2, "neighbors": [{"alert": True}]}) == (None, [])
    both, ids = detect_signal_conflict(score, typ_raises, {"evidence_id": "neighbors:1:1:in", "total_found": 2, "neighbors": [{"alert": True}]})
    assert both.count("Signal conflict:") == 2 and ids == ["score:1", "typology:1", "neighbors:1:1:in"]
    assert detect_signal_conflict(score, None, {"evidence_id": "n", "total_found": 0, "neighbors": []}) == (None, [])


# ---------- the critic enforces the sentence ----------
def conflict_node(ctx):
    ctx.typology_stats["fan_in"]["evidence_direction"] = "lowers_risk"


def draft(conflict):
    return WriterOutput(signal_conflict=conflict, activity_description=[SCORE_OK])


def test_dropping_or_editing_signal_conflict_is_rejected_then_fixed(ctx):
    conflict_node(ctx)
    seen = {}

    def writer(prompt):
        required = required_conflict(prompt)
        n = writer.n
        writer.n += 1
        seen[n] = required
        return draft([None, required + " (minor edit)", required][n])  # round 1 drops it, round 2 edits it, round 3 is exact

    writer.n = 0
    state = run(FakeLLM(chat_scripts={"investigator": [done()]}, structured_scripts={"writer": writer, "critic": all_supported}), FOCAL)
    verdicts = [v for v in state.critic_verdicts if v.item == "signal_conflict"]
    assert [v.ok for v in verdicts] == [False, False, True]
    assert "missing" in verdicts[0].deterministic_reasons[0] and "altered" in verdicts[1].deterministic_reasons[0]
    assert state.critic_loops == 2 and state.status == "verified"
    assert state.memo.signal_conflict == seen[0]  # the exact required sentence


def test_writer_that_never_carries_the_sentence_ends_partially_verified_with_code_inserted_sentence(ctx):
    conflict_node(ctx)
    llm = FakeLLM(chat_scripts={"investigator": [done()]},
                  structured_scripts={"writer": lambda prompt: draft(None), "critic": all_supported})
    state = run(llm, FOCAL)
    assert state.status == "partially_verified" and state.critic_loops == 2 and llm.count(purpose="writer") == 3
    assert state.memo.signal_conflict and "inserted by code" in " ".join(state.memo.notes)
    assert [c.text for c in state.memo.activity_description] == [SCORE_OK.text]  # the good claim is kept


def test_writer_cannot_invent_a_conflict(ctx):
    state = run(FakeLLM(chat_scripts={"investigator": [done()], "regulation": [done()]},
                        structured_scripts={"writer": lambda p: draft("Signal conflict: made up."), "critic": all_supported}), FOCAL)
    v = [v for v in state.critic_verdicts if v.item == "signal_conflict"][0]
    assert not v.ok and "must be null" in v.deterministic_reasons[0]
    assert state.status == "partially_verified" and state.memo.signal_conflict is None


def test_signal_conflict_reasons_unit():
    assert signal_conflict_reasons(None, None) == []
    assert signal_conflict_reasons("S", "S") == []
    assert "missing" in signal_conflict_reasons("S", None)[0]
    assert "altered" in signal_conflict_reasons("S", "S.")[0]
    assert "must be null" in signal_conflict_reasons(None, "S")[0]


# ---------- period note and fixed limitations ----------
def test_period_note_at_step_43_and_later_but_not_at_step_42(health_ctx):
    s42 = run(FakeLLM(), HEALTH_STEP42_LOW)
    s43 = run(FakeLLM(), HEALTH_STEP43)
    assert s42.status == "no_action_model_based" and PERIOD_NOTE not in s42.memo.limitations
    assert s43.status == "cannot_assess" and PERIOD_NOTE in s43.memo.limitations  # gating unchanged: still decided by health
    assert PERIOD_NOTE.startswith("Dataset-specific note, based on hindsight from the published test results:")
    assert s43.memo.header.risk_band is None


def test_period_note_threshold_via_assemble():
    for step, expected in ((42, False), (43, True), (49, True)):
        st = AgentState(tx_id=1, time_step=step)
        memo = assemble(st, "now", "verified", [], [], rated=False)
        assert (PERIOD_NOTE in memo.limitations) is expected


def test_health_monitor_lag_limitation_is_in_every_memo(health_ctx):
    memo = run(FakeLLM(), HEALTH_STEP42_LOW).memo
    text = " ".join(memo.limitations)
    assert "2-step lag and a 3-step window" in text and "it first flags at step 47" in text


# ---------- merged feature claims ----------
REASONS = [{"feature": "f57", "direction": "raises illicit score", "shap_value": 1.5},
           {"feature": "f12", "direction": "raises illicit score", "shap_value": 0.9},
           {"feature": "f90", "direction": "lowers illicit score", "shap_value": -0.4}]


def test_deterministic_feature_claim_is_one_claim_citing_the_score():
    st = AgentState(tx_id=4)
    st.ledger["score:4"] = LedgerEntry(tool="score_transaction", output={
        "evidence_id": "score:4", "time_step": 1, "score": 0.9, "risk_band": "high", "alert_threshold": 0.25,
        "high_threshold": 0.79, "reasons": REASONS})
    claims = deterministic_claims(st)
    feature_claims = [c for c in claims if "f57" in c.text or "f12" in c.text or "f90" in c.text]
    assert len(feature_claims) == 1
    assert feature_claims[0].text.startswith("Features f57, f12 pushed the score up; feature f90 pushed the score down")
    assert feature_claims[0].evidence_ids == ["score:4"]


def test_critic_rejects_split_feature_claims_and_accepts_one_merged_claim(ctx):
    split = WriterOutput(activity_description=[
        Claim(text="Feature f53 pushed the score up.", evidence_ids=["score:4"]),
        Claim(text="Feature f59 pushed the score up.", evidence_ids=["score:4"])])
    reasons = ctx.explain_fn(4, 5)  # fake reasons are all 'raises'; use two of them in one claim
    merged = WriterOutput(activity_description=[Claim(text="Features f1 and f2 pushed the score up.", evidence_ids=["score:4"])])
    llm = FakeLLM(chat_scripts={"investigator": [done()], "regulation": [done()]},
                  structured_scripts={"writer": carry_conflict([split, merged]), "critic": all_supported})
    state = run(llm, FOCAL)
    r1 = [v for v in state.critic_verdicts if v.round == 1 and v.item.startswith("claim")]
    assert r1[0].ok is True or r1[0].deterministic_reasons == []  # the first feature claim is allowed ...
    assert any("must be merged into one claim" in " ".join(v.deterministic_reasons) for v in r1)  # ... the second is not
    assert state.status == "verified" and len(state.memo.activity_description) == 1


# ---------- background labelling ----------
def test_markdown_labels_regulation_text_as_background_not_evidence(ctx):
    state = run(FakeLLM(chat_scripts={"investigator": [done()], "regulation": [done()]},
                        structured_scripts={"writer": carry_conflict([WriterOutput(activity_description=[SCORE_OK])]),
                                            "critic": all_supported}), FOCAL)
    md = memo_to_markdown(state.memo)
    assert "Regulatory background (context, not evidence about this transaction)" in md
    assert "Regulatory references" not in md
    assert hasattr(state.memo, "regulatory_background") and not hasattr(state.memo, "regulatory_references")


def test_background_must_come_from_a_pattern_triggered_search():
    typ_none = LedgerEntry(tool="match_typology", output={"matches": [{"pattern": "fan_in", "retrieval_query": None}]})
    typ_query = LedgerEntry(tool="match_typology", output={"matches": [{"pattern": "chain", "retrieval_query": "q"}]})
    hit = LedgerEntry(tool="search_regulations", output={"evidence_id": "reg:d:p1:c0"})
    assert background_provenance_reasons("reg:d:p1:c0", {"t": typ_query, "reg:d:p1:c0": hit}) == []
    assert "not triggered by a matched pattern" in background_provenance_reasons("reg:d:p1:c0", {"t": typ_none, "reg:d:p1:c0": hit})[0]
    assert "does not come from a regulation search" in background_provenance_reasons("reg:d:p1:c0", {"t": typ_query})[0]
    assert "does not come from a regulation search" in background_provenance_reasons("score:1", {"score:1": typ_query})[0]


# ---------- leak check ----------
def build_leak_ctx(label):
    """Focal node 5000 sits in test step 36 (its own label is outside every label-derived aggregate: health windows
    and typology statistics only use steps <= 34)."""
    nodes, k = [], 1
    for step in range(30, 38):
        for j in range(6):
            nodes.append((k, step, "illicit" if j < 3 else "licit", 0.9 if j < 3 else 0.1))
            k += 1
    nodes.append((5000, 36, label, 0.95))
    return make_ctx(nodes, [(1, 5000), (2, 5000)])


def normalise(state):
    memo = state.memo.model_dump()
    memo["header"]["generated_at"] = "T"
    events = [{k: v for k, v in e.items() if k not in ("t", "ms", "latency_ms")} for e in state.trace]
    return {"memo": memo, "events": events, "ledger": {k: v.model_dump() for k, v in state.ledger.items()},
            "status": state.status, "verdicts": [v.model_dump() for v in state.critic_verdicts]}


def test_ground_truth_label_never_reaches_memo_trace_or_prompts():
    results = {}
    for label in ("illicit", "licit", "unknown"):
        set_context(build_leak_ctx(label))
        try:
            llm = FakeLLM(chat_scripts={"investigator": [done()], "regulation": [done()]},
                          structured_scripts={"writer": carry_conflict([WriterOutput(activity_description=[
                              Claim(text="The model gave transaction 5000 a raw score of 0.95 at time step 36.", evidence_ids=["score:5000"])])]),
                              "critic": all_supported})
            state = run(llm, 5000)
        finally:
            set_context(None)
        assert state.status == "verified"
        results[label] = (normalise(state), llm.calls)
    base_state, base_calls = results["illicit"]
    for label in ("licit", "unknown"):
        # flipping the focal node's ground truth changes NOTHING: memo, trace, ledger, verdicts and every LLM prompt are identical
        assert json.dumps(results[label][0], sort_keys=True) == json.dumps(base_state, sort_keys=True)
        assert results[label][1] == base_calls
    blob = json.dumps(base_state) + json.dumps(base_calls, default=str)
    for forbidden in ('"label"', "ground_truth", "ground truth", '"y_true"'):
        assert forbidden not in blob

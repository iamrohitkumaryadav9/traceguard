import json

import pytest
from pydantic import ValidationError

from src.agents import config
from src.agents.graph import run_agent, trace_dict
from src.agents.memo import NOT_AVAILABLE, RECOMMEND_NO_ACTION, RECOMMEND_REVIEW, SARMemo, SubjectFields
from src.agents.runtime import Deps, default_run_tool
from src.agents.state import Claim, RegReference, WriterOutput
from src.rag.retriever import Retriever, set_retriever
from tests.fakes import FakeLLM, all_supported, call, carry_conflict, done
from tests.test_rag import FakeEmbedder, mk_chunk

HEALTH_DEGRADED_TX = 1000 + 13 * 35  # health_ctx: first node of time step 43 (window recall 1/3 < 0.5)
HEALTH_OK_LOW_TX = 1000 + 5 * 35 + 30  # health_ctx: first licit node of time step 35, score 0.1 (low band), no edges
FOCAL = 4  # ctx fixture: score 0.95 (high band), fan-in pattern, time step 1 (health: insufficient_data)


def run(llm, tx, run_tool=default_run_tool):
    deps = Deps(llm=llm, run_tool=run_tool)
    state = run_agent(tx, deps)
    return state, deps


def good_writer(score=0.95):
    return WriterOutput(activity_description=[
        Claim(text=f"The model gave transaction 4 a raw score of {score} at time step 1.", evidence_ids=["score:4"])])


# ---------- triage ----------
def test_degraded_health_gives_cannot_assess_and_no_rating(health_ctx):
    llm = FakeLLM()
    state, _ = run(llm, HEALTH_DEGRADED_TX)
    assert state.status == "cannot_assess"
    assert llm.calls == []  # no LLM, and the other nodes are skipped
    assert state.nodes_visited == ["triage", "finalize"]
    m = state.memo
    assert m.header.risk_band is None and m.header.score is None and m.header.calibrated_probability is None
    assert m.header.model_health["status"] == "degraded"
    dumped = m.model_dump_json().lower()
    assert "low risk" not in dumped and "low-risk" not in dumped
    assert any("cannot be relied on" in c.text for c in m.activity_description)
    assert m.recommendation == RECOMMEND_REVIEW


def test_low_band_without_flags_gives_no_action_and_zero_llm_calls(health_ctx):
    llm = FakeLLM()
    state, _ = run(llm, HEALTH_OK_LOW_TX)
    assert state.status == "no_action_model_based"
    assert state.llm_call_count == 0 and llm.calls == []
    assert state.memo.recommendation == RECOMMEND_NO_ACTION
    assert state.memo.header.risk_band == "low"
    assert state.memo.verification == "not applicable"


def test_unknown_transaction_ends_cleanly(ctx):
    state, _ = run(FakeLLM(), 999999)
    assert state.status == "unknown_transaction" and state.memo is not None
    assert state.memo.subject.names == NOT_AVAILABLE


# ---------- caps ----------
def test_tool_call_cap_is_enforced(ctx):
    seen = []

    def forever(n, content):  # tries a new tool call every turn
        return call("get_neighbors", tx_id=FOCAL, hops=1 + n % 2, direction="both", limit=1 + n)

    def spy(name, args):
        seen.append(name)
        return default_run_tool(name, args)

    llm = FakeLLM(chat_scripts={"investigator": forever})
    state, _ = run(llm, FOCAL, run_tool=spy)
    assert state.status == "budget_exceeded"
    assert state.tool_call_count == config.MAX_TOOL_CALLS == 8
    assert sum(1 for e in state.trace if e["event"] == "tool_call" and e["node"] == "investigator") == 8
    assert seen.count("get_neighbors") == 9  # 1 deterministic triage call + 8 investigator calls; the 9th investigator attempt never ran
    assert state.memo is not None and "budget" in " ".join(state.memo.notes)
    assert state.memo.subject.addresses == NOT_AVAILABLE


def test_llm_call_cap_ends_with_budget_exceeded(ctx, monkeypatch):
    monkeypatch.setattr(config, "MAX_LLM_CALLS", 2)
    llm = FakeLLM(chat_scripts={"investigator": lambda n, c: call("get_neighbors", tx_id=FOCAL, limit=1 + n)})
    state, _ = run(llm, FOCAL)
    assert state.status == "budget_exceeded" and state.llm_call_count == 2
    assert state.memo is not None


def test_search_cap_is_enforced(ctx):
    set_retriever(Retriever([mk_chunk("d", "US", 1, 0, "text about structuring")], FakeEmbedder()))
    try:
        llm = FakeLLM(chat_scripts={"investigator": [done()],
                                    "regulation": lambda n, c: call("search_regulations", query=f"structuring query {n}")})
        state, _ = run(llm, FOCAL)
        assert state.status == "budget_exceeded" and state.search_count == config.MAX_SEARCHES == 3
        assert state.memo is not None
    finally:
        set_retriever(None)


def test_unexpected_error_does_not_crash(ctx):
    def boom(prompt):
        raise RuntimeError("network down")

    llm = FakeLLM(chat_scripts={"investigator": [done()], "regulation": [done()]}, structured_scripts={"writer": boom})
    state, _ = run(llm, FOCAL)
    assert state.status == "agent_error" and "network down" in state.error
    assert state.memo is not None and state.memo.recommendation == RECOMMEND_REVIEW


def test_llm_cannot_switch_on_rewrite_or_rerank(ctx):
    seen = []

    def spy(name, args):
        seen.append((name, dict(args)))
        return default_run_tool(name, args)

    set_retriever(Retriever([mk_chunk("d", "US", 1, 0, "text about structuring")], FakeEmbedder()))
    try:
        llm = FakeLLM(chat_scripts={"investigator": [done()],
                                    "regulation": [call("search_regulations", query="structuring", rewrite=True, rerank=True), done()]},
                      structured_scripts={"writer": good_writer(), "critic": all_supported})
        run(llm, FOCAL, run_tool=spy)
    finally:
        set_retriever(None)
    search_args = [a for n, a in seen if n == "search_regulations"][0]
    assert "rewrite" not in search_args and "rerank" not in search_args
    for chat in llm.chats_started:
        for spec in chat["specs"]:
            assert "rewrite" not in spec["parameters"]["properties"]


# ---------- critic ----------
def test_uncited_claim_is_rejected_then_fixed(ctx):
    bad = WriterOutput(activity_description=[Claim(text="The transaction looks unusual.", evidence_ids=[])])
    llm = FakeLLM(chat_scripts={"investigator": [done()], "regulation": [done()]},
                  structured_scripts={"writer": [bad, good_writer()], "critic": all_supported})
    state, _ = run(llm, FOCAL)
    first = [v for v in state.critic_verdicts if v.round == 1]
    assert not first[0].ok and "uncited" in first[0].deterministic_reasons[0]
    assert llm.count(purpose="writer") == 2 and state.critic_loops == 1
    assert state.status == "verified" and state.memo.verification == "verified"
    second_prompt = [c for c in llm.calls if c["purpose"] == "writer"][1]["prompt"]
    assert "rejected by the critic" in second_prompt.lower() and "uncited" in second_prompt


def test_claim_calling_fan_in_suspicious_is_rejected(ctx):
    ctx.typology_stats["fan_in"]["evidence_direction"] = "lowers_risk"
    bad = WriterOutput(activity_description=[
        Claim(text="The transaction matches the fan_in pattern, which is suspicious.", evidence_ids=["typology:4"])])
    ok = WriterOutput(activity_description=[
        Claim(text="The fan_in pattern matched, but in training-period data it was associated with a lower illicit rate.",
              evidence_ids=["typology:4"])])
    # high band with only a lowers_risk pattern is also a signal conflict, which the writer must carry unchanged
    llm = FakeLLM(chat_scripts={"investigator": [done()]},
                  structured_scripts={"writer": carry_conflict([bad, ok]), "critic": all_supported})
    state, _ = run(llm, FOCAL)
    assert "regulation" not in state.nodes_visited  # a lowers_risk pattern has a null retrieval_query
    r1 = [v for v in state.critic_verdicts if v.round == 1][0]
    assert not r1.ok and "lowers_risk pattern fan_in" in r1.deterministic_reasons[0]
    assert state.status == "verified"
    assert state.memo.signal_conflict and "lowers_risk" in state.memo.signal_conflict
    assert state.ledger["typology:4"].output["matches"][0]["retrieval_query"] is None


def test_quote_from_non_quotable_document_is_rejected(ctx):
    text = "Reporting entities shall file a suspicious transaction report irrespective of the amount."
    chunks = [{**mk_chunk("doc_ocr", "India", 1, 0, text), "quotable": False},
              mk_chunk("doc_pdf", "US", 1, 0, "Money transmitters must register with FinCEN before operating.")]
    set_retriever(Retriever(chunks, FakeEmbedder()))
    try:
        search = call("search_regulations", query="reporting entities file a suspicious transaction report", k=2)
        bad = WriterOutput(activity_description=[
            Claim(text="The transaction matched a pattern.", evidence_ids=["typology:4"])],
            regulatory_background=[RegReference(paraphrase="Reports are due regardless of amount.",
                                                evidence_id="reg:doc_ocr:p1:c0", quote="suspicious transaction report irrespective of the amount")])
        ok = WriterOutput(activity_description=bad.activity_description,
                          regulatory_background=[RegReference(paraphrase="Reports are due regardless of amount.",
                                                              evidence_id="reg:doc_ocr:p1:c0")])
        llm = FakeLLM(chat_scripts={"investigator": [done()], "regulation": [search, done()]},
                      structured_scripts={"writer": [bad, ok], "critic": all_supported})
        state, _ = run(llm, FOCAL)
    finally:
        set_retriever(None)
    r1 = [v for v in state.critic_verdicts if v.round == 1 and v.item == "background 1"][0]
    assert not r1.ok and "quotable=false" in r1.deterministic_reasons[0]
    assert state.status == "verified"
    ref = state.memo.regulatory_background[0]
    assert ref.quotable is False and ref.quote is None


def test_verbatim_quote_from_quotable_chunk_is_accepted(ctx):
    text = "Money transmitters must register with FinCEN before operating."
    set_retriever(Retriever([mk_chunk("doc_pdf", "US", 1, 0, text)], FakeEmbedder()))
    try:
        out = WriterOutput(activity_description=[Claim(text="The transaction matched a pattern.", evidence_ids=["typology:4"])],
                           regulatory_background=[RegReference(paraphrase="Transmitters must register.",
                                                               evidence_id="reg:doc_pdf:p1:c0", quote="must register with FinCEN")])
        llm = FakeLLM(chat_scripts={"investigator": [done()], "regulation": [call("search_regulations", query="money transmitters register"), done()]},
                      structured_scripts={"writer": [out], "critic": all_supported})
        state, _ = run(llm, FOCAL)
    finally:
        set_retriever(None)
    assert state.status == "verified" and state.memo.regulatory_background[0].quote == "must register with FinCEN"
    assert state.memo.regulatory_background[0].quotable is True


def test_two_failed_critic_loops_end_partially_verified(ctx):
    draft = WriterOutput(activity_description=[
        Claim(text="Something unsupported happened.", evidence_ids=[]),
        Claim(text="The model gave transaction 4 a raw score of 0.95 at time step 1.", evidence_ids=["score:4"])])
    llm = FakeLLM(chat_scripts={"investigator": [done()], "regulation": [done()]},
                  structured_scripts={"writer": [draft], "critic": all_supported})
    state, _ = run(llm, FOCAL)
    assert state.critic_loops == config.MAX_CRITIC_LOOPS == 2
    assert llm.count(purpose="writer") == 3  # first draft + 2 rewrites, then it stops
    assert state.status == "partially_verified" and state.memo.verification == "partially verified"
    assert [c.text for c in state.memo.activity_description] == ["The model gave transaction 4 a raw score of 0.95 at time step 1."]
    assert len(state.memo.removed_claims) == 1 and "Something unsupported" in state.memo.removed_claims[0]


def test_llm_verdict_unsupported_or_overstated_fails_the_claim(ctx):
    from src.agents.nodes.critic import JudgeOutput, JudgeVerdict

    def overstated(prompt):
        return JudgeOutput(verdicts=[JudgeVerdict(index=0, verdict="overstated", reason="stronger than the evidence")])

    llm = FakeLLM(chat_scripts={"investigator": [done()], "regulation": [done()]},
                  structured_scripts={"writer": [good_writer()], "critic": overstated})
    state, _ = run(llm, FOCAL)
    assert state.status == "partially_verified" and state.memo.activity_description == []
    assert state.critic_verdicts[0].llm_verdict == "overstated"


def test_numbers_must_match_the_cited_evidence(ctx):
    bad = WriterOutput(activity_description=[Claim(text="The model gave a raw score of 0.99 at time step 1.", evidence_ids=["score:4"])])
    llm = FakeLLM(chat_scripts={"investigator": [done()], "regulation": [done()]},
                  structured_scripts={"writer": [bad, good_writer()], "critic": all_supported})
    state, _ = run(llm, FOCAL)
    r1 = state.critic_verdicts[0]
    assert not r1.ok and "numbers not found" in r1.deterministic_reasons[0] and "0.99" in r1.deterministic_reasons[0]
    assert state.status == "verified"


# ---------- memo ----------
def test_subject_fields_are_always_not_available_in_dataset(ctx, health_ctx):
    assert SubjectFields().model_dump() == {k: NOT_AVAILABLE for k in ("names", "addresses", "amounts", "dates")}
    with pytest.raises(ValidationError):
        SubjectFields(names="John Doe")
    with pytest.raises(ValidationError):
        SARMemo.model_validate({**_minimal_memo(), "subject": {"names": "x", "addresses": NOT_AVAILABLE, "amounts": NOT_AVAILABLE, "dates": NOT_AVAILABLE}})
    llm = FakeLLM(chat_scripts={"investigator": [done()], "regulation": [done()]},
                  structured_scripts={"writer": [good_writer()], "critic": all_supported})
    state, _ = run(llm, FOCAL)
    assert state.memo.subject.model_dump() == {k: NOT_AVAILABLE for k in ("names", "addresses", "amounts", "dates")}
    assert state.memo.recommendation in (RECOMMEND_REVIEW, RECOMMEND_NO_ACTION)
    assert "file a sar" not in state.memo.model_dump_json().lower()


def _minimal_memo():
    return {"header": {"tx_id": 1, "time_step": 1, "risk_band": None, "score": None, "calibrated_probability": None,
                       "thresholds": {}, "model_health": {}, "generated_at": "now"},
            "activity_description": [], "evidence_table": [], "regulatory_background": [],
            "recommendation": RECOMMEND_REVIEW, "status": "x", "verification": "not applicable"}


def test_memo_limitations_cover_the_required_topics(ctx):
    llm = FakeLLM(chat_scripts={"investigator": [done()], "regulation": [done()]},
                  structured_scripts={"writer": [good_writer()], "critic": all_supported})
    state, _ = run(llm, FOCAL)
    text = " ".join(state.memo.limitations).lower()
    for needle in ("addresses", "after time step 43", "retrieval is weak", "not calibrated probabilities", "draft for a human analyst"):
        assert needle in text


# ---------- trace ----------
def test_trace_is_written_with_counts_and_verdicts(ctx, tmp_path):
    from src.agents.run import run_one

    llm = FakeLLM(chat_scripts={"investigator": [call("get_neighbors", tx_id=FOCAL), done()], "regulation": [done()]},
                  structured_scripts={"writer": [good_writer()], "critic": all_supported})
    state, trace = run_one(FOCAL, llm=llm, trace_dir=tmp_path)
    saved = json.loads((tmp_path / f"{FOCAL}.json").read_text(encoding="utf-8"))
    assert saved["status"] == "verified" and saved["counts"]["tool_calls_investigator"] == 1
    assert saved["nodes_visited"][0] == "triage" and "critic" in saved["nodes_visited"]
    assert saved["critic_verdicts"][0]["ok"] is True and saved["memo"]["status"] == "verified"
    assert saved["latency_s"]["total"] >= 0 and saved["caps"]["tool_calls"] == 8

"""Phase 5c, design v3: facts rendered by code. FakeLLM only (the conftest fixture selects design v3 for this module)."""
import json
import re
from pathlib import Path

import pytest

from src.agents import config
from src.agents.facts import (
    best_sentence, bm25_scores, code_built_claims, excerpt, facts_problems, neighbour_claims, regulation_chunks,
    render_background, typology_claims,
)
from src.agents.graph import run_agent
from src.agents.memo import memo_to_markdown
from src.agents.nodes.critic import critic
from src.agents.nodes.triage import triage
from src.agents.nodes.writer import writer
from src.agents.runtime import Deps, default_run_tool
from src.agents.state import AgentState, Claim, DraftV3, LedgerEntry, SummaryDraft
from tests.fakes import FakeLLM, all_supported, done

RAW_DIR = Path("eval/results/agent_eval/raw")
FOCAL = 4  # conftest ctx: score 0.95 (high band), fan_in pattern (uninformative), in-neighbours 1, 2, 3
GOOD_SUMMARY = SummaryDraft(executive_summary=[
    Claim(text="The model gave transaction 4 a raw score of 0.95 at time step 1.", evidence_ids=["score:4"])],
    next_checks=["Check whether the neighbouring transactions share an owner."])


def state_from_raw(tx):
    raw = json.loads((RAW_DIR / f"{tx}.json").read_text(encoding="utf-8"))
    return AgentState(tx_id=tx, time_step=raw["time_step"], ledger={k: LedgerEntry(**v) for k, v in raw["ledger"].items()},
                      signal_conflict=raw.get("signal_conflict"))


def run(llm, tx=FOCAL):
    return run_agent(tx, Deps(llm=llm, run_tool=default_run_tool))


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


# ---------- regression: the seven claims the human flagged cannot be produced any more ----------
@pytest.mark.parametrize("tx", [54925062, 69777051, 114502915, 13383872, 73002915])
def test_neighbour_claims_state_direction_and_hop_and_never_merge_counts(tx):
    st = state_from_raw(tx)
    claims = neighbour_claims(st)
    nbr_entries = [e for e in st.ledger.values() if e.tool == "get_neighbors" and "error" not in e.output]
    assert len([c for c in claims if c.text.startswith("get_neighbors")]) == len(nbr_entries) > 0
    for c in claims:
        assert len(c.evidence_ids) == 1 and c.evidence_ids[0] in st.ledger  # one tool output per claim: counts are never merged
    for e in nbr_entries:
        o = e.output
        claim = next(c for c in claims if c.evidence_ids == [o["evidence_id"]])
        assert f"direction {o['direction']}" in claim.text and f"up to {o['hops']} hop" in claim.text
        assert "directed chains only" in claim.text
        if o["direction"] == "in":  # 73002915-05 ("0 outgoing neighbors" from an incoming-only output) cannot be produced
            assert "outgoing" not in claim.text
        if o["direction"] == "out":
            assert "incoming" not in claim.text
        for n in o["neighbors"][:5]:  # 114502915-06 / 13383872-05: every neighbour carries its OWN hop and direction
            word = {"in": "incoming", "out": "outgoing", "both": "incoming and outgoing"}[n["direction"]]
            assert f"{n['tx_id']} ({word} neighbour at hop {n['hop']}, score {n['score']}," in claim.text
        assert f"{o['total_found']} " in claim.text or "no " in claim.text
    for c in claims:  # the old 'N incoming and M outgoing neighbours' wording is impossible
        assert not re.search(r"\d+ incoming neighbou?rs? and \d+ outgoing", c.text)
        assert "within 1 hop" not in c.text


def test_subgraph_claims_state_their_own_parameters():
    st = state_from_raw(114502915)
    subs = [c for c in neighbour_claims(st) if c.text.startswith("subgraph_summary")]
    for c in subs:
        o = st.ledger[c.evidence_ids[0]].output
        assert "undirected" in c.text and f"subgraph of radius {o['radius']}" in c.text and "focal node itself" in c.text
        assert f"{o['n_nodes']} nodes" in c.text


@pytest.mark.parametrize("tx,step", [(114502915, 38), (54925062, 35)])
def test_health_claim_cites_its_evidence_id(tx, step):  # 114502915-11 had no citation
    st = state_from_raw(tx)
    health = [c for c in code_built_claims(st) if c.text.startswith("Model health check")]
    assert len(health) == 1 and health[0].evidence_ids == [f"health:{step}"] and f"health:{step}" in st.ledger


@pytest.mark.parametrize("tx", [54925062, 69777051])
def test_typology_claims_never_show_the_retrieval_query(tx):  # 54925062-06 presented the query as a finding
    st = state_from_raw(tx)
    typ = st.ledger[f"typology:{tx}"].output
    claims = typology_claims(st)
    assert len(claims) == len(typ["matches"]) > 0
    for m, c in zip(typ["matches"], claims):
        assert m["pattern"] in c.text and f"evidence direction {m['evidence_direction']}" in c.text
        assert c.evidence_ids == [typ["evidence_id"]]
        assert f"{m['n_flagged']} labeled nodes" in c.text and str(m["observed_illicit_rate"]) in c.text
        if m.get("retrieval_query"):
            assert m["retrieval_query"] not in c.text and "sequential transfers" not in c.text.lower()


@pytest.mark.parametrize("tx", [54925062, 69777051, 114502915])
def test_regulatory_background_is_a_verbatim_excerpt_or_a_label_never_a_paraphrase(tx):
    st = state_from_raw(tx)  # 54925062-04 / 69777051-04 paraphrased the chunk and invented 'multiple jurisdictions'
    chunks = {cid: text for cid, text, _ in regulation_chunks(st)}
    items = render_background(st)
    for r in items:
        o = st.ledger[r.evidence_id].output
        assert r.paraphrase.startswith(f"Retrieved guidance: {o['title']}, ") and r.evidence_id in r.paraphrase
        if o.get("quotable", True):
            assert r.quote and len(r.quote.split()) <= 40 and norm(r.quote) in norm(chunks[r.evidence_id])
        else:
            assert r.quote is None and r.paraphrase == f"Retrieved guidance: {o['title']}, " + r.paraphrase.split(", ", 1)[1]
        text = (r.paraphrase + " " + (r.quote or "")).lower()
        if "multiple jurisdictions" in text:
            assert "multiple jurisdictions" in chunks[r.evidence_id].lower()  # only if the chunk itself says it


# ---------- code-built claims are present whenever the facts exist ----------
def test_code_built_claims_cover_every_fact_in_the_ledger():
    st = state_from_raw(54925062)
    claims = code_built_claims(st)
    ids = {i for c in claims for i in c.evidence_ids}
    typ = st.ledger["typology:54925062"].output
    assert len([c for c in claims if c.text.startswith("Structural pattern")]) == len(typ["matches"])  # required whenever a pattern matched
    for e in st.ledger.values():
        if e.tool in ("score_transaction", "check_model_health", "match_typology", "get_neighbors", "subgraph_summary") \
                and "error" not in e.output and e.output.get("tx_id", st.tx_id) == st.tx_id or e.tool == "check_model_health":
            if e.tool == "match_typology" and not e.output.get("matches"):
                continue
            assert e.output["evidence_id"] in ids, e.output["evidence_id"]
    assert all(c.source == "code" for c in claims)


def test_no_code_built_claims_for_missing_facts():
    st = AgentState(tx_id=1, ledger={})
    assert code_built_claims(st) == [] and render_background(st) == []


# ---------- excerpt selection ----------
LONG = " ".join(f"word{i}" for i in range(60)) + " money laundering layering indicator."


def test_excerpt_is_at_most_40_words_cut_at_a_word_boundary_and_verbatim():
    ex, cut = excerpt(LONG)
    assert cut and len(ex.split()) == 40 and LONG.startswith(ex)
    short_ex, cut2 = excerpt("A short sentence about layering.")
    assert not cut2 and short_ex == "A short sentence about layering."


def test_best_sentence_is_deterministic_and_ties_go_to_the_first():
    chunks = [("reg:a:p1:c0", "Mixing services obscure funds. Cash deposits are reported. Layering moves funds between accounts."),
              ("reg:b:p2:c0", "Layering moves funds between accounts. Other text here.")]
    first = best_sentence("layering between accounts", chunks)
    assert first == best_sentence("layering between accounts", chunks)
    assert first[0] == "reg:a:p1:c0" and first[1] == "Layering moves funds between accounts."  # identical score in both chunks: first wins
    assert best_sentence("zebra giraffe", chunks) is None  # no overlap: nothing is rendered
    assert bm25_scores(["a"], [["a"], ["b"]])[0] > bm25_scores(["a"], [["a"], ["b"]])[1] == 0.0


def background_state(quotable, text="Banks should watch for sequential transfers through chains of intermediary accounts. Other sentence."):
    st = AgentState(tx_id=7)
    st.ledger["typology:7"] = LedgerEntry(tool="match_typology", output={"evidence_id": "typology:7", "matches": [
        {"pattern": "chain", "evidence_direction": "raises_risk", "retrieval_query": "sequential transfers chain of intermediary accounts",
         "evidence": "chain_len 4 >= 4", "n_flagged": 3, "observed_illicit_rate": 0.2, "baseline_illicit_rate": 0.1, "lift": 2.0}]})
    st.ledger["reg:d:p3:c1"] = LedgerEntry(tool="search_regulations", output={
        "evidence_id": "reg:d:p3:c1", "doc_id": "d", "title": "Guidance Title", "publisher": "Pub", "year": 2020, "page_start": 3, "page_end": 3,
        "text": text, "text_truncated": False, "quotable": quotable})
    return st


def test_quotable_document_renders_a_verbatim_excerpt_with_title_page_and_chunk_id():
    (r,) = render_background(background_state(True))
    assert r.paraphrase == "Retrieved guidance: Guidance Title, page 3, reg:d:p3:c1 (quotable)"
    assert r.quote == "Banks should watch for sequential transfers through chains of intermediary accounts."


def test_non_quotable_document_renders_no_text_at_all():
    (r,) = render_background(background_state(False))
    assert r.paraphrase == "Retrieved guidance: Guidance Title, page 3, reg:d:p3:c1" and r.quote is None
    assert "Banks" not in r.paraphrase and "sequential" not in r.paraphrase


def test_a_truncated_ledger_copy_is_replaced_by_the_exact_stored_text():
    st = background_state(True, text="Banks should watch ... (truncated")
    st.ledger["reg:d:p3:c1"].output["text_truncated"] = True
    seen = []

    def fetch(cid):
        seen.append(cid)
        return {"text": "Banks should watch for sequential transfers through chains of intermediary accounts."}

    (r,) = render_background(st, fetch)
    assert seen == ["reg:d:p3:c1"] and r.quote.endswith("intermediary accounts.")


def test_a_pattern_without_retrieval_query_renders_nothing():
    st = background_state(True)
    st.ledger["typology:7"].output["matches"][0]["retrieval_query"] = None
    assert render_background(st) == []


# ---------- the v3 writer and critic ----------
def test_the_llm_is_only_offered_a_summary_and_next_checks(ctx):
    llm = FakeLLM(chat_scripts={"investigator": [done()], "regulation": [done()]},
                  structured_scripts={"writer": [GOOD_SUMMARY], "critic": all_supported})
    state = run(llm)
    schema = next(c["schema"] for c in llm.calls if c["purpose"] == "writer")
    assert schema is SummaryDraft and set(schema.model_json_schema()["properties"]) == {"executive_summary", "next_checks"}
    assert state.status == "verified" and state.memo.design == "v3"
    m = state.memo
    assert [c.text for c in m.executive_summary] == [GOOD_SUMMARY.executive_summary[0].text]
    assert m.next_checks == GOOD_SUMMARY.next_checks
    assert m.activity_description and all(c.source == "code" for c in m.activity_description)
    assert all(c.source == "llm" for c in m.executive_summary)
    expected = [c.text for c in code_built_claims(state)]
    assert [c.text for c in m.activity_description] == expected  # the memo's facts are exactly what code rendered
    md = memo_to_markdown(m)
    assert "Executive summary" in md and "rendered by code from the tool outputs" in md and "What a human analyst could check next" in md


def test_the_writer_prompt_shows_the_facts_and_never_the_raw_ledger(ctx):
    llm = FakeLLM(chat_scripts={"investigator": [done()], "regulation": [done()]},
                  structured_scripts={"writer": [GOOD_SUMMARY], "critic": all_supported})
    run(llm)
    prompt = next(c["prompt"] for c in llm.calls if c["purpose"] == "writer")
    assert prompt.startswith("THE MEMO'S FACTS") and "get_neighbors for this transaction: direction in" in prompt
    assert "EVIDENCE LEDGER" not in prompt and "retrieval_query" not in prompt


def prepared_state(ctx):
    """A state after triage and the v3 writer, ready to be handed to the critic (FakeLLM, real tools on the ctx fixture)."""
    llm = FakeLLM(structured_scripts={"writer": [GOOD_SUMMARY], "critic": all_supported})
    deps = Deps(llm=llm, run_tool=default_run_tool)
    st = AgentState(tx_id=FOCAL)
    triage(st, deps)
    st.status = "investigating"
    writer(st, deps)
    return st, deps


def test_the_critic_rejects_a_draft_that_drops_a_code_built_claim(ctx):
    st, deps = prepared_state(ctx)
    dropped = st.draft3.facts[1]
    st.draft3.facts = [c for c in st.draft3.facts if c is not dropped]
    critic(st, deps)
    v = next(x for x in st.critic_verdicts if x.item == "code-built facts")
    assert not v.ok and "missing or was edited" in v.deterministic_reasons[0]
    assert st.status == "partially_verified" and st.critic_loops == 0  # the writer is not asked to fix what it did not write
    assert dropped.text in [c.text for c in st.memo.activity_description]  # code restored it
    assert "altered or incomplete" in " ".join(st.memo.notes)


def test_the_critic_rejects_a_draft_that_edits_a_code_built_claim(ctx):
    st, deps = prepared_state(ctx)
    original = st.draft3.facts[0].text
    st.draft3.facts[0].text = original.replace("raw score", "raw score of about")
    critic(st, deps)
    v = next(x for x in st.critic_verdicts if x.item == "code-built facts")
    assert not v.ok and st.status == "partially_verified"
    assert original in [c.text for c in st.memo.activity_description]
    assert "about" not in " ".join(c.text for c in st.memo.activity_description)


def test_the_critic_rejects_a_tampered_rendered_background():
    st = background_state(True)
    exp = render_background(st)
    assert facts_problems([], [], exp, exp) == []
    bad = [exp[0].model_copy(update={"quote": exp[0].quote + " and more words"})]
    assert "rendered regulatory background differs" in facts_problems([], [], bad, exp)[0]
    assert facts_problems([], [], [], exp)  # a dropped background item is a problem too


def test_summary_longer_than_three_sentences_and_bad_next_checks_are_rejected(ctx):
    four = SummaryDraft(executive_summary=[Claim(text=f"The model gave transaction 4 a raw score of 0.95 at time step 1 ({i}).", evidence_ids=["score:4"])
                                           for i in range(4)] + [], next_checks=["Check the 3 neighbours.", "Check whether the owner is the same."])
    fixed = SummaryDraft(executive_summary=GOOD_SUMMARY.executive_summary, next_checks=["Check whether the owner is the same."])
    llm = FakeLLM(chat_scripts={"investigator": [done()], "regulation": [done()]},
                  structured_scripts={"writer": [four, fixed], "critic": all_supported})
    state = run(llm)
    r1 = [v for v in state.critic_verdicts if v.round == 1]
    assert any("at most 3 sentences" in " ".join(v.deterministic_reasons) for v in r1 if v.item == "summary 4")
    assert any(v.item == "check 1" and not v.ok and "contains a number" in v.deterministic_reasons[0] for v in r1)
    assert state.critic_loops == 1 and state.status == "verified"
    assert state.memo.next_checks == fixed.next_checks and len(state.memo.executive_summary) == 1


def test_summary_sentence_calling_a_lowers_risk_pattern_suspicious_is_rejected(ctx):
    ctx.typology_stats["fan_in"]["evidence_direction"] = "lowers_risk"
    bad = SummaryDraft(executive_summary=[Claim(text="The fan_in pattern is suspicious.", evidence_ids=["typology:4"])])
    ok = SummaryDraft(executive_summary=[Claim(text="The fan_in pattern matched, with an illicit rate below the baseline.", evidence_ids=["typology:4"])])
    llm = FakeLLM(chat_scripts={"investigator": [done()]}, structured_scripts={"writer": [bad, ok], "critic": all_supported})
    state = run(llm)
    assert "lowers_risk pattern fan_in" in [v for v in state.critic_verdicts if v.round == 1][0].deterministic_reasons[0]
    assert state.status == "verified"
    claim = next(c for c in state.memo.activity_description if c.text.startswith("Structural pattern fan_in"))
    assert "evidence direction lowers_risk" in claim.text and "not treated as suspicious" in claim.text
    assert state.memo.regulatory_background == []  # a lowers_risk pattern has no retrieval query, so nothing is rendered


def test_the_old_design_is_still_available_behind_the_flag(ctx, monkeypatch):
    monkeypatch.setattr(config, "DESIGN", "v2")
    from src.agents.state import WriterOutput

    old = WriterOutput(activity_description=[Claim(text="The model gave transaction 4 a raw score of 0.95 at time step 1.", evidence_ids=["score:4"])])
    state = run(FakeLLM(chat_scripts={"investigator": [done()], "regulation": [done()]}, structured_scripts={"writer": [old], "critic": all_supported}))
    assert state.memo.design == "v2" and state.memo.executive_summary == []

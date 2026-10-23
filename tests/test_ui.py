"""Phase 6b: the Streamlit UI. Render logic is tested as plain functions; the page code is run headless with streamlit's AppTest.
No network, no data files, no API key."""
import json
from pathlib import Path

import pytest

from src.ui import render
from src.ui.data import ApiSource, BundleSource

BUNDLE = Path("demo/memos.json")
DATA = json.loads(BUNDLE.read_text(encoding="utf-8"))
HIGH, CHAIN, FANIN, CANNOT, LOW = 73002538, 200950509, 114656654, 125406909, 200638883


def entry(tx):
    return DATA["memos"][str(tx)]


# ---------- status banners ----------
@pytest.mark.parametrize("status,level,words", [
    ("verified", "success", "passed the critic"),
    ("partially_verified", "warning", "removed"),
    ("cannot_assess", "error", "cannot be relied on"),
    ("no_action_model_based", "info", "No action suggested"),
    ("budget_exceeded", "warning", "cap"),
    ("something_new", "warning", "something new")])
def test_status_banner_levels_and_words(status, level, words):
    b = render.status_banner(status, time_step=47)
    assert b.level == level and words in (b.title + " " + b.text)


def test_cannot_assess_is_a_red_banner_that_names_the_time_step_and_withholds_the_score():
    v = render.memo_view(entry(CANNOT))
    assert v.banner.level == "error" and "time step 47" in v.banner.text and "cannot be relied on" in v.banner.title
    assert v.risk[0].startswith("Risk band: not rated") and v.risk[1] == "Raw score: withheld"
    assert not any(c.chips for c in v.claims if c.text.startswith("Risk band"))


def test_verified_memo_shows_band_score_thresholds_and_a_separate_calibrated_estimate():
    v = render.memo_view(entry(HIGH))
    assert v.banner.level == "success"
    assert v.risk[0] == "Risk band: high" and "Raw model score: 0.999979" in v.risk[1] and "not a probability" in v.risk[1]
    assert "alert 0.258181" in v.risk[2] and "high 0.788025" in v.risk[2] and v.risk[3].startswith("Calibrated estimate")


def test_no_action_memo_has_the_info_banner_and_the_recommendation_in_words():
    v = render.memo_view(entry(LOW))
    assert v.banner.level == "info" and v.recommendation == "no action suggested, model-based only"


# ---------- the signal-conflict callout ----------
def test_conflict_callout_is_the_code_built_sentence_or_none():
    assert render.memo_view(entry(CHAIN)).conflict.startswith("Signal conflict: the model score 0.127512")
    no_conflict = next(e for e in DATA["memos"].values() if not e["memo"]["signal_conflict"])
    assert render.memo_view(no_conflict).conflict is None
    assert render.memo_view(entry(HIGH)).conflict.startswith("Signal conflict:")  # a high band with unalerted neighbours
    assert render.conflict_callout({"signal_conflict": None}) is None and render.conflict_callout({}) is None


# ---------- evidence chips ----------
def test_every_cited_id_becomes_a_chip_with_the_cited_tool_output():
    v = render.memo_view(entry(CHAIN))
    chips = [c for cl in v.claims + v.summary for c in cl.chips]
    assert chips and all(c.text and "not available" not in c.text for c in chips)
    score = next(c for c in chips if c.evidence_id == f"score:{CHAIN}")
    assert score.tool == "score_transaction" and "score: 0.127512" in score.text and "risk_band: low" in score.text
    nb = next(c for c in chips if c.evidence_id.startswith("neighbors:"))
    assert nb.tool == "get_neighbors" and "total_found:" in nb.text
    first = v.claims[0]
    assert [c.evidence_id for c in first.chips] == entry(CHAIN)["memo"]["activity_description"][0]["evidence_ids"]


def test_a_chip_for_unknown_evidence_says_so_and_chips_never_show_feature_values():
    assert "not available" in render.evidence_chips(["x:1"], {})[0].text
    ev = {"score:1": {"tool": "score_transaction", "output": {"score": 0.5, "evidence_id": "score:1", "reasons": [
        {"feature": "f1", "direction": "raises illicit score", "shap_value": 1.2, "feature_value": 0.3}]}}}
    text = render.evidence_chips(["score:1"], ev)[0].text
    assert "f1" in text and "shap_value" not in text and "feature_value" not in text and "evidence_id" not in text


def test_claims_carry_their_source_so_the_page_can_say_who_wrote_them():
    v = render.memo_view(entry(CHAIN))
    assert {c.source for c in v.claims} == {"code"} and {c.source for c in v.summary} == {"llm"}


# ---------- background, subject fields, limitations ----------
def test_regulatory_background_is_marked_as_context_and_subject_fields_are_not_available():
    v = render.memo_view(entry(CHAIN))
    assert v.background and all(b["heading"] == "Context, not evidence about this transaction" for b in v.background)
    assert v.subject == {k: "not available in dataset" for k in ("names", "addresses", "amounts", "dates")}
    assert v.limitations and any("draft for a human analyst" in x for x in v.limitations)


# ---------- no label is ever rendered ----------
def test_no_bundled_memo_view_or_example_card_has_a_label_field():
    for tx in DATA["memos"]:
        v = render.memo_view(DATA["memos"][tx])
        assert render.label_keys(v) == []
    cards = render.example_cards(BundleSource(BUNDLE).examples())
    assert len(cards) == 5 and render.label_keys(cards) == []
    assert set(cards[0]) == {"tx_id", "description", "status", "time_step"}


def test_injected_label_fields_are_ignored_by_the_whitelist_and_caught_by_the_guard():
    e = json.loads(json.dumps(entry(HIGH)))
    e["label"], e["ground_truth"], e["memo"]["label"], e["memo"]["header"]["class"] = "illicit", "illicit", 1, 1
    e["evidence"][f"score:{HIGH}"]["output"]["label"] = 1
    v = render.memo_view(e)  # does not raise: nothing outside the whitelist is read
    assert "illicit" not in json.dumps(vars(v), default=lambda o: vars(o)).split("Raw model")[0]
    assert all("label" not in c.text for cl in v.claims for c in cl.chips)
    with pytest.raises(ValueError):
        render.assert_no_labels({"claims": [{"text": "x", "ground_truth": "licit"}]})
    with pytest.raises(ValueError):
        render.assert_no_labels(render.ClaimView("x", "code", []).__dict__ | {"label": 1})


def test_example_cards_drop_any_extra_fields():
    cards = render.example_cards([{"tx_id": 1, "description": "d", "status": "verified", "time_step": 3, "label": "illicit"}])
    assert cards == [{"tx_id": 1, "description": "d", "status": "verified", "time_step": 3}]


# ---------- data sources ----------
def test_bundle_source_lists_examples_and_memos_and_api_source_uses_the_endpoints():
    src = BundleSource(BUNDLE)
    assert len(src.examples()) == 5 and len(src.memos()) == 30 and src.get(HIGH)["status"] == "verified"
    calls = []

    def fetch(method, path, body):
        calls.append((method, path, body))
        return {"/health": {"mode": "live"}, "/examples": {"examples": []}, "/memos": {"memos": []},
                "/investigate": {"error": {"code": "unknown_transaction", "message": "nope"}}}[path]

    api = ApiSource("http://x", fetch=fetch)
    assert api.live is True and api.examples() == [] and api.memos() == []
    with pytest.raises(LookupError):
        api.get(5)
    assert ("POST", "/investigate", {"tx_id": 5}) in calls


def test_evaluation_page_figures_exist_and_cite_report_files():
    for item in render.EVAL_ITEMS:
        assert Path(item["report"]).exists()
        if item["figure"]:
            assert Path(item["figure"]).exists()
    assert any("not met" in i["title"] and "+0.031" in i["text"] for i in render.EVAL_ITEMS)
    assert any("4 of 7" in i["text"] and "98 of the 98" in i["text"] for i in render.EVAL_ITEMS)
    assert len(render.LIMITS) >= 7


# ---------- the pages, run headless ----------
def run_app(**state):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"), default_timeout=60)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_examples_page_lists_five_cards_with_open_buttons():
    at = run_app()
    assert at.header[0].value == "Examples"
    assert len([b for b in at.button if b.label == "Open memo"]) == 5
    assert any("Transaction 125406909" in m.value for m in at.markdown)


def test_open_memo_button_switches_to_the_memo_page():
    at = run_app()
    next(b for b in at.button if b.label == "Open memo").click().run()
    assert not at.exception and at.header[0].value == "Memo view" and at.session_state["tx"] == 73002538
    assert any("Verified" in s.value for s in at.success)


def test_memo_page_shows_the_red_banner_for_cannot_assess_and_the_conflict_callout():
    at = run_app(page="Memo view", tx=CANNOT)
    assert any("cannot be relied on" in e.value for e in at.error)
    at = run_app(page="Memo view", tx=CHAIN)
    assert any("Signal conflict" in w.value for w in at.warning)
    assert any(e.label.startswith("evidence · score:") for e in at.expander)
    assert any("Context, not evidence about this transaction" in str(c.value) for c in at.caption)


def test_evaluation_and_limits_pages_render():
    at = run_app(page="How it was evaluated")
    assert at.header[0].value == "How it was evaluated" and len(at.subheader) == len(render.EVAL_ITEMS)
    at = run_app(page="Known limits")
    assert len([m for m in at.markdown if m.value.startswith("- ")]) == len(render.LIMITS)


# ---------- Phase 6b polish ----------
def test_cannot_assess_recommendation_is_replaced_for_display_only():
    e = entry(CANNOT)
    stored = e["memo"]["recommendation"]
    v = render.memo_view(e)
    assert v.recommendation == "The model cannot assess this transaction; any review must proceed without model input."
    assert e["memo"]["recommendation"] == stored != v.recommendation  # the stored memo is unchanged
    assert render.memo_view(entry(LOW)).recommendation == entry(LOW)["memo"]["recommendation"]
    assert render.memo_view(entry(HIGH)).recommendation == entry(HIGH)["memo"]["recommendation"]


def test_excerpt_is_cut_at_a_clause_boundary_ends_with_an_ellipsis_and_has_the_general_background_note():
    q = ("the SAR narrative should include information about both the structuring and outbound transfers "
         "(including dates, destinations, amounts, accounts, frequency, and")
    shown = render.display_excerpt(q)
    assert shown.endswith("frequency…") and not shown.endswith("and…") and shown.count("…") == 1
    assert render.display_excerpt("Funds moved through several accounts and the") == "Funds moved through several accounts…"
    assert render.display_excerpt("A complete sentence.") == "A complete sentence…"
    assert render.display_excerpt("") == ""
    v = render.memo_view(entry(CHAIN))
    assert v.background[0]["quote"].endswith("…") and v.background[0]["note"] == (
        "General background from the retrieved guidance; it may not match this specific pattern.")
    assert entry(CHAIN)["memo"]["regulatory_background"][0]["quote"].endswith("and")  # stored text unchanged


def test_at_a_glance_has_status_band_score_conflict_and_model_health():
    g = dict(render.memo_view(entry(CHAIN)).glance)
    assert g == {"Status": "Verified", "Risk band": "low", "Raw score": 0.127512, "Signal conflict": "yes", "Model health": "ok"}
    c = dict(render.memo_view(entry(CANNOT)).glance)
    assert c["Risk band"] == "not rated" and c["Raw score"] == "withheld" and c["Status"].startswith("The model cannot be relied on")
    assert render.label_keys(render.memo_view(entry(CHAIN))) == []


def test_limits_page_says_next_checks_are_unconstrained_llm_text():
    assert any("what a human analyst could check next" in x and "unconstrained LLM text" in x and "may not be actionable" in x for x in render.LIMITS)


def test_memo_page_layout_glance_box_single_conflict_prefix_and_collapsed_activity_description():
    at = run_app(page="Memo view", tx=CHAIN)
    assert any("At a glance" in m.value for m in at.markdown) and len(at.metric) == 5
    conflicts = [w.value for w in at.warning if "Signal conflict" in w.value]
    assert len(conflicts) == 1 and conflicts[0].count("Signal conflict") == 1 and not conflicts[0].startswith("**")
    act = next(e for e in at.expander if e.label.startswith("Activity description"))
    assert act.proto.expanded is False
    assert any("General background from the retrieved guidance" in str(c.value) for c in at.caption)
    at = run_app(page="Memo view", tx=CANNOT)
    assert any("any review must proceed without model input" in m.value for m in at.markdown)

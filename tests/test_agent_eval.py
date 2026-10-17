"""CI-safe checks of the evaluation module and runner with a scripted FakeLLM (no network)."""
import json

import numpy as np
import pytest

from eval.agent_eval import metrics as M
from eval.agent_eval import report
from eval.agent_eval.arms import build_a0, build_a1, build_a2
from eval.agent_eval.scoring import score_arm
from eval.agent_eval.selection import DEMO_NODES, STRATA, select_cases, stratum_masks
from eval.run_agent_eval import run_case
from src.agents.memo import LIMITATIONS, PERIOD_NOTE
from src.agents.state import Claim, WriterOutput
from tests.conftest import make_ctx
from tests.fakes import FakeLLM, all_supported, carry_conflict, done

FOCAL = 4  # conftest ctx: high band, fan_in pattern (uninformative), in-neighbour 3 alerted
LOW = 1  # conftest ctx: low band, no pattern


# ---------- smoke: the whole pipeline on synthetic data ----------
def good_and_bad_drafts():
    bad = WriterOutput(activity_description=[
        Claim(text="The model gave transaction 4 a raw score of 0.99 at time step 1.", evidence_ids=["score:4"])])
    good = WriterOutput(activity_description=[
        Claim(text="The model gave transaction 4 a raw score of 0.95 at time step 1.", evidence_ids=["score:4"]),
        Claim(text="The fan_in pattern matched and is uninformative in the training-period data.", evidence_ids=["typology:4"]),
        Claim(text="Neighbourhood analysis found 3 incoming neighbours.", evidence_ids=["neighbors:4:1:in"])])
    return bad, good


def test_pipeline_smoke_with_fake_llm(ctx, tmp_path):
    bad, good = good_and_bad_drafts()
    llm = FakeLLM(chat_scripts={"investigator": [done()], "regulation": [done()]},
                  structured_scripts={"writer": carry_conflict([bad, good]), "critic": all_supported})
    cases = [{"stratum": "S1", "tx_id": FOCAL, "time_step": 1, "ground_truth": "illicit", "score": 0.95, "candidates": 1, "k": 1, "note": ""},
             {"stratum": "S2", "tx_id": LOW, "time_step": 1, "ground_truth": "licit", "score": 0.1, "candidates": 1, "k": 1, "note": ""},
             {"stratum": "S2", "tx_id": 999999, "time_step": 1, "ground_truth": "unknown", "score": 0.0, "candidates": 1, "k": 1, "note": ""}]
    (tmp_path).mkdir(exist_ok=True)
    (tmp_path / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
    raws = [run_case(c, llm, tmp_path) for c in cases]

    assert [r["status"] for r in raws] == ["verified", "no_action_model_based", "unknown_transaction"]
    for tx in (FOCAL, LOW):
        for arm in ("A0", "A1", "A2"):
            assert (tmp_path / f"{tx}_{arm}.json").exists()
        assert (tmp_path / f"{tx}_run.json").exists() and (tmp_path / "raw" / f"{tx}.json").exists()
    assert not (tmp_path / "999999_A2.json").exists()  # failed runs get a raw record only

    a = {arm: json.loads((tmp_path / f"{FOCAL}_{arm}.json").read_text(encoding="utf-8")) for arm in ("A0", "A1", "A2")}
    # A1 is the writer's FIRST draft (the bad number), A2 the final memo after the critic loop
    assert a["A1"]["memo"]["claims"][0]["text"].endswith("0.99 at time step 1.")
    assert a["A2"]["memo"]["claims"][0]["text"].endswith("0.95 at time step 1.")
    assert a["A1"]["metrics"]["groundedness"]["grounded"] < a["A1"]["metrics"]["groundedness"]["total"]
    assert a["A2"]["metrics"]["groundedness"]["grounded"] == a["A2"]["metrics"]["groundedness"]["total"] == 3
    assert a["A2"]["metrics"]["completeness"]["share"] > a["A0"]["metrics"]["completeness"]["share"]
    assert a["A0"]["metrics"]["cost"]["llm_calls"] == 0 and a["A2"]["metrics"]["cost"]["llm_calls"] > a["A1"]["metrics"]["cost"]["llm_calls"] > 0
    # no-writer case: A1 and A2 are the same memo by construction
    low = {arm: json.loads((tmp_path / f"{LOW}_{arm}.json").read_text(encoding="utf-8"))["memo"] for arm in ("A1", "A2")}
    assert {k: v for k, v in low["A1"].items() if k != "arm"} == {k: v for k, v in low["A2"].items() if k != "arm"}

    text = report.build_report(tmp_path)
    assert "Cases selected: 3; included in the aggregates: 2; failed or skipped: 1" in text
    assert "unknown_transaction" in text and "Policy violations" in text and "Limitations" in text
    assert "Critic statistics" in text and "numbers not in evidence" in text  # the bad first draft was rejected by number
    assert report.build_report(tmp_path) == text  # deterministic (bootstrap seed 42)


# ---------- metrics ----------
LEDGER = {
    "score:1": {"tool": "score_transaction", "args": {}, "output": {"evidence_id": "score:1", "score": 0.951234, "time_step": 41}},
    "reg:d:p1:c0": {"tool": "search_regulations", "args": {}, "output": {"evidence_id": "reg:d:p1:c0", "quotable": False, "text": "Entities must file."}},
    "error:x:y": {"tool": "x", "args": {}, "output": {"error": "boom"}},
}


def test_groundedness_rules():
    assert M.claim_grounded("score 0.95 at step 41", ["score:1"], LEDGER)[0]
    assert not M.claim_grounded("score 0.97", ["score:1"], LEDGER)[0]
    assert not M.claim_grounded("no citation", [], LEDGER)[0]
    assert not M.claim_grounded("x", ["missing"], LEDGER)[0]
    assert not M.claim_grounded("x", ["error:x:y"], LEDGER)[0]
    assert M.claim_grounded("feature f57 pushed it up", ["score:1"], LEDGER)[0]  # feature names are not numbers
    assert M.claim_grounded("95.12% weight", ["score:1"], LEDGER)[0]  # percent compared with value x 100


def test_scientific_notation_scores_are_one_number():
    led = {"score:2": {"tool": "score_transaction", "args": {}, "output": {"evidence_id": "score:2", "score": 9e-06}}}
    assert M.text_numbers("a raw score of 9e-06 at step 40") == [(9e-06, None, False), (40.0, 0, False)]
    assert M.claim_grounded("a raw score of 9e-06 (risk band low)", ["score:2"], led)[0]
    assert not M.claim_grounded("a raw score of 8e-06", ["score:2"], led)[0]
    assert M.claim_grounded("a raw score of 5.1e-05", ["score:3"], {"score:3": {"tool": "t", "args": {}, "output": {"evidence_id": "score:3", "score": 5.1e-05}}})[0]


def test_label_rule_flags_statements_about_the_transaction_not_the_health_sentence():
    health = ("Model health check at time step 42: status ok; recall in the delayed-label window was 0.84 against a "
              "reference recall of 0.98 over 232 labeled illicit nodes.")
    raw = {"ledger": {}, "trace": [], "tx_id": 1, "time_step": 42}
    info = {"lowers": [], "degraded": False, "matches": []}

    def leaks(text):
        memo = {"claims": [{"text": text, "evidence_ids": []}], "background": [], "notes": [], "recommendation": "r", "header": {}}
        return M.policy_violations(memo, raw, info)["d_label_leak"]

    assert leaks(health) == []  # describes the monitoring window, not the transaction
    assert leaks("The transaction is illicit.") and leaks("This transaction was labeled licit.") and leaks("Transaction 12 is confirmed illicit.")
    assert leaks("The model raises the illicit score for features f1 and f2.") == []


def test_policy_violation_detectors():
    raw = {"ledger": LEDGER, "trace": [], "tx_id": 1, "time_step": 41}
    info = {"lowers": ["fan_in"], "degraded": False, "matches": []}
    memo = {"claims": [{"text": "The fan_in pattern is suspicious.", "evidence_ids": ["score:1"]},
                       {"text": "The fan_in pattern is not treated as suspicious.", "evidence_ids": ["score:1"]},
                       {"text": 'The text says "Entities must file reports now" in general.', "evidence_ids": ["reg:d:p1:c0"]},
                       {"text": "The transaction is illicit.", "evidence_ids": ["score:1"]}],
            "background": [{"paraphrase": "p", "evidence_id": "reg:d:p1:c0", "quote": "Entities must file"}],
            "notes": [], "recommendation": "refer for human review", "header": {}}
    v = M.policy_violations(memo, raw, info)
    assert len(v["a_lowers_risk_called_suspicious"]) == 1
    assert len(v["c_quote_from_non_quotable"]) == 2  # the background quote and the inline quotation
    assert len(v["d_label_leak"]) == 1
    degraded = {"lowers": [], "degraded": True, "matches": []}
    m2 = {"claims": [{"text": "Risk band low, nothing to see.", "evidence_ids": []}], "background": [], "notes": [],
          "recommendation": "no action suggested, model-based only", "header": {"risk_band": "low", "score": 0.1}}
    assert len(M.policy_violations(m2, raw, degraded)["b_low_risk_while_degraded"]) == 3
    raw_leak = {**raw, "ledger": {"x": {"tool": "t", "args": {}, "output": {"label": "illicit"}}}}
    assert M.policy_violations({"claims": [], "background": [], "notes": [], "recommendation": "r", "header": {}}, raw_leak, info)["d_label_leak"]


def test_completeness_elements_and_applicability():
    info = {"matches": [{"pattern": "fan_in", "evidence_direction": "lowers_risk"}], "degraded": False,
            "required_conflict": "S", "reached_writer": True, "step": 43, "lowers": ["fan_in"], "raises": [], "uninformative": []}
    memo = {"tx_id": 4, "header": {"score": 0.9, "risk_band": "high", "model_health": {"status": "ok"}},
            "claims": [{"text": "The fan_in pattern matched; it lowers risk in training data.", "evidence_ids": ["typology:4"]},
                       {"text": "Neighbours found.", "evidence_ids": ["neighbors:4:1:in"]}],
            "signal_conflict": "S", "limitations": LIMITATIONS + [PERIOD_NOTE],
            "subject": {k: "not available in dataset" for k in ("names", "addresses", "amounts", "dates")}}
    el = M.completeness(memo, info, LIMITATIONS, PERIOD_NOTE)
    assert all(a and p for a, p in el.values()) and M.completeness_share(el) == 1.0
    memo2 = {**memo, "signal_conflict": "edited", "claims": [{"text": "The fan_in pattern matched.", "evidence_ids": ["typology:4"]}]}
    el2 = M.completeness(memo2, info, LIMITATIONS, PERIOD_NOTE)
    assert not el2["signal conflict"][1] and not el2["typology evidence with direction"][1] and not el2["neighbour summary"][1]
    assert 0 < M.completeness_share(el2) < 1
    cannot = {**info, "degraded": True}
    assert M.completeness(memo, cannot, LIMITATIONS, PERIOD_NOTE)["score"][0] is False  # score not applicable when degraded


def test_tool_correctness_r2_and_invalid_arguments():
    info = {"matches": [{"pattern": "fan_in"}], "raises": [], "lowers": [], "uninformative": ["fan_in"]}
    raw = {"status": "verified", "counts": {"searches": 1, "tool_calls_investigator": 3, "chunk_fetches": 0, "critic_loops": 0, "llm_calls": 6},
           "trace": [{"event": "tool_call", "node": "triage", "tool": "match_typology"},
                     {"event": "tool_call", "node": "investigator", "tool": "get_neighbors", "error": "invalid_argument"},
                     {"event": "tool_call", "node": "regulation", "tool": "search_regulations", "error": None}]}
    t = M.tool_correctness(raw, info)
    assert t["r1_ok"] and t["r2_violation"] and t["search_with_only_uninformative"] and not t["search_with_lowers_risk_only"]
    assert t["within_caps"] and t["agent_tool_calls"] == 2 and t["invalid_argument_calls"] == 1
    over = {**raw, "counts": {**raw["counts"], "searches": 4}}
    assert not M.tool_correctness(over, info)["within_caps"]


def test_cost_a1_cuts_at_the_first_critic_event():
    raw = {"first_draft": {}, "counts": {"llm_calls": 4}, "tokens": {"prompt": 100, "output": 10}, "latency_s": {"total": 20.0, "llm_throttle_wait": 8.0},
           "trace": [{"t": 1, "node": "investigator", "event": "llm_call"}, {"t": 2, "node": "investigator", "event": "llm_result", "usage": {"prompt_tokens": 10, "output_tokens": 1}},
                     {"t": 3, "node": "writer", "event": "llm_call"}, {"t": 5, "node": "writer", "event": "llm_result", "usage": {"prompt_tokens": 30, "output_tokens": 5}},
                     {"t": 6, "node": "critic", "event": "llm_call"}, {"t": 9, "node": "critic", "event": "llm_result", "usage": {"prompt_tokens": 50, "output_tokens": 2}}]}
    a1, a2 = M.cost(raw, "A1"), M.cost(raw, "A2")
    assert (a1["llm_calls"], a1["prompt_tokens"], a1["wall_s"]) == (2, 40, 5)
    assert (a2["llm_calls"], a2["wall_s"], a2["wall_s_no_wait"]) == (4, 20.0, 12.0)
    assert M.cost(raw, "A0")["llm_calls"] == 0


def test_critic_stats_and_false_rejection_candidate():
    assert M.reviewer_false_rejection_candidate("The claim states the calibrated probability is 0.25, but the cited evidence shows 0.25.")
    assert not M.reviewer_false_rejection_candidate("The claim states 0.25 but the evidence shows 0.31.")
    assert not M.reviewer_false_rejection_candidate("No numbers here, but fine.")
    raw = {"critic_verdicts": [
        {"round": 1, "item": "claim 1", "ok": False, "deterministic_reasons": ["numbers not found in the cited evidence: 3"], "llm_verdict": None},
        {"round": 1, "item": "claim 2", "ok": False, "deterministic_reasons": [], "llm_verdict": "unsupported",
         "llm_reason": "states 0.25 but the evidence shows 0.25"},
        {"round": 2, "item": "claim 1", "ok": True, "deterministic_reasons": [], "llm_verdict": "supported"}]}
    c = M.critic_stats(raw)
    assert c["rounds"] == 2 and c["deterministic"] == {"numbers not in evidence": 1} and c["llm"]["unsupported"] == 1
    assert len(c["false_rejection_candidates"]) == 1 and c["rejected_items"] == 2


def test_bootstrap_is_seeded_and_brackets_the_point_estimate():
    num, den = [3, 5, 2, 4, 6, 1], [4, 6, 3, 5, 6, 2]
    a, b = M.bootstrap_ci(num, den), M.bootstrap_ci(num, den)
    assert a == b
    point, lo, hi = a
    assert point == pytest.approx(sum(num) / sum(den)) and lo <= point <= hi
    d = M.bootstrap_ci(num, den, paired_with=([1, 1, 1, 1, 1, 1], den))
    assert d[0] == pytest.approx(sum(num) / sum(den) - 6 / sum(den))
    assert np.isnan(M.bootstrap_ci([], [])[0])


# ---------- selection ----------
def selection_ctx():
    rng = np.random.default_rng(0)
    nodes, k = [], 100
    for step in range(35, 50):
        for _ in range(30):
            nodes.append((k, step, str(rng.choice(["illicit", "licit", "unknown"])), float(rng.random())))
            k += 1
    nodes += [(d, 40, "licit", 0.95) for d in DEMO_NODES]  # the demo nodes qualify for stratum 2 but must be excluded
    return make_ctx(nodes, [])


def test_selection_follows_the_pre_registered_rules(ctx):
    c = selection_ctx()
    cases = select_cases(c)
    assert cases == select_cases(c)  # deterministic
    ids = [x["tx_id"] for x in cases]
    assert len(ids) == len(set(ids)) and not set(ids) & set(DEMO_NODES)
    by = {}
    for x in cases:
        by.setdefault(x["stratum"], []).append(x)
    for (name, k), mask in zip(STRATA, stratum_masks(c)):
        n_cand = int(mask.sum())
        assert len(by.get(name, [])) <= min(k, n_cand)
    assert all(x["ground_truth"] == "illicit" and 35 <= x["time_step"] <= 42 for x in by["1 high band, illicit"])
    assert all(x["ground_truth"] == "licit" for x in by["2 high band, licit"])
    assert all(43 <= x["time_step"] <= 49 and x["score"] >= c.alert_threshold for x in by["7 late steps (43-49), band at least medium"])
    assert "4 chain-flagged" not in by  # no edges in the synthetic ctx: zero candidates, nothing to draw

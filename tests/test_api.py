"""Phase 6a: the API, demo mode and live-mode guardrails. TestClient and FakeLLM only (no network, no Gemini, no data files)."""
import json
import subprocess
import sys
import time
from datetime import date

import pytest
from fastapi.testclient import TestClient

from scripts.build_demo_bundle import EXAMPLES, build
from src.agents import config
from src.agents.state import Claim, SummaryDraft
from src.api.bundle import Bundle, label_keys_in
from src.api.guard import DailyBudget, RateLimiter, seconds_until_quota_reset
from src.api.main import create_app
from src.api.settings import DESIGN_VERSION, Settings
from tests.fakes import FakeLLM, all_supported, done

BUNDLE = "demo/memos.json"
FOCAL = 4


def settings(mode="live", **kw):
    return Settings(mode=mode, bundle_path=BUNDLE, **kw)


def fake_run(status="verified", llm_calls=2, error=None, sleep=0.0, calls=None):
    def run(tx_id):
        if calls is not None:
            calls.append(tx_id)
        time.sleep(sleep)
        return {"status": status, "error": error, "memo": {"header": {"tx_id": tx_id}},
                "trace_summary": {"tool_calls": 3, "llm_calls": llm_calls, "critic_rounds": 1, "latency_s": 1.0, "nodes_visited": ["triage"]}}
    return run


def live_client(runner=None, **kw):
    return TestClient(create_app(settings("live", **kw), runner=runner or fake_run(), readiness=lambda: []))


def demo_client():
    def boom(tx_id):
        raise AssertionError("demo mode must never run the agent")
    return TestClient(create_app(settings("demo"), runner=boom))


# ---------- health, examples, memo ----------
def test_health_reports_status_design_version_and_mode():
    for mode, c in (("demo", demo_client()), ("live", live_client())):
        body = c.get("/health").json()
        assert body["status"] == "ok" and body["design_version"] == DESIGN_VERSION == "agent-v3.1" and body["mode"] == mode
    assert c.get("/health").json()["live_ready"] is True
    missing = TestClient(create_app(settings("live"), runner=fake_run(), readiness=lambda: ["GEMINI_API_KEY"])).get("/health").json()
    assert missing["live_ready"] is False and missing["missing"] == ["GEMINI_API_KEY"]


def test_examples_cover_the_five_case_types_and_contain_no_labels():
    ex = demo_client().get("/examples").json()["examples"]
    assert [e["tx_id"] for e in ex] == [t for t, _ in EXAMPLES] and len(ex) == 5
    assert {e["status"] for e in ex} == {"verified", "no_action_model_based", "cannot_assess"}
    text = json.dumps(ex).lower()
    for word in ("ground_truth", "label", "stratum", '"illicit"', '"licit"'):
        assert word not in text
    assert all(set(e) == {"tx_id", "description", "status", "time_step"} for e in ex)


def test_the_bundle_and_every_response_are_free_of_label_keys():
    data = json.loads(open(BUNDLE, encoding="utf-8").read())
    assert label_keys_in(data) == [] and len(data["memos"]) == 30
    c = demo_client()
    for tx in list(data["memos"])[:5]:
        assert label_keys_in(c.get(f"/memo/{tx}").json()) == []
        assert label_keys_in(c.post("/investigate", json={"tx_id": int(tx)}).json()) == []


def test_the_bundle_holds_no_text_from_a_non_quotable_document():
    """The India FIU-IND guidelines are OCR text (quotable=false): neither a chunk text nor a quote of it may be published."""
    data = json.loads(open(BUNDLE, encoding="utf-8").read())
    seen = 0

    def walk(x):
        if isinstance(x, dict):
            assert x.get("quotable", True) is not False or not (x.get("text") or x.get("quote")), x.get("evidence_id")
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    for e in data["memos"].values():
        walk(e)
        for ev_id, ev in e["evidence"].items():
            if ev_id.startswith("reg:"):
                seen += 1
                assert ev["output"]["doc_id"] != "fiu_ind_2026_vda" and "fiu" not in ev["output"]["doc_id"].lower(), ev_id
        for b in e["memo"]["regulatory_background"]:
            assert b["quotable"] or not b["quote"]
    assert seen >= 1 and "FIU-IND" not in json.dumps(data)


def test_memo_endpoint_returns_a_cached_memo_or_a_structured_404():
    c = demo_client()
    ok = c.get(f"/memo/{EXAMPLES[0][0]}")
    assert ok.status_code == 200 and ok.json()["source"] == "precomputed" and ok.json()["memo"]["header"]["tx_id"] == EXAMPLES[0][0]
    r = c.get("/memo/12345")
    assert r.status_code == 404 and r.json()["error"]["code"] == "memo_not_found"


def test_memos_lists_the_bundle_and_memo_payloads_carry_sanitized_evidence():
    c = demo_client()
    ms = c.get("/memos").json()["memos"]
    assert len(ms) == 30 and set(ms[0]) == {"tx_id", "status", "time_step"}
    body = c.post("/investigate", json={"tx_id": EXAMPLES[1][0]}).json()
    ev = body["evidence"][f"score:{EXAMPLES[1][0]}"]
    assert ev["tool"] == "score_transaction" and "shap_value" not in json.dumps(body) and "feature_value" not in json.dumps(body)
    assert c.get(f"/memo/{EXAMPLES[1][0]}").json()["evidence"] == body["evidence"]


def test_invalid_body_is_a_structured_422():
    r = demo_client().post("/investigate", json={"tx_id": "abc"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"


# ---------- demo mode ----------
def test_demo_mode_serves_precomputed_memos_and_refuses_live_runs():
    c = demo_client()
    ok = c.post("/investigate", json={"tx_id": EXAMPLES[0][0]})
    assert ok.status_code == 200 and ok.json()["source"] == "precomputed" and ok.json()["trace_summary"]["llm_calls"] >= 0
    no = c.post("/investigate", json={"tx_id": 12345})
    assert no.status_code == 403 and no.json()["error"]["code"] == "live_runs_disabled"
    assert c.post("/investigate", json={"tx_id": 12345, "refresh": True}).status_code == 403


def test_demo_mode_imports_no_agent_data_or_model_code():
    code = ("import sys; from fastapi.testclient import TestClient; from src.api.main import create_app; from src.api.settings import Settings; "
            "c = TestClient(create_app(Settings(mode='demo'))); assert c.get('/health').status_code == 200; "
            "assert c.post('/investigate', json={'tx_id': 1}).status_code == 403; "
            "bad = [m for m in ('src.agents', 'src.tools', 'lightgbm', 'pandas', 'google.genai', 'torch') if m in sys.modules]; "
            "print('LOADED', bad); assert not bad")
    done_ = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done_.returncode == 0, done_.stderr[-500:] + done_.stdout[-300:]


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        Settings.from_env(mode="production")


# ---------- live mode with the real graph and a FakeLLM ----------
def test_live_investigation_runs_the_agent_caches_and_returns_a_trace_summary(ctx, monkeypatch):
    monkeypatch.setattr(config, "DESIGN", "v3")
    summary = SummaryDraft(executive_summary=[
        Claim(text="The model gave transaction 4 a raw score of 0.95 at time step 1.", evidence_ids=["score:4"])],
        next_checks=["Check whether the neighbouring transactions share an owner."])
    llms = []

    def factory():
        llms.append(FakeLLM(chat_scripts={"investigator": [done()], "regulation": [done()]},
                            structured_scripts={"writer": [summary], "critic": all_supported}))
        return llms[-1]

    c = TestClient(create_app(settings("live"), llm_factory=factory, readiness=lambda: []))
    r = c.post("/investigate", json={"tx_id": FOCAL})
    body = r.json()
    assert r.status_code == 200 and body["status"] == "verified" and body["source"] == "live" and body["mode"] == "live"
    assert body["memo"]["header"]["tx_id"] == FOCAL and body["memo"]["design"] == "v3"
    ts = body["trace_summary"]
    assert ts["llm_calls"] >= 3 and ts["tool_calls"] >= 1 and ts["latency_s"] >= 0 and "writer" in ts["nodes_visited"]
    assert c.get("/health").json()["daily_llm_calls_used"] == ts["llm_calls"]
    assert c.get(f"/memo/{FOCAL}").json()["source"] == "live"
    again = c.post("/investigate", json={"tx_id": FOCAL}).json()  # served from the cache: no new LLM client, no budget used
    assert again["source"] == "live" and len(llms) == 1 and c.get("/health").json()["daily_llm_calls_used"] == ts["llm_calls"]


def test_unknown_transaction_is_a_structured_404_in_live_mode(ctx):
    llm = FakeLLM()
    c = TestClient(create_app(settings("live"), llm_factory=lambda: llm, readiness=lambda: []))
    r = c.post("/investigate", json={"tx_id": 999999})
    assert r.status_code == 404 and r.json()["error"]["code"] == "unknown_transaction" and r.json()["error"]["tx_id"] == 999999
    assert llm.calls == [] and c.get("/memo/999999").status_code == 404


def test_live_mode_that_is_not_ready_returns_503_without_running_anything():
    calls = []
    c = TestClient(create_app(settings("live"), runner=fake_run(calls=calls), readiness=lambda: ["GEMINI_API_KEY", "models/phase2b_model.pkl"]))
    r = c.post("/investigate", json={"tx_id": 1})
    assert r.status_code == 503 and r.json()["error"]["code"] == "not_ready" and "Retry-After" in r.headers and calls == []
    assert "GEMINI_API_KEY" in r.json()["error"]["message"]


# ---------- guardrails ----------
def test_rate_limit_per_ip_returns_429_with_retry_after():
    calls = []
    c = live_client(fake_run(calls=calls), rate_limit_per_hour=2)
    assert [c.post("/investigate", json={"tx_id": t}).status_code for t in (1, 2)] == [200, 200]
    r = c.post("/investigate", json={"tx_id": 3})
    assert r.status_code == 429 and r.json()["error"]["code"] == "rate_limited" and 1 <= int(r.headers["Retry-After"]) <= 3600
    assert calls == [1, 2]


def test_rate_limiter_window_slides_and_is_per_key():
    now = [0.0]
    rl = RateLimiter(2, 100.0, clock=lambda: now[0])
    assert rl.check("a")[0] and rl.check("a")[0] and not rl.check("a")[0] and rl.check("b")[0]
    now[0] = 101.0
    assert rl.check("a")[0]


def test_budget_exhausted_returns_503_with_retry_after_and_stops_runs():
    calls = []
    c = live_client(fake_run(llm_calls=3, calls=calls), daily_llm_budget=3)
    assert c.post("/investigate", json={"tx_id": 1}).status_code == 200
    r = c.post("/investigate", json={"tx_id": 2})
    assert r.status_code == 503 and r.json()["error"]["code"] == "daily_budget_exhausted" and int(r.headers["Retry-After"]) >= 1
    assert calls == [1]
    assert c.post("/investigate", json={"tx_id": 1}).status_code == 200  # a cached memo costs nothing


def test_daily_budget_rolls_over_with_the_quota_day():
    day = [date(2026, 1, 1)]
    b = DailyBudget(10, today=lambda: day[0])
    b.add(7)
    assert b.remaining() == 3
    day[0] = date(2026, 1, 2)
    assert b.remaining() == 10 and 1 <= seconds_until_quota_reset() <= 86400


def test_quota_error_from_the_llm_is_503_with_retry_after_and_is_not_cached():
    c = live_client(fake_run(status="agent_error", error="DailyQuotaExceeded: 429 per day"))
    r = c.post("/investigate", json={"tx_id": 1})
    assert r.status_code == 503 and r.json()["error"]["code"] == "llm_quota_exhausted" and int(r.headers["Retry-After"]) >= 1
    assert c.get("/memo/1").status_code == 404


def test_quota_exception_from_the_runner_is_503_too():
    class DailyQuotaExceeded(RuntimeError):
        pass

    def run(tx):
        raise DailyQuotaExceeded("429")

    r = live_client(run).post("/investigate", json={"tx_id": 1})
    assert r.status_code == 503 and r.json()["error"]["code"] == "llm_quota_exhausted"


def test_request_timeout_is_504_and_concurrency_is_one_until_the_run_really_ends():
    with live_client(fake_run(sleep=0.6), request_timeout_s=0.1) as c:  # one event loop for all requests, as under uvicorn
        r = c.post("/investigate", json={"tx_id": 1})
        assert r.status_code == 504 and r.json()["error"]["code"] == "timeout" and "Retry-After" in r.headers
        busy = c.post("/investigate", json={"tx_id": 2})  # the first run is still going in its thread
        assert busy.status_code == 503 and busy.json()["error"]["code"] == "busy" and "Retry-After" in busy.headers
        time.sleep(0.8)
        c.app.state.settings.request_timeout_s = 5.0
        assert c.post("/investigate", json={"tx_id": 2}).status_code == 200
        assert c.get("/health").json()["daily_llm_calls_used"] == 4  # the timed-out run was still counted against the budget


# ---------- the bundle builder ----------
def test_bundle_builder_copies_only_memo_and_trace_summary(tmp_path):
    raw = {"tx_id": 7, "status": "verified", "time_step": 5, "stratum": "1 high band, illicit", "ground_truth": "illicit",
           "ledger": {"score:7": {"tool": "score_transaction", "output": {"label": 1, "score": 0.5, "reasons": [
               {"feature": "f1", "direction": "raises illicit score", "shap_value": 1.2, "feature_value": 0.3}]}},
                      "neighbors:7:1:in": {"tool": "get_neighbors", "output": {"total_found": 0}}},
           "trace": [{"event": "x"}], "nodes_visited": ["triage"],
           "counts": {"tool_calls_total": 2, "llm_calls": 3, "critic_loops": 0}, "latency_s": {"total": 1.5},
           "memo": {"header": {"tx_id": 7}, "activity_description": [{"text": "t", "evidence_ids": ["score:7"]}],
                    "evidence_table": [{"evidence_id": "score:7", "tool": "score_transaction"}]}}
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "7_v3.json").write_text(json.dumps(raw), encoding="utf-8")
    out = tmp_path / "memos.json"
    bundle = build(tmp_path / "raw", out)
    text = out.read_text(encoding="utf-8")
    assert list(bundle["memos"]["7"]) == ["tx_id", "status", "time_step", "memo", "trace_summary", "evidence"]
    assert list(bundle["memos"]["7"]["evidence"]) == ["score:7"]  # only the cited tool output, not the uncited one
    assert bundle["memos"]["7"]["evidence"]["score:7"]["output"]["score"] == 0.5
    for leaked in ("stratum", "ground_truth", "ledger", "label", "event", "shap_value", "feature_value", "neighbors:7"):
        assert leaked not in text
    assert Bundle(out).get(7)["trace_summary"]["llm_calls"] == 3

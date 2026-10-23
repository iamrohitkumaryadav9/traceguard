"""TraceGuard API. Run: TRACEGUARD_MODE=demo uvicorn src.api.main:app   (or live; see the README).

demo: serves only pre-computed memos from demo/memos.json; needs no data, no model and no key, and never runs the agent
      (the agent modules are not even imported).
live: runs the agent (tag agent-v3.1) on request; needs GEMINI_API_KEY and the local data and models. Guardrails: per-IP
      rate limit, a global daily LLM-call budget, a request timeout, one agent run at a time, quota errors as HTTP 503.
No endpoint returns a ground-truth label.
"""
import asyncio
import math
import os
import threading

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.bundle import Bundle
from src.api.guard import DailyBudget, RateLimiter, seconds_until_quota_reset
from src.api.settings import DESIGN_VERSION, Settings


class InvestigateRequest(BaseModel):
    tx_id: int
    refresh: bool = False  # live mode: run again even if this process already holds a memo for the node


def error(status, code, message, headers=None, **extra):
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, **extra}}, headers=headers)


def default_readiness():
    """Names of what live mode is missing (never values): the key and the local data and models."""
    from dotenv import load_dotenv

    load_dotenv()
    from src.tools.context import CALIBRATOR_PATH, EDGES_PATH, MODEL_PATH, NODES_PATH, THRESHOLD_PATH

    missing = []
    key = os.environ.get("GEMINI_API_KEY")
    if not key or key == "your_key_here":
        missing.append("GEMINI_API_KEY")
    missing += [str(p) for p in (MODEL_PATH, CALIBRATOR_PATH, THRESHOLD_PATH, NODES_PATH, EDGES_PATH) if not p.exists()]
    return missing


class AgentRunner:
    """Runs the agent graph on one node with one shared LLM client (so its throttle state persists across requests)."""

    def __init__(self, llm_factory=None):
        self._factory, self._llm = llm_factory, None

    def __call__(self, tx_id):
        from src.agents.graph import run_agent, trace_dict
        from src.agents.runtime import Deps

        if self._llm is None:
            if self._factory is None:
                from src.agents.llm import GeminiLLM

                self._factory = GeminiLLM
            self._llm = self._factory()
        deps = Deps(llm=self._llm)
        state = run_agent(tx_id, deps)
        t = trace_dict(state, deps)
        from src.api.bundle import cited_ids, public_evidence

        ledger = {k: v.model_dump() for k, v in state.ledger.items()}
        evidence = public_evidence(ledger, cited_ids(t["memo"])) if t["memo"] else {}
        return {"status": state.status, "error": state.error, "memo": t["memo"], "trace_summary": trace_summary(t),
                "evidence": evidence}


def json_safe(obj):
    """NaN and infinity are not valid JSON (a metric that is undefined, e.g. recall with no labeled nodes): send null."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


def trace_summary(t):
    return {"tool_calls": t["counts"]["tool_calls_total"], "llm_calls": t["counts"]["llm_calls"],
            "critic_rounds": t["counts"]["critic_loops"], "latency_s": t["latency_s"]["total"],
            "nodes_visited": t["nodes_visited"]}


def create_app(settings=None, runner=None, llm_factory=None, readiness=None):
    s = settings or Settings.from_env()
    bundle = Bundle(s.bundle_path)
    live_cache = {}
    limiter = RateLimiter(s.rate_limit_per_hour, 3600.0)
    budget = DailyBudget(s.daily_llm_budget)
    run_lock = threading.Semaphore(s.max_concurrent_runs)
    if s.mode == "live":
        runner = runner or AgentRunner(llm_factory)
        readiness = readiness or default_readiness

    app = FastAPI(title="TraceGuard", version=DESIGN_VERSION,
                  description="Flags suspicious Bitcoin transactions (Elliptic) and writes grounded investigation memos.")
    app.state.settings, app.state.budget, app.state.live_cache = s, budget, live_cache

    @app.exception_handler(RequestValidationError)
    async def _validation(_, exc):
        return error(422, "invalid_request", "The request body or path is invalid.", detail=[e["msg"] for e in exc.errors()])

    @app.exception_handler(StarletteHTTPException)
    async def _http(_, exc):
        return error(exc.status_code, "http_error", str(exc.detail))

    def payload(tx_id, status, memo, trace, source, evidence=None):
        return {"tx_id": tx_id, "status": status, "memo": json_safe(memo), "trace_summary": trace, "evidence": json_safe(evidence or {}),
                "mode": s.mode, "design_version": DESIGN_VERSION, "source": source}

    @app.get("/health")
    def health():
        body = {"status": "ok", "design_version": DESIGN_VERSION, "mode": s.mode, "precomputed_memos": len(bundle)}
        if s.mode == "live":
            missing = readiness()
            body.update(live_ready=not missing, missing=missing, daily_llm_budget=s.daily_llm_budget,
                        daily_llm_calls_used=budget.used())
        return body

    @app.get("/examples")
    def examples():
        out = []
        for e in bundle.examples:
            entry = bundle.get(e["tx_id"]) or {}
            out.append({"tx_id": e["tx_id"], "description": e["description"], "status": entry.get("status"),
                        "time_step": entry.get("time_step")})
        return {"examples": out}

    @app.get("/memo/{tx_id}")
    def memo(tx_id: int):
        if tx_id in live_cache:
            return live_cache[tx_id]
        entry = bundle.get(tx_id)
        if entry is None:
            return error(404, "memo_not_found", f"No cached memo for transaction {tx_id}.", tx_id=tx_id)
        return payload(tx_id, entry["status"], entry["memo"], entry["trace_summary"], "precomputed", entry.get("evidence"))

    @app.get("/memos")
    def memos():
        return {"memos": [{"tx_id": t, "status": e["status"], "time_step": e["time_step"]} for t, e in sorted(bundle.entries.items())]}

    @app.post("/investigate")
    async def investigate(req: InvestigateRequest, request: Request):
        tx = req.tx_id
        if s.mode == "demo":
            entry = bundle.get(tx)
            if entry is None:
                return error(403, "live_runs_disabled",
                             "This server runs in demo mode: it only serves pre-computed memos and never runs the agent. "
                             "See GET /examples for the available nodes.", tx_id=tx)
            return payload(tx, entry["status"], entry["memo"], entry["trace_summary"], "precomputed", entry.get("evidence"))
        if tx in live_cache and not req.refresh:
            return live_cache[tx]
        missing = readiness()
        if missing:
            return error(503, "not_ready", "Live mode is missing: " + ", ".join(missing), headers={"Retry-After": "300"})
        ok, retry = limiter.check(request.client.host if request.client else "unknown")
        if not ok:
            return error(429, "rate_limited", "Too many live investigations from this address.", headers={"Retry-After": str(retry)},
                         retry_after_s=retry)
        if budget.remaining() <= 0:
            retry = seconds_until_quota_reset()
            return error(503, "daily_budget_exhausted", "The daily LLM-call budget of this service is used up.",
                         headers={"Retry-After": str(retry)}, retry_after_s=retry)
        if not run_lock.acquire(blocking=False):
            return error(503, "busy", "Another investigation is running (one at a time); retry shortly.",
                         headers={"Retry-After": "30"}, retry_after_s=30)

        def work():
            try:
                res = runner(tx)
                budget.add((res.get("trace_summary") or {}).get("llm_calls", 0))
                return res
            finally:
                run_lock.release()  # held until the run really ends, even if the request timed out

        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(None, work)
        try:
            res = await asyncio.wait_for(asyncio.shield(fut), timeout=s.request_timeout_s)
        except asyncio.TimeoutError:
            return error(504, "timeout", f"The investigation did not finish within {s.request_timeout_s:g} s.",
                         headers={"Retry-After": "60"}, retry_after_s=60)
        except Exception as e:
            if type(e).__name__ == "DailyQuotaExceeded":
                return quota_503(str(e))
            return error(500, "agent_failure", f"The agent failed: {type(e).__name__}.")
        if (res.get("error") or "").startswith("DailyQuotaExceeded"):
            return quota_503(res["error"])
        if res["status"] == "unknown_transaction":
            return error(404, "unknown_transaction", f"Transaction {tx} is not in the dataset.", tx_id=tx)
        out = payload(tx, res["status"], res["memo"], res["trace_summary"], "live", res.get("evidence"))
        if res["status"] != "agent_error":
            live_cache[tx] = out
        return out

    def quota_503(msg):
        retry = seconds_until_quota_reset()
        return error(503, "llm_quota_exhausted", "The LLM provider's daily quota is exhausted.",
                     headers={"Retry-After": str(retry)}, retry_after_s=retry)

    return app


app = create_app()

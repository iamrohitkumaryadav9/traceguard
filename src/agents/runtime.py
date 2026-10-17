"""Run-time helpers shared by the nodes: budgets, tool execution with validation, tracing."""
import time
from dataclasses import dataclass, field
from typing import Callable

from pydantic import ValidationError

from src.agents import config
from src.agents.state import AgentState, LedgerEntry


class BudgetExceeded(Exception):
    """A cap was hit (tool calls, searches, chunk fetches or LLM calls). Ends the run with a partial memo."""

    def __init__(self, which):
        super().__init__(which)
        self.which = which


def default_run_tool(name, arguments):
    """src.tools.run_tool with argument validation errors returned as structured error objects."""
    from src.tools import TOOLS, run_tool

    if name not in TOOLS:
        return {"error": "invalid_argument", "message": f"unknown tool {name!r}", "tool": name,
                "evidence_id": f"error:{name}:unknown_tool"}
    try:
        return run_tool(name, arguments)
    except ValidationError as e:
        errs = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())
        return {"error": "invalid_argument", "message": errs, "tool": name, "evidence_id": f"error:{name}:invalid_arguments"}


@dataclass
class Deps:
    llm: object
    run_tool: Callable = default_run_tool
    clock: Callable = time.perf_counter
    now_iso: Callable = lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z")
    t0: float = field(default_factory=time.perf_counter)

    def elapsed(self):
        return round(self.clock() - self.t0, 3)


def log(state: AgentState, deps: Deps, node, event, **data):
    state.trace.append({"t": deps.elapsed(), "node": node, "event": event, **data})


def call_tool(state: AgentState, deps: Deps, node, name, args, count=None):
    """Execute a tool, store its output in the ledger under its evidence_id and log it. `count` names the
    state counter to enforce and increment (checked BEFORE the call, so a call beyond the cap never runs)."""
    if count:
        caps = {"tool_call_count": config.MAX_TOOL_CALLS, "search_count": config.MAX_SEARCHES,
                "chunk_fetch_count": config.MAX_CHUNK_FETCHES}
        if getattr(state, count) >= caps[count]:
            raise BudgetExceeded(count)
        setattr(state, count, getattr(state, count) + 1)
    t0 = deps.clock()
    out = deps.run_tool(name, args)
    ms = round(1000 * (deps.clock() - t0), 1)
    eid = out.get("evidence_id", f"error:{name}:no_evidence_id")
    state.ledger[eid] = LedgerEntry(tool=name, args=args, output=out)
    log(state, deps, node, "tool_call", tool=name, args=args, evidence_id=eid, ms=ms, error=out.get("error"))
    return out


def count_llm_call(state: AgentState, deps: Deps, node, info=None, purpose=""):
    if state.llm_call_count >= config.MAX_LLM_CALLS:
        raise BudgetExceeded("llm_call_count")
    state.llm_call_count += 1
    log(state, deps, node, "llm_call", purpose=purpose, **(info or {}))

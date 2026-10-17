"""The LangGraph investigation graph.

triage -> (cannot_assess | no_action | unknown) -> finalize
triage -> investigator -> [regulation] -> writer -> critic -> (writer again, at most 2 times) -> end
Every node is wrapped so that a budget cap or an unexpected error ends the run with a partial memo, never a crash.
"""
from langgraph.graph import END, StateGraph

from src.agents import config
from src.agents.build import deterministic_memo
from src.agents.nodes.critic import critic
from src.agents.nodes.investigator import investigator
from src.agents.nodes.regulation import regulation, retrieval_queries
from src.agents.nodes.triage import triage
from src.agents.nodes.writer import writer
from src.agents.runtime import BudgetExceeded, Deps, log
from src.agents.state import AgentState

TERMINAL_STATUSES = {"cannot_assess", "no_action_model_based", "unknown_transaction", "budget_exceeded", "agent_error"}


def safe(fn, deps: Deps):
    """Run a node on a copy of the state; turn BudgetExceeded and unexpected errors into terminal statuses."""
    def wrapped(state: AgentState):
        s = state.model_copy(deep=True)
        try:
            s = fn(s, deps)
        except BudgetExceeded as e:
            s.status = "budget_exceeded"
            s.error = f"budget exceeded: {e.which}"
            log(s, deps, fn.__name__, "budget_exceeded", which=e.which)
        except Exception as e:  # never crash a run: record it and end with a partial memo
            s.status = "agent_error"
            s.error = f"{type(e).__name__}: {str(e)[:300]}"
            log(s, deps, fn.__name__, "error", error=s.error)
        return {k: getattr(s, k) for k in AgentState.model_fields}
    wrapped.__name__ = fn.__name__
    return wrapped


def finalize(state: AgentState, deps: Deps) -> AgentState:
    state.nodes_visited.append("finalize")
    if state.memo is None:
        notes = [state.error] if state.error else []
        if state.status == "budget_exceeded":
            notes.append("The run was stopped by a budget cap; this memo contains only what the evidence ledger already supports.")
        state.memo = deterministic_memo(state, deps.now_iso(), state.status, notes=notes)
    return state


def route_after_triage(state: AgentState):
    return "finalize" if state.status in TERMINAL_STATUSES else "investigator"


def route_after_investigator(state: AgentState):
    if state.status in TERMINAL_STATUSES:
        return "finalize"
    return "regulation" if retrieval_queries(state) else "writer"


def route_after_regulation(state: AgentState):
    return "finalize" if state.status in TERMINAL_STATUSES else "writer"


def route_after_writer(state: AgentState):
    return "finalize" if state.status in TERMINAL_STATUSES else "critic"


def route_after_critic(state: AgentState):
    if state.status == "rewriting":
        return "writer"
    return "finalize"


def build_graph(deps: Deps):
    g = StateGraph(AgentState)
    for name, fn in (("triage", triage), ("investigator", investigator), ("regulation", regulation),
                     ("writer", writer), ("critic", critic), ("finalize", finalize)):
        g.add_node(name, safe(fn, deps))
    g.set_entry_point("triage")
    g.add_conditional_edges("triage", route_after_triage, {"finalize": "finalize", "investigator": "investigator"})
    g.add_conditional_edges("investigator", route_after_investigator,
                            {"finalize": "finalize", "regulation": "regulation", "writer": "writer"})
    g.add_conditional_edges("regulation", route_after_regulation, {"finalize": "finalize", "writer": "writer"})
    g.add_conditional_edges("writer", route_after_writer, {"finalize": "finalize", "critic": "critic"})
    g.add_conditional_edges("critic", route_after_critic, {"writer": "writer", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


def run_agent(tx_id: int, deps: Deps) -> AgentState:
    """Run the graph for one transaction and return the final state (always with a memo)."""
    graph = build_graph(deps)
    out = graph.invoke(AgentState(tx_id=tx_id), config={"recursion_limit": 60})
    state = out if isinstance(out, AgentState) else AgentState.model_validate(out)
    return state


def trace_dict(state: AgentState, deps: Deps) -> dict:
    llm = deps.llm
    return {
        "tx_id": state.tx_id, "time_step": state.time_step, "status": state.status, "error": state.error,
        "model": config.LLM_MODEL, "temperature": config.LLM_TEMPERATURE,
        "nodes_visited": state.nodes_visited,
        "counts": {"tool_calls_investigator": state.tool_call_count, "searches": state.search_count,
                   "chunk_fetches": state.chunk_fetch_count, "llm_calls": state.llm_call_count,
                   "critic_loops": state.critic_loops,
                   "tool_calls_total": sum(1 for e in state.trace if e["event"] == "tool_call")},
        "caps": {"tool_calls": config.MAX_TOOL_CALLS, "searches": config.MAX_SEARCHES, "critic_loops": config.MAX_CRITIC_LOOPS,
                 "llm_calls": config.MAX_LLM_CALLS},
        "latency_s": {"total": deps.elapsed(), "llm_throttle_wait": round(getattr(llm, "throttle_wait_s", 0.0), 2)},
        "tokens": {"prompt": sum((e.get("usage") or {}).get("prompt_tokens") or 0 for e in state.trace),
                   "output": sum((e.get("usage") or {}).get("output_tokens") or 0 for e in state.trace)},
        "critic_verdicts": [v.model_dump() for v in state.critic_verdicts],
        "events": state.trace,
        "memo": state.memo.model_dump() if state.memo else None,
    }

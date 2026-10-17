"""Investigator: an LLM tool-calling loop over the graph tools, capped at MAX_TOOL_CALLS."""
import json

from src.agents import config, prompts
from src.agents.llm import ToolResult, tool_specs_for
from src.agents.runtime import Deps, call_tool, count_llm_call, log
from src.agents.state import AgentState


def compact(obj, limit=6000):
    s = json.dumps(obj, separators=(",", ":"))
    return s if len(s) <= limit else s[:limit] + "...(truncated)"


def run_tool_loop(state: AgentState, deps: Deps, node, allowed, system, first_message, counters, tools):
    """Shared LLM tool loop. `counters` maps a tool name to the state counter enforced for it (or None)."""
    chat = deps.llm.start_chat(system, tool_specs_for([tools[n] for n in allowed]), purpose=node)
    message = first_message
    while True:
        count_llm_call(state, deps, node, purpose="tool loop")
        turn = chat.send(message)
        log(state, deps, node, "llm_result", latency_ms=round(turn.latency_ms, 1), usage=turn.usage,
            tool_calls=[c.name for c in turn.tool_calls])
        if not turn.tool_calls:
            return
        results = []
        for call in turn.tool_calls:
            if call.name not in allowed:
                out = {"error": "invalid_argument", "message": f"tool {call.name!r} is not available here; use one of {list(allowed)}",
                       "tool": call.name, "evidence_id": f"error:{call.name}:not_allowed"}
                log(state, deps, node, "tool_refused", tool=call.name)
            else:
                args = {k: v for k, v in call.args.items() if k not in config.HIDDEN_TOOL_ARGS}  # retrieval is frozen
                out = call_tool(state, deps, node, call.name, args, count=counters.get(call.name))
            results.append(ToolResult(call.name, out))
        message = results


def investigator(state: AgentState, deps: Deps) -> AgentState:
    from src.tools import TOOLS

    state.nodes_visited.append("investigator")
    triage_ev = {k: v.output for k, v in state.ledger.items() if v.tool in ("score_transaction", "match_typology")}
    first = (f"Transaction {state.tx_id} at time step {state.time_step} was flagged. Evidence already collected:\n"
             f"{compact(triage_ev)}\nInvestigate its neighbourhood. At most {config.MAX_TOOL_CALLS} tool calls.")
    run_tool_loop(state, deps, "investigator", config.INVESTIGATOR_TOOLS,
                  prompts.INVESTIGATOR.format(max_calls=config.MAX_TOOL_CALLS), first,
                  {n: "tool_call_count" for n in config.INVESTIGATOR_TOOLS}, TOOLS)
    return state

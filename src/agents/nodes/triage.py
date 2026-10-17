"""Triage: deterministic, no LLM. Decides whether the investigation runs at all, and computes the signal conflict."""
from src.agents.conflict import detect_signal_conflict
from src.agents.runtime import Deps, call_tool, log
from src.agents.state import AgentState


def triage(state: AgentState, deps: Deps) -> AgentState:
    state.nodes_visited.append("triage")
    score = call_tool(state, deps, "triage", "score_transaction", {"tx_id": state.tx_id})
    if "error" in score:
        state.status = "unknown_transaction"
        state.error = score.get("message")
        log(state, deps, "triage", "decision", status=state.status)
        return state
    state.time_step = score["time_step"]
    health = call_tool(state, deps, "triage", "check_model_health", {"time_step": state.time_step})
    typ = call_tool(state, deps, "triage", "match_typology", {"tx_id": state.tx_id})

    if health.get("status") == "degraded":
        state.status = "cannot_assess"  # the model cannot be relied on here: never a "low risk" statement
    elif score["risk_band"] == "low" and not any(m["evidence_direction"] == "raises_risk" for m in typ.get("matches", [])):
        state.status = "no_action_model_based"
    else:
        state.status = "investigating"

    if state.status != "cannot_assess":  # no score is shown for cannot_assess, so there is nothing to conflict with
        # in-neighbour alerts are part of the conflict test; this deterministic call is not an investigator call
        nbrs = call_tool(state, deps, "triage", "get_neighbors", {"tx_id": state.tx_id, "hops": 1, "direction": "in"})
        state.signal_conflict, state.signal_conflict_evidence_ids = detect_signal_conflict(score, typ, nbrs)
    log(state, deps, "triage", "decision", status=state.status, risk_band=score["risk_band"],
        health=health.get("status"), patterns=[(m["pattern"], m["evidence_direction"]) for m in typ.get("matches", [])],
        signal_conflict=state.signal_conflict)
    return state

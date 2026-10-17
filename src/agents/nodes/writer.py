"""Writer: structured LLM output (claims with evidence ids). It sees the ledger and any critic feedback."""
import json

from src.agents import prompts
from src.agents.runtime import Deps, count_llm_call, log
from src.agents.state import AgentState, WriterOutput


def ledger_for_prompt(state: AgentState, limit=2500):
    out = {}
    for eid, e in state.ledger.items():
        if "error" in e.output:
            continue  # tool errors are not evidence
        s = json.dumps(e.output, separators=(",", ":"))
        out[eid] = {"tool": e.tool, "output": e.output if len(s) <= limit else s[:limit] + "...(truncated)"}
    return out


def writer(state: AgentState, deps: Deps) -> AgentState:
    state.nodes_visited.append("writer")
    prompt = ["EVIDENCE LEDGER (evidence_id -> tool output):", json.dumps(ledger_for_prompt(state), separators=(",", ":"))]
    if state.critic_feedback:
        prompt += ["", "YOUR PREVIOUS DRAFT WAS REJECTED BY THE CRITIC. Fix every problem below, or drop the claim:",
                   *[f"- {x}" for x in state.critic_feedback]]
    prompt += ["", "Write the activity_description claims and regulatory_references now."]
    count_llm_call(state, deps, "writer", purpose=f"draft {state.critic_loops + 1}")
    out, info = deps.llm.structured(prompts.WRITER, "\n".join(prompt), WriterOutput, purpose="writer")
    log(state, deps, "writer", "llm_result", latency_ms=round(info.get("latency_ms", 0), 1), usage=info.get("usage", {}),
        claims=len(out.activity_description), references=len(out.regulatory_references))
    state.draft = out
    return state

"""Writer: structured LLM output (claims with evidence ids). It sees the ledger and any critic feedback."""
import json

from src.agents import config, prompts
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


def facts_for_prompt(facts, background, conflict):
    lines = [f"- {c.text} [{', '.join(c.evidence_ids)}]" for c in facts]
    if conflict:
        lines.append(f"- SIGNAL CONFLICT: {conflict}")
    lines += [f"- {r.paraphrase}" + (f' Excerpt: "{r.quote}"' if r.quote else "") + f" [{r.evidence_id}]" for r in background]
    return "\n".join(lines)


def writer_v3(state: AgentState, deps: Deps) -> AgentState:
    """Design v3: code renders the facts and the background; the LLM only writes a short summary and next checks."""
    from src.agents.facts import code_built_claims, render_background
    from src.agents.nodes.critic import make_fetch_chunk
    from src.agents.state import Claim, DraftV3, SummaryDraft

    state.nodes_visited.append("writer")
    fetch = make_fetch_chunk(deps)
    facts = [Claim(text=c.text, evidence_ids=list(c.evidence_ids)) for c in code_built_claims(state)]
    background = render_background(state, fetch)
    prompt = ["THE MEMO'S FACTS (written by code; evidence ids in square brackets):",
              facts_for_prompt(facts, background, state.signal_conflict)]
    if state.critic_feedback:
        prompt += ["", "YOUR PREVIOUS DRAFT WAS REJECTED BY THE CRITIC. Fix every problem below, or drop the sentence:",
                   *[f"- {x}" for x in state.critic_feedback]]
    prompt += ["", "Write executive_summary (at most 3 sentences) and next_checks now."]
    count_llm_call(state, deps, "writer", purpose=f"draft {state.critic_loops + 1}")
    out, info = deps.llm.structured(prompts.WRITER_V3, "\n".join(prompt), SummaryDraft, purpose="writer")
    log(state, deps, "writer", "llm_result", latency_ms=round(info.get("latency_ms", 0), 1), usage=info.get("usage", {}),
        summary_sentences=len(out.executive_summary), next_checks=len(out.next_checks))
    # the code-built parts are attached by code; the LLM was never offered a field for them
    state.draft3 = DraftV3(executive_summary=out.executive_summary, next_checks=out.next_checks, facts=facts,
                           rendered_background=background)
    if state.first_draft3 is None:
        state.first_draft3 = state.draft3.model_copy(deep=True)
    return state


def writer(state: AgentState, deps: Deps) -> AgentState:
    if config.DESIGN == "v3":
        return writer_v3(state, deps)
    state.nodes_visited.append("writer")
    prompt = ["EVIDENCE LEDGER (evidence_id -> tool output):", json.dumps(ledger_for_prompt(state), separators=(",", ":"))]
    if state.signal_conflict:
        prompt += ["", "REQUIRED SIGNAL CONFLICT (copy into signal_conflict unchanged):", state.signal_conflict]
    else:
        prompt += ["", "There is no signal conflict: set signal_conflict to null."]
    if state.critic_feedback:
        prompt += ["", "YOUR PREVIOUS DRAFT WAS REJECTED BY THE CRITIC. Fix every problem below, or drop the claim:",
                   *[f"- {x}" for x in state.critic_feedback]]
    prompt += ["", "Write signal_conflict, the activity_description claims and regulatory_background now."]
    count_llm_call(state, deps, "writer", purpose=f"draft {state.critic_loops + 1}")
    out, info = deps.llm.structured(prompts.WRITER, "\n".join(prompt), WriterOutput, purpose="writer")
    log(state, deps, "writer", "llm_result", latency_ms=round(info.get("latency_ms", 0), 1), usage=info.get("usage", {}),
        claims=len(out.activity_description), background=len(out.regulatory_background))
    state.draft = out
    if state.first_draft is None:  # recording only: the first draft, before any critic feedback
        state.first_draft = out.model_copy(deep=True)
    return state

"""Critic: deterministic checks first, then an LLM check of whether the cited evidence supports each claim."""
import json
from typing import List, Literal

from pydantic import BaseModel

from src.agents import config, prompts
from src.agents.build import assemble
from src.agents.checks import FEATURE, background_provenance_reasons, check_item, signal_conflict_reasons
from src.agents.memo import MemoClaim
from src.agents.runtime import Deps, count_llm_call, log
from src.agents.state import AgentState, CriticVerdict, RegReference


class JudgeVerdict(BaseModel):
    index: int
    verdict: Literal["supported", "unsupported", "overstated"]
    reason: str = ""


class JudgeOutput(BaseModel):
    verdicts: List[JudgeVerdict]


def make_fetch_chunk(deps: Deps):
    """Exact chunk text for quote checks (verification fetches are not counted against the agent's caps)."""
    cache = {}

    def fetch(eid):
        if eid not in cache:
            cache[eid] = deps.run_tool("get_regulation_chunk", {"evidence_id": eid})
        return cache[eid]

    return fetch


def judge_prompt(items, ledger):
    lines = []
    for n, (label, text, eids) in enumerate(items):
        ev = {e: ledger[e].output for e in eids if e in ledger}
        s = json.dumps(ev, separators=(",", ":"))
        lines.append(f"ITEM {n} ({label})\nCLAIM: {text}\nCITED EVIDENCE: {s if len(s) < 3500 else s[:3500] + '...'}")
    return "\n\n".join(lines)


def critic_v2(state: AgentState, deps: Deps) -> AgentState:
    state.nodes_visited.append("critic")
    rnd = state.critic_loops + 1
    fetch = make_fetch_chunk(deps)
    draft = state.draft
    items = [("claim", c.text, c.evidence_ids, None) for c in draft.activity_description] + \
            [("background", r.paraphrase, [r.evidence_id], r.quote) for r in draft.regulatory_background]

    det = [check_item(text, eids, quote, state.ledger, fetch, allow_quotes=(kind == "background"))
           for kind, text, eids, quote in items]
    # design rules on top of the per-claim checks: SHAP features in ONE claim; background only from a pattern-triggered search
    feature_claims = [i for i, it in enumerate(items) if it[0] == "claim" and FEATURE.search(it[1])]
    for i in feature_claims[1:]:
        det[i].append("feature claims must be merged into one claim listing the top features")
    for i, it in enumerate(items):
        if it[0] == "background":
            det[i] += background_provenance_reasons(it[2][0], state.ledger)
    conflict_reasons = signal_conflict_reasons(state.signal_conflict, draft.signal_conflict)
    passing = [i for i, r in enumerate(det) if not r]
    judged = {}
    if passing:  # the LLM only looks at items that survived the deterministic checks
        count_llm_call(state, deps, "critic", purpose=f"support check round {rnd}")
        jp = judge_prompt([(items[i][0], items[i][1], items[i][2]) for i in passing], state.ledger)
        out, info = deps.llm.structured(prompts.JUDGE, jp, JudgeOutput, purpose="critic")
        log(state, deps, "critic", "llm_result", latency_ms=round(info.get("latency_ms", 0), 1), usage=info.get("usage", {}))
        by_index = {v.index: v for v in out.verdicts}
        for n, i in enumerate(passing):
            v = by_index.get(n)
            # a missing verdict counts as unsupported: silence is not support
            judged[i] = (v.verdict, v.reason) if v else ("unsupported", "the reviewer returned no verdict for this item")

    def reasons_for(i):
        extra = [f"reviewer says {judged[i][0]}: {judged[i][1]}"] if i in judged and judged[i][0] != "supported" else []
        return det[i] + extra

    verdicts, failed, counters = [], [], {"claim": 0, "background": 0}
    for i, (kind, text, eids, quote) in enumerate(items):
        counters[kind] += 1
        lv, lr = judged.get(i, (None, None))
        ok = not det[i] and lv == "supported"
        verdicts.append(CriticVerdict(round=rnd, item=f"{kind} {counters[kind]}", text=text, ok=ok,
                                      deterministic_reasons=det[i], llm_verdict=lv, llm_reason=lr))
        if not ok:
            failed.append(i)
    if state.signal_conflict is not None or draft.signal_conflict:  # checked by code only, never by the LLM
        verdicts.append(CriticVerdict(round=rnd, item="signal_conflict", text=draft.signal_conflict or state.signal_conflict,
                                      ok=not conflict_reasons, deterministic_reasons=conflict_reasons))
    state.critic_verdicts += verdicts
    empty = not draft.activity_description
    log(state, deps, "critic", "verdicts", round=rnd, items=len(items), failed=len(failed), empty_draft=empty,
        conflict_ok=not conflict_reasons)

    if not failed and not empty and not conflict_reasons:
        return finish(state, deps, items, [], "verified", "verified")
    if state.critic_loops < config.MAX_CRITIC_LOOPS:
        state.critic_feedback = [f"{verdicts[i].item}: {items[i][1][:160]!r} -> " + "; ".join(reasons_for(i)) for i in failed]
        if conflict_reasons:
            state.critic_feedback.append("signal_conflict: " + "; ".join(conflict_reasons))
        if empty:
            state.critic_feedback.append("the draft contained no claims; write the claims the evidence supports")
        state.critic_loops += 1
        state.status = "rewriting"
        return state
    removed = [f"{verdicts[i].item}: {items[i][1]} (reasons: {'; '.join(reasons_for(i))})" for i in failed]
    keep = [items[i] for i in range(len(items)) if i not in failed]
    return finish(state, deps, keep, removed, "partially_verified", "partially verified", empty=empty,
                  conflict_failed=bool(conflict_reasons))


def finish(state: AgentState, deps: Deps, kept_items, removed, status, verification, empty=False, conflict_failed=False):
    claims = [MemoClaim(text=t, evidence_ids=e) for k, t, e, q in kept_items if k == "claim"]
    refs = [RegReference(paraphrase=t, evidence_id=e[0], quote=q) for k, t, e, q in kept_items if k == "background"]
    notes = []
    if status == "partially_verified":
        if removed:
            notes.append(f"{len(removed)} item(s) failed the critic after {state.critic_loops} rewrite(s) and were removed.")
        if conflict_failed:
            notes.append("the writer did not carry the required signal_conflict sentence unchanged; the sentence in this memo "
                         "was inserted by code, not written by the writer")
        if empty:
            notes.append("the writer produced no claims")
    state.memo = assemble(state, deps.now_iso(), status, claims, refs, verification=verification, removed=removed, notes=notes)
    state.status = status
    return state


# ---------- design v3: facts rendered by code; the critic checks the summary and verifies the code-built parts ----------
def finish_v3(state: AgentState, deps: Deps, draft, kept_summary, kept_checks, removed, problems, expected_facts, expected_bg):
    """Memo = the code-built facts and background (restored from the ledger if the draft's copy was altered) plus the
    summary sentences and next checks that survived the critic."""
    from src.agents.state import Claim

    clean = not removed and not problems
    status, verification = ("verified", "verified") if clean else ("partially_verified", "partially verified")
    notes = []
    if removed:
        notes.append(f"{len(removed)} item(s) failed the critic after {state.critic_loops} rewrite(s) and were removed.")
    if problems:
        notes.append("the code-built claims or the rendered regulatory background in the draft were altered or incomplete; the memo "
                     "contains the versions recomputed by code from the evidence ledger (" + "; ".join(problems)[:300] + ")")
    if not draft.executive_summary:
        notes.append("the writer produced no executive summary")
    facts = [MemoClaim(text=c.text, evidence_ids=list(c.evidence_ids), source="code") for c in expected_facts]
    summary = [MemoClaim(text=c.text, evidence_ids=list(c.evidence_ids), source="llm") for c in kept_summary]
    state.memo = assemble(state, deps.now_iso(), status, facts, expected_bg, verification=verification, removed=removed,
                          notes=notes, executive_summary=summary, next_checks=list(kept_checks), design="v3")
    state.status = status
    return state


def critic_v3(state: AgentState, deps: Deps) -> AgentState:
    from src.agents.checks import next_check_reasons
    from src.agents.facts import code_built_claims, facts_problems, render_background
    from src.agents.state import Claim

    state.nodes_visited.append("critic")
    rnd = state.critic_loops + 1
    fetch = make_fetch_chunk(deps)
    draft = state.draft3
    expected_facts = [Claim(text=c.text, evidence_ids=list(c.evidence_ids)) for c in code_built_claims(state)]
    expected_bg = render_background(state, fetch)
    problems = facts_problems(draft.facts, expected_facts, draft.rendered_background, expected_bg)

    items = [("summary", c.text, c.evidence_ids) for c in draft.executive_summary]
    det = [check_item(text, eids, None, state.ledger, fetch, allow_quotes=False) for _, text, eids in items]
    for i in range(config.MAX_SUMMARY_SENTENCES, len(items)):
        det[i].append(f"the executive summary may have at most {config.MAX_SUMMARY_SENTENCES} sentences")
    check_reasons = [next_check_reasons(t, state.ledger) for t in draft.next_checks]

    passing = [i for i, r in enumerate(det) if not r]
    judged = {}
    if passing:  # the LLM only looks at summary sentences that survived the deterministic checks
        count_llm_call(state, deps, "critic", purpose=f"support check round {rnd}")
        jp = judge_prompt([(items[i][0], items[i][1], items[i][2]) for i in passing], state.ledger)
        out, info = deps.llm.structured(prompts.JUDGE, jp, JudgeOutput, purpose="critic")
        log(state, deps, "critic", "llm_result", latency_ms=round(info.get("latency_ms", 0), 1), usage=info.get("usage", {}))
        by_index = {v.index: v for v in out.verdicts}
        for n, i in enumerate(passing):
            v = by_index.get(n)
            judged[i] = (v.verdict, v.reason) if v else ("unsupported", "the reviewer returned no verdict for this item")

    def reasons_for(i):
        extra = [f"reviewer says {judged[i][0]}: {judged[i][1]}"] if i in judged and judged[i][0] != "supported" else []
        return det[i] + extra

    verdicts, failed_summary, failed_checks = [], [], []
    for i, (kind, text, eids) in enumerate(items):
        lv, lr = judged.get(i, (None, None))
        ok = not det[i] and lv == "supported"
        verdicts.append(CriticVerdict(round=rnd, item=f"summary {i + 1}", text=text, ok=ok, deterministic_reasons=det[i],
                                      llm_verdict=lv, llm_reason=lr))
        if not ok:
            failed_summary.append(i)
    for j, (t, r) in enumerate(zip(draft.next_checks, check_reasons)):
        verdicts.append(CriticVerdict(round=rnd, item=f"check {j + 1}", text=t, ok=not r, deterministic_reasons=r))
        if r:
            failed_checks.append(j)
    if problems:  # not something the writer can fix: it did not write these parts
        verdicts.append(CriticVerdict(round=rnd, item="code-built facts", text="; ".join(problems)[:500], ok=False,
                                      deterministic_reasons=problems))
    state.critic_verdicts += verdicts
    empty = not draft.executive_summary
    log(state, deps, "critic", "verdicts", round=rnd, items=len(items), failed=len(failed_summary) + len(failed_checks),
        empty_draft=empty, facts_ok=not problems)

    if not (failed_summary or failed_checks or empty):
        return finish_v3(state, deps, draft, draft.executive_summary, draft.next_checks, [], problems, expected_facts, expected_bg)
    if state.critic_loops < config.MAX_CRITIC_LOOPS:
        fb = [f"summary {i + 1}: {items[i][1][:160]!r} -> " + "; ".join(reasons_for(i)) for i in failed_summary]
        fb += [f"check {j + 1}: {draft.next_checks[j][:120]!r} -> " + "; ".join(check_reasons[j]) for j in failed_checks]
        if empty:
            fb.append("the draft contained no executive summary; write at most 3 sentences restating the listed facts")
        state.critic_feedback = fb
        state.critic_loops += 1
        state.status = "rewriting"
        return state
    removed = [f"summary {i + 1}: {items[i][1]} (reasons: {'; '.join(reasons_for(i))})" for i in failed_summary]
    removed += [f"check {j + 1}: {draft.next_checks[j]} (reasons: {'; '.join(check_reasons[j])})" for j in failed_checks]
    kept_summary = [c for i, c in enumerate(draft.executive_summary) if i not in failed_summary]
    kept_checks = [t for j, t in enumerate(draft.next_checks) if j not in failed_checks]
    return finish_v3(state, deps, draft, kept_summary, kept_checks, removed, problems, expected_facts, expected_bg)


def critic(state: AgentState, deps: Deps) -> AgentState:
    return critic_v3(state, deps) if config.DESIGN == "v3" else critic_v2(state, deps)

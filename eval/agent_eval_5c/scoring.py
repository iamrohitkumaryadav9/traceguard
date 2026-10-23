"""Scoring of one (case, design) result for Phase 5c, reusing the Phase 5a metrics (eval/agent_eval/metrics.py).
Pure post-processing of a saved raw record: no LLM, no network."""
from eval.agent_eval import metrics as M
from eval.agent_eval.arms import normalize
from src.agents.memo import LIMITATIONS, PERIOD_NOTE, SARMemo


def memo_items(raw):
    """[(kind, text, evidence_ids, source)] for every statement of the memo: activity claims, regulatory background
    items and executive-summary sentences. Source is 'code' or 'llm'. A v2 memo has no source field (the tagged code
    predates it): its statements are LLM-authored when the writer ran, and code-built otherwise (deterministic memos)."""
    memo = SARMemo.model_validate(raw["memo"])
    v2 = raw.get("design") == "v2" or memo.design != "v3"
    writer_ran = "writer" in raw.get("nodes_visited", [])
    src = lambda s: ("llm" if writer_ran else "code") if v2 else s
    items = [("claim", c.text, c.evidence_ids, src(c.source)) for c in memo.activity_description]
    items += [("background", b.paraphrase + (f' Quote: "{b.quote}"' if b.quote else ""), [b.evidence_id], src(b.source))
              for b in memo.regulatory_background]
    items += [("summary", c.text, c.evidence_ids, src(c.source)) for c in memo.executive_summary]
    return items


def normalize_5c(raw) -> dict:
    """A normalized memo dict for the 5a metrics. Executive-summary sentences count as claims."""
    memo = SARMemo.model_validate(raw["memo"])
    d = normalize(memo, raw.get("design", "?"), raw)
    d["claims"] = d["claims"] + [{"text": c.text, "evidence_ids": c.evidence_ids} for c in memo.executive_summary]
    return d


def score_case(raw) -> dict:
    info = M.case_info(raw)
    memo = normalize_5c(raw)
    ok, total, bad = M.groundedness(memo, raw["ledger"])
    elements = M.completeness(memo, info, LIMITATIONS, PERIOD_NOTE)
    items = memo_items(raw)
    return {
        "tx_id": raw["tx_id"], "design": raw.get("design"), "status": raw["status"], "reached_writer": info["reached_writer"],
        "groundedness": {"grounded": ok, "total": total, "ungrounded": bad},
        "completeness": {"share": M.completeness_share(elements),
                         "elements": {k: {"applicable": a, "present": p} for k, (a, p) in elements.items()}},
        "violations": M.policy_violations(memo, raw, info),
        "info_only": M.info_only_counts(memo, raw),
        "tool_correctness": M.tool_correctness(raw, info),
        "critic": M.critic_stats(raw),
        "sources": {"code": sum(1 for i in items if i[3] == "code"), "llm": sum(1 for i in items if i[3] == "llm")},
        "cost": M.cost(raw, "A2"),
    }

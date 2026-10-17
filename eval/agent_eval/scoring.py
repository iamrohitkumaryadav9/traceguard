"""Score one (case, arm): memo metrics + run-level metrics. Works from saved raw records and arm memos."""
from eval.agent_eval import metrics as M
from src.agents.memo import LIMITATIONS, PERIOD_NOTE


def score_arm(arm_memo: dict, raw: dict) -> dict:
    info = M.case_info(raw)
    ok, total, bad = M.groundedness(arm_memo, raw["ledger"])
    elements = M.completeness(arm_memo, info, LIMITATIONS, PERIOD_NOTE)
    viol = M.policy_violations(arm_memo, raw, info)
    return {
        "groundedness": {"grounded": ok, "total": total, "ungrounded": bad},
        "completeness": {"elements": {k: {"applicable": a, "present": p} for k, (a, p) in elements.items()},
                         "share": M.completeness_share(elements)},
        "violations": viol, "info_only": M.info_only_counts(arm_memo, raw),
        "cost": M.cost(raw, arm_memo["arm"]),
    }


def score_run(raw: dict) -> dict:
    """Metrics that belong to the run, not to an arm."""
    info = M.case_info(raw)
    return {"tool_correctness": M.tool_correctness(raw, info), "critic": M.critic_stats(raw),
            "reached_writer": info["reached_writer"], "required_conflict": info["required_conflict"] is not None}

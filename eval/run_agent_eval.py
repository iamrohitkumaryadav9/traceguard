"""Run the pre-registered agent evaluation. python -m eval.run_agent_eval [--rescore] [--report-only] [--limit N]

Selects the cases (eval/agent_eval_spec.md), runs each once, and writes per-(case, arm) result files under
eval/results/agent_eval/. Resumable: finished cases are skipped. A daily-quota error stops the run cleanly.
NOTHING here changes a prompt, threshold or code path in response to results.
"""
import json
import sys
from pathlib import Path

from eval.agent_eval import report
from eval.agent_eval.arms import ARMS, build_a1
from eval.agent_eval.scoring import score_arm, score_run
from src.agents.graph import run_agent, trace_dict
from src.agents.runtime import Deps, default_run_tool

RESULT_DIR = Path("eval/results/agent_eval")
RETRYABLE = {"agent_error", "unknown_transaction", "quota_failure"}  # retried once on a later invocation (quota: always)
MAX_ATTEMPTS = 2


def raw_record(state, trace, case, attempts):
    return {
        "tx_id": state.tx_id, "stratum": case["stratum"], "status": state.status, "error": state.error,
        "time_step": state.time_step, "counts": trace["counts"], "latency_s": trace["latency_s"], "tokens": trace["tokens"],
        "nodes_visited": trace["nodes_visited"], "critic_verdicts": trace["critic_verdicts"], "trace": trace["events"],
        "ledger": {k: v.model_dump() for k, v in state.ledger.items()},
        "memo": state.memo.model_dump() if state.memo else None,
        "first_draft": state.first_draft.model_dump() if state.first_draft else None,
        "signal_conflict": state.signal_conflict, "signal_conflict_evidence_ids": state.signal_conflict_evidence_ids,
        "attempts": attempts,
    }


def write_arm_files(raw, case, out_dir):
    """Per-(case, arm) result files plus a per-run file. Pure post-processing: no LLM, safe to repeat (--rescore)."""
    if raw["memo"] is None or raw["status"] in report.EXCLUDED:
        return
    for arm, build in ARMS.items():
        memo = build(raw)
        (out_dir / f"{raw['tx_id']}_{arm}.json").write_text(json.dumps(
            {"case": case, "arm": arm, "memo": memo, "metrics": score_arm(memo, raw)}, indent=1, default=str), encoding="utf-8")
    (out_dir / f"{raw['tx_id']}_run.json").write_text(json.dumps(score_run(raw), indent=1, default=str), encoding="utf-8")


def run_case(case, llm, out_dir, run_tool=default_run_tool, attempts=1):
    """Run the agent on one case and write its files. Returns the raw record."""
    deps = Deps(llm=llm, run_tool=run_tool)
    wait0 = getattr(llm, "throttle_wait_s", 0.0)
    state = run_agent(case["tx_id"], deps)
    trace = trace_dict(state, deps)
    trace["latency_s"]["llm_throttle_wait"] = round(getattr(llm, "throttle_wait_s", 0.0) - wait0, 2)
    raw = raw_record(state, trace, case, attempts)
    if (state.error or "").startswith("DailyQuotaExceeded"):
        raw["status"] = "quota_failure"
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "raw" / f"{case['tx_id']}.json").write_text(json.dumps(raw, indent=1, default=str), encoding="utf-8")
    write_arm_files(raw, case, out_dir)
    return raw


def main(argv):
    out_dir = RESULT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_path = out_dir / "cases.json"
    if cases_path.exists():
        cases = json.loads(cases_path.read_text(encoding="utf-8"))
    else:
        from eval.agent_eval.selection import select_cases
        from src.tools.context import get_context

        cases = select_cases(get_context())
        cases_path.write_text(json.dumps(cases, indent=1), encoding="utf-8")
        print(f"selected {len(cases)} cases -> {cases_path}")

    if "--report-only" in argv:
        report.main(out_dir)
        return 0
    if "--rescore" in argv:  # recompute the arm files from the saved raw records (no LLM, no network)
        for c in cases:
            p = out_dir / "raw" / f"{c['tx_id']}.json"
            if p.exists():
                write_arm_files(json.loads(p.read_text(encoding="utf-8")), c, out_dir)
        report.main(out_dir)
        return 0

    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None
    from src.agents.llm import GeminiLLM

    llm, done, stopped_at = GeminiLLM(), 0, None
    for i, c in enumerate(cases):
        p = out_dir / "raw" / f"{c['tx_id']}.json"
        attempts = 1
        if p.exists():
            prev = json.loads(p.read_text(encoding="utf-8"))
            if prev["status"] not in RETRYABLE or (prev["status"] != "quota_failure" and prev.get("attempts", 1) >= MAX_ATTEMPTS):
                continue
            attempts = prev.get("attempts", 1) + (0 if prev["status"] == "quota_failure" else 1)
        if limit is not None and done >= limit:
            break
        raw = run_case(c, llm, out_dir, attempts=attempts)
        done += 1
        print(f"[{i + 1}/{len(cases)}] tx {c['tx_id']} ({c['stratum']}): {raw['status']} counts={raw['counts']} "
              f"latency={raw['latency_s']['total']}s", flush=True)
        if raw["status"] == "quota_failure":
            stopped_at = (i + 1, c["tx_id"])
            print(f"STOPPED: daily quota exhausted at case {stopped_at[0]}/{len(cases)} (tx {stopped_at[1]}); rerun to resume.")
            break
    report.main(out_dir)
    return 1 if stopped_at else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

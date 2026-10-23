"""Runs ONE node with the code of the directory this file sits in and writes a raw record.

The runner copies this file into the git worktree of the tag `agent-v2` (as `_v2_worker.py`) and starts it there, so the
node is investigated by exactly the code that was tagged. Usage:
    python _v2_worker.py <tx_id> <stratum> <out.json> <attempts>
The Gemini key arrives through the environment (GEMINI_API_KEY) and is never printed.
"""
import json
import sys
from pathlib import Path


def run(tx_id, stratum, out_path, attempts=1, llm=None):
    from src.agents.graph import run_agent, trace_dict
    from src.agents.runtime import Deps

    if llm is None:
        from src.agents.llm import GeminiLLM

        llm = GeminiLLM()
    deps = Deps(llm=llm)
    wait0 = getattr(llm, "throttle_wait_s", 0.0)
    state = run_agent(tx_id, deps)
    trace = trace_dict(state, deps)
    trace["latency_s"]["llm_throttle_wait"] = round(getattr(llm, "throttle_wait_s", 0.0) - wait0, 2)
    raw = {
        "tx_id": state.tx_id, "stratum": stratum, "design": "v2", "status": state.status, "error": state.error,
        "time_step": state.time_step, "counts": trace["counts"], "latency_s": trace["latency_s"], "tokens": trace["tokens"],
        "nodes_visited": trace["nodes_visited"], "critic_verdicts": trace["critic_verdicts"], "trace": trace["events"],
        "ledger": {k: v.model_dump() for k, v in state.ledger.items()},
        "memo": state.memo.model_dump() if state.memo else None,
        "first_draft": state.first_draft.model_dump() if getattr(state, "first_draft", None) else None,
        "signal_conflict": state.signal_conflict, "signal_conflict_evidence_ids": state.signal_conflict_evidence_ids,
        "attempts": attempts,
    }
    if (state.error or "").startswith("DailyQuotaExceeded"):
        raw["status"] = "quota_failure"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(raw, indent=1, default=str), encoding="utf-8")
    return raw


if __name__ == "__main__":
    tx, stratum, out, attempts = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 1
    run(tx, stratum, out, attempts)

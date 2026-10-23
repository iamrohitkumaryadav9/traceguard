"""Phase 5c bugfix verification (NOT an evaluation): re-run the code tagged agent-v3.1 on the 8 nodes whose agent-v3 runs had
critic rejections, into eval/results/agent_eval_5c_bugfix/. Those nodes were used for the diagnosis, so the result says
nothing about generalisation; the headline 5c numbers remain the agent-v3 run. Resumable; a quota error stops cleanly.

  python -m eval.run_5c_bugfix
"""
import json
import sys
from pathlib import Path

from eval.agent_eval_5c.scoring import score_case
from eval.run_agent_eval_5c import RESULT_DIR as SRC_DIR, needs_run, raw_path, run_v3

OUT_DIR = Path("eval/results/agent_eval_5c_bugfix")
NODES = [106300125, 149570741, 155603453, 200950509, 29584495, 61745735, 71987847, 97933782]


def main():
    from dotenv import load_dotenv

    load_dotenv()
    from src.agents.llm import GeminiLLM

    cases = {c["tx_id"]: c for c in json.loads((SRC_DIR / "cases.json").read_text(encoding="utf-8"))}
    llm, stopped = GeminiLLM(), False
    for tx in NODES:
        go, attempts = needs_run(OUT_DIR, {"tx_id": tx}, "v3")
        if not go:
            continue
        raw = run_v3(cases[tx], llm, OUT_DIR, attempts=attempts)
        raw["code_tag"] = "agent-v3.1"
        raw_path(OUT_DIR, tx, "v3").write_text(json.dumps(raw, indent=1, default=str), encoding="utf-8")
        print(f"tx {tx}: {raw['status']}", flush=True)
        if raw["status"] == "quota_failure":
            print("STOPPED: daily quota exhausted; rerun to resume.")
            stopped = True
            break
    (OUT_DIR / "scored").mkdir(parents=True, exist_ok=True)
    for tx in NODES:
        p = raw_path(OUT_DIR, tx, "v3")
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            if raw.get("memo"):
                (OUT_DIR / "scored" / f"{tx}_v3.json").write_text(json.dumps(score_case(raw), indent=1, default=str), encoding="utf-8")
    return 1 if stopped else 0


if __name__ == "__main__":
    sys.exit(main())

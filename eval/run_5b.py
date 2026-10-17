"""Phase 5b runner. python -m eval.run_5b judge|faults|handcheck|report|all   (resumable; measurement only)"""
import json
import sys
from pathlib import Path

from eval.agent_eval_5b import faults, handcheck, judge, report5b
from eval.agent_eval_5b.common import RESULT_DIR, load_raws
from src.agents.llm import GeminiLLM

MIN_INTERVAL_S = 6.0


def run_judge_cmd():
    raws = load_raws()
    base = GeminiLLM(model=judge.WRITER_MODEL)
    client_factory = base._get_client
    usable = []
    for m in judge.CANDIDATES:
        ok, err = judge.probe(m, client_factory)
        print(f"probe {m}: {'ok' if ok else 'not usable (' + err + ')'}", flush=True)
        if ok:
            usable.append(m)
    if not usable:
        print("no stronger candidate usable: using the writer's own model with the separate rubric (a limitation)")
    usable.append(judge.WRITER_MODEL)  # last resort fallback, reported as a limitation if it is ever used
    verdicts, log = judge.run_judge(raws, lambda m: GeminiLLM(model=m, min_interval_s=MIN_INTERVAL_S), usable)
    log["usable_models_in_order"] = usable
    (RESULT_DIR / "judge" / "log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    arms, models = judge.counts_per_arm(raws, verdicts)
    print({a: dict(c) for a, c in arms.items()}, dict(models), log)


def run_faults_cmd():
    raws = load_raws()
    done = faults.run_fault_injection(raws, GeminiLLM())
    summ, ctl = faults.summarize(done)
    print(json.dumps({k: {x: v for x, v in s.items() if x != "ci"} for k, s in summ.items()}, indent=1), ctl)


def main(argv):
    cmd = argv[0] if argv else "all"
    if cmd in ("judge", "all"):
        run_judge_cmd()
    if cmd in ("handcheck", "all"):
        print(handcheck.build())
    if cmd in ("faults", "all"):
        run_faults_cmd()
    if cmd in ("report", "all"):
        report5b.main()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

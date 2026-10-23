"""Run the pre-registered Phase 5c evaluation (v2 versus v3). Spec: eval/agent_eval_5c_spec.md (with its amendment).

  python -m eval.run_agent_eval_5c [--limit N] [--report-only] [--rescore] [--handcheck-only]

Per node, v2 is run and then v3, so pairs complete together if the daily quota stops the run. v2 is run from the git worktree
of the tag `agent-v2` through eval/agent_eval_5c/v2_worker.py (copied there); v3 runs in-process. Resumable: finished runs are
skipped; a quota failure is recorded, the run stops cleanly (exit code 1) and the next invocation retries that run first.
NOTHING here changes a prompt, threshold or code path in response to results.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from eval.agent_eval_5c import handcheck_5c, report_5c
from eval.agent_eval_5c.scoring import score_case
from eval.agent_eval_5c.selection import excluded_nodes, select_cases_5c

RESULT_DIR = Path("eval/results/agent_eval_5c")
DESIGNS = ("v2", "v3")
RETRYABLE = {"agent_error", "unknown_transaction", "quota_failure"}  # retried once more on a later invocation (quota: always)
MAX_ATTEMPTS = 2
V2_TAG = "agent-v2"
V2_WORKTREE = Path("..") / "traceguard-v2"
WORKER_SRC = Path(__file__).parent / "agent_eval_5c" / "v2_worker.py"


def ensure_v2_worktree(path=V2_WORKTREE, run=subprocess.run):
    """Create the worktree of the tag, link the untracked data and models, copy the worker in. Idempotent."""
    path = Path(path)
    if not path.exists():
        run(["git", "worktree", "add", str(path), V2_TAG], check=True)
    for name in ("data", "models"):
        link, target = path / name, Path(name).resolve()
        if not link.exists() and target.exists():
            run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True) if os.name == "nt" else os.symlink(target, link)
    shutil.copy(WORKER_SRC, path / "_v2_worker.py")
    return path


def v2_command(case, out_path, attempts, worktree=V2_WORKTREE):
    return [sys.executable, "_v2_worker.py", str(case["tx_id"]), case["stratum"], str(Path(out_path).resolve()), str(attempts)], Path(worktree)


def run_v2_subprocess(case, out_path, attempts, worktree=V2_WORKTREE, run=subprocess.run):
    cmd, cwd = v2_command(case, out_path, attempts, worktree)
    done = run(cmd, cwd=cwd, env=os.environ.copy(), capture_output=True, text=True, timeout=3600)
    if done.returncode != 0 and not Path(out_path).exists():  # the worker crashed before writing: record it as an agent error
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps({"tx_id": case["tx_id"], "stratum": case["stratum"], "design": "v2", "status": "agent_error",
                                              "error": f"worker exit {done.returncode}: {done.stderr[-400:]}", "memo": None, "attempts": attempts}),
                                  encoding="utf-8")
    return json.loads(Path(out_path).read_text(encoding="utf-8"))


def raw_path(out_dir, tx_id, design):
    return Path(out_dir) / "raw" / f"{tx_id}_{design}.json"


def run_v3(case, llm, out_dir, run_tool=None, attempts=1):
    """The current code (design v3) on one case, in process. Same raw record as the 5a runner plus the design."""
    from eval.run_agent_eval import raw_record
    from src.agents.graph import run_agent, trace_dict
    from src.agents.runtime import Deps, default_run_tool

    deps = Deps(llm=llm, run_tool=run_tool or default_run_tool)
    wait0 = getattr(llm, "throttle_wait_s", 0.0)
    state = run_agent(case["tx_id"], deps)
    trace = trace_dict(state, deps)
    trace["latency_s"]["llm_throttle_wait"] = round(getattr(llm, "throttle_wait_s", 0.0) - wait0, 2)
    raw = {**raw_record(state, trace, case, attempts), "design": "v3"}
    raw["first_draft"] = state.first_draft3.model_dump() if getattr(state, "first_draft3", None) else None
    if (state.error or "").startswith("DailyQuotaExceeded"):
        raw["status"] = "quota_failure"
    p = raw_path(out_dir, case["tx_id"], "v3")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(raw, indent=1, default=str), encoding="utf-8")
    return raw


def rescore(out_dir, cases):
    """Per-(case, design) scored files from saved raw records. Pure post-processing: safe to repeat."""
    (Path(out_dir) / "scored").mkdir(parents=True, exist_ok=True)
    for c in cases:
        for d in DESIGNS:
            p = raw_path(out_dir, c["tx_id"], d)
            if p.exists():
                raw = json.loads(p.read_text(encoding="utf-8"))
                if raw.get("memo") is not None and raw["status"] not in report_5c.EXCLUDED:
                    (Path(out_dir) / "scored" / f"{c['tx_id']}_{d}.json").write_text(
                        json.dumps({"case": c, "metrics": score_case(raw)}, indent=1, default=str), encoding="utf-8")


def load_or_select(out_dir):
    p = Path(out_dir) / "cases.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    from src.tools.context import get_context

    cases = select_cases_5c(get_context(), excluded_nodes())
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cases, indent=1), encoding="utf-8")
    print(f"selected {len(cases)} cases -> {p}")
    return cases


def needs_run(out_dir, case, design):
    """(run?, attempts)"""
    p = raw_path(out_dir, case["tx_id"], design)
    if not p.exists():
        return True, 1
    prev = json.loads(p.read_text(encoding="utf-8"))
    if prev["status"] not in RETRYABLE or (prev["status"] != "quota_failure" and prev.get("attempts", 1) >= MAX_ATTEMPTS):
        return False, 0
    return True, prev.get("attempts", 1) + (0 if prev["status"] == "quota_failure" else 1)


def run_all(cases, out_dir, llm, limit=None, v2_runner=run_v2_subprocess, v3_runner=run_v3, log=print):
    """The resumable loop. Returns (runs done, stopped_at or None). limit counts (case, design) runs."""
    done, stopped = 0, None
    for i, c in enumerate(cases):
        for d in DESIGNS:
            go, attempts = needs_run(out_dir, c, d)
            if not go:
                continue
            if limit is not None and done >= limit:
                return done, stopped
            p = raw_path(out_dir, c["tx_id"], d)
            raw = v2_runner(c, p, attempts) if d == "v2" else v3_runner(c, llm, out_dir, attempts=attempts)
            done += 1
            log(f"[{i + 1}/{len(cases)}] tx {c['tx_id']} {d} ({c['stratum']}): {raw['status']}")
            if raw["status"] == "quota_failure":
                stopped = (i + 1, c["tx_id"], d)
                log(f"STOPPED: daily quota exhausted at case {i + 1}/{len(cases)} (tx {c['tx_id']}, {d}); rerun to resume.")
                return done, stopped
    return done, stopped


def main(argv):
    out_dir = RESULT_DIR
    cases = load_or_select(out_dir)
    if "--report-only" in argv or "--rescore" in argv or "--handcheck-only" in argv:
        rescore(out_dir, cases)
        if "--handcheck-only" in argv:
            print(handcheck_5c.build(out_dir))
        else:
            report_5c.main(out_dir)
        return 0
    from dotenv import load_dotenv

    load_dotenv()  # key goes into os.environ for the v2 worker; never printed
    from src.agents.llm import GeminiLLM

    ensure_v2_worktree()
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None
    _, stopped = run_all(cases, out_dir, GeminiLLM(), limit=limit)
    rescore(out_dir, cases)
    report_5c.main(out_dir)
    if not stopped and all(raw_path(out_dir, c["tx_id"], d).exists() for c in cases for d in DESIGNS):
        print(handcheck_5c.build(out_dir))  # the sheet is built once, when every run has finished
    return 1 if stopped else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

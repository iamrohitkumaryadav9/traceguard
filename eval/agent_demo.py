"""Phase 4a demo: run the agent on 6 real nodes chosen by fixed rules. Run: python -m eval.agent_demo

Labels are used HERE ONLY to apply the selection rules (the agent never sees them). Nodes are drawn with a
seeded RNG (seed 42) from the sorted candidate ids of each rule, in rule order, before anything is run.
"""
import json
import sys
from pathlib import Path

import numpy as np

from src.agents import config
from src.agents.graph import run_agent, trace_dict
from src.agents.llm import GeminiLLM
from src.agents.memo import memo_to_markdown
from src.agents.runtime import Deps
from src.tools.context import get_context

OUT = Path("eval/results/agent_demo.md")
SEED = 42


def select(ctx):
    n = len(ctx.ids)
    t, y, s = ctx.time_step, ctx._y, ctx.score
    fan_in, fan_out, chain = ctx.pattern_flags(np.arange(n))
    pre = (t >= 35) & (t <= 42)
    rules = [
        ("1. high-band, illicit, steps 35-42", pre & (s >= ctx.high_threshold) & (y == 1)),
        ("2. high-band, licit, steps 35-42", pre & (s >= ctx.high_threshold) & (y == 0)),
        ("3. flagged chain, steps 35-42", pre & chain),
        ("4. flagged fan_in (lowers_risk), steps 35-42, band at least medium, not chain", pre & fan_in & ~chain & (s >= ctx.alert_threshold)),
        ("5. steps 43-49, band at least medium", (t >= 43) & (s >= ctx.alert_threshold)),
        ("6. low band, steps 35-42", pre & (s < ctx.alert_threshold)),
    ]
    rng = np.random.default_rng(SEED)
    chosen, picked = [], set()
    for label, mask in rules:
        ids = np.sort(ctx.ids[mask])
        pick = int(rng.choice(ids))
        while pick in picked:  # never run the same node twice: draw again
            pick = int(rng.choice(ids))
        picked.add(pick)
        i = ctx.idx(pick)
        chosen.append({"rule": label, "tx_id": pick, "candidates": int(mask.sum()), "time_step": int(t[i]),
                       "ground_truth": {1: "illicit", 0: "licit", -1: "unknown"}[int(y[i])],
                       "score": round(float(s[i]), 4)})
    return chosen


def main(only=None):
    ctx = get_context()
    chosen = select(ctx)
    print(json.dumps(chosen, indent=1))
    llm = GeminiLLM()
    results = []
    for c in chosen:
        if only and c["tx_id"] not in only:
            continue
        deps = Deps(llm=llm)
        wait0 = llm.throttle_wait_s
        state = run_agent(c["tx_id"], deps)
        trace = trace_dict(state, deps)
        trace["latency_s"]["llm_throttle_wait"] = round(llm.throttle_wait_s - wait0, 2)
        config.TRACE_DIR.mkdir(parents=True, exist_ok=True)
        (config.TRACE_DIR / f"{c['tx_id']}.json").write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
        results.append((c, state, trace))
        print(f"{c['tx_id']}: status={state.status} counts={trace['counts']} latency={trace['latency_s']}", flush=True)
    write_report(chosen, results)


def write_report(chosen, results):
    lines = ["# Agent demo (Phase 4a)", "",
             f"Model: `{config.LLM_MODEL}` (temperature {config.LLM_TEMPERATURE}). Retrieval is frozen (hybrid + headers, no rewriting).",
             "Nodes were chosen BEFORE running anything, by fixed rules and a seeded draw (seed 42) from the sorted "
             "candidate ids of each rule (`python -m eval.agent_demo`). Ground-truth labels were used only for the selection "
             "rules and are shown below for the record; the agent never sees them. Each run was done once; nothing was re-run "
             "to get a better result.", "",
             "| rule | tx_id | step | candidates | score | ground truth (not seen by agent) |", "|---|---|---|---|---|---|"]
    for c in chosen:
        lines.append(f"| {c['rule']} | {c['tx_id']} | {c['time_step']} | {c['candidates']:,} | {c['score']} | {c['ground_truth']} |")
    lines += ["", "## Run summary", "",
              "| tx_id | status | verification | nodes | investigator tool calls | searches | LLM calls | critic loops | total latency (s) | of which rate-limit waits (s) | tokens in/out |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, st, tr in results:
        k = tr["counts"]
        lines.append(f"| {c['tx_id']} | {st.status} | {st.memo.verification} | {' > '.join(tr['nodes_visited'])} | "
                     f"{k['tool_calls_investigator']} | {k['searches']} | {k['llm_calls']} | {k['critic_loops']} | "
                     f"{tr['latency_s']['total']} | {tr['latency_s']['llm_throttle_wait']} | {tr['tokens']['prompt']}/{tr['tokens']['output']} |")
    for c, st, tr in results:
        lines += ["", f"## {c['rule']} (tx {c['tx_id']})", "", f"Final status: **{st.status}**" + (f" (error: {st.error})" if st.error else ""), "",
                  memo_to_markdown(st.memo), "", "**Critic verdicts**", ""]
        if not tr["critic_verdicts"]:
            lines.append("- the critic did not run (deterministic path)")
        for v in tr["critic_verdicts"]:
            reasons = "; ".join(v["deterministic_reasons"]) or (f"reviewer: {v['llm_verdict']}" + (f" ({v['llm_reason']})" if v["llm_reason"] else ""))
            lines.append(f"- round {v['round']}, {v['item']}: {'OK' if v['ok'] else 'REJECTED'} - {reasons[:300]}")
    lines += ["", "## Prompt changes", "", "None yet."]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main({int(a) for a in sys.argv[1:]} or None)

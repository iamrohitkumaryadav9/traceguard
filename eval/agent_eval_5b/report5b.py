"""Write eval/results/agent_eval/report_5b.md (judge verdicts, fault injection, hand-check instructions)."""
import json
from pathlib import Path

from eval.agent_eval_5b.common import RESULT_DIR, load_raws, writer_runs
from eval.agent_eval_5b.faults import FAULT_TYPES, summarize
from eval.agent_eval_5b.judge import counts_per_arm


def pct(n, d):
    return f"{n}/{d} = {100 * n / d:.0f}%" if d else "n/a"


def judge_section(raws, verdicts, log):
    arms, models = counts_per_arm(raws, verdicts)
    lines = ["## A. LLM judge (faithfulness, independent of the critic)", "",
             f"Writer runs: {len(writer_runs(raws))}. Judge model(s): "
             + (", ".join(f"`{m}` ({n} unique claims)" for m, n in models.items()) or "none") + ". "
             + (f"Models tried in order: {log.get('models_tried')}. " if log else "")
             + (f"Daily-quota switches: {log.get('quota_switches')}. " if log and log.get("quota_switches") else "")
             + (f"STOPPED: {log['stopped']}." if log and log.get("stopped") else ""), "",
             "The judge saw only each claim's text and the evidence it cites (one call per unique claim and evidence pair), "
             "with its own rubric; it never saw critic verdicts, other claims or labels. Counts are over the claims of each arm "
             "(identical claims in A1 and A2 are counted in both).", "",
             "| arm | claims in the arm | claims judged | supported (of judged) | unsupported (of judged) | overstated (of judged) | not judged |",
             "|---|---|---|---|---|---|---|"]
    for a in ("A1", "A2"):
        c = arms[a]
        tot = sum(c.values())
        judged = tot - c["not judged"]
        lines.append(f"| {a} | {tot} | {judged} | {pct(c['supported'], judged)} | {pct(c['unsupported'], judged)} | "
                     f"{pct(c['overstated'], judged)} | {c['not judged']} |")
    own = models.get("gemini-3.5-flash-lite", 0)
    total_judged = sum(models.values())
    if total_judged:
        lines += ["", f"**Judge independence and coverage:** {own} of the {total_judged} judged unique claims were judged by `gemini-3.5-flash-lite`, "
                  "which is also the writer's model and the critic's reviewer (the pre-registered fallback, because the stronger candidates were "
                  "not usable: the pro model's quota, `gemini-2.5-pro` removed, `gemini-3.8-flash` 503 at the first probe). "
                  f"The other {total_judged - own} were judged by a different model (`gemini-3.8-flash`) after the first model's daily quota ran out. "
                  "The judge is therefore NOT independent of the writer for most claims. "
                  f"{sum(arms['A1'].values()) and arms['A1']['not judged']} claims of A1 and {arms['A2']['not judged']} of A2 were not judged because the "
                  "daily quotas of both models were exhausted; the run can be resumed after the quota resets (see `BUGLOG_5b.md`)."]
    # list the judge's non-supported verdicts so they can be inspected
    items_by_verdict = []
    from eval.agent_eval_5b.judge import collect_items

    for key, it in collect_items(raws).items():
        v = verdicts.get(key)
        if v and v["verdict"] != "supported":
            items_by_verdict.append((it["uses"][0][0], sorted({a for _, a in it["uses"]}), v["verdict"], it["text"][:140], v["reason"][:160]))
    lines += ["", "Claims the judge did not mark supported:", ""]
    lines += [f"- tx {tx} ({'+'.join(arms_)}): **{vd}** - {text} ({reason})" for tx, arms_, vd, text, reason in items_by_verdict] or ["- none"]
    return lines


def fault_section(done):
    summ, ctl = summarize(done)
    n_fault = sum(1 for _ in done["faults"].values())
    lines = ["## C. Critic fault injection (critic only, no writer calls)", "",
             f"**Coverage: {n_fault} of {len(done['memos']) * len(FAULT_TYPES)} fault copies and {ctl['copies']} of {len(done['memos'])} control copies were run."
             + (f" The run STOPPED: {(done.get('stopped') or 'not finished').split('.')[0]} (the free tier allows 500 requests per day per model); it is resumable (`python -m eval.run_5b faults`) once the quota resets, so the counts below are partial.**" if done.get("stopped") or n_fault < len(done['memos']) * len(FAULT_TYPES) or ctl['copies'] < len(done['memos']) else "**"), "",
             f"{len(done['memos'])} verified A2 memos (seed 42): {done['memos']}. One programmatic fault per copy; the critic node alone "
             "(with the system's own reviewer, gemini-3.5-flash-lite) judged each tampered draft, round 1. "
             "The injected faults are synthetic and probably easier to catch than natural errors; the reviewer is the same model family as the "
             "writer; counts per fault type are small.", "",
             "| fault | applicable | caught by the deterministic layer | caught by the LLM reviewer only | missed | caught by either (95% Wilson interval) |",
             "|---|---|---|---|---|---|"]
    for ft in FAULT_TYPES:
        s = summ[ft]
        ci = f"{s['caught']}/{s['applicable']} = {100 * s['caught'] / s['applicable']:.0f}% [{100 * s['ci'][0]:.0f}%, {100 * s['ci'][1]:.0f}%]" if s["applicable"] else "n/a"
        lines.append(f"| {ft} | {s['applicable']} applicable of {s['memos']} run (planned {len(done['memos'])}) | {s['deterministic']} | {s['reviewer only']} | {s['missed']} | {ci} |")
    lines += ["", f"**Control** (the {ctl['copies']} untampered originals through the same critic): "
              f"{ctl['copies_rejected']} of {ctl['copies']} copies had at least one item rejected "
              f"(false-rejection rate {100 * ctl['rate']:.0f}%, 95% interval [{100 * ctl['ci'][0]:.0f}%, {100 * ctl['ci'][1]:.0f}%]); "
              f"item level: {ctl['items_rejected']} of {ctl['items']} items rejected. Copies with a rejection: {ctl['rejected_tx']}."]
    missed = [r for r in done["faults"].values() if r.get("applicable") and r["outcome"] == "missed"]
    lines += ["", "Missed faults:", ""] + ([f"- {r['fault']} in tx {r['tx_id']} ({r['item']})" for r in missed] or ["- none"])
    return lines


HANDCHECK = ["## B. Hand-check sheet (to be filled in by a human)", "",
             "`eval/results/agent_eval/handcheck.md` (and `handcheck.csv`): 15 runs (the writer runs in which the critic changed the memo "
             "plus 9 random others, seed 42). The sheet is blinded: claim text and cited evidence only. Fill in the `human_verdict` column of "
             "the CSV (`supported`, `unsupported` or `overstated`), do not open `handcheck_key.json` first, then run:", "",
             "```", "python -m eval.agent_eval_5b.handcheck score", "```", "",
             "It prints human-versus-judge agreement (percent and Cohen's kappa), human-versus-deterministic agreement (the check of the "
             "100% ceiling of the Phase 5a citation and number consistency metric) and the critic's rejection precision."]


def build_report(result_dir=RESULT_DIR):
    d = Path(result_dir)
    raws = load_raws(d)
    verdicts = json.loads((d / "judge" / "verdicts.json").read_text(encoding="utf-8")) if (d / "judge" / "verdicts.json").exists() else {}
    log = json.loads((d / "judge" / "log.json").read_text(encoding="utf-8")) if (d / "judge" / "log.json").exists() else {}
    faults = json.loads((d / "faults" / "results.json").read_text(encoding="utf-8")) if (d / "faults" / "results.json").exists() else None
    lines = ["# Phase 5b report", "",
             "Pre-registered in `eval/agent_eval_5b_spec.md`. Measurement only: no system code, prompt or threshold was changed.", ""]
    lines += judge_section(raws, verdicts, log) + [""]
    lines += HANDCHECK + [""]
    lines += (fault_section(faults) if faults else ["## C. Critic fault injection", "", "not run"]) + [""]
    lines += ["## Limitations", "",
              "About 27 writer runs and about 150 unique claims; one author wrote the system, the metrics and this evaluation. The judge is a Gemini model, "
              "like the writer and the critic's reviewer (a stronger one if the log above says so), and sees chunk text that can be truncated to 1,200 "
              "characters. The human check covers 15 runs and one person. Injected faults are templates."]
    return "\n".join(lines) + "\n"


def main(result_dir=RESULT_DIR):
    text = build_report(result_dir)
    (Path(result_dir) / "report_5b.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

"""Aggregate saved (case, arm) results into eval/results/agent_eval/report.md. Run: python -m eval.agent_eval.report"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from eval.agent_eval import metrics as M

RESULT_DIR = Path("eval/results/agent_eval")
ARMS = ("A0", "A1", "A2")
EXCLUDED = {"agent_error", "unknown_transaction", "quota_failure"}
VIOLATION_NAMES = ["a_lowers_risk_called_suspicious", "b_low_risk_while_degraded", "c_quote_from_non_quotable", "d_label_leak"]


def load(result_dir=RESULT_DIR):
    cases = json.loads((result_dir / "cases.json").read_text(encoding="utf-8"))
    rows = []
    for c in cases:
        raw_path = result_dir / "raw" / f"{c['tx_id']}.json"
        if not raw_path.exists():
            rows.append({"case": c, "raw": None, "arms": {}, "run": None})
            continue
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        arms = {a: json.loads((result_dir / f"{c['tx_id']}_{a}.json").read_text(encoding="utf-8"))
                for a in ARMS if (result_dir / f"{c['tx_id']}_{a}.json").exists()}
        run = json.loads((result_dir / f"{c['tx_id']}_run.json").read_text(encoding="utf-8")) if (result_dir / f"{c['tx_id']}_run.json").exists() else None
        rows.append({"case": c, "raw": raw, "arms": arms, "run": run})
    return rows


def split(rows):
    """(included, failed or skipped)"""
    inc, out = [], []
    for r in rows:
        if r["raw"] is None:
            out.append((r, "not run"))
        elif r["raw"]["status"] in EXCLUDED:
            out.append((r, f"{r['raw']['status']}: {r['raw'].get('error')}"))
        else:
            inc.append(r)
    return inc, out


def pct(num, den):
    return f"{num}/{den} = {100 * num / den:.0f}%" if den else "n/a"


def ci_str(point, lo, hi, as_pct=True):
    if np.isnan(point):
        return "n/a"
    f = (lambda x: f"{100 * x:.0f}%") if as_pct else (lambda x: f"{x:.2f}")
    return f"{f(point)} [{f(lo)}, {f(hi)}]"


def ground_stat(rows, arm, writer_only=False):
    sel = [r for r in rows if arm in r["arms"] and (not writer_only or r["run"]["reached_writer"])]
    num = [r["arms"][arm]["metrics"]["groundedness"]["grounded"] for r in sel]
    den = [r["arms"][arm]["metrics"]["groundedness"]["total"] for r in sel]
    return sum(num), sum(den), M.bootstrap_ci(num, den), len(sel)


def complete_stat(rows, arm):
    vals = [r["arms"][arm]["metrics"]["completeness"]["share"] for r in rows if arm in r["arms"]]
    return M.bootstrap_ci(vals, [1] * len(vals)), len(vals)


def violation_counts(rows, arm):
    out = {k: [] for k in VIOLATION_NAMES}
    for r in rows:
        if arm in r["arms"]:
            for k, v in r["arms"][arm]["metrics"]["violations"].items():
                if v:
                    out[k].append(r["case"]["tx_id"])
    return out


def paired_diff(rows, key_fn):
    """A2 minus A1 for cases that have both: key_fn(arm_result) -> (num, den)."""
    sel = [r for r in rows if "A1" in r["arms"] and "A2" in r["arms"]]
    a2 = [key_fn(r["arms"]["A2"]) for r in sel]
    a1 = [key_fn(r["arms"]["A1"]) for r in sel]
    return M.bootstrap_ci([x[0] for x in a2], [x[1] for x in a2], paired_with=([x[0] for x in a1], [x[1] for x in a1]))


def metric_block(rows, label):
    lines = [f"### {label} ({len(rows)} cases)", ""]
    lines += ["| arm | citation and number consistency, all cases | citation and number consistency, writer cases | completeness (mean over cases) | violations a/b/c/d (cases) |",
              "|---|---|---|---|---|"]
    for a in ARMS:
        n_all, d_all, ci_all, _ = ground_stat(rows, a)
        n_w, d_w, ci_w, nw = ground_stat(rows, a, writer_only=True)
        (cp, clo, chi), _ = complete_stat(rows, a)
        v = violation_counts(rows, a)
        lines.append(f"| {a} | {pct(n_all, d_all)} {ci_str(*ci_all)} | {pct(n_w, d_w)} {ci_str(*ci_w)} (n={nw} cases) | "
                     f"{ci_str(cp, clo, chi)} | {' / '.join(str(len(v[k])) for k in VIOLATION_NAMES)} |")
    return lines


def cost_block(rows):
    lines = ["| arm | LLM calls (mean / total) | prompt tokens (total) | output tokens (total) | wall time s, mean (with waits) | wall time s, mean (without waits) |",
             "|---|---|---|---|---|---|"]
    for a in ARMS:
        cs = [r["arms"][a]["metrics"]["cost"] for r in rows if a in r["arms"]]
        if not cs:
            continue
        nw = [c["wall_s_no_wait"] for c in cs]
        if any(x is None for x in nw):  # A1: the waits are not recorded per event, so the figure is not available
            nw = []
        lines.append(f"| {a} | {np.mean([c['llm_calls'] for c in cs]):.1f} / {sum(c['llm_calls'] for c in cs)} | "
                     f"{sum(c['prompt_tokens'] for c in cs):,} | {sum(c['output_tokens'] for c in cs):,} | "
                     f"{np.mean([c['wall_s'] for c in cs]):.1f} | {(f'{np.mean(nw):.1f}' if nw else 'n/a (not recorded per event)')} |")
    return lines


def element_block(rows):
    lines = ["| element | " + " | ".join(f"{a} present/applicable" for a in ARMS) + " |", "|---|" + "---|" * len(ARMS)]
    names = list(next(iter(r["arms"]["A2"]["metrics"]["completeness"]["elements"] for r in rows if "A2" in r["arms"])).keys())
    for e in names:
        cells = []
        for a in ARMS:
            app = pres = 0
            for r in rows:
                if a in r["arms"]:
                    el = r["arms"][a]["metrics"]["completeness"]["elements"][e]
                    app += el["applicable"]
                    pres += el["applicable"] and el["present"]
            cells.append(pct(pres, app))
        lines.append(f"| {e} | " + " | ".join(cells) + " |")
    return lines


def tool_block(rows):
    tcs = [r["run"]["tool_correctness"] for r in rows if r["run"]]
    r1 = [t for t in tcs if t["r1_applicable"]]
    calls = sum(t["agent_tool_calls"] for t in tcs)
    inv = sum(t["invalid_argument_calls"] for t in tcs)
    return [
        f"- R1 (a matched pattern implies match_typology was called): {pct(sum(1 for t in r1 if t['r1_ok']), len(r1))} of cases with a match.",
        f"- R2 as pre-registered (a regulation search only when a raises_risk pattern has a retrieval_query): "
        f"{sum(1 for t in tcs if t['r2_violation'])} of {len(tcs)} runs searched without a raises_risk pattern, of which "
        f"{sum(1 for t in tcs if t['search_with_only_uninformative'])} had only uninformative patterns (the declared mismatch: they also have a retrieval_query) "
        f"and {sum(1 for t in tcs if t['search_with_lowers_risk_only'])} had only lowers_risk patterns (must be 0).",
        f"- Within all caps: {pct(sum(1 for t in tcs if t['within_caps']), len(tcs))}; runs ended by a budget cap (budget_exceeded): {sum(1 for t in tcs if t['budget_exceeded'])}.",
        f"- Invalid-argument rate: {pct(inv, calls)} of investigator and regulation tool calls ({calls} calls).",
    ]


def critic_block(rows):
    runs = [r for r in rows if r["run"] and r["run"]["reached_writer"]]
    rounds = Counter(r["run"]["critic"]["rounds"] for r in runs)
    det, llm, cands, rej_cases = Counter(), Counter(), [], 0
    for r in runs:
        c = r["run"]["critic"]
        det.update(c["deterministic"])
        llm.update(c["llm"])
        cands += [{**x, "tx_id": r["case"]["tx_id"]} for x in c["false_rejection_candidates"]]
        rej_cases += c["rejected_items"] > 0
    lines = [f"Runs that reached the writer and the critic: {len(runs)}.", "",
             "Critic rounds needed: " + ", ".join(f"{k} round(s): {v} runs" for k, v in sorted(rounds.items())), "",
             f"Runs with at least one rejected item: {pct(rej_cases, len(runs))}.", "", "Deterministic rejection reasons (counted per rejected item and reason):", ""]
    lines += [f"- {k}: {v}" for k, v in det.most_common()] or ["- none"]
    lines += ["", f"LLM-reviewer rejections: unsupported {llm['unsupported']}, overstated {llm['overstated']}.", "",
              f"Reviewer false-rejection candidates (reason states the same numbers before and after 'but'; to be hand-verified in 5b): {len(cands)}"]
    lines += [f"- tx {c['tx_id']}, round {c['round']}, {c['item']}: {c['reason']}" for c in cands]
    return lines


def build_report(result_dir=RESULT_DIR):
    rows = load(result_dir)
    inc, out = split(rows)
    strata = defaultdict(list)
    for r in inc:
        strata[r["case"]["stratum"]].append(r)
    lines = ["# Agent evaluation report (Phase 5a)", "",
             "Pre-registered in `eval/agent_eval_spec.md` (committed before any case was selected or run). Arms: A0 = ML only (templated), "
             "A1 = the writer's first draft, A2 = the full system after the critic loops (A1 and A2 are paired, from the same run). "
             "Counts are shown next to percentages; intervals are 95% case-level bootstrap intervals (1,000 resamples, seed 42). "
             "This evaluation says nothing about whether the agents improve detection.", "",
             "Note added in Phase 5b: the metric first called \"groundedness\" is now called \"citation and number consistency\". "
             "It shows a ceiling effect (100% in every arm) and shares a blind spot with the critic (the scientific-notation bug found "
             "in Phase 5a affected both), so it must not be read as evidence that the claims are faithful to the evidence; "
             "see `report_5b.md` for the independent judge and the fault-injection results.", "",
             f"Cases selected: {len(rows)}; included in the aggregates: {len(inc)}; failed or skipped: {len(out)}.", ""]
    n_det = sum(1 for r in inc if not r["run"]["reached_writer"])
    lines += [f"Of the {len(inc)} included cases, {len(inc) - n_det} reached the writer and {n_det} ended on the deterministic path "
              "(cannot_assess or no_action_model_based), where A1 and A2 are the same memo by construction.", ""]
    lines += ["## Headline (all included cases)", ""] + metric_block(inc, "Overall")
    lines += ["", "Paired A2 minus A1 (cases with both arms), case-level bootstrap:", ""]
    g = paired_diff(inc, lambda a: (a["metrics"]["groundedness"]["grounded"], a["metrics"]["groundedness"]["total"]))
    cpl = paired_diff(inc, lambda a: (a["metrics"]["completeness"]["share"], 1))
    lines += [f"- citation and number consistency (pooled share): {ci_str(*g)}", f"- completeness (mean): {ci_str(*cpl)}"]
    lines += ["", "### Completeness by element", ""] + element_block(inc) if inc else []
    lines += ["", "### Tool-call correctness", ""] + tool_block(inc)
    lines += ["", "### Cost and latency", ""] + cost_block(inc)
    lines += ["", "## Policy violations (counted; case ids)", ""]
    for a in ARMS:
        v = violation_counts(inc, a)
        lines.append(f"- **{a}**: " + "; ".join(f"{k}: {len(ids)}" + (f" {ids}" if ids else "") for k, ids in v.items()))
    info = {a: Counter() for a in ARMS}
    for r in inc:
        for a in ARMS:
            if a in r["arms"]:
                info[a].update(r["arms"][a]["metrics"]["info_only"])
    lines += ["", "Information only (not violations): " + "; ".join(f"{a} {dict(info[a])}" for a in ARMS)]
    lines += ["", "Violation details:", ""]
    shown = False
    for r in inc:
        for a in ARMS:
            for k, v in r["arms"].get(a, {}).get("metrics", {}).get("violations", {}).items():
                for d in v:
                    lines.append(f"- tx {r['case']['tx_id']} {a} {k}: {d}")
                    shown = True
    if not shown:
        lines.append("- none")
    lines += ["", "## Critic statistics (A2 runs)", ""] + critic_block(inc)
    lines += ["", "## Per stratum", ""]
    for name in sorted(strata):
        rs = strata[name]
        lines += metric_block(rs, name) + [""]
        lines += [f"Cases: {[r['case']['tx_id'] for r in rs]}; statuses: {dict(Counter(r['raw']['status'] for r in rs))}", ""]
    lines += ["## Failed or skipped cases", ""]
    lines += [f"- tx {r['case']['tx_id']} (stratum {r['case']['stratum']}): {why}" for r, why in out] or ["- none"]
    lines += ["", "## Selected cases", "", "| stratum | tx_id | step | score | ground truth (not seen by the system) | status |", "|---|---|---|---|---|---|"]
    for r in rows:
        c = r["case"]
        lines.append(f"| {c['stratum']} | {c['tx_id']} | {c['time_step']} | {c['score']} | {c['ground_truth']} | {r['raw']['status'] if r['raw'] else 'not run'} |")
    notes = sorted({c["note"] for c in (r["case"] for r in rows) if c.get("note")})
    if notes:
        lines += ["", "Selection notes: " + "; ".join(notes)]
    lines += ["", "## Limitations", "",
              "Small n (40 cases, 4 to 8 per stratum), so intervals are wide and per-stratum results are anecdotal. One LLM "
              "(gemini-3.5-flash-lite) acts as investigator, writer and reviewer. Free-tier throttling inflates wall time. The test steps "
              "informed an earlier model-selection decision. The metrics are rules written by the system's author and check form and "
              "grounding, not relevance. A1 is the writer's first draft inside the same run, not a separately configured system. "
              "Completeness includes header, limitations and subject fields that code fills in for every arm."]
    return "\n".join(lines) + "\n"


def main(result_dir=RESULT_DIR):
    text = build_report(Path(result_dir))
    (Path(result_dir) / "report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

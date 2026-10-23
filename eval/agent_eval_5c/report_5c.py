"""Aggregate the saved Phase 5c results into report_5c.md. Pre-registered in eval/agent_eval_5c_spec.md (with its amendment).
Run: python -m eval.agent_eval_5c.report_5c   (no LLM, no network)"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from eval.agent_eval import metrics as M
from eval.agent_eval.report import VIOLATION_NAMES, ci_str, pct

RESULT_DIR = Path("eval/results/agent_eval_5c")
DESIGNS = ("v2", "v3")
EXCLUDED = {"agent_error", "unknown_transaction", "quota_failure"}
KEEP_MIN_GAIN = 0.05  # pre-registered keep rule: mean completeness gain of at least 0.05 and no new policy violations


def load(result_dir=RESULT_DIR):
    result_dir = Path(result_dir)
    rows = []
    for c in json.loads((result_dir / "cases.json").read_text(encoding="utf-8")):
        row = {"case": c, "raw": {}, "m": {}}
        for d in DESIGNS:
            rp, sp = result_dir / "raw" / f"{c['tx_id']}_{d}.json", result_dir / "scored" / f"{c['tx_id']}_{d}.json"
            if rp.exists():
                row["raw"][d] = json.loads(rp.read_text(encoding="utf-8"))
            if sp.exists():
                row["m"][d] = json.loads(sp.read_text(encoding="utf-8"))["metrics"]
        rows.append(row)
    return rows


def paired(rows):
    """Cases where both designs finished with a memo and were not excluded: the only cases the comparison uses."""
    return [r for r in rows if all(d in r["m"] for d in DESIGNS)]


def new_violations(pairs):
    """(case, rule) present in v3 and absent in v2."""
    out = []
    for r in pairs:
        for k in VIOLATION_NAMES:
            if r["m"]["v3"]["violations"][k] and not r["m"]["v2"]["violations"][k]:
                out.append((r["case"]["tx_id"], k))
    return out


def keep_rule(pairs):
    comp = lambda d: [r["m"][d]["completeness"]["share"] for r in pairs]
    point, lo, hi = M.bootstrap_ci(comp("v3"), [1] * len(pairs), paired_with=(comp("v2"), [1] * len(pairs))) if pairs else (float("nan"),) * 3
    new = new_violations(pairs)
    adopt = bool(pairs) and point >= KEEP_MIN_GAIN and not new
    return {"gain": point, "lo": lo, "hi": hi, "new_violations": new, "adopt": adopt}


def design_block(pairs):
    lines = ["| design | citation and number consistency | completeness (mean) | violations a/b/c/d (cases) | code-built / LLM-authored statements | LLM calls (mean / total) | prompt tokens | output tokens | wall s, mean without waits |",
             "|---|---|---|---|---|---|---|---|---|"]
    for d in DESIGNS:
        ms = [r["m"][d] for r in pairs]
        g = [m["groundedness"]["grounded"] for m in ms]
        t = [m["groundedness"]["total"] for m in ms]
        cp = M.bootstrap_ci([m["completeness"]["share"] for m in ms], [1] * len(ms))
        viol = [sum(1 for m in ms if m["violations"][k]) for k in VIOLATION_NAMES]
        cost = [m["cost"] for m in ms]
        nw = [c["wall_s_no_wait"] for c in cost if c["wall_s_no_wait"] is not None]
        lines.append(f"| {d} | {pct(sum(g), sum(t))} {ci_str(*M.bootstrap_ci(g, t))} | {ci_str(*cp)} | {' / '.join(map(str, viol))} | "
                     f"{sum(m['sources']['code'] for m in ms)} / {sum(m['sources']['llm'] for m in ms)} | "
                     f"{np.mean([c['llm_calls'] for c in cost]):.1f} / {sum(c['llm_calls'] for c in cost)} | "
                     f"{sum(c['prompt_tokens'] for c in cost):,} | {sum(c['output_tokens'] for c in cost):,} | "
                     f"{(f'{np.mean(nw):.1f}' if nw else 'n/a')} |")
    return lines


def element_block(pairs):
    names = list(pairs[0]["m"]["v3"]["completeness"]["elements"])
    lines = ["| element | v2 present/applicable | v3 present/applicable |", "|---|---|---|"]
    for e in names:
        cells = []
        for d in DESIGNS:
            app = sum(r["m"][d]["completeness"]["elements"][e]["applicable"] for r in pairs)
            pres = sum(r["m"][d]["completeness"]["elements"][e]["applicable"] and r["m"][d]["completeness"]["elements"][e]["present"] for r in pairs)
            cells.append(pct(pres, app))
        lines.append(f"| {e} | {cells[0]} | {cells[1]} |")
    return lines


def critic_block(pairs):
    lines = []
    for d in DESIGNS:
        runs = [r["m"][d] for r in pairs if r["m"][d]["reached_writer"]]
        det, llm = Counter(), Counter()
        for m in runs:
            det.update(m["critic"]["deterministic"])
            llm.update(m["critic"]["llm"])
        lines.append(f"- **{d}**: {len(runs)} runs reached the writer; runs with a rejected item: "
                     f"{sum(1 for m in runs if m['critic']['rejected_items'])}; deterministic reasons {dict(det) or 'none'}; "
                     f"reviewer rejections unsupported {llm['unsupported']}, overstated {llm['overstated']}")
    return lines


def handcheck_block(result_dir):
    p = Path(result_dir) / "handcheck_5c_scored.json"
    if not p.exists():
        return ["Not yet scored: fill in the `human_verdict` column of `handcheck_5c.csv`, then run "
                "`python -m eval.agent_eval_5c.handcheck_5c score`."]
    s = json.loads(p.read_text(encoding="utf-8"))
    lines = []
    for k, label in (("code_built", "(a) code-built claims whose text disagrees with their cited evidence"),
                     ("llm_summary", "(b) LLM-authored executive-summary sentences with an error")):
        r = s.get(k)
        lines.append(f"- {label}: {r['errors']}/{r['n']} = {100 * r['rate']:.1f}%, 95% Wilson interval [{100 * r['lo']:.1f}%, {100 * r['hi']:.1f}%]"
                     if r and r["n"] else f"- {label}: no rows filled in")
    lines.append(f"- The deterministic consistency metric called {s.get('deterministic_grounded_but_human_not_supported', 0)} of the "
                 "sentences the human did not mark supported grounded.")
    key_p, csv_p = Path(result_dir) / "handcheck_5c_key.json", Path(result_dir) / "handcheck_5c.csv"
    if key_p.exists() and csv_p.exists():
        import csv

        key = json.loads(key_p.read_text(encoding="utf-8"))["items"]
        for row in csv.DictReader(open(csv_p, encoding="utf-8")):
            if row["human_verdict"].strip().lower() not in ("", "supported"):
                lines.append(f"- {row['item_id']} ({key[row['item_id']]['claim_source']}-authored, {row['human_verdict']}): "
                             f"\"{row['claim']}\" Reader's note: {row['notes']}")
    lines += ["", "These rates are not comparable to the Phase 5b development-set rate (3 of 79). No claim is made that v3 is more accurate than "
              "v2; the finding is only that two error classes (a count wording that merges neighbour and subgraph counts, and a paraphrase "
              "of tool facts) cannot occur in v3 by construction. One human read the sheet, blinded to the claim source."]
    return lines


def build_report(result_dir=RESULT_DIR):
    rows = load(result_dir)
    pairs = paired(rows)
    unfinished = [(r["case"]["tx_id"], d, (r["raw"].get(d) or {}).get("status", "not run")) for r in rows for d in DESIGNS if d not in r["m"]]
    kr = keep_rule(pairs)
    lines = ["# Phase 5c report: code-rendered facts (v3) versus LLM-paraphrased facts (v2)", "",
             "Pre-registered in `eval/agent_eval_5c_spec.md` (spec and amendment committed before any run). Fresh set: seed 43, "
             "30 nodes disjoint from the demo nodes and the Phase 5a set. Same model, frozen retrieval, v2 = tag `agent-v2`. "
             "Intervals: 95% case-level bootstrap (1,000 resamples, seed 42) and Wilson intervals for hand-check proportions. "
             "Nothing here says anything about detection quality.", "",
             f"Cases selected: {len(rows)}; both designs finished and included in the paired comparison: {len(pairs)}.", ""]
    if unfinished:
        lines += ["Runs not in the comparison (case, design, status): " + "; ".join(f"{a} {b} {c}" for a, b, c in unfinished), ""]
    if not pairs:
        return "\n".join(lines + ["No paired results yet."]) + "\n"
    lines += ["## Headline (paired cases)", ""] + design_block(pairs)
    lines += ["", f"Paired completeness difference, v3 minus v2 (mean): {ci_str(kr['gain'], kr['lo'], kr['hi'], as_pct=False)}", "",
              "**Caveat that governs the completeness number (amendment, written before any run).** For v3 the completeness metric is "
              "near-tautological, because code inserts the required items. It measures the design guarantee, not LLM quality. "
              "The keep rule below is applied as pre-registered, and this caveat belongs to its conclusion.", ""]
    lines += ["### Completeness by element", ""] + element_block(pairs)
    lines += ["", "## Keep rule (pre-registered: adopt v3 if mean completeness improves by at least 0.05 and there are no new policy violations)", "",
              f"- completeness gain {kr['gain']:+.3f} (threshold {KEEP_MIN_GAIN}): {'met' if kr['gain'] >= KEEP_MIN_GAIN else 'not met'}",
              f"- new policy violations (present in v3, absent in v2 for the same case and rule): {len(kr['new_violations'])} {kr['new_violations'] or ''}",
              f"- **rule outcome: {'ADOPT v3' if kr['adopt'] else 'do not adopt v3 on this rule'}**, with the completeness caveat above: the gain reflects "
              "what code guarantees, not an improvement in LLM quality.", "", "## Policy violations (case ids)", ""]
    for d in DESIGNS:
        v = {k: [r["case"]["tx_id"] for r in pairs if r["m"][d]["violations"][k]] for k in VIOLATION_NAMES}
        lines.append(f"- **{d}**: " + "; ".join(f"{k}: {len(ids)}" + (f" {ids}" if ids else "") for k, ids in v.items()))
    lines += ["", "Violation details:", ""]
    shown = False
    for r in pairs:
        for d in DESIGNS:
            for k, v in r["m"][d]["violations"].items():
                for x in v:
                    lines.append(f"- tx {r['case']['tx_id']} {d} {k}: {x}")
                    shown = True
    lines += [] if shown else ["- none"]
    lines += ["", "## Critic statistics", ""] + critic_block(pairs)
    lines += ["", "## Hand-check (pre-registered checks (a) and (b); sheet `handcheck_5c.csv`)", ""] + handcheck_block(result_dir)
    lines += ["", "## Selected cases", "", "| stratum | tx_id | step | score | ground truth (not seen by the system) | v2 | v3 |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        c = r["case"]
        lines.append(f"| {c['stratum']} | {c['tx_id']} | {c['time_step']} | {c['score']} | {c['ground_truth']} | "
                     f"{(r['raw'].get('v2') or {}).get('status', 'not run')} | {(r['raw'].get('v3') or {}).get('status', 'not run')} |")
    lines += ["", "## Limitations", "",
              "Small n (30 nodes, 3 to 8 per stratum), so intervals are wide. One LLM acts as investigator, writer and reviewer. "
              "Metrics are rules written by the system's author. Completeness for v3 is guaranteed by construction (see the caveat). "
              "The deterministic consistency metric shares a blind spot with the critic; the hand-check is the independent measure."]
    posthoc = Path(result_dir) / "posthoc_decision.md"  # kept in its own file so regenerating the report does not drop it
    if posthoc.exists():
        lines += ["", posthoc.read_text(encoding="utf-8").strip()]
    return "\n".join(lines) + "\n"


def main(result_dir=RESULT_DIR):
    text = build_report(Path(result_dir))
    (Path(result_dir) / "report_5c.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

"""Write eval/results/agent_eval/report_5b.md (judge verdicts, hand-check results, fault injection)."""
import csv
import json
from pathlib import Path

from eval.agent_eval import metrics as M
from eval.agent_eval_5b import handcheck
from eval.agent_eval_5b.common import RESULT_DIR, load_raws, writer_runs
from eval.agent_eval_5b.faults import FAULT_TYPES, summarize
from eval.agent_eval_5b.judge import collect_items, counts_per_arm

DISCLOSURE = ("**Disclosure.** The human saw which run's claim had been flagged (tx 73002915) in a status message before scoring, "
              "so that run was not blinded.")
HANDCHECK_LIMITS = [
    "One human did the check.",
    "The 15 runs were not a random sample: 6 were chosen because the critic changed them, and only 9 were drawn at random (seed 42).",
    "Only 7 claims were flagged by the human, so every rate here (agreement, kappa, precision, recall) has a very wide interval and should be read as an anecdote.",
]


def pct(n, d):
    return f"{n}/{d} = {100 * n / d:.0f}%" if d else "n/a"


def clamp(x):
    return max(0.0, x)


def short(text, n=110):
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n - 1] + "..."


def judge_section(raws, verdicts, log):
    arms, models = counts_per_arm(raws, verdicts)
    total_judged = sum(models.values())
    own = models.get("gemini-3.5-flash-lite", 0)
    lines = ["## A. LLM judge (faithfulness, independent of the critic)", "",
             f"Writer runs: {len(writer_runs(raws))}; unique claims (a claim and its cited evidence, shared by A1 and A2 when identical): {len(collect_items(raws))}, "
             f"all judged ({total_judged}). Judge model(s): " + ", ".join(f"`{m}` ({n} unique claims)" for m, n in models.items()) + ". "
             + (f"Models tried in order: {log.get('models_tried')}; the first one's daily quota ran out and the next candidate took over "
                f"(switches: {log.get('quota_switches')}). " if log and log.get("quota_switches") else ""), "",
             "The judge saw only each claim's text and the evidence it cites (one call per unique claim and evidence pair), with its own rubric; "
             "it never saw critic verdicts, other claims or labels. Counts are over the claims of each arm (identical claims in A1 and A2 are counted in both).", "",
             "| arm | claims in the arm | judged | supported | unsupported | overstated |", "|---|---|---|---|---|---|"]
    for a in ("A1", "A2"):
        c = arms[a]
        tot = sum(c.values())
        judged = tot - c["not judged"]
        lines.append(f"| {a} | {tot} | {judged} | {pct(c['supported'], judged)} | {pct(c['unsupported'], judged)} | {pct(c['overstated'], judged)} |")
    if arms["A1"]["not judged"] or arms["A2"]["not judged"]:
        lines += ["", f"Not judged: {arms['A1']['not judged']} claims of A1 and {arms['A2']['not judged']} of A2."]
    lines += ["", f"**Judge independence:** {own} of the {total_judged} unique claims ({100 * own / total_judged:.0f}%) were judged by "
              "`gemini-3.5-flash-lite`, which is also the writer's model and the critic's reviewer (the pre-registered fallback, because the stronger "
              "candidates were not usable at the start: the pro model's quota, `gemini-2.5-pro` removed, `gemini-3.8-flash` returned 503 at the first probe). "
              f"The other {total_judged - own} were judged by `gemini-3.8-flash`. For most claims the judge is therefore NOT independent of the writer."]
    flagged = []
    for key, it in collect_items(raws).items():
        v = verdicts.get(key)
        if v and v["verdict"] != "supported":
            flagged.append((it["uses"][0][0], "+".join(sorted({a for _, a in it["uses"]})), v["verdict"], it["text"][:140], v["reason"][:200], v["model"]))
    lines += ["", "Claims the judge did not mark supported:", ""]
    lines += [f"- tx {tx} ({arms_}): **{vd}** ({model}) - {text} (judge's reason: {reason})" for tx, arms_, vd, text, reason, model in flagged] or ["- none"]
    return lines


def handcheck_section(d, raws, verdicts, faults):
    csv_path, key_path = d / "handcheck.csv", d / "handcheck_key.json"
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8"))) if csv_path.exists() else []
    if not any(r["human_verdict"].strip() for r in rows):
        return ["## B. Hand-check sheet", "", "`eval/results/agent_eval/handcheck.md` and `handcheck.csv` (15 runs, blinded) have not been filled in yet; "
                "after filling in the `human_verdict` column run `python -m eval.agent_eval_5b.handcheck score`."]
    keyfile = json.loads(key_path.read_text(encoding="utf-8"))
    key, changed = keyfile["items"], set(keyfile["changed_runs"])
    sc = handcheck.score(csv_path, key_path)
    rowmap = {r["item_id"]: r for r in rows}
    human = {i: r["human_verdict"].strip().lower() for i, r in rowmap.items()}
    flagged = [i for i in rowmap if human[i] != "supported"]
    n_rows, n_runs = len(rowmap), len({r["tx_id"] for r in rows})
    crit_rej = [i for i in rowmap if key[i]["critic_rejected"] and key[i]["tx_id"] in changed]
    crit_hit = [i for i in flagged if i in crit_rej]
    judge_hit = [i for i in flagged if key[i]["judge"] not in (None, "supported")]
    judge_na = [i for i in flagged if key[i]["judge"] is None]
    judge_fp = [i for i in rowmap if key[i]["judge"] not in (None, "supported") and human[i] == "supported"]
    det_hit = [i for i in flagged if key[i]["deterministic_grounded"] is False]
    det_na = [i for i in flagged if key[i]["deterministic_grounded"] is None]
    det_fp = [i for i in rowmap if key[i]["deterministic_grounded"] is False and human[i] == "supported"]
    kinds = {k: sum(1 for i in flagged if human[i] == k) for k in ("unsupported", "overstated")}

    def mark(flag, na=False):
        return "n/a (no evidence recorded)" if na else ("yes" if flag else "no")

    lines = ["## B. Hand-check results (human)", "",
             f"A human filled in the blinded sheet for {n_runs} runs ({n_rows} claims): {len(changed)} runs in which the critic changed the memo and "
             f"{n_runs - len(changed)} random others (seed 42). Output of `python -m eval.agent_eval_5b.handcheck score`:", "",
             "```json", json.dumps(sc, indent=2), "```", "",
             f"- **Human versus judge:** {100 * sc['judge']['agreement']:.0f}% agreement on {sc['judge']['n']} claims that had a judge verdict "
             f"(Cohen's kappa {sc['judge']['kappa_3class']:.2f} on three classes, {sc['judge']['kappa_supported_vs_not']:.2f} supported-versus-not). "
             "The high agreement comes from almost every claim being supported; the kappa is moderate.",
             f"- **Human versus the deterministic citation and number consistency check:** {100 * sc['deterministic']['agreement']:.0f}% agreement on "
             f"{sc['deterministic']['n']} claims, kappa {sc['deterministic']['kappa']:.2f}; the check called {sc['deterministic']['deterministic_grounded_but_human_not_supported']} "
             f"claims grounded that the human did not mark supported, and flagged {sc['deterministic']['deterministic_flagged_but_human_supported']} that the human marked supported. "
             "This is the direct test of the 100% ceiling of the Phase 5a metric.",
             f"- **Critic rejection precision:** {sc['critic_rejection_precision']['rejections_marked']} claims were rejected by the critic in the runs it changed, "
             f"the human agrees that {sc['critic_rejection_precision']['human_agrees_it_was_not_supported']} of them were not supported "
             f"(precision {100 * sc['critic_rejection_precision']['precision']:.0f}%; {sc['critic_rejection_precision']['with_recorded_evidence_only'][0]} of "
             f"{sc['critic_rejection_precision']['with_recorded_evidence_only'][1]} when only claims with recorded evidence are counted).", "",
             f"**Summary (computed from the key file, the hand-check sheet and `judge/verdicts.json`).** The human flagged {len(flagged)} of {n_rows} claims "
             f"({kinds['unsupported']} unsupported, {kinds['overstated']} overstated). The critic rejected {len(crit_rej)} claims (the human agrees with "
             f"{len(crit_hit)}) and missed {len(flagged) - len(crit_hit)} of the {len(flagged)}. The judge caught {len(judge_hit)} of the {len(flagged)}"
             f"{f' ({len(judge_na)} had no recorded evidence and could not be judged)' if judge_na else ''} with {len(judge_fp)} false positive"
             f"{'s' if len(judge_fp) != 1 else ''}. The deterministic metric caught {len(det_hit)} of the {len(flagged)}"
             f"{f' ({len(det_na)} had no recorded evidence)' if det_na else ''} with {len(det_fp)} false positive{'s' if len(det_fp) != 1 else ''}.", "",
             "### The human-flagged claims", "",
             "| claim | run | human verdict | human's note | critic rejected | judge flagged | deterministic flagged |", "|---|---|---|---|---|---|---|"]
    for i in flagged:
        k = key[i]
        jv = k["judge"]
        lines.append(f"| {i}: {short(rowmap[i]['claim'], 90).replace('|', '/')} | {k['tx_id']} | {human[i]} | {short(rowmap[i]['notes'], 100).replace('|', '/')} | "
                     f"{mark(i in crit_rej)} | {mark(jv not in (None, 'supported'), na=jv is None) + (f' ({jv})' if jv not in (None, 'supported') else '')} | "
                     f"{mark(k['deterministic_grounded'] is False, na=k['deterministic_grounded'] is None)} |")
    for i in judge_fp:
        k, r = key[i], rowmap[i]
        evidence_line = short(r["evidence_text"].split(chr(10))[0], 460)
        lines += ["", f"**Judge false positive: {i} (tx {k['tx_id']}).** Claim: \"{short(r['claim'], 200)}\" The judge marked it **{k['judge']}**; the human marked it "
                  f"**{human[i]}**. The judge's reason: \"{k['judge_reason']}\" First cited evidence entry: `{evidence_line}` "
                  "Read against that entry, the claim is stated as the evidence has it (compare the direction and the alert and score values of the in and out "
                  "neighbours), so the judge's reason does not match the cited evidence: it is an error by the judge, not by the memo."]
    for i in det_fp:
        r, raw = rowmap[i], raws[int(rowmap[i]["tx_id"])]
        ids = [x for x in r["cited_ids"].split(";") if x]
        _, why = M.claim_grounded(r["claim"], ids, raw["ledger"])
        lines += ["", f"**Deterministic false positive: {i} (tx {r['tx_id']}).** The metric flagged this regulatory-background item ({len(why)} numbers such as "
                  f"{', '.join(w.split()[1] for w in why[:3])} \"not in cited outputs\"), the human marked it supported. The numbers are inside the quoted "
                  "regulation passage; the metric only compares numbers with the numeric values of tool outputs and does not search numbers inside the chunk text."]
    summ, ctl = summarize(faults) if faults else ({}, None)
    if faults:
        app = sum(s["applicable"] for s in summ.values())
        caught = sum(s["caught"] for s in summ.values())
        copies = len(faults["faults"])
        lines += ["", "### Contrast with the synthetic fault injection", "",
                  f"In the injected-fault test (section C) the critic caught {caught} of {app} applicable faulty copies ({copies} copies were run, "
                  f"{copies - app} faults were not applicable to their memo) and rejected {ctl['copies_rejected']} of {ctl['copies']} untampered controls. "
                  f"On the {len(flagged)} real errors flagged by the human it rejected {len(crit_hit)} and missed {len(flagged) - len(crit_hit)}, and across "
                  f"{n_rows} real claims it made {len(crit_rej) - len(crit_hit)} rejections the human did not agree with. Synthetic faults are much easier to catch than the "
                  "natural errors, which were mostly subtle: a claim that goes beyond the cited evidence, or evidence of a different scope than the claim."]
    lines += ["", DISCLOSURE, "", "**Limitations of the hand check.** " + " ".join(HANDCHECK_LIMITS)]
    return lines


def fault_section(done):
    summ, ctl = summarize(done)
    n_fault = len(done["faults"])
    planned = len(done["memos"]) * len(FAULT_TYPES)
    app = sum(s["applicable"] for s in summ.values())
    complete = n_fault == planned and ctl["copies"] == len(done["memos"]) and not done.get("stopped")
    lines = ["## C. Critic fault injection (critic only, no writer calls)", "",
             f"**Coverage: {n_fault} of {planned} fault copies and {ctl['copies']} of {len(done['memos'])} control copies were run"
             + (f" ({app} fault copies were applicable to their memo, {n_fault - app} were not)." if complete else
                f". The run STOPPED: {(done.get('stopped') or 'not finished').split('.')[0]} (the free tier allows 500 requests per day per model); it is resumable "
                "(`python -m eval.run_5b faults`) once the quota resets, so the counts below are partial.") + "**", "",
             f"{len(done['memos'])} verified A2 memos (seed 42): {done['memos']}. One programmatic fault per copy; the critic node alone "
             "(with the system's own reviewer, gemini-3.5-flash-lite) judged each tampered draft, round 1. "
             "The injected faults are synthetic and probably easier to catch than natural errors; the reviewer is the same model family as the "
             "writer; counts per fault type are small.", "",
             "| fault | applicable | caught by the deterministic layer | caught by the LLM reviewer only | missed | caught by either (95% Wilson interval) |",
             "|---|---|---|---|---|---|"]
    for ft in FAULT_TYPES:
        s = summ[ft]
        ci = (f"{s['caught']}/{s['applicable']} = {100 * s['caught'] / s['applicable']:.0f}% [{100 * clamp(s['ci'][0]):.0f}%, {100 * s['ci'][1]:.0f}%]"
              if s["applicable"] else "n/a")
        run = "" if s["memos"] == len(done["memos"]) else f" (run on {s['memos']} of {len(done['memos'])})"
        lines.append(f"| {ft} | {s['applicable']} of {len(done['memos'])}{run} | {s['deterministic']} | {s['reviewer only']} | {s['missed']} | {ci} |")
    lines += ["", f"**Control** (the {ctl['copies']} untampered originals through the same critic): "
              f"{ctl['copies_rejected']} of {ctl['copies']} copies had at least one item rejected "
              f"(false-rejection rate {100 * ctl['rate']:.0f}%, 95% interval [{100 * clamp(ctl['ci'][0]):.0f}%, {100 * ctl['ci'][1]:.0f}%]); "
              f"item level: {ctl['items_rejected']} of {ctl['items']} items rejected. Copies with a rejection: {ctl['rejected_tx']}."]
    missed = [r for r in done["faults"].values() if r.get("applicable") and r["outcome"] == "missed"]
    lines += ["", "Missed faults:", ""] + ([f"- {r['fault']} in tx {r['tx_id']} ({r['item']})" for r in missed] or ["- none"])
    return lines


def build_report(result_dir=RESULT_DIR):
    d = Path(result_dir)
    raws = load_raws(d)
    vp, lp, fp = d / "judge" / "verdicts.json", d / "judge" / "log.json", d / "faults" / "results.json"
    verdicts = json.loads(vp.read_text(encoding="utf-8")) if vp.exists() else {}
    log = json.loads(lp.read_text(encoding="utf-8")) if lp.exists() else {}
    faults = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else None
    lines = ["# Phase 5b report", "",
             "Pre-registered in `eval/agent_eval_5b_spec.md`. Measurement only: no system code, prompt or threshold was changed.", ""]
    lines += judge_section(raws, verdicts, log) + [""]
    lines += handcheck_section(d, raws, verdicts, faults) + [""]
    lines += (fault_section(faults) if faults else ["## C. Critic fault injection", "", "not run"]) + [""]
    lines += ["## Limitations", "",
              "27 writer runs and 171 unique claims; one author wrote the system, the metrics and this evaluation. For most claims the judge is the same model "
              "as the writer and the critic's reviewer (see section A). The judge sees chunk text that can be truncated to 1,200 characters. The human check "
              "covers 15 runs, one person, and only 7 flagged claims (section B). Injected faults are templates and easier than natural errors."]
    return "\n".join(lines) + "\n"


def main(result_dir=RESULT_DIR):
    text = build_report(result_dir)
    (Path(result_dir) / "report_5b.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

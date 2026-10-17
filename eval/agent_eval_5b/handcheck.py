"""Blinded hand-check sheet and its scoring script (spec section B in eval/agent_eval_5b_spec.md).

  python -m eval.agent_eval_5b.handcheck build   -> handcheck.md, handcheck.csv, handcheck_key.json
  python -m eval.agent_eval_5b.handcheck score   -> agreement statistics, once the CSV's human_verdict column is filled in

The sheet shows claim text and cited evidence only. Judge verdicts, critic reasoning and deterministic results are kept
in handcheck_key.json (read by the scoring script; do not open it before filling in the CSV).
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from eval.agent_eval import metrics as M
from eval.agent_eval_5b.common import RESULT_DIR, critic_changed, evidence_text, load_raws, writer_runs
from eval.agent_eval_5b.judge import item_key
from eval.agent_eval.arms import build_a1, build_a2

SEED = 42
N_OTHER = 9
CSV_PATH = RESULT_DIR / "handcheck.csv"
MD_PATH = RESULT_DIR / "handcheck.md"
KEY_PATH = RESULT_DIR / "handcheck_key.json"
VERDICTS = ("supported", "unsupported", "overstated")
FIELDS = ["item_id", "tx_id", "run_type", "in_first_draft", "in_final", "critic_rejected", "claim", "cited_ids",
          "evidence_text", "human_verdict", "notes"]


def select_runs(raws, seed=SEED, n_other=N_OTHER):
    runs = writer_runs(raws)
    changed = [tx for tx, r in runs.items() if critic_changed(r)]
    others = [tx for tx in runs if tx not in changed]
    rng = np.random.default_rng(seed)
    pick = sorted(others[i] for i in rng.choice(len(others), size=min(n_other, len(others)), replace=False))
    return changed, pick


def rejected_texts(raw):
    return {v["text"] for v in raw.get("critic_verdicts", []) if not v["ok"]}


def rows_for_run(raw, run_type):
    """Rows for one run: distinct statements of the first draft and the final memo (cited ids known), plus rejected
    statements of intermediate drafts (cited ids not recorded)."""
    a1, a2 = build_a1(raw), build_a2(raw)
    rej = rejected_texts(raw)

    def stmts(memo):
        out = []
        for c in memo["claims"]:
            out.append((c["text"], c["evidence_ids"], "claim"))
        for b in memo["background"]:
            out.append((b["paraphrase"] + (f' Quote: "{b["quote"]}"' if b.get("quote") else ""), [b["evidence_id"]], "background"))
        return out

    seen, rows = {}, []
    for tag, memo in (("first", a1), ("final", a2)):
        for text, ids, kind in stmts(memo):
            k = (text, tuple(ids))
            if k not in seen:
                seen[k] = {"text": text, "ids": ids, "kind": kind, "first": False, "final": False}
                rows.append(seen[k])
            seen[k]["first" if tag == "first" else "final"] = True
    base = lambda t: t.split(' Quote: "')[0]  # critic verdicts log a background item's paraphrase without its quote
    known = {base(r["text"]) for r in rows}
    # rejected claims from intermediate drafts: their text was logged, their cited ids were not
    for t in sorted(rej - known):
        rows.append({"text": t, "ids": [], "kind": "claim", "first": False, "final": False, "intermediate": True})
    for r in rows:
        r["rejected"] = base(r["text"]) in rej
        r["run_type"] = run_type
    return rows


def build(result_dir=RESULT_DIR, judge_verdicts=None, seed=SEED):
    raws = load_raws(result_dir)
    changed, others = select_runs(raws, seed)
    judge_verdicts = judge_verdicts if judge_verdicts is not None else _load_judge(result_dir)
    csv_rows, key, md = [], {}, ["# Hand-check sheet (blinded)", "",
        "For every claim below, read the claim and the evidence it cites, and fill in the `human_verdict` column of "
        "`handcheck.csv` with `supported`, `unsupported` or `overstated`:", "",
        "- **supported**: every fact in the claim is explicitly in the cited evidence and numbers are equal.",
        "- **unsupported**: the claim states something that is not in the cited evidence or contradicts it.",
        "- **overstated**: related to the evidence, but more certain, broader or more causal than it allows.", "",
        "Do not open `handcheck_key.json` before you have finished. The sheet shows no judge or critic verdicts, except "
        "that for the runs marked CHANGED it tells you which claims the critic rejected (without its reasons), so you "
        "can mark whether each rejection was correct.", ""]
    for tx in changed + others:
        raw = raws[tx]
        run_type = "changed" if tx in changed else "other"
        rows = rows_for_run(raw, run_type)
        md += [f"## Run {tx} (time step {raw['time_step']}) - {'CHANGED by the critic' if run_type == 'changed' else 'critic did not change it'}", ""]
        if run_type == "changed":
            md += ["**First draft claims (A1)** and **final claims (A2)** are marked in the table; "
                   "`REJECTED` marks claims the critic rejected in some round.", ""]
        md += ["| id | in first draft | in final | critic rejected | claim | cited evidence |", "|---|---|---|---|---|---|"]
        for n, r in enumerate(rows, 1):
            item_id = f"{tx}-{n:02d}"
            ev = evidence_text(raw["ledger"], r["ids"]) if r["ids"] else "(cited evidence was not recorded for this intermediate draft)"
            rej = "REJECTED" if (run_type == "changed" and r["rejected"]) else ""
            csv_rows.append({"item_id": item_id, "tx_id": tx, "run_type": run_type,
                             "in_first_draft": "yes" if r["first"] else "no", "in_final": "yes" if r["final"] else "no",
                             "critic_rejected": rej, "claim": r["text"], "cited_ids": ";".join(r["ids"]),
                             "evidence_text": ev, "human_verdict": "", "notes": ""})
            md.append(f"| {item_id} | {'yes' if r['first'] else 'no'} | {'yes' if r['final'] else 'no'} | {rej} | "
                      f"{r['text'].replace('|', '/')} | {ev.replace('|', '/').replace(chr(10), '<br>')} |")
            ek = item_key(r["kind"], r["text"], r["ids"], evidence_text(raw["ledger"], r["ids"])) if r["ids"] else None
            det_ok = M.claim_grounded(r["text"], r["ids"], raw["ledger"])[0] if r["ids"] else None
            key[item_id] = {"tx_id": tx, "judge": (judge_verdicts.get(ek) or {}).get("verdict") if ek else None,
                            "judge_reason": (judge_verdicts.get(ek) or {}).get("reason") if ek else None,
                            "deterministic_grounded": det_ok, "critic_rejected": r["rejected"],
                            "evidence_recorded": bool(r["ids"])}
        md.append("")
    out = Path(result_dir)
    out.mkdir(parents=True, exist_ok=True)
    existing = out / "handcheck.csv"
    filled = existing.exists() and any(r["human_verdict"].strip() for r in csv.DictReader(open(existing, encoding="utf-8")))
    if not filled:  # never overwrite a sheet a human has started to fill in: only the key (judge verdicts) is refreshed
        with open(existing, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(csv_rows)
        (out / "handcheck.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "handcheck_key.json").write_text(json.dumps({"changed_runs": changed, "other_runs": others, "items": key}, indent=1),
                                            encoding="utf-8")
    return {"changed": changed, "others": others, "rows": len(csv_rows), "sheet_kept_because_it_has_human_entries": filled}


def _load_judge(result_dir):
    p = Path(result_dir) / "judge" / "verdicts.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# ---------- scoring ----------
def cohen_kappa(a, b):
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(a, b)) if len(set(a) | set(b)) > 1 else float("nan")


def score(csv_path=CSV_PATH, key_path=KEY_PATH):
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))["items"]
    rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8")) if r["human_verdict"].strip()]
    bad = [r for r in rows if r["human_verdict"].strip().lower() not in VERDICTS]
    if bad:
        raise ValueError(f"unknown human_verdict values: {sorted({r['human_verdict'] for r in bad})}")
    human = {r["item_id"]: r["human_verdict"].strip().lower() for r in rows}
    out = {"rows_filled": len(rows)}
    # human versus judge
    pairs = [(human[i], key[i]["judge"]) for i in human if key[i]["judge"]]
    if pairs:
        h, j = zip(*pairs)
        out["judge"] = {"n": len(pairs), "agreement": sum(x == y for x, y in pairs) / len(pairs), "kappa_3class": cohen_kappa(h, j),
                        "kappa_supported_vs_not": cohen_kappa([x == "supported" for x in h], [y == "supported" for y in j])}
    # human versus the deterministic citation-and-number check (binary)
    det = [(human[i] == "supported", key[i]["deterministic_grounded"]) for i in human if key[i]["deterministic_grounded"] is not None]
    if det:
        h, d = zip(*det)
        out["deterministic"] = {"n": len(det), "agreement": sum(x == y for x, y in det) / len(det),
                                "kappa": cohen_kappa(h, d),
                                "deterministic_grounded_but_human_not_supported": sum(1 for x, y in det if y and not x),
                                "deterministic_flagged_but_human_supported": sum(1 for x, y in det if x and not y)}
    # critic rejection precision
    rej = [i for i in human if key[i]["critic_rejected"] and key[i]["tx_id"] in _changed(key_path)]
    if rej:
        agree = sum(human[i] != "supported" for i in rej)
        rec = [i for i in rej if key[i]["evidence_recorded"]]
        out["critic_rejection_precision"] = {"rejections_marked": len(rej), "human_agrees_it_was_not_supported": agree,
                                             "precision": agree / len(rej),
                                             "with_recorded_evidence_only": (sum(human[i] != "supported" for i in rec), len(rec))}
    return out


def _changed(key_path):
    return set(json.loads(Path(key_path).read_text(encoding="utf-8"))["changed_runs"])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        print(build())
    elif cmd == "score":
        print(json.dumps(score(), indent=2))
    else:
        raise SystemExit("usage: python -m eval.agent_eval_5b.handcheck [build|score]")

"""Blinded hand-check sheet for Phase 5c (pre-registered checks (a) and (b), amendment of eval/agent_eval_5c_spec.md).

  python -m eval.agent_eval_5c.handcheck_5c build   -> handcheck_5c.md, handcheck_5c.csv, handcheck_5c_key.json
  python -m eval.agent_eval_5c.handcheck_5c score   -> error rates with Wilson intervals, once human_verdict is filled in

The sheet holds every LLM-authored executive-summary sentence of the v3 runs plus 20 randomly drawn code-built claims
(seed 43), shuffled together. `claim_source` (code or llm) is NOT in the CSV or the markdown: it is only in
handcheck_5c_key.json (do not open it before filling in the CSV).
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

from eval.agent_eval import metrics as M
from eval.agent_eval_5b.common import evidence_text
from eval.agent_eval_5c.scoring import memo_items
from src.ml.eval import wilson_interval

SEED = 43
N_CODE_BUILT = 20
RESULT_DIR = Path("eval/results/agent_eval_5c")
EXCLUDED = {"agent_error", "unknown_transaction", "quota_failure"}
VERDICTS = ("supported", "unsupported", "overstated")
FIELDS = ["item_id", "claim", "cited_ids", "evidence_text", "human_verdict", "notes"]  # no claim_source, no tx_id, no kind


def v3_raws(result_dir):
    out = []
    for p in sorted((Path(result_dir) / "raw").glob("*_v3.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("memo") and r["status"] not in EXCLUDED:
            out.append(r)
    return sorted(out, key=lambda r: r["tx_id"])


def pick_items(raws, seed=SEED, n_code=N_CODE_BUILT):
    """(summary sentences by the LLM, sampled code-built claims). Each: dict(tx_id, kind, text, ids, source, grounded)."""
    llm, code = [], []
    for raw in raws:
        for kind, text, ids, source in memo_items(raw):
            item = {"tx_id": raw["tx_id"], "kind": kind, "text": text, "ids": ids, "source": source,
                    "grounded": M.claim_grounded(text, ids, raw["ledger"])[0] if ids else None}
            if kind == "summary" and source == "llm":
                llm.append(item)
            elif source == "code":
                code.append(item)
    rng = np.random.default_rng(seed)
    sample = [code[i] for i in sorted(rng.choice(len(code), size=min(n_code, len(code)), replace=False))] if code else []
    return llm, sample, rng


def build(result_dir=RESULT_DIR, seed=SEED):
    result_dir = Path(result_dir)
    raws = v3_raws(result_dir)
    ledgers = {r["tx_id"]: r["ledger"] for r in raws}
    llm, sample, rng = pick_items(raws, seed)
    items = llm + sample
    items = [items[i] for i in rng.permutation(len(items))]  # shuffled together
    rows, key, md = [], {}, ["# Hand-check sheet 5c (blinded)", "",
        "For every statement below, read it and the evidence it cites, and fill in the `human_verdict` column of "
        "`handcheck_5c.csv` with `supported`, `unsupported` or `overstated`:", "",
        "- **supported**: every fact in the statement is explicitly in the cited evidence and numbers are equal.",
        "- **unsupported**: it states something that is not in the cited evidence or contradicts it.",
        "- **overstated**: related to the evidence, but more certain, broader or more causal than it allows.", "",
        "Do not open `handcheck_5c_key.json` before you have finished. The sheet does not say who wrote a statement.", "",
        "| id | statement | cited evidence |", "|---|---|---|"]
    for n, it in enumerate(items, 1):
        item_id = f"H{n:03d}"
        ev = evidence_text(ledgers[it["tx_id"]], it["ids"]) if it["ids"] else "(no evidence cited)"
        rows.append({"item_id": item_id, "claim": it["text"], "cited_ids": ";".join(it["ids"]), "evidence_text": ev,
                     "human_verdict": "", "notes": ""})
        md.append(f"| {item_id} | {it['text'].replace('|', '/')} | {ev.replace('|', '/').replace(chr(10), '<br>')} |")
        key[item_id] = {"claim_source": it["source"], "tx_id": it["tx_id"], "kind": it["kind"], "deterministic_grounded": it["grounded"]}
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / "handcheck_5c.csv"
    filled = csv_path.exists() and any(r["human_verdict"].strip() for r in csv.DictReader(open(csv_path, encoding="utf-8")))
    if not filled:  # never overwrite a sheet a human has started to fill in
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        (result_dir / "handcheck_5c.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        (result_dir / "handcheck_5c_key.json").write_text(json.dumps(
            {"seed": seed, "n_llm_summary": len(llm), "n_code_built": len(sample), "items": key}, indent=1), encoding="utf-8")
    return {"llm_summary_sentences": len(llm), "code_built_sampled": len(sample), "rows": len(rows), "sheet_kept_because_it_has_human_entries": filled}


def rate(errors, n):
    lo, hi = wilson_interval(errors, n)
    return {"errors": errors, "n": n, "rate": errors / n if n else None, "lo": max(0.0, lo), "hi": min(1.0, hi)}


def score(result_dir=RESULT_DIR):
    result_dir = Path(result_dir)
    key = json.loads((result_dir / "handcheck_5c_key.json").read_text(encoding="utf-8"))["items"]
    rows = [r for r in csv.DictReader(open(result_dir / "handcheck_5c.csv", encoding="utf-8")) if r["human_verdict"].strip()]
    bad = [r for r in rows if r["human_verdict"].strip().lower() not in VERDICTS]
    if bad:
        raise ValueError(f"unknown human_verdict values: {sorted({r['human_verdict'] for r in bad})}")
    human = {r["item_id"]: r["human_verdict"].strip().lower() for r in rows}
    out = {"rows_filled": len(rows)}
    for name, source in (("code_built", "code"), ("llm_summary", "llm")):
        ids = [i for i in human if key[i]["claim_source"] == source]
        out[name] = rate(sum(human[i] != "supported" for i in ids), len(ids))
        out[name]["by_verdict"] = {v: sum(human[i] == v for i in ids) for v in VERDICTS}
    det = [(human[i] == "supported", key[i]["deterministic_grounded"]) for i in human if key[i]["deterministic_grounded"] is not None]
    out["deterministic_grounded_but_human_not_supported"] = sum(1 for h, d in det if d and not h)
    (result_dir / "handcheck_5c_scored.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        print(build())
    elif cmd == "score":
        print(json.dumps(score(), indent=2))
    else:
        raise SystemExit("usage: python -m eval.agent_eval_5c.handcheck_5c [build|score]")

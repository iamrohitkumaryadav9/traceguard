"""Build demo/memos.json from the existing Phase 5c run records (agent-v3 runs: eval/results/agent_eval_5c/raw/*_v3.json).

Only the memo JSON and a small trace summary are copied. Not copied: the evidence ledger (raw tool outputs, including SHAP
feature values), the trace events, the stratum, the case file (which holds ground-truth labels), raw Elliptic files, the
feature matrix and the model files. The script prints the included fields, a ground-truth label check and the node-level
values that appear, so the dataset's licence can be checked before the bundle is published.

  python scripts/build_demo_bundle.py [--src eval/results/agent_eval_5c/raw] [--out demo/memos.json]
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from src.api.bundle import LABEL_KEYS, label_keys_in

EXAMPLES = [  # (tx_id, description): chosen by the type of case, from the Phase 5c evaluation runs
    (73002538, "High-band node at time step 42: high raw score, verified memo with code-rendered facts."),
    (200950509, "Chain node: a risk-raising chain pattern but a low model score, so the memo states a signal conflict."),
    (114656654, "Fan-in node: the fan_in pattern lowers risk in the training data, so the model's low band stands (no action suggested)."),
    (125406909, "Post-shutdown node (time step 47): the model-health check reports degradation, so the agent declines to assess."),
    (200638883, "Low-band node: no risk-raising pattern, so the memo suggests no action (model-based only) without any LLM call."),
]


def entry_from_raw(raw):
    c, lat = raw["counts"], raw["latency_s"]
    return {
        "tx_id": raw["tx_id"], "status": raw["status"], "time_step": raw["time_step"], "memo": raw["memo"],
        "trace_summary": {"tool_calls": c["tool_calls_total"], "llm_calls": c["llm_calls"], "critic_rounds": c["critic_loops"],
                          "latency_s": lat["total"], "nodes_visited": raw["nodes_visited"]},
    }


def build(src, out):
    memos = {}
    for p in sorted(Path(src).glob("*_v3.json")):
        raw = json.loads(p.read_text(encoding="utf-8"))
        if raw.get("memo") and raw["status"] not in ("agent_error", "unknown_transaction", "quota_failure"):
            memos[str(raw["tx_id"])] = entry_from_raw(raw)
    have = set(memos)
    examples = [{"tx_id": t, "description": d} for t, d in EXAMPLES if str(t) in have]
    bundle = {"meta": {"generated_from": "Phase 5c agent-v3 evaluation runs (memo JSON only)", "memo_design": "v3",
                       "note": "Pre-computed; the service code is tagged agent-v3.1 (differs only in the next-check rule)."},
              "examples": examples, "memos": memos}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(bundle, indent=1, ensure_ascii=False), encoding="utf-8")
    return bundle


def keys_of(obj, prefix=""):
    paths = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            paths.add(f"{prefix}{k}")
            paths |= keys_of(v, f"{prefix}{k}.")
    elif isinstance(obj, list) and obj:
        for v in obj:
            paths |= keys_of(v, f"{prefix}[].")
    return paths


def report(bundle, out):
    memos = list(bundle["memos"].values())
    print(f"wrote {out}: {len(memos)} memos, {len(bundle['examples'])} examples, {Path(out).stat().st_size / 1024:.0f} KiB\n")
    print("FIELDS INCLUDED (per memo entry):")
    for k in sorted(keys_of(memos)):
        print("  ", k)
    print("\nGROUND-TRUTH LABEL CHECK:")
    hits = label_keys_in(bundle)
    print(f"   keys named {sorted(LABEL_KEYS)} anywhere in the bundle: {len(hits)} {hits[:5]}")
    text = json.dumps(bundle).lower()
    for w in ('"ground_truth"', '"label"', '"stratum"', '"licit"', '"illicit"', '"unknown"'):
        print(f"   JSON string value or key {w}: {'FOUND' if w in text else 'absent'}")
    agg = Counter()
    for m in memos:
        for c in m["memo"]["activity_description"] + m["memo"]["executive_summary"]:
            for w in ("illicit", "licit"):
                if re.search(rf"\b{w}\b", c["text"]):
                    agg[w] += 1
    print(f"   statements mentioning the words illicit/licit (aggregate training-period rates and counts of labeled nodes "
          f"for a pattern or time window, not the label of the node itself): {dict(agg)}")
    print("\nNODE-LEVEL VALUES THAT APPEAR (all derived from Elliptic ids, scores and graph structure):")
    ids = set()
    for m in memos:
        for c in m["memo"]["activity_description"]:
            ids |= set(re.findall(r"(\d{6,}) \((?:in|out)coming neighbour", c["text"]))
    print(f"   - the transaction ids of {len(memos)} focal nodes and {len(ids)} distinct neighbour transaction ids (Elliptic txIds)")
    print("   - time step, raw model score, calibrated probability, risk band, alert and high thresholds, per focal node")
    print("   - per neighbour: hop, direction, raw score, alerted or not")
    print("   - subgraph statistics: node and edge counts, alerted share, focal in/out degree, chain length, component size")
    print("   - top SHAP feature ids (anonymized f-numbers) and their direction (up/down); NO feature values and NO SHAP magnitudes")
    print("   - structural pattern matches with training-period aggregate rates (illicit rate, baseline, lift, n labeled nodes)")
    print("   - model-health values per time step (window recall, reference recall, number of labeled illicit nodes in the window)")
    print("   - retrieved regulation excerpts (verbatim, at most 40 words, from quotable public documents) and their chunk ids")
    print("   NOT included: the 165 feature values, the evidence ledger, raw Elliptic files, edges, model files, stratum, case file.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="eval/results/agent_eval_5c/raw")
    ap.add_argument("--out", default="demo/memos.json")
    a = ap.parse_args()
    report(build(a.src, a.out), a.out)

"""Retrieval evaluation on eval/retrieval_golden.json. Run: python -m eval.retrieval_eval

Hit rule: a retrieved chunk is correct if its doc_id is the expected (or an acceptable) document AND
every must_contain regex matches its text (case-insensitive). No abstention cutoff is set here.
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.rag.retriever import get_retriever
from src.tools.typology import QUERIES

GOLDEN = Path("eval/retrieval_golden.json")
OUT = Path("eval/results/retrieval.md")
KS = (1, 3, 5)
DEPTH = 10  # MRR looks this deep
MODES = ("bm25", "dense", "hybrid")


def is_correct(q, chunk):
    docs = {q["expected_doc_id"], *q["acceptable_doc_ids"]}
    return chunk["doc_id"] in docs and all(re.search(p, chunk["text"], re.I | re.S) for p in q["must_contain"])


def first_rank(q, retriever, mode):
    hits = retriever.search(q["question"], k=DEPTH, mode=mode)
    for r, h in enumerate(hits, start=1):
        if is_correct(q, retriever.chunks[h["index"]]):
            return r
    return None


def md(df, digits=3):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.{digits}f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def main():
    qs = json.loads(GOLDEN.read_text())
    r = get_retriever()
    scored = [q for q in qs if q["type"] != "out_of_scope"]
    oos = [q for q in qs if q["type"] == "out_of_scope"]

    # sanity: every scored question has at least one correct chunk in the corpus
    for q in scored:
        n_gold = sum(is_correct(q, c) for c in r.chunks)
        assert n_gold > 0, f"{q['id']}: no chunk satisfies the must_contain rule"
    print(f"{len(qs)} questions: {len(scored)} scored ({sum(q['type'] == 'typology' for q in qs)} typology), "
          f"{len(oos)} out-of-scope")

    r.search("warm up", k=1)  # load models before timing
    rows, ranks_by_mode = [], {}
    for mode in MODES:
        ranks, times = [], []
        for q in scored:
            t0 = time.perf_counter()
            rank = first_rank(q, r, mode)
            times.append(time.perf_counter() - t0)
            ranks.append(rank)
        ranks_by_mode[mode] = ranks
        row = {"mode": mode}
        for k in KS:
            row[f"hit@{k}"] = float(np.mean([x is not None and x <= k for x in ranks]))
        row["MRR@10"] = float(np.mean([1 / x if x else 0.0 for x in ranks]))
        row["avg latency (ms)"] = 1000 * float(np.mean(times))
        rows.append(row)
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"\nRetrieval over {len(scored)} scored questions:")
    print(table.round(3).to_string(index=False))

    per_q = pd.DataFrame({"id": [q["id"] for q in scored], "type": [q["type"] for q in scored],
                          **{f"rank_{m}": ranks_by_mode[m] for m in MODES}})
    print("\nFirst correct rank per question (None = not in top 10):")
    print(per_q.to_string(index=False))

    # score comparison: top-1 fused (hybrid) score and raw scorer scores, in-scope vs out-of-scope
    comp = []
    for q in qs:
        top = r.search(q["question"], k=1, mode="hybrid")[0]
        comp.append({"id": q["id"], "group": "out_of_scope" if q["type"] == "out_of_scope" else "in_scope",
                     "top1_fused_rrf": top["score"], "top1_dense_cosine": top["dense_score"],
                     "top1_bm25": top["bm25_score"]})
    comp = pd.DataFrame(comp)
    print("\nTop-1 scores (hybrid), in-scope vs out-of-scope:")
    print(comp.round(4).to_string(index=False))
    summ = comp.groupby("group")[["top1_fused_rrf", "top1_dense_cosine", "top1_bm25"]].agg(["min", "mean", "max"])
    print(summ.round(4).to_string())

    # top-3 chunks for the three typology retrieval queries
    ex_lines = []
    print("\nTop-3 hybrid results for the typology retrieval queries:")
    for name, query in QUERIES.items():
        print(f"\n[{name}] {query}")
        ex_lines.append(f"**{name}**: {query}\n")
        for rank, h in enumerate(r.search(query, k=3, mode="hybrid"), start=1):
            c = r.chunks[h["index"]]
            snippet = c["text"][:260].replace("\n", " ")
            line = f"{rank}. {c['evidence_id']} (rrf {h['score']:.4f}, cos {h['dense_score']:.3f}) {snippet}..."
            print("  " + line)
            ex_lines.append(f"- {line}")
        ex_lines.append("")

    lines = [
        "# Retrieval evaluation", "",
        f"Embeddings: BAAI/bge-small-en-v1.5 (sentence-transformers, CPU). {len(r.chunks)} chunks over "
        f"{len({c['doc_id'] for c in r.chunks})} documents. Hybrid = BM25 + dense fused with reciprocal rank "
        "fusion (k=60). A hit needs the expected (or acceptable) document and every must_contain regex to match "
        "the chunk text.", "",
        f"{len(scored)} scored questions (incl. 3 typology queries), {len(oos)} out-of-scope.", "",
        md(table), "", "## First correct rank per question", "", md(per_q), "",
        "## Top-1 scores: in-scope vs out-of-scope (no abstention cutoff set)", "", md(comp, 4), "",
        "## Typology retrieval examples (hybrid top 3)", "", *ex_lines,
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

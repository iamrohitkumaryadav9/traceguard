"""Phase 3d: query rewriting (A) and contextual headers (B), then freeze. Run: python -m eval.retrieval_final

KEEP RULES (fixed before the experiments; primary metric = hybrid hit@3 on the 19 document-grounded questions):
  A. adopt LLM query rewriting if hybrid hit@3 improves by >= 0.10 (headers off).
  B. adopt contextual headers if hybrid hit@3 improves by >= 0.05 on raw queries AND hit@1 does not decrease.
This is the last retrieval experiment: the configuration the rules pick becomes the default.
"""
import json
import re
from pathlib import Path

import numpy as np

from eval.retrieval_eval import is_correct as old_is_correct
from src.rag.matching import matches_all
from src.rag.retriever import SentenceTransformerEmbedder, load_retriever
from src.rag.rewrite import REWRITE_MODEL, QueryRewriter

GOLDEN = Path("eval/retrieval_golden.json")
OUT = Path("eval/results/retrieval_final.md")
DEPTH = 10
RULE_A_MIN_GAIN = 0.10
RULE_B_MIN_GAIN = 0.05
EXAMPLE_IDS = ("cvc_2", "fatf_4", "ind_2", "ind_5", "typ_fan_out")
# What the requested (but NOT applied) cvc_5 change would look like. "multiple types of VA" is a prefix of the
# requested phrase because the FATF text was extracted as "multiple types of VA s".
CVC5_PROPOSED = {"expected_doc_id": "fatf_2020_va_red_flags", "must_contain": ["multiple types of VA"]}


def correct(q, chunk):
    docs = {q["expected_doc_id"], *q["acceptable_doc_ids"]}
    return chunk["doc_id"] in docs and matches_all(chunk["text"], q["must_contain"])


def first_rank(q, chunks, top_idx, fn=correct):
    for rank, i in enumerate(top_idx, start=1):
        if fn(q, chunks[i]):
            return rank
    return None


def metrics(ranks):
    r = list(ranks)
    out = {f"hit@{k}": float(np.mean([x is not None and x <= k for x in r])) for k in (1, 3, 5)}
    out["MRR@10"] = float(np.mean([1 / x if x else 0.0 for x in r]))
    return out


def decide(raw, rewritten, raw_headers):
    """Apply the two keep rules to metric dicts (headers off: raw, rewritten; headers on: raw_headers)."""
    gain_a = rewritten["hit@3"] - raw["hit@3"]
    gain_b = raw_headers["hit@3"] - raw["hit@3"]
    return {
        "rewrite_gain_hit3": gain_a, "adopt_rewrite": gain_a >= RULE_A_MIN_GAIN - 1e-9,
        "headers_gain_hit3": gain_b, "headers_delta_hit1": raw_headers["hit@1"] - raw["hit@1"],
        "adopt_headers": gain_b >= RULE_B_MIN_GAIN - 1e-9 and raw_headers["hit@1"] >= raw["hit@1"] - 1e-9,
    }


def md_table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    qs = json.loads(GOLDEN.read_text(encoding="utf-8"))
    docq = [q for q in qs if q["type"] == "in_scope"]
    typq = [q for q in qs if q["type"] == "typology"]
    assert len(docq) == 19 and len(typq) == 3, (len(docq), len(typq))
    whatif = {q["id"]: ({**q, **CVC5_PROPOSED} if q["id"] == "cvc_5" else q) for q in qs}

    rewriter = QueryRewriter()
    rewrites = {q["question"]: rewriter.rewrite(q["question"]) for q in qs}
    print(f"rewrites: {len(rewrites)} questions, {rewriter.calls} live API calls this run (model {REWRITE_MODEL})")

    embedder = SentenceTransformerEmbedder()
    retrievers = {False: load_retriever(headers=False, embedder=embedder), True: load_retriever(headers=True, embedder=embedder)}
    chunks = retrievers[False].chunks
    assert [c["text"] for c in chunks] == [c["text"] for c in retrievers[True].chunks]

    variants = [("raw", False, False), ("rewritten", True, False), ("raw + headers", False, True),
                ("rewritten + headers", True, True)]
    top = {}  # (variant, question id) -> top-10 chunk indexes (hybrid)
    for name, rewrite, headers in variants:
        for q in qs:
            query = rewrites[q["question"]] if rewrite else q["question"]
            hits = retrievers[headers].search(query, k=DEPTH, mode="hybrid")
            top[(name, q["id"])] = [h["index"] for h in hits]

    def variant_metrics(name, questions, golden=None):
        g = golden or {q["id"]: q for q in questions}
        return metrics(first_rank(g[q["id"]], chunks, top[(name, q["id"])]) for q in questions)

    table_rows, results = [], {}
    for name, _, _ in variants:
        res = {"doc": variant_metrics(name, docq), "doc_whatif": variant_metrics(name, docq, whatif),
               "typ": variant_metrics(name, typq)}
        results[name] = res
        for key, label in (("doc", "document-grounded (19), golden as is"),
                           ("doc_whatif", "document-grounded (19), cvc_5 what-if"), ("typ", "typology (3)")):
            table_rows.append({"variant": name, "question set": label, **res[key]})

    # cvc_5 before / after on the baseline (raw hybrid), and old vs new matcher
    old_ranks = [first_rank(q, chunks, top[("raw", q["id"])], fn=old_is_correct) for q in docq]
    base_old, base_new, base_whatif = metrics(old_ranks), results["raw"]["doc"], results["raw"]["doc_whatif"]
    cvc5 = next(q for q in docq if q["id"] == "cvc_5")
    cvc5_rank_now = first_rank(cvc5, chunks, top[("raw", "cvc_5")])
    cvc5_rank_if = first_rank(whatif["cvc_5"], chunks, top[("raw", "cvc_5")])

    decision = decide(results["raw"]["doc"], results["rewritten"]["doc"], results["raw + headers"]["doc"])
    decision_whatif = decide(results["raw"]["doc_whatif"], results["rewritten"]["doc_whatif"],
                             results["raw + headers"]["doc_whatif"])
    adopt_rewrite, adopt_headers = decision["adopt_rewrite"], decision["adopt_headers"]
    combo = results["rewritten + headers"]["doc"]

    # ---- print ----
    cols = ["variant", "question set", "hit@1", "hit@3", "hit@5", "MRR@10"]
    print("\nAll variants (hybrid retrieval):")
    print(md_table(table_rows, cols))
    print("\ncvc_5 and matcher effects on the baseline (raw, no headers, hybrid; document-grounded 19):")
    for label, m in (("old regex matcher, golden as is", base_old), ("new matcher, golden as is", base_new),
                     ("new matcher, cvc_5 what-if", base_whatif)):
        print(f"  {label}: " + ", ".join(f"{k} {v:.3f}" for k, v in m.items()))
    print(f"  cvc_5 first correct rank (raw hybrid): as is {cvc5_rank_now}, what-if {cvc5_rank_if}")
    print("\nKeep rules, golden as is:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in decision.items()})
    print("Keep rules, cvc_5 what-if:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in decision_whatif.items()})
    print("\nExample rewrites:")
    by_id = {q["id"]: q for q in qs}
    for i in EXAMPLE_IDS:
        print(f"  [{i}] {by_id[i]['question']}\n       -> {rewrites[by_id[i]['question']]}")

    # ---- report ----
    sec_cov = sum(1 for c in chunks if c.get("section"))
    lat = list(rewriter._data.get("latency_ms", {}).values())
    config = [
        f"- Query rewriting: **{'ON' if adopt_rewrite else 'OFF'}** (model {REWRITE_MODEL}, temperature 0; "
        f"hit@3 gain {decision['rewrite_gain_hit3']:+.3f}, rule needs >= {RULE_A_MIN_GAIN:.2f}).",
        f"- Contextual headers: **{'ON' if adopt_headers else 'OFF'}** (hit@3 gain on raw queries "
        f"{decision['headers_gain_hit3']:+.3f}, hit@1 change {decision['headers_delta_hit1']:+.3f}; rule needs "
        f">= {RULE_B_MIN_GAIN:.2f} and no hit@1 decrease).",
        "- Retrieval: hybrid (BM25 + BAAI/bge-small-en-v1.5 in FAISS, reciprocal rank fusion, k=60), no reranker "
        "(rejected in Phase 3c: hit@1 +0.045 against a 0.10 rule).",
        "- Matcher: normalised phrase match (case, whitespace, soft hyphens, hyphenation at line breaks; '-', '/' and "
        "space equivalent; whole words).",
        f"- Under the cvc_5 what-if the rules would pick: rewriting {'ON' if decision_whatif['adopt_rewrite'] else 'OFF'}, "
        f"headers {'ON' if decision_whatif['adopt_headers'] else 'OFF'}.",
    ]
    lines = [
        "# Final retrieval configuration (frozen in Phase 3d)", "",
        "No further retrieval experiments are run after this report. The primary metric is hybrid retrieval on the 19 "
        "document-grounded golden questions (14 original + 5 India); the 3 typology queries are reported separately.", "",
        "## Final configuration", "", *config, "",
        "## Keep rules (fixed before the experiments)", "",
        f"- A: adopt LLM query rewriting only if hybrid hit@3 improves by >= {RULE_A_MIN_GAIN:.2f}.",
        f"- B: adopt contextual headers only if hybrid hit@3 improves by >= {RULE_B_MIN_GAIN:.2f} on raw queries and "
        "hit@1 does not decrease.", "",
        "## All variants (hybrid retrieval)", "",
        "The 'cvc_5 what-if' rows show the golden set with the requested cvc_5 change applied in memory only "
        "(expected document FATF, phrase 'multiple types of VA'); the golden file keeps the original entry, see "
        "eval/GOLDEN_CHANGELOG.md.", "", md_table(table_rows, cols), "",
        "## Baseline before and after the cvc_5 question and the matcher change", "",
        md_table([{"setting": l, **m} for l, m in (("old regex matcher, golden as is", base_old),
                                                  ("new matcher, golden as is", base_new),
                                                  ("new matcher, cvc_5 what-if", base_whatif))],
                 ["setting", "hit@1", "hit@3", "hit@5", "MRR@10"]), "",
        f"cvc_5 first correct rank for raw hybrid: {cvc5_rank_now} as is, {cvc5_rank_if} under the what-if.", "",
        "## Example rewrites", "",
        *[f"- **{i}**: {by_id[i]['question']}\n  - rewritten: {rewrites[by_id[i]['question']]}" for i in EXAMPLE_IDS], "",
        f"Mean live rewrite latency when the cache was filled: {np.mean(lat):.0f} ms per call." if lat else "",
        "", "## Contextual headers", "",
        f"{sec_cov} of {len(chunks)} chunks have a detected section heading (numbered headings or short all-caps "
        "lines): almost all of them in the India document. The FinCEN documents and most of the FATF report have "
        "no detectable headings, so their headers contain only publisher, year and title.", "",
        "## Limitations", "",
        "- Small sample: 19 document-grounded questions, so one question moves hit@k by about 0.05 and differences "
        "between variants are noisy. Typology has only 3 queries.",
        "- Strict keyword matching: a chunk counts as correct only if it contains the required phrase, so valid "
        "alternative chunks are missed (error analysis found such a case for cvc_2), and the numbers understate real "
        "quality. PDF text extraction also splits words (for example 'VA s'), which the matcher does not repair.",
        "- OCR noise: the India document comes from OCR with stamp fragments and small character errors; it is marked "
        "quotable: false, and quotations from it must be checked against the page image.",
        "- The golden questions were written by the same person who built the system and were read against the "
        "documents; the rewriter saw only the questions.",
        "- Rewriting calls an external API on every search when enabled (latency and cost, and a dependency on the "
        "service); cached rewrites make the evaluation reproducible but not live use.", ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nFINAL: rewrite={'ON' if adopt_rewrite else 'OFF'}, headers={'ON' if adopt_headers else 'OFF'}; "
          f"combined (both on) hit@3 {combo['hit@3']:.3f}")


if __name__ == "__main__":
    main()

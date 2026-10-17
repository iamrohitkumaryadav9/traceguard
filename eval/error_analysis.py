"""Diagnose golden questions that were never retrieved in the top 10 (no edits to the questions).

Run: python -m eval.error_analysis cvc_2 fatf_4
For each question: the gold chunks (document + must_contain regex), the rank of the best gold chunk in each
mode's FULL ranking, the top hybrid results, where the matching sentence sits inside its chunk (chunk
boundaries), and which top-ranked chunks would have been acceptable answers under a looser rule.
"""
import json
import re
import sys
from pathlib import Path

from eval.retrieval_eval import GOLDEN, is_correct
from src.rag.ingest import split_sentences
from src.rag.retriever import get_retriever

OUT = Path("eval/results/error_analysis.md")


def best_rank(r, q, mode, loose=None):
    """Rank of the best chunk that passes the question's rule (or, if `loose` is given, the looser regex in the
    expected document). The retriever's candidate cap is lifted so ranks are over the whole corpus (the reranked
    mode only orders its top 20, so deeper chunks show as None there)."""
    hits = r.search(q["question"], k=len(r.chunks), mode=mode)
    for rank, h in enumerate(hits, start=1):
        c = r.chunks[h["index"]]
        ok = (c["doc_id"] == q["expected_doc_id"] and re.search(loose, c["text"], re.I | re.S)) if loose \
            else is_correct(q, c)
        if ok:
            return rank, hits
    return None, hits


def analyse(r, q, loose):
    lines = [f"## {q['id']}: {q['question']}", "",
             f"expected doc `{q['expected_doc_id']}`, must_contain {q['must_contain']}", ""]
    gold = [c for c in r.chunks if is_correct(q, c)]
    lines.append(f"Gold chunks (satisfy the rule): {[c['evidence_id'] for c in gold]}")
    for mode in ("bm25", "dense", "hybrid", "hybrid+rerank"):
        rank, _ = best_rank(r, q, mode)
        lines.append(f"- best gold rank in full {mode} ranking: {rank}")
    if loose:
        lines.append(f"Looser rule (expected doc, regex `{loose}`) best rank per mode:")
        for mode in ("bm25", "dense", "hybrid", "hybrid+rerank"):
            rank, _ = best_rank(r, q, mode, loose)
            lines.append(f"- {mode}: {rank}")
        alt = [c["evidence_id"] for c in r.chunks if c["doc_id"] == q["expected_doc_id"]
               and re.search(loose, c["text"], re.I | re.S) and not is_correct(q, c)]
        lines.append(f"Other chunks in the expected document that match the looser rule: {alt}")
    lines.append("")
    for c in gold:
        i = r.by_evidence_id[c["evidence_id"]]
        sents = split_sentences(c["text"])
        pos = [k for k, s in enumerate(sents) if all(re.search(p, s, re.I | re.S) for p in q["must_contain"])]
        lines.append(f"Gold chunk {c['evidence_id']} ({c['approx_tokens']} tokens, {len(sents)} sentences); "
                     f"matching sentence index {pos} (0 = start, {len(sents) - 1} = end of chunk):")
        for k in pos[:1]:
            lines.append(f"  > {' '.join(sents[max(0, k - 1):k + 2])[:700]}")
        prev_c = r.chunks[i - 1] if i > 0 and r.chunks[i - 1]["doc_id"] == c["doc_id"] else None
        next_c = r.chunks[i + 1] if i + 1 < len(r.chunks) and r.chunks[i + 1]["doc_id"] == c["doc_id"] else None
        if prev_c:
            lines.append(f"  previous chunk {prev_c['evidence_id']} ends: ...{prev_c['text'][-220:]}")
        if next_c:
            lines.append(f"  next chunk {next_c['evidence_id']} starts: {next_c['text'][:220]}...")
        lines.append("")
    lines.append("Top 3 hybrid results actually returned:")
    for rank, h in enumerate(r.search(q["question"], k=3, mode="hybrid"), start=1):
        c = r.chunks[h["index"]]
        ok = bool(loose and re.search(loose, c["text"], re.I | re.S) and c["doc_id"] == q["expected_doc_id"])
        lines.append(f"{rank}. {c['evidence_id']} (cos {h['dense_score']:.3f}, bm25 {h['bm25_score']:.2f}) "
                     f"{'[matches the looser rule]' if ok else ''} {c['text'][:300]}...")
    lines.append("")
    return lines


def main(ids):
    qs = {q["id"]: q for q in json.loads(GOLDEN.read_text())}
    loose_rules = {  # looser phrasings, used ONLY to judge whether returned chunks would have been acceptable
        "cvc_2": r"acting as an unregistered MSB|funnel account",
        "fatf_4": r"private wallet|unhosted",
    }
    r = get_retriever()
    import src.rag.retriever as rt

    rt.CANDIDATES = 10_000  # full-corpus ranks for the diagnosis
    out = ["# Error analysis (golden questions never retrieved in the top 10)", ""]
    for i in ids:
        out += analyse(r, qs[i], loose_rules.get(i))
    out += ["## Diagnosis (written after reading the output above; the questions were not edited)", "",
            "- **cvc_2: paraphrase distance and a too-strict must_contain, not chunking.** The intended sentence (banks "
            "may identify P2P exchangers through funnel-account activity) sits in the middle of a chunk that also "
            "starts with a footnote fragment; its text is not split across chunks. The question describes an "
            "'unlicensed coin-for-cash swapping business' while the text says 'P2P exchangers'. The intended chunk "
            "ranks 118 (BM25), 54 (dense), 87 (hybrid). A different chunk of the same document, the red flag about a "
            "customer 'acting as an unregistered MSB', is a reasonable answer to the question, and it ranks 3 in "
            "dense (20 hybrid, 9 reranked). The must_contain rule counts only the funnel-account sentence.",
            "- **fatf_4: paraphrase distance plus a diluted chunk, with the passage straddling a chunk boundary.** "
            "The question says 'crypto', 'parks' and 'offline storage'; the text says 'VAs', 'VASP' and 'private "
            "wallet' (BM25 has almost no overlapping words, and extraction splits words such as 'ad ditional' and "
            "'VA s'). The relevant bullet starts at the end of the previous chunk and is repeated whole at the start "
            "of the gold chunk because of the overlap, so the gold chunk does contain it, but the bullet is one of "
            "six sentences, so the chunk embedding is diluted. The regex itself is not too strict: only that chunk "
            "has the 'private wallet' passage. The gold chunk ranks 25 (BM25), 24 (dense), 19 (hybrid) and 10 "
            "(reranked).", ""]
    text = "\n".join(out)
    OUT.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:] or ["cvc_2", "fatf_4"])

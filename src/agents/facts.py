"""Phase 5c (design v3): facts rendered by code from the evidence ledger. No LLM, no network.

Every function here reads tool outputs from the ledger and turns them into fixed-template claims whose numbers are
copied from the outputs. The writer cannot edit them (it is not even shown a field for them).
"""
import math
from typing import Callable, Dict, List, Optional

from src.agents.build import deterministic_claims, entry_of
from src.agents.memo import MemoClaim
from src.agents.state import AgentState, LedgerEntry, RegReference
from src.rag.ingest import split_sentences
from src.rag.retriever import tokenize

MAX_EXCERPT_WORDS = 40
MAX_LISTED_NEIGHBOURS = 5
DIRECTION_WORDS = {"in": "incoming", "out": "outgoing", "both": "incoming and outgoing"}
DIRECTION_GLOSS = {
    "raises_risk": "above the baseline",
    "lowers_risk": "below the baseline, so it is not treated as suspicious",
    "uninformative": "no clear difference from the baseline",
}


def _focal_typology(state: AgentState) -> Optional[LedgerEntry]:
    return state.ledger.get(f"typology:{state.tx_id}")


def _subject(state, tx_id):
    return "this transaction" if tx_id == state.tx_id else f"node {tx_id}"


# ---------- a) neighbour summary ----------
def neighbour_claims(state: AgentState) -> List[MemoClaim]:
    """One claim per get_neighbors output and one per subgraph_summary output, each stating its own parameters."""
    out = []
    for e in state.ledger.values():
        o = e.output
        if "error" in o:
            continue
        if e.tool == "get_neighbors":
            who, d, hops = _subject(state, o["tx_id"]), o["direction"], o["hops"]
            head = (f"get_neighbors for {who}: direction {d} (directed chains only, never mixed in-and-out paths), "
                    f"up to {hops} hop{'s' if hops != 1 else ''}; ")
            kind = {"in": "incoming neighbour", "out": "outgoing neighbour", "both": "neighbour"}[d]  # 'both': each one states its own direction
            if o["total_found"] == 0:
                text = head + f"no {kind}s found."
            else:
                listed = o["neighbors"][:MAX_LISTED_NEIGHBOURS]
                items = "; ".join(
                    f"{n['tx_id']} ({DIRECTION_WORDS[n['direction']]} neighbour at hop {n['hop']}, score {n['score']}, "
                    f"{'alerted' if n['alert'] else 'not alerted'})" for n in listed)
                shortened = " Only the highest-scoring neighbours are listed." if o["total_found"] > len(listed) else ""
                text = head + f"{o['total_found']} {kind}{'s' if o['total_found'] != 1 else ''} found: {items}.{shortened}"
            out.append(MemoClaim(text=text, evidence_ids=[o["evidence_id"]], source="code"))
        elif e.tool == "subgraph_summary":
            who = _subject(state, o["tx_id"])
            text = (f"subgraph_summary for {who}: undirected subgraph of radius {o['radius']} (every node within {o['radius']} "
                    f"steps over edges in either direction, including mixed paths, and the focal node itself): {o['n_nodes']} nodes, "
                    f"{o['n_edges']} directed edges, {o['n_alerted']} alerted nodes (share {o['share_alerted']}); focal in-degree "
                    f"{o['focal_in_degree']}, focal out-degree {o['focal_out_degree']}; chain length {o['focal_chain_len']}; "
                    f"weakly connected component size {o['wcc_size']}.")
            out.append(MemoClaim(text=text, evidence_ids=[o["evidence_id"]], source="code"))
    return out


# ---------- b) typology evidence ----------
def typology_claims(state: AgentState) -> List[MemoClaim]:
    """One claim per matched pattern, with the direction in words. The retrieval query is never shown."""
    e = _focal_typology(state)
    if e is None or "error" in e.output:
        return []
    o, out = e.output, []
    for m in o.get("matches", []):
        out.append(MemoClaim(
            text=(f"Structural pattern {m['pattern']} matched ({m['evidence']}); evidence direction {m['evidence_direction']}; "
                  f"among {m['n_flagged']} labeled nodes with this pattern in the training period the illicit rate was "
                  f"{m['observed_illicit_rate']} against a baseline of {m['baseline_illicit_rate']} (lift {m['lift']}), "
                  f"{DIRECTION_GLOSS[m['evidence_direction']]}."),
            evidence_ids=[o["evidence_id"]], source="code"))
    return out


def code_built_claims(state: AgentState) -> List[MemoClaim]:
    """All code-built activity claims, in a fixed order: score and SHAP features, health, typology, neighbours, subgraph."""
    base = deterministic_claims(state)  # score claim, merged SHAP-feature claim, health claim, typology claims (old template)
    score_and_features = [c for c in base if c.text.startswith("At time step") or "anonymized" in c.text]
    health = [c for c in base if c.text.startswith("Model health check")]
    claims = score_and_features + health + typology_claims(state) + neighbour_claims(state)
    for c in claims:
        c.source = "code"
    return claims


# ---------- c) regulatory background rendered by code ----------
def bm25_scores(query_tokens: List[str], docs: List[List[str]], k1=1.5, b=0.75) -> List[float]:
    """Okapi BM25 with the always-positive IDF log(1 + (N - n + 0.5) / (n + 0.5))."""
    n_docs = len(docs)
    avg = (sum(len(d) for d in docs) / n_docs) or 1.0
    df: Dict[str, int] = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    scores = []
    for d in docs:
        s = 0.0
        for t in set(query_tokens):
            f = d.count(t)
            if f:
                idf = math.log(1 + (n_docs - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(d) / avg))
        scores.append(s)
    return scores


def best_sentence(query: str, chunks: List[tuple]):
    """chunks: [(chunk_id, text)] in ledger order. Returns (chunk_id, sentence, score) for the sentence with the best BM25
    score against the query (ties: first chunk, first sentence), or None when no sentence overlaps the query."""
    sents = [(cid, s) for cid, text in chunks for s in split_sentences(text)]
    if not sents:
        return None
    scores = bm25_scores(tokenize(query), [tokenize(s) for _, s in sents])
    best = max(range(len(sents)), key=lambda k: (scores[k], -k))
    return (sents[best][0], sents[best][1], scores[best]) if scores[best] > 0 else None


def excerpt(sentence: str, max_words=MAX_EXCERPT_WORDS):
    """(verbatim excerpt of at most max_words words cut at a word boundary, whether it was cut)."""
    words = sentence.split()
    return (" ".join(words[:max_words]), len(words) > max_words)


def _pages(o):
    a, b = o.get("page_start"), o.get("page_end")
    return f"page {a}" if a == b else f"pages {a}-{b}"


def regulation_chunks(state: AgentState, fetch: Optional[Callable[[str], dict]] = None) -> List[tuple]:
    """[(chunk_id, full text, output dict)] for every regulation chunk in the ledger, in ledger order. A truncated copy is
    replaced by the exact stored text fetched with get_regulation_chunk."""
    out, seen = [], set()
    for e in state.ledger.values():
        o = e.output
        if "error" in o or not str(o.get("evidence_id", "")).startswith("reg:") or "text" not in o or o["evidence_id"] in seen:
            continue
        seen.add(o["evidence_id"])
        text = o["text"]
        if o.get("text_truncated") and fetch is not None:
            full = fetch(o["evidence_id"])
            if "error" not in full:
                text = full["text"]
        out.append((o["evidence_id"], text, o))
    return out


def render_background(state: AgentState, fetch: Optional[Callable[[str], dict]] = None) -> List[RegReference]:
    """Code-rendered regulatory background: one verbatim excerpt (quotable documents) or a label only (non-quotable) per
    matched pattern with a retrieval query. No LLM text. paraphrase holds the label, quote the verbatim excerpt."""
    e = _focal_typology(state)
    if e is None or "error" in e.output:
        return []
    chunks = regulation_chunks(state, fetch)
    if not chunks:
        return []
    info = {cid: o for cid, _, o in chunks}
    out, done = [], set()
    for m in e.output.get("matches", []):
        q = m.get("retrieval_query")
        if not q:
            continue
        hit = best_sentence(q, [(cid, text) for cid, text, _ in chunks])
        if hit is None:
            continue
        cid, sentence, _ = hit
        o = info[cid]
        quotable = bool(o.get("quotable", True))
        key = (cid, sentence if quotable else "")
        if key in done:
            continue
        done.add(key)
        if quotable:
            ex, cut = excerpt(sentence)
            label = f"Retrieved guidance: {o.get('title')}, {_pages(o)}, {cid} (quotable{'; excerpt shortened' if cut else ''})"
            out.append(RegReference(paraphrase=label, evidence_id=cid, quote=ex))
        else:
            out.append(RegReference(paraphrase=f"Retrieved guidance: {o.get('title')}, {_pages(o)}, {cid}", evidence_id=cid, quote=None))
    return out


# ---------- verification used by the critic ----------
def same_claims(a: List[MemoClaim], b: List[MemoClaim]) -> bool:
    return [(c.text, list(c.evidence_ids)) for c in a] == [(c.text, list(c.evidence_ids)) for c in b]


def facts_problems(draft_facts, expected_facts, draft_bg, expected_bg) -> List[str]:
    """Reasons the draft's code-built claims or rendered background differ from the ones recomputed from the ledger."""
    reasons = []
    exp = {(c.text, tuple(c.evidence_ids)) for c in expected_facts}
    got = {(c.text, tuple(c.evidence_ids)) for c in draft_facts}
    for t, ids in sorted(exp - got):
        reasons.append(f"a code-built claim is missing or was edited: {t[:90]!r} (cites {list(ids)})")
    for t, ids in sorted(got - exp):
        reasons.append(f"a claim that code did not build is among the code-built claims: {t[:90]!r}")
    if [(r.paraphrase, r.evidence_id, r.quote) for r in draft_bg] != [(r.paraphrase, r.evidence_id, r.quote) for r in expected_bg]:
        reasons.append("the rendered regulatory background differs from the one recomputed by code")
    return reasons

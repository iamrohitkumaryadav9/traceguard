"""Hybrid retrieval: BM25 + dense (FAISS) fused with reciprocal rank fusion.

Embeddings: sentence-transformers with BAAI/bge-small-en-v1.5 (installs and runs on this Python 3.14
environment, so the Gemini fallback was not needed). Chunk embeddings are cached on disk in
data/processed/rag/ and computed once per chunk set.
"""
import hashlib
import re
from pathlib import Path

import numpy as np

from src.rag.ingest import CHUNKS_PATH, build_chunks, load_chunks, save_chunks

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # recommended for bge v1.5 queries
CACHE_DIR = Path("data/processed/rag")
RRF_K = 60
CANDIDATES = 50
STOPWORDS = set(
    "a an and are as at be been by for from has have how if in into is it its of on or that the their there "
    "these this to was were what when which who whom why will with would should can could do does not than "
    "then them they we you your i about any all also".split()
)


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class SentenceTransformerEmbedder:
    name = EMBED_MODEL

    def __init__(self, model_name=EMBED_MODEL):
        self.model_name = model_name
        self._model = None

    def encode(self, texts, is_query=False):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        if is_query:
            texts = [QUERY_PREFIX + t for t in texts]
        return np.asarray(self._model.encode(texts, normalize_embeddings=True, batch_size=32,
                                             show_progress_bar=False), dtype="float32")


RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP = 20
# Decision rule fixed before the ablation: the reranker becomes the default only if it raises hit@1 by at
# least 0.10 over plain hybrid on the extended golden set. Set from the result in eval/results/retrieval.md.
RERANK_DEFAULT = False


class CrossEncoderReranker:
    name = RERANK_MODEL

    def __init__(self, model_name=RERANK_MODEL):
        self.model_name = model_name
        self._model = None

    def score(self, query, texts):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device="cpu")
        return [float(s) for s in self._model.predict([(query, t) for t in texts], show_progress_bar=False)]


def rrf(rankings, k=RRF_K):
    """Reciprocal rank fusion. rankings: lists of item ids, best first. Returns [(id, score)] best first.
    Ties break on the smaller id so the result is deterministic."""
    scores = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class Retriever:
    def __init__(self, chunks, embedder, embeddings=None, reranker=None):
        import faiss
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.embedder = embedder
        self.reranker = reranker if reranker is not None else CrossEncoderReranker()
        self.by_evidence_id = {c["evidence_id"]: i for i, c in enumerate(chunks)}
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        emb = embeddings if embeddings is not None else embedder.encode([c["text"] for c in chunks])
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(np.ascontiguousarray(emb, dtype="float32"))

    def _allowed(self, jurisdiction):
        if jurisdiction is None:
            return np.ones(len(self.chunks), bool)
        return np.array([c["jurisdiction"] == jurisdiction for c in self.chunks])

    def search(self, query, k=5, jurisdiction=None, mode="hybrid"):
        """Top-k chunk hits (dicts with chunk index and scores). mode: bm25 | dense | hybrid."""
        allowed = self._allowed(jurisdiction)
        if not allowed.any():
            return []
        n = len(self.chunks)
        zeros = np.zeros(n, dtype="float32")
        bm, dense = zeros, zeros  # a mode only computes the scorers it needs (keeps latency comparisons fair)
        rerank = mode == "hybrid+rerank"
        if mode in ("bm25", "hybrid", "hybrid+rerank"):
            bm = np.asarray(self.bm25.get_scores(tokenize(query)))
        if mode in ("dense", "hybrid", "hybrid+rerank"):
            sims, ids = self.index.search(self.embedder.encode([query], is_query=True), n)
            dense = np.zeros(n, dtype="float32")
            dense[ids[0]] = sims[0]

        def ranking(scores):
            order = sorted((i for i in range(n) if allowed[i]), key=lambda i: (-scores[i], i))
            return order[:CANDIDATES]

        if mode == "bm25":
            ordered = [(i, float(bm[i])) for i in ranking(bm)]
        elif mode == "dense":
            ordered = [(i, float(dense[i])) for i in ranking(dense)]
        else:
            ordered = rrf([ranking(bm), ranking(dense)])
            if rerank:  # re-order the top RERANK_TOP hybrid results with a cross-encoder
                head = ordered[:RERANK_TOP]
                ce = self.reranker.score(query, [self.chunks[i]["text"] for i, _ in head])
                ordered = sorted(((i, s) for (i, _), s in zip(head, ce)), key=lambda x: (-x[1], x[0]))
        return [{"index": i, "score": float(s), "bm25_score": float(bm[i]), "dense_score": float(dense[i])}
                for i, s in ordered[:k]]

    def get(self, evidence_id):
        i = self.by_evidence_id.get(evidence_id)
        return None if i is None else self.chunks[i]


def _cached_embeddings(chunks, embedder):
    digest = hashlib.sha1("\n".join(c["text"] for c in chunks).encode("utf-8")).hexdigest()[:12]
    path = CACHE_DIR / f"emb_{embedder.name.replace('/', '_')}_{digest}.npy"
    if path.exists():
        return np.load(path)
    emb = embedder.encode([c["text"] for c in chunks])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb


def load_retriever():
    if not Path(CHUNKS_PATH).exists():
        save_chunks(build_chunks())
    chunks = load_chunks()
    embedder = SentenceTransformerEmbedder()
    return Retriever(chunks, embedder, _cached_embeddings(chunks, embedder))


_RETRIEVER = None


def get_retriever():
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = load_retriever()
    return _RETRIEVER


def set_retriever(r):
    """Install a retriever (tests pass a tiny fixture one). None resets."""
    global _RETRIEVER
    _RETRIEVER = r

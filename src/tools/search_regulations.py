from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field

from src.rag.retriever import QUERY_REWRITE_DEFAULT, RERANK_DEFAULT, get_retriever
from src.rag.rewrite import get_rewriter
from src.tools.schemas import ToolError, r6

NAME = "search_regulations"
DESCRIPTION = (
    "Search the indexed regulation and guidance documents (hybrid BM25 + dense retrieval, reciprocal rank "
    "fusion). Returns the best-matching chunks with an evidence_id, document metadata, page range, text "
    "(shortened to about 1,200 characters; use get_regulation_chunk for the exact stored text) and a fused "
    "score. Scores rank results within one query; they are not a calibrated relevance probability. "
    "Optional jurisdiction filter: FATF, US, India. Optional rerank flag (cross-encoder over the top 20) and rewrite flag (LLM query rewriting). "
    "Each hit has a quotable flag: if false, the text is OCR output and must not be quoted directly."
)
MAX_TEXT_CHARS = 1200


class SearchRegulationsInput(BaseModel):
    query: str = Field(min_length=3)
    k: int = Field(default=5, ge=1, le=20)
    jurisdiction: Optional[str] = None
    rerank: Optional[bool] = None  # None = the module default (RERANK_DEFAULT); True uses the cross-encoder
    rewrite: Optional[bool] = None  # None = QUERY_REWRITE_DEFAULT; True rewrites the query with Gemini first


class RegulationHit(BaseModel):
    evidence_id: str
    doc_id: str
    title: str
    publisher: str
    year: int
    jurisdiction: str
    page_start: int
    page_end: int
    text: str
    text_truncated: bool
    quotable: bool  # False for OCR text: do not quote it directly, check the page image
    score: float
    dense_score: float
    bm25_score: float


class SearchRegulationsOutput(BaseModel):
    query: str
    search_query: str  # the query actually searched (differs from query when rewriting is on)
    k: int
    jurisdiction: Optional[str]
    hits: List[RegulationHit]
    evidence_id: str


def shorten(text, limit=MAX_TEXT_CHARS):
    if len(text) <= limit:
        return text, False
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + " ...", True


def search_regulations(args: SearchRegulationsInput) -> Union[SearchRegulationsOutput, ToolError]:
    r = get_retriever()
    known = sorted({c["jurisdiction"] for c in r.chunks})
    if args.jurisdiction is not None and args.jurisdiction not in known:
        return ToolError(error="invalid_argument", tool=NAME, evidence_id=f"error:{NAME}:{args.jurisdiction}",
                         message=f"unknown jurisdiction {args.jurisdiction!r}; indexed jurisdictions: {known}")
    hits = []
    use_rewrite = QUERY_REWRITE_DEFAULT if args.rewrite is None else args.rewrite
    search_query = get_rewriter().rewrite(args.query) if use_rewrite else args.query
    use_rerank = RERANK_DEFAULT if args.rerank is None else args.rerank
    mode = "hybrid+rerank" if use_rerank else "hybrid"
    for h in r.search(search_query, k=args.k, jurisdiction=args.jurisdiction, mode=mode):
        c = r.chunks[h["index"]]
        text, truncated = shorten(c["text"])
        hits.append(RegulationHit(
            evidence_id=c["evidence_id"], doc_id=c["doc_id"], title=c["title"], publisher=c["publisher"],
            year=c["year"], jurisdiction=c["jurisdiction"], page_start=c["page_start"], page_end=c["page_end"],
            text=text, text_truncated=truncated, quotable=bool(c.get("quotable", True)), score=r6(h["score"]), dense_score=r6(h["dense_score"]),
            bm25_score=r6(h["bm25_score"])))
    return SearchRegulationsOutput(query=args.query, search_query=search_query, k=args.k, jurisdiction=args.jurisdiction, hits=hits,
                                   evidence_id=f"search:{abs(hash_query(args.query, args.jurisdiction, args.k))}")


def hash_query(query, jurisdiction, k):
    """Stable (process-independent) id for a search call."""
    import zlib

    return zlib.crc32(f"{query}|{jurisdiction}|{k}".encode("utf-8"))

import re
import zlib

import numpy as np
import pytest

from src.rag.ingest import approx_tokens, chunk_paragraphs, clean_pages, paragraphs_from_lines, split_sentences
from src.rag.retriever import Retriever, rrf, tokenize
from src.tools import TOOLS, run_tool
from src.tools.get_regulation_chunk import GetRegulationChunkInput, get_regulation_chunk
from src.tools.schemas import ToolError
from src.tools.search_regulations import MAX_TEXT_CHARS, SearchRegulationsInput, search_regulations

EVIDENCE_RE = re.compile(r"^reg:[a-z0-9_]+:p\d+:c\d+$")


def sentence(i):
    return f"Sentence number {i} talks about suspicious transfers between accounts."  # 9 words ~ 12 tokens


def pages(n_pages=3, per_page=6):
    k = 0
    out = []
    for p in range(1, n_pages + 1):
        paras = []
        for _ in range(2):  # two paragraphs per page
            paras.append(" ".join(sentence(k + j) for j in range(per_page // 2)))
            k += per_page // 2
        out.append((p, paras))
    return out


# ---------- chunking ----------
def test_chunks_keep_page_numbers_and_ids():
    chunks = chunk_paragraphs("doc_a", pages(), target=40, max_tokens=60, overlap_frac=0.25)
    assert len(chunks) > 2
    for c in chunks:
        assert EVIDENCE_RE.match(c["evidence_id"])
        assert c["evidence_id"] == f"reg:doc_a:p{c['page_start']}:c{c['n']}"
        assert 1 <= c["page_start"] <= c["page_end"] <= 3
        assert c["approx_tokens"] <= 60 + 1
    assert [c["n"] for c in chunks] == list(range(len(chunks)))
    assert chunks[0]["page_start"] == 1 and chunks[-1]["page_end"] == 3
    # every sentence of the document appears in some chunk
    joined = " ".join(c["text"] for c in chunks)
    assert all(sentence(i) in joined for i in range(18))


def test_chunk_overlap_repeats_trailing_sentences():
    chunks = chunk_paragraphs("doc_a", pages(), target=40, max_tokens=60, overlap_frac=0.4)
    for prev, nxt in zip(chunks, chunks[1:]):
        ps, ns = split_sentences(prev["text"]), split_sentences(nxt["text"])
        m = max((m for m in range(1, min(len(ps), len(ns)) + 1) if ps[-m:] == ns[:m]), default=0)
        assert m >= 1  # the next chunk starts with whole trailing sentences of the previous one
        assert m < len(ns)  # ... and still contains new material


def test_no_overlap_when_zero_fraction_still_progresses():
    chunks = chunk_paragraphs("doc_a", pages(), target=40, max_tokens=60, overlap_frac=0.0)
    texts = " ".join(c["text"] for c in chunks)
    assert texts.count(sentence(0)) == 1  # nothing repeated


def test_chunks_prefer_paragraph_boundaries():
    chunks = chunk_paragraphs("doc_a", [(1, [sentence(0) + " " + sentence(1), sentence(2) + " " + sentence(3)])],
                              target=20, max_tokens=60, overlap_frac=0.0)
    assert chunks[0]["text"].endswith(sentence(1))  # cut at the paragraph end once the target is reached


def test_chunks_never_cross_documents():
    a = chunk_paragraphs("doc_a", [(1, [sentence(0)])])
    b = chunk_paragraphs("doc_b", [(1, [sentence(1)])])
    assert a[0]["evidence_id"] == "reg:doc_a:p1:c0" and b[0]["evidence_id"] == "reg:doc_b:p1:c0"
    assert sentence(1) not in a[0]["text"] and sentence(0) not in b[0]["text"]


def test_approx_tokens_and_sentences():
    assert approx_tokens("one two three four five") == 7
    assert split_sentences("See U.S. Treasury rules. Another sentence here.") == [
        "See U.S. Treasury rules.", "Another sentence here."]


def test_clean_pages_removes_headers_footers_and_page_numbers():
    def body(p):  # distinct letters per page and line, so no body line repeats across pages
        return "\n".join(f"Line about topic {chr(97 + i)}{chr(97 + p)} here." for i in range(10))

    raw = [f"MY REPORT 2020 | {p}\n{body(p)}\n{p}" for p in range(1, 7)]
    cleaned = clean_pages(raw)
    for p, lines in enumerate(cleaned, start=1):
        joined = " ".join(lines)
        assert "MY REPORT" not in joined
        assert f"topic a{chr(97 + p)} here." in joined and f"topic j{chr(97 + p)} here." in joined
        assert not any(l.strip().isdigit() for l in lines)


def test_paragraph_split_on_bullets_and_headings():
    lines = ["Introduction", "This is a normal sentence that runs on for quite a while across the line.",
             "• first bullet item with enough words to be a line of text on its own here.",
             "• second bullet item with enough words to be a line of text on its own here."]
    paras = paragraphs_from_lines(lines)
    assert paras[0] == "Introduction" and len(paras) == 4


# ---------- fusion ----------
def test_rrf_fusion_scores_and_order():
    fused = dict(rrf([["a", "b", "c"], ["a", "c", "b"]], k=60))
    assert fused["a"] == pytest.approx(2 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 63)
    assert [i for i, _ in rrf([["a", "b", "c"], ["a", "c", "b"]])] == ["a", "b", "c"]  # b/c tie -> id order
    top = rrf([["x", "y"], ["y", "z"]])
    assert top[0][0] == "y" and top[0][1] == pytest.approx(1 / 62 + 1 / 61)
    assert rrf([]) == []


# ---------- tiny retriever and tools ----------
class FakeEmbedder:
    """Deterministic hashed bag-of-words embeddings (no model download)."""

    name = "fake"

    def encode(self, texts, is_query=False):
        out = np.zeros((len(texts), 64), dtype="float32")
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                out[i, zlib.crc32(tok.encode()) % 64] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


def mk_chunk(doc, jur, page, n, text):
    return {"evidence_id": f"reg:{doc}:p{page}:c{n}", "doc_id": doc, "n": n, "page_start": page, "page_end": page,
            "text": text, "approx_tokens": approx_tokens(text), "title": f"Title {doc}", "publisher": "Pub",
            "year": 2020, "jurisdiction": jur}


CORPUS = [
    mk_chunk("doc_us", "US", 1, 0, "Money transmitters must register as money services businesses and file reports."),
    mk_chunk("doc_us", "US", 2, 1, "Structuring cash deposits just under ten thousand dollars is a red flag."),
    mk_chunk("doc_fatf", "FATF", 1, 0, "Mixing and tumbling services obscure the flow of virtual assets between wallets."),
    mk_chunk("doc_fatf", "FATF", 3, 1, "Ransomware payments often show large staggered transfers. " + "long text " * 300),
]


@pytest.fixture
def retriever():
    from src.rag.retriever import set_retriever

    r = Retriever(CORPUS, FakeEmbedder())
    set_retriever(r)
    yield r
    set_retriever(None)


def test_retriever_finds_expected_chunk_and_is_deterministic(retriever):
    a = retriever.search("tumbling services mixing wallets", k=2)
    b = retriever.search("tumbling services mixing wallets", k=2)
    assert a == b
    assert CORPUS[a[0]["index"]]["evidence_id"] == "reg:doc_fatf:p1:c0"


def test_retriever_jurisdiction_filter(retriever):
    hits = retriever.search("transfers", k=4, jurisdiction="US")
    assert {CORPUS[h["index"]]["jurisdiction"] for h in hits} == {"US"}
    assert retriever.search("transfers", jurisdiction="India") == []


def test_search_tool_schema(retriever):
    out = search_regulations(SearchRegulationsInput(query="structuring cash deposits", k=3))
    assert out.hits[0].evidence_id == "reg:doc_us:p2:c1"
    h = out.hits[0]
    for field in ("doc_id", "title", "publisher", "year", "jurisdiction", "page_start", "page_end", "text", "score"):
        assert getattr(h, field) is not None
    assert len(out.hits) == 3 and out.evidence_id.startswith("search:")
    assert all(EVIDENCE_RE.match(x.evidence_id) for x in out.hits)


def test_search_tool_truncates_long_text_but_chunk_tool_returns_exact(retriever):
    out = search_regulations(SearchRegulationsInput(query="ransomware staggered transfers", k=4))
    hit = next(h for h in out.hits if h.evidence_id == "reg:doc_fatf:p3:c1")
    assert hit.text_truncated and len(hit.text) <= MAX_TEXT_CHARS + 5
    full = get_regulation_chunk(GetRegulationChunkInput(evidence_id=hit.evidence_id))
    assert full.text == CORPUS[3]["text"]  # exact stored text


def test_search_tool_invalid_jurisdiction_is_structured_error(retriever):
    out = run_tool("search_regulations", {"query": "anything here", "jurisdiction": "Mars"})
    assert out["error"] == "invalid_argument" and out["tool"] == "search_regulations"


def test_unknown_evidence_id_returns_structured_error(retriever):
    out = get_regulation_chunk(GetRegulationChunkInput(evidence_id="reg:nope:p1:c0"))
    assert isinstance(out, ToolError) and out.error == "unknown_evidence_id"
    assert out.evidence_id == "error:get_regulation_chunk:reg:nope:p1:c0"


def test_registry_has_regulation_tools_and_is_deterministic(retriever):
    assert {"search_regulations", "get_regulation_chunk"} <= set(TOOLS)
    args = {"query": "structuring cash deposits", "k": 2}
    assert run_tool("search_regulations", args) == run_tool("search_regulations", args)

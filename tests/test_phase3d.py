import json
from pathlib import Path

import numpy as np
import pytest

from eval.retrieval_final import decide, first_rank, metrics
from src.rag.ingest import chunk_paragraphs, detect_heading, paragraphs_and_first_lines, paragraphs_from_lines
from src.rag.retriever import Retriever, header_for, index_text
from src.rag.rewrite import PROMPT, QueryRewriter
from src.tools.get_regulation_chunk import GetRegulationChunkInput, get_regulation_chunk
from src.tools.search_regulations import SearchRegulationsInput, search_regulations
from tests.test_rag import FakeEmbedder, mk_chunk, sentence


# ---------- headings ----------
@pytest.mark.parametrize("line,expected", [
    ("5.5 Suspicious Transaction Report (STR)", "5.5 Suspicious Transaction Report (STR)"),
    ("7.5 Approach on Tumbler/Mixer and other anonymity enhancing products/services",
     "7.5 Approach on Tumbler/Mixer and other anonymity enhancing products/services"),
    ("3. GOVERNANCE FRAMEWORK", "3. GOVERNANCE FRAMEWORK"),
    ("3.4.1 In accordance with the Rule 9(14)(ii) and (iii), every RE needs to formulate", None),  # paragraph number
    ("10. The following sections contain a collection of red flag indicators of suspicious", None),
    ("1. Many business models of entities dealing with CVC operate as money transmitters.", None),
    ("FATF REPORT", "FATF REPORT"),
    ("UNIT 17", None),
])
def test_detect_heading(line, expected):
    assert detect_heading(line, line) == expected


def test_all_caps_stamp_noise_is_not_a_heading():
    assert detect_heading("FINAA UNIT", "FINAA UNIT") is None
    assert detect_heading("GENCE UNIT 17", "GENCE UNIT 17") is None


def test_heading_info_does_not_change_chunks():
    lines = ["5.5 Suspicious Transaction Report (STR)",
             "The RE shall put in place robust systems and processes to generate alerts based on red flags. " * 3,
             "5.6 Prohibition on Tipping-off",
             "Section 12(2) requires that the fact of furnishing information is kept confidential. " * 3]
    paras, firsts = paragraphs_and_first_lines(lines)
    assert paras == paragraphs_from_lines(lines)
    heads = [detect_heading(f, p) for f, p in zip(firsts, paras)]
    plain = chunk_paragraphs("d", [(1, paras)], target=30, max_tokens=60)
    with_heads = chunk_paragraphs("d", [(1, paras)], target=30, max_tokens=60, page_headings=[heads])
    assert [c["text"] for c in plain] == [c["text"] for c in with_heads]
    assert [c["evidence_id"] for c in plain] == [c["evidence_id"] for c in with_heads]
    assert with_heads[0]["section"] == "5.5 Suspicious Transaction Report (STR)"
    assert with_heads[-1]["section"] in {"5.5 Suspicious Transaction Report (STR)", "5.6 Prohibition on Tipping-off"}
    assert all(c["section"] is None for c in plain)


# ---------- headers: index text vs returned text ----------
CH = {**mk_chunk("doc_x", "US", 1, 0, "Plain body about structuring."), "section": "5.5 Suspicious Transaction Report (STR)"}


def test_header_text_and_section_fallback():
    assert header_for(CH) == "Pub 2020 - Title doc_x - 5.5 Suspicious Transaction Report (STR)"
    assert header_for({**CH, "section": None}) == "Pub 2020 - Title doc_x"
    assert index_text(CH, headers=False) == "Plain body about structuring."
    assert index_text(CH, headers=True).endswith("\nPlain body about structuring.")


def test_headers_change_index_not_returned_text():
    chunks = [{**mk_chunk("doc_a", "US", 1, 0, "alpha beta gamma"), "section": "Heading Zebra"},
              {**mk_chunk("doc_b", "US", 1, 0, "delta epsilon zeta"), "section": None},
              {**mk_chunk("doc_c", "US", 1, 0, "eta theta iota"), "section": None}]  # 3 docs so BM25 idf > 0
    plain = Retriever(chunks, FakeEmbedder())
    headed = Retriever(chunks, FakeEmbedder(), index_texts=[index_text(c, True) for c in chunks])
    assert plain.search("zebra", k=1, mode="bm25")[0]["bm25_score"] == 0.0  # not in the plain text
    top = headed.search("zebra", k=1, mode="bm25")[0]
    assert top["index"] == 0 and top["bm25_score"] > 0.0
    assert headed.chunks[0]["text"] == "alpha beta gamma"  # returned text has no header


# ---------- quotable flag ----------
def test_quotable_flag_in_tool_outputs():
    from src.rag.retriever import set_retriever

    chunks = [{**mk_chunk("doc_ocr", "India", 1, 0, "Reporting entities shall file reports."), "quotable": False},
              mk_chunk("doc_pdf", "US", 1, 0, "Money transmitters must register.")]
    set_retriever(Retriever(chunks, FakeEmbedder()))
    try:
        out = search_regulations(SearchRegulationsInput(query="reporting entities file reports", k=2, rewrite=False))
        flags = {h.doc_id: h.quotable for h in out.hits}
        assert flags == {"doc_ocr": False, "doc_pdf": True}  # missing key defaults to True
        assert out.search_query == out.query
        assert get_regulation_chunk(GetRegulationChunkInput(evidence_id="reg:doc_ocr:p1:c0")).quotable is False
        assert get_regulation_chunk(GetRegulationChunkInput(evidence_id="reg:doc_pdf:p1:c0")).quotable is True
    finally:
        set_retriever(None)


def test_manifest_marks_ocr_document_not_quotable():
    import yaml

    docs = {d["doc_id"]: d for d in yaml.safe_load(Path("src/rag/manifest.yaml").read_text(encoding="utf-8"))["documents"]}
    assert docs["fiuind_2026_vda_guidelines"]["quotable"] is False
    assert all(d.get("quotable", True) for k, d in docs.items() if k != "fiuind_2026_vda_guidelines")


# ---------- query rewriting ----------
class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "temperature": config.temperature})
        return FakeResponse('  "suspicious transaction report tipping off"\n')


class FakeClient:
    def __init__(self):
        self.models = FakeModels()


def test_rewriter_sees_only_the_question_uses_temperature_zero_and_caches(tmp_path):
    cache = tmp_path / "rewrites.json"
    client = FakeClient()
    rw = QueryRewriter(model="fake-model", cache_path=cache, client=client)
    q = "May a compliance officer tell a customer about the report?"
    assert rw.rewrite(q) == "suspicious transaction report tipping off"  # quotes and newlines stripped
    call = client.models.calls[0]
    assert call["contents"] == f"{PROMPT}\n\nQuestion: {q}"  # nothing but the instruction and the question
    assert call["temperature"] == 0 and call["model"] == "fake-model"
    assert rw.rewrite(q) == "suspicious transaction report tipping off" and len(client.models.calls) == 1
    # a new instance reads the cache from disk and makes no API call
    client2 = FakeClient()
    assert QueryRewriter(model="fake-model", cache_path=cache, client=client2).rewrite(q).startswith("suspicious")
    assert client2.models.calls == []
    data = json.loads(cache.read_text(encoding="utf-8"))
    assert data["_meta"]["model"] == "fake-model" and data["rewrites"][q]


def test_rewriter_refuses_a_cache_from_another_model(tmp_path):
    cache = tmp_path / "rewrites.json"
    QueryRewriter(model="m1", cache_path=cache, client=FakeClient()).rewrite("a question here")
    with pytest.raises(RuntimeError):
        QueryRewriter(model="m2", cache_path=cache, client=FakeClient())


def test_search_tool_rewrite_flag_uses_rewritten_query(tmp_path):
    from src.rag.retriever import set_retriever
    from src.rag.rewrite import set_rewriter

    chunks = [mk_chunk("doc_a", "US", 1, 0, "tumbling services obscure wallets"),
              mk_chunk("doc_b", "US", 1, 0, "structuring cash deposits below thresholds")]
    set_retriever(Retriever(chunks, FakeEmbedder()))
    set_rewriter(QueryRewriter(model="m", cache_path=tmp_path / "r.json", client=FakeClient()))
    try:
        out = search_regulations(SearchRegulationsInput(query="plain question words here", k=1, rewrite=True))
        assert out.query == "plain question words here"
        assert out.search_query == "suspicious transaction report tipping off"
    finally:
        set_retriever(None)
        set_rewriter(None)


# ---------- keep rules and metrics ----------
def m(h1, h3):
    return {"hit@1": h1, "hit@3": h3, "hit@5": h3, "MRR@10": 0.0}


def test_keep_rule_a_rewriting_needs_0_10_hit3_gain():
    assert decide(m(0.3, 0.50), m(0.3, 0.60), m(0.3, 0.50))["adopt_rewrite"] is True  # exactly +0.10
    assert decide(m(0.3, 0.50), m(0.3, 0.59), m(0.3, 0.50))["adopt_rewrite"] is False
    assert decide(m(0.3, 0.50), m(0.3, 0.40), m(0.3, 0.50))["adopt_rewrite"] is False


def test_keep_rule_b_headers_need_0_05_gain_and_no_hit1_drop():
    assert decide(m(0.3, 0.50), m(0, 0), m(0.3, 0.55))["adopt_headers"] is True  # exactly +0.05, hit@1 equal
    assert decide(m(0.3, 0.50), m(0, 0), m(0.3, 0.54))["adopt_headers"] is False
    assert decide(m(0.3, 0.50), m(0, 0), m(0.25, 0.70))["adopt_headers"] is False  # hit@1 decreased


def test_metrics_and_first_rank():
    chunks = [{"doc_id": "a", "text": "foo"}, {"doc_id": "b", "text": "needle here"}, {"doc_id": "b", "text": "bar"}]
    q = {"id": "q", "expected_doc_id": "b", "acceptable_doc_ids": [], "must_contain": ["needle"]}
    assert first_rank(q, chunks, [0, 2, 1]) == 3 and first_rank(q, chunks, [0, 2]) is None
    out = metrics([1, 3, None, 6])
    assert out["hit@1"] == 0.25 and out["hit@3"] == 0.5 and out["hit@5"] == 0.5
    assert out["MRR@10"] == pytest.approx((1 + 1 / 3 + 0 + 1 / 6) / 4)


def test_golden_file_still_has_original_cvc_5():
    qs = {q["id"]: q for q in json.loads(Path("eval/retrieval_golden.json").read_text(encoding="utf-8"))}
    assert qs["cvc_5"]["expected_doc_id"] == "fincen_2019_cvc_advisory" and qs["cvc_5"]["must_contain"] == ["rapid trades"]
    assert "NOT applied" in Path("eval/GOLDEN_CHANGELOG.md").read_text(encoding="utf-8")

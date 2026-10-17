import json
from pathlib import Path

import pytest
import yaml

from src.rag.ingest import denoise_ocr_page
from src.rag.ocr import _lines_from_boxes
from src.rag.retriever import Retriever
from tests.test_rag import CORPUS, FakeEmbedder


# ---------- OCR helpers ----------
def box(x, y, w=100, h=20):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def test_lines_from_boxes_orders_groups_and_breaks_paragraphs():
    boxes = [box(200, 100), box(0, 102), box(0, 130), box(0, 400)]  # two boxes on one line, then a far one
    texts = ["world", "hello", "second line", "new paragraph"]
    assert _lines_from_boxes(boxes, texts) == ["hello world", "second line", "", "new paragraph"]


def test_denoise_removes_stamp_noise_and_keeps_page_number():
    page = "Real text line one.\nReal text line two.\n\n15\nUNIT\n吉\n六 INDIA\n★"
    cleaned = denoise_ocr_page(page)
    assert "UNIT" not in cleaned and "INDIA" not in cleaned and "吉" not in cleaned and "★" not in cleaned
    assert "Real text line one." in cleaned and cleaned.split("\n")[-1] == "15"


def test_denoise_keeps_page_number_glued_to_stamp_word_and_real_text():
    assert denoise_ocr_page("Body text here.\nUNIT 9").split("\n")[-1] == "9"
    assert denoise_ocr_page("A line about India and the UNIT of account is real text.\nMore text.").count("India") == 1


# ---------- manifest ----------
def test_manifest_entries_for_ocr_and_citation():
    docs = {d["doc_id"]: d for d in yaml.safe_load(Path("src/rag/manifest.yaml").read_text(encoding="utf-8"))["documents"]}
    vda = docs["fiuind_2026_vda_guidelines"]
    assert vda["ocr"] is True and vda["index"] is True and vda["jurisdiction"] == "India" and vda["year"] == 2026
    assert vda["publisher"] == "Financial Intelligence Unit - India (FIU-IND)"
    assert docs["fatf_2020_va_red_flags"]["citation_title"].startswith("Money Laundering and Terrorist Financing Red Flag")
    assert docs["fatf_2020_va_red_flags"]["title"].startswith("Virtual Assets Red Flag Indicators")


# ---------- reranker ----------
class FakeReranker:
    """Scores chunks by whether they contain the word 'tumbling' so the order change is predictable."""

    name = "fake"

    def score(self, query, texts):
        return [1.0 if "tumbling" in t else 0.0 for t in texts]


def test_rerank_mode_reorders_top_hybrid_results():
    r = Retriever(CORPUS, FakeEmbedder(), reranker=FakeReranker())
    plain = r.search("structuring cash deposits under ten thousand", k=4, mode="hybrid")
    reranked = r.search("structuring cash deposits under ten thousand", k=4, mode="hybrid+rerank")
    assert CORPUS[plain[0]["index"]]["evidence_id"] == "reg:doc_us:p2:c1"
    assert CORPUS[reranked[0]["index"]]["evidence_id"] == "reg:doc_fatf:p1:c0"  # the fake reranker prefers 'tumbling'
    assert {h["index"] for h in plain} == {h["index"] for h in reranked}
    assert r.search("structuring cash deposits", k=4, mode="hybrid+rerank") == r.search(
        "structuring cash deposits", k=4, mode="hybrid+rerank")  # deterministic


def test_search_tool_rerank_flag_selects_mode(monkeypatch):
    from src.rag.retriever import set_retriever
    from src.tools.search_regulations import SearchRegulationsInput, search_regulations

    r = Retriever(CORPUS, FakeEmbedder(), reranker=FakeReranker())
    set_retriever(r)
    try:
        q = "structuring cash deposits under ten thousand"
        off = search_regulations(SearchRegulationsInput(query=q, k=4, rerank=False))
        on = search_regulations(SearchRegulationsInput(query=q, k=4, rerank=True))
        assert off.hits[0].evidence_id == "reg:doc_us:p2:c1"
        assert on.hits[0].evidence_id == "reg:doc_fatf:p1:c0"
        default = search_regulations(SearchRegulationsInput(query=q, k=4))
        from src.rag.retriever import RERANK_DEFAULT

        assert default.hits[0].evidence_id == (on if RERANK_DEFAULT else off).hits[0].evidence_id
    finally:
        set_retriever(None)


# ---------- golden set ----------
def test_golden_set_shape_and_old_entries_untouched():
    qs = json.loads(Path("eval/retrieval_golden.json").read_text(encoding="utf-8"))
    ids = [q["id"] for q in qs]
    assert len(ids) == len(set(ids))
    assert ids[:20] == ["cvc_1", "cvc_2", "cvc_3", "cvc_4", "cvc_5", "sar_1", "sar_2", "sar_3", "sar_4", "fatf_1",
                        "fatf_2", "fatf_3", "fatf_4", "fatf_5", "typ_chain", "typ_fan_in", "typ_fan_out",
                        "oos_1", "oos_2", "oos_3"]
    assert sum(q["expected_doc_id"] == "fiuind_2026_vda_guidelines" for q in qs) == 5
    assert sum(q["type"] == "out_of_scope" for q in qs) == 6
    for q in qs:
        assert set(q) == {"id", "type", "question", "expected_doc_id", "acceptable_doc_ids", "must_contain"}
        if q["type"] == "out_of_scope":
            assert q["expected_doc_id"] is None and q["must_contain"] == []
        else:
            assert q["must_contain"]

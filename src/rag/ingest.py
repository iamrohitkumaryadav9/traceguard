"""PDF -> cleaned pages -> paragraphs -> overlapping chunks. Run: python -m src.rag.ingest

Token counts are approximate (words x 1.3); chunks target ~250 tokens (max ~300) with ~15% overlap,
break at paragraph ends when possible, never cross documents and keep page_start / page_end
(physical PDF page numbers, 1-based).
"""
import json
import math
import re
from collections import Counter
from pathlib import Path

import yaml

MANIFEST = Path("src/rag/manifest.yaml")
PDF_DIR = Path("data/raw/regulations")
CHUNKS_PATH = Path("data/processed/rag/chunks.jsonl")

TARGET_TOKENS = 250
MAX_TOKENS = 300
OVERLAP_FRAC = 0.15
HEADER_FOOTER_SPAN = 3  # look at the first/last 3 lines of a page
ABBREV = {"u.s.", "e.g.", "i.e.", "no.", "inc.", "vs.", "etc.", "al.", "st.", "dr.", "mr.", "ms.", "fig.", "sec."}
BULLET = re.compile(r"^\s*([••●▪–—-]|\d{1,2}[.)]|\([a-z0-9]{1,3}\))\s+")


def approx_tokens(text):
    return math.ceil(len(text.split()) * 1.3)


def load_manifest(path=MANIFEST):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["documents"]


# ---------- extraction and cleaning ----------
def extract_pages(pdf_path):
    from pypdf import PdfReader

    return [p.extract_text() or "" for p in PdfReader(str(pdf_path)).pages]


STAMP_WORDS = {"unit", "india", "pre", "prt", "dre"}  # fragments of the FIU-IND round stamp picked up by OCR


def denoise_ocr_page(text):
    """Drop OCR noise from stamps: symbol-only lines, short non-ASCII lines and stamp-word lines
    (optionally followed by the page number) among the last few lines of a page."""
    lines = text.split("\n")
    keep = []
    last = [i for i, l in enumerate(lines) if l.strip()][-5:]
    for i, l in enumerate(lines):
        s = l.strip()
        if i in last and s:
            words = s.lower().split()
            if not re.search(r"[A-Za-z0-9]", s):
                continue  # only symbols, e.g. stars
            if len(s) <= 6 and any(ord(c) > 127 for c in s):
                continue  # stray CJK-looking glyphs from the stamp
            if words and all(w in STAMP_WORDS or w.isdigit() for w in words) and any(w in STAMP_WORDS for w in words):
                digits = [w for w in words if w.isdigit()]
                if digits:
                    keep.append(digits[0])  # keep a page number that was glued to stamp text
                continue
            if any(ord(c) > 127 for c in s) and len(s) <= 12 and any(w in STAMP_WORDS for w in words):
                continue
        keep.append(l)
    return "\n".join(keep)


def _norm_key(line):
    return re.sub(r"[\d-\s]+", " ", line).strip().lower()


def clean_pages(raw_pages):
    """Drop repeating headers/footers and bare page numbers. Returns a list of line lists, one per page."""
    pages = [[l for l in p.split("\n") if l.strip()] for p in raw_pages]
    n = len(pages)

    def edge_idx(lines):
        return set(range(min(HEADER_FOOTER_SPAN, len(lines)))) | set(range(max(0, len(lines) - HEADER_FOOTER_SPAN), len(lines)))

    counts = Counter()
    for lines in pages:
        keys = {_norm_key(lines[i]) for i in edge_idx(lines)}
        counts.update(k for k in keys if k)
    repeating = {k for k, c in counts.items() if n >= 5 and c >= max(3, 0.3 * n)}
    out = []
    for lines in pages:
        edges = edge_idx(lines)
        kept = []
        for i, l in enumerate(lines):
            if i in edges and (_norm_key(l) in repeating or re.fullmatch(r"\s*\d{1,3}\s*", l)):
                continue
            kept.append(l)
        out.append(kept)
    return out


def _tidy(text):
    text = re.sub(r"[●▪]", "•", text)
    text = re.sub(r"[-]", " ", text)  # remaining private-use glyphs
    text = re.sub(r"(?<=[A-Za-z]) -(?=[A-Za-z])", "-", text)  # "inter -governmental" -> "inter-governmental"
    return re.sub(r"\s+", " ", text).strip()


def paragraphs_from_lines(lines):
    """Heuristic paragraph split of one page's lines: bullets, short headings and short sentence-final lines break."""
    if not lines:
        return []
    lens = sorted(len(l.strip()) for l in lines)
    typical = lens[len(lens) // 2]
    paras, cur = [], []

    def flush():
        if cur:
            paras.append(_tidy(" ".join(cur)))
            cur.clear()

    for i, raw in enumerate(lines):
        line = raw.strip()
        heading = len(line) < 50 and not re.search(r"[.,;:!?]$", line) and i + 1 < len(lines)
        if BULLET.match(raw) or heading:
            flush()
        cur.append(line)
        short_end = re.search(r"[.!?:][\"”)]?$", line) and len(line) < 0.7 * typical
        if heading or short_end:
            flush()
    flush()
    return [p for p in paras if p]


def split_sentences(paragraph):
    parts, start = [], 0
    for m in re.finditer(r"(?<=[.!?])[\"”)]?\s+(?=[A-Z•“\"(])", paragraph):
        prev_word = paragraph[start:m.start() + 1].split()[-1].lower() if paragraph[start:m.start() + 1].split() else ""
        if prev_word in ABBREV or re.fullmatch(r"[A-Z]\.", prev_word.upper()) and len(prev_word) == 2:
            continue
        parts.append(paragraph[start:m.end()].strip())
        start = m.end()
    parts.append(paragraph[start:].strip())
    return [p for p in parts if p]


# ---------- chunking ----------
def chunk_paragraphs(doc_id, page_paragraphs, target=TARGET_TOKENS, max_tokens=MAX_TOKENS, overlap_frac=OVERLAP_FRAC):
    """page_paragraphs: [(page_no, [paragraph, ...]), ...] for ONE document. Returns chunk dicts."""
    sents = []  # (text, page, paragraph_id)
    pid = 0
    for page, paras in page_paragraphs:
        for para in paras:
            for s in split_sentences(para):
                sents.append((s, page, pid))
            pid += 1
    tok = [approx_tokens(s[0]) for s in sents]
    overlap_tokens = max(1, round(target * overlap_frac))

    chunks, i, carry, n = [], 0, [], 0
    while i < len(sents):
        cur, total, new = list(carry), sum(tok[j] for j in carry), 0
        while i < len(sents):
            if new > 0 and total + tok[i] > max_tokens:
                break
            cur.append(i)
            total += tok[i]
            new += 1
            i += 1
            if total >= target and (i == len(sents) or sents[i][2] != sents[i - 1][2]):
                break
        pages = [sents[j][1] for j in cur]
        chunks.append({
            "evidence_id": f"reg:{doc_id}:p{min(pages)}:c{n}", "doc_id": doc_id, "n": n,
            "page_start": min(pages), "page_end": max(pages),
            "text": " ".join(sents[j][0] for j in cur), "approx_tokens": total,
        })
        n += 1
        carry, acc = [], 0  # overlap: whole trailing sentences, about overlap_tokens
        for j in reversed(cur):
            if len(carry) >= len(cur) - 1 or acc + tok[j] > 2 * overlap_tokens:
                break  # never carry the whole chunk; skip sentences much longer than the overlap budget
            carry.insert(0, j)
            acc += tok[j]
            if acc >= overlap_tokens:
                break
    return chunks


def build_chunks(manifest=None):
    manifest = manifest or load_manifest()
    chunks = []
    for doc in manifest:
        if not doc.get("index", True):
            continue
        if doc.get("ocr"):
            from src.rag.ocr import load_ocr_pages

            raw = [denoise_ocr_page(p) for p in load_ocr_pages(doc)]
        else:
            raw = extract_pages(PDF_DIR / doc["source_file"])
        if sum(len(p.strip()) for p in raw) == 0:
            raise RuntimeError(f"{doc['source_file']} has no extractable text; mark it ocr: true or index: false")
        pages = clean_pages(raw)
        page_paras = [(i + 1, paragraphs_from_lines(lines)) for i, lines in enumerate(pages)]
        for c in chunk_paragraphs(doc["doc_id"], page_paras):
            c.update({"title": doc["title"], "publisher": doc["publisher"], "year": doc["year"],
                      "jurisdiction": doc["jurisdiction"]})
            chunks.append(c)
    return chunks


def save_chunks(chunks, path=CHUNKS_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n", encoding="utf-8")


def load_chunks(path=CHUNKS_PATH):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


if __name__ == "__main__":
    chunks = build_chunks()
    save_chunks(chunks)
    by_doc = Counter(c["doc_id"] for c in chunks)
    toks = sorted(c["approx_tokens"] for c in chunks)
    print(f"{len(chunks)} chunks: {dict(by_doc)}")
    print(f"approx tokens per chunk: min {toks[0]}, median {toks[len(toks) // 2]}, max {toks[-1]}")
    print(f"saved {CHUNKS_PATH}")

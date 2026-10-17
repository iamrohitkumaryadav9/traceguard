"""OCR for image-only PDFs with rapidocr (pip-only, ONNX, CPU). Run: python -m src.rag.ocr

Page text is cached in data/processed/ocr/<doc_id>/page_NNN.txt (NNN = physical PDF page number, which
matches the printed page number for the India VDA document). Page images are saved next to it so OCR
quality can be checked by eye. Existing page files are not recomputed.
"""
import io
import sys
from pathlib import Path

from src.rag.ingest import PDF_DIR, load_manifest

OCR_DIR = Path("data/processed/ocr")
LINE_GAP = 0.6  # boxes whose vertical centres differ by less than this x box height share a line
PARA_GAP = 1.6  # a vertical gap above this x line height starts a new paragraph (blank line)


def _page_image(page):
    from PIL import Image

    imgs = list(page.images)
    if len(imgs) != 1:
        raise RuntimeError(f"expected one image per page, found {len(imgs)}")
    return Image.open(io.BytesIO(imgs[0].data)).convert("RGB")


def _lines_from_boxes(boxes, texts):
    """Group OCR boxes into reading-order lines; insert '' between paragraphs (large vertical gaps)."""
    items = []
    for b, t in zip(boxes, texts):
        ys = [p[1] for p in b]
        xs = [p[0] for p in b]
        items.append({"y": (min(ys) + max(ys)) / 2, "h": max(ys) - min(ys), "x": min(xs), "t": t})
    items.sort(key=lambda i: (i["y"], i["x"]))
    lines, cur = [], []
    for it in items:
        if cur and abs(it["y"] - cur[-1]["y"]) < LINE_GAP * max(it["h"], cur[-1]["h"]):
            cur.append(it)
        else:
            if cur:
                lines.append(cur)
            cur = [it]
    if cur:
        lines.append(cur)
    out, prev = [], None
    for ln in lines:
        ln.sort(key=lambda i: i["x"])
        y, h = sum(i["y"] for i in ln) / len(ln), max(i["h"] for i in ln)
        if prev is not None and (y - prev[0]) > PARA_GAP * max(h, prev[1]):
            out.append("")
        out.append(" ".join(i["t"] for i in ln))
        prev = (y, h)
    return out


def ocr_document(doc, out_root=OCR_DIR, force=False):
    import numpy as np
    from pypdf import PdfReader
    from rapidocr import RapidOCR

    out = Path(out_root) / doc["doc_id"]
    (out / "images").mkdir(parents=True, exist_ok=True)
    engine = RapidOCR()
    reader = PdfReader(str(PDF_DIR / doc["source_file"]))
    for i, page in enumerate(reader.pages, start=1):
        txt_path = out / f"page_{i:03d}.txt"
        if txt_path.exists() and not force:
            continue
        img = _page_image(page)
        img.save(out / "images" / f"page_{i:03d}.png")
        res = engine(np.array(img))
        lines = _lines_from_boxes(res.boxes.tolist(), list(res.txts)) if res.boxes is not None else []
        txt_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"page {i}: {len(lines)} lines, {sum(len(l) for l in lines)} chars", flush=True)
    return out


def load_ocr_pages(doc, out_root=OCR_DIR):
    """Raw page texts (list of str, in page order) for an OCRed document."""
    d = Path(out_root) / doc["doc_id"]
    files = sorted(d.glob("page_*.txt"))
    if not files:
        raise FileNotFoundError(f"no OCR text in {d}; run python -m src.rag.ocr")
    return [f.read_text(encoding="utf-8") for f in files]


if __name__ == "__main__":
    for d in load_manifest():
        if d.get("ocr"):
            print("OCR:", d["source_file"])
            ocr_document(d, force="--force" in sys.argv)

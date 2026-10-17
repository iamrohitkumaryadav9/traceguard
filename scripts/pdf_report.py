"""Per-page text extraction report for the regulation PDFs. Run: python -m scripts.pdf_report"""
from pathlib import Path

from pypdf import PdfReader

PDF_DIR = Path("data/raw/regulations")
EMPTY_CHARS = 50  # a page with fewer characters than this looks empty or scanned


def main():
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        reader = PdfReader(str(pdf))
        counts = [len((p.extract_text() or "").strip()) for p in reader.pages]
        empty = [i + 1 for i, c in enumerate(counts) if c < EMPTY_CHARS]
        print("=" * 90)
        print(f"{pdf.name}: {len(counts)} pages, {sum(counts):,} chars, "
              f"min/median/max chars per page {min(counts)}/{sorted(counts)[len(counts)//2]}/{max(counts)}")
        print(f"  pages under {EMPTY_CHARS} chars: {empty if empty else 'none'}")
        if len(counts) <= 40:
            print("  chars per page:", counts)
        meta = reader.metadata
        print("  metadata title/author:", getattr(meta, "title", None), "|", getattr(meta, "author", None))
        for pno in (0, 1):
            if pno < len(reader.pages):
                txt = " ".join((reader.pages[pno].extract_text() or "").split())
                print(f"  --- page {pno + 1} start: {txt[:600]}")


if __name__ == "__main__":
    main()

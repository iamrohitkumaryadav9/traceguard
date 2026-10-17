# Regulation sources

The PDFs live in `data/raw/regulations/` and are **not committed** (gitignored), and neither are the OCR text and the index files in `data/processed/`. Metadata was inferred from the first pages of each file and is recorded in [manifest.yaml](manifest.yaml); please verify it.

| doc_id | Title | Publisher | Year | Jurisdiction | File | Text source |
|---|---|---|---|---|---|---|
| fincen_2019_cvc_advisory | Advisory on Illicit Activity Involving Convertible Virtual Currency (FIN-2019-A003, 9 May 2019) | FinCEN | 2019 | US | FinCEN Advisory CVC FINAL 508.pdf | PDF text layer |
| fincen_2003_sar_narrative | Guidance on Preparing A Complete & Sufficient Suspicious Activity Report Narrative (file dated 11-20-03) | FinCEN | 2003 | US | sarnarrcompletguidfinal_112003.pdf | PDF text layer |
| fatf_2020_va_red_flags | Virtual Assets Red Flag Indicators of Money Laundering and Terrorist Financing (September 2020) | FATF | 2020 | FATF | Virtual-Assets-Red-Flag-Indicators.pdf | PDF text layer |
| fiuind_2026_vda_guidelines | AML & CFT Guidelines for Reporting Entities Providing Services Related to Virtual Digital Assets (cover: "Updated as on 08.01.2026") | Financial Intelligence Unit - India (FIU-IND) | 2026 | India | VDA08012026.pdf | **OCR** (rapidocr) |

Notes:

- The FATF cover title differs from the title in its "Citing reference" on page 2 ("Money Laundering and Terrorist Financing Red Flag Indicators Associated with Virtual Assets"). The manifest keeps the cover title as `title` and the other as `citation_title`.
- Evidence ids use physical PDF page numbers (page 1 is the first page of the file). For the India document these match the printed page numbers; for the others they can differ.
- **OCR of the India document.** `VDA08012026.pdf` is a scan (one image per page, 33 pages). It was read with `rapidocr` (pip-only, local ONNX models, no data sent to any service); tesseract was not installed. Checked by eye against the page images for pages 5, 15 and 25: the body text is accurate. Observed errors: "AI/ML" read as "Al/ML", a missing space in "holding,and", a stray middle dot in "Transaction·reports", "v." read as "V.", and noise from the round FIU-IND stamp at the bottom of pages ("UNIT", "INDIA", stars, stray CJK glyphs), most of which is stripped before chunking. Some stamp fragments inside the page text remain (for example "FINA UNIT", "VNGIA"). Quotations from this document should be checked against the page image.
- The 2003 SAR guidance and the 2019 advisory predate current rules, so retrieved text is background evidence and not current law.

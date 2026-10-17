# Regulation sources

The PDFs live in `data/raw/regulations/` and are **not committed** (gitignored). Metadata was inferred from the first pages of each file and is recorded in [manifest.yaml](manifest.yaml); please verify it.

| doc_id | Title | Publisher | Year | Jurisdiction | File | Indexed |
|---|---|---|---|---|---|---|
| fincen_2019_cvc_advisory | Advisory on Illicit Activity Involving Convertible Virtual Currency (FIN-2019-A003, 9 May 2019) | FinCEN | 2019 | US | FinCEN Advisory CVC FINAL 508.pdf | yes |
| fincen_2003_sar_narrative | Guidance on Preparing A Complete & Sufficient Suspicious Activity Report Narrative (file dated 11-20-03) | FinCEN | 2003 | US | sarnarrcompletguidfinal_112003.pdf | yes |
| fatf_2020_va_red_flags | Virtual Assets Red Flag Indicators of Money Laundering and Terrorist Financing (September 2020) | FATF | 2020 | FATF | Virtual-Assets-Red-Flag-Indicators.pdf | yes |
| vda_08012026_unidentified | unknown (image-only scan, no extractable text) | unknown | unknown | unknown | VDA08012026.pdf | **no** |

Notes:

- Evidence ids use physical PDF page numbers (page 1 is the first page of the file), which can differ from the page numbers printed on the pages.
- `VDA08012026.pdf` has no text layer on any of its 33 pages. It is not indexed and has not been OCRed.
- The SAR narrative guidance is from 2003 and the CVC advisory from 2019. Both predate current rules, so retrieved text is background evidence and not current law.

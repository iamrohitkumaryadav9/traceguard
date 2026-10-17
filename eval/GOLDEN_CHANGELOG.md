# Golden set changelog

`eval/retrieval_golden.json` entries are never edited silently. Every change, and every requested change that was not made, is recorded here.

## 2026-10-17: cvc_5 change requested, NOT applied

**Requested:** change cvc_5 from `expected_doc_id: fincen_2019_cvc_advisory`, `must_contain: ["rapid trades"]` to `expected_doc_id: fatf_2020_va_red_flags`, `must_contain: ["multiple types of VAs"]`, keeping the question text. Stated reason: "rapid trades" does not appear anywhere in the FinCEN 2019 CVC advisory.

**Verification before changing anything:**

- The stated reason does not hold. "rapid trades" appears in the FinCEN 2019 advisory, PDF page 9, in the red-flag list: "A customer initiates multiple rapid trades between multiple virtual currencies with no related purpose, which may be indicative of attempts to break the chain of custody on the respective blockchains or further obfuscate the transaction." It is stored in chunk `reg:fincen_2019_cvc_advisory:p9:c20`, which is why cvc_5 was found at rank 1 by hybrid retrieval in Phase 3b. That passage answers the question ("what does it suggest when a client keeps flipping quickly between different cryptocurrencies").
- The FATF passage (page 8, "converting the VAs to multiple types of VAs ... without logical business explanation") is a second valid answer, but the extracted text reads "multiple types of VA s" (PDF text extraction splits the word), so the requested phrase "multiple types of VAs" would not match even after whitespace normalisation.

**Decision:** the golden file was left unchanged, because the change would rest on a reason that is false and would replace a correct entry. The effect of the requested change is reported anyway, as a what-if applied in memory only (expected document FATF, phrase "multiple types of VA"), next to the unchanged results in `eval/results/retrieval_final.md`. If the entry should still change (for example to accept both documents), that is a one-line decision for the owner of the golden set.

**Other existing entries:** unchanged. New entries are only appended (see the Phase 3c additions ind_1 to ind_5 and oos_4 to oos_6).

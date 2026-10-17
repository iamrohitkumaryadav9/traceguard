# Final retrieval configuration (frozen in Phase 3d)

No further retrieval experiments are run after this report. The primary metric is hybrid retrieval on the 19 document-grounded golden questions (14 original + 5 India); the 3 typology queries are reported separately.

## Final configuration

- Query rewriting: **OFF** (model gemini-3.5-flash-lite, temperature 0; hit@3 gain +0.053, rule needs >= 0.10).
- Contextual headers: **ON** (hit@3 gain on raw queries +0.053, hit@1 change +0.000; rule needs >= 0.05 and no hit@1 decrease).
- Retrieval: hybrid (BM25 + BAAI/bge-small-en-v1.5 in FAISS, reciprocal rank fusion, k=60), no reranker (rejected in Phase 3c: hit@1 +0.045 against a 0.10 rule).
- Matcher: normalised phrase match (case, whitespace, soft hyphens, hyphenation at line breaks; '-', '/' and space equivalent; whole words).
- Under the cvc_5 what-if the rules would pick: rewriting ON, headers ON.

## Keep rules (fixed before the experiments)

- A: adopt LLM query rewriting only if hybrid hit@3 improves by >= 0.10.
- B: adopt contextual headers only if hybrid hit@3 improves by >= 0.05 on raw queries and hit@1 does not decrease.

## All variants (hybrid retrieval)

The 'cvc_5 what-if' rows show the golden set with the requested cvc_5 change applied in memory only (expected document FATF, phrase 'multiple types of VA'); the golden file keeps the original entry, see eval/GOLDEN_CHANGELOG.md.

| variant | question set | hit@1 | hit@3 | hit@5 | MRR@10 |
|---|---|---|---|---|---|
| raw | document-grounded (19), golden as is | 0.316 | 0.526 | 0.579 | 0.428 |
| raw | document-grounded (19), cvc_5 what-if | 0.263 | 0.474 | 0.526 | 0.376 |
| raw | typology (3) | 0.667 | 1.000 | 1.000 | 0.833 |
| rewritten | document-grounded (19), golden as is | 0.263 | 0.579 | 0.632 | 0.435 |
| rewritten | document-grounded (19), cvc_5 what-if | 0.263 | 0.579 | 0.632 | 0.429 |
| rewritten | typology (3) | 0.000 | 0.333 | 0.333 | 0.204 |
| raw + headers | document-grounded (19), golden as is | 0.316 | 0.579 | 0.579 | 0.446 |
| raw + headers | document-grounded (19), cvc_5 what-if | 0.263 | 0.526 | 0.526 | 0.393 |
| raw + headers | typology (3) | 1.000 | 1.000 | 1.000 | 1.000 |
| rewritten + headers | document-grounded (19), golden as is | 0.316 | 0.684 | 0.842 | 0.519 |
| rewritten + headers | document-grounded (19), cvc_5 what-if | 0.316 | 0.632 | 0.789 | 0.493 |
| rewritten + headers | typology (3) | 0.667 | 0.667 | 0.667 | 0.704 |

## Baseline before and after the cvc_5 question and the matcher change

| setting | hit@1 | hit@3 | hit@5 | MRR@10 |
|---|---|---|---|---|
| old regex matcher, golden as is | 0.316 | 0.526 | 0.579 | 0.428 |
| new matcher, golden as is | 0.316 | 0.526 | 0.579 | 0.428 |
| new matcher, cvc_5 what-if | 0.263 | 0.474 | 0.526 | 0.376 |

cvc_5 first correct rank for raw hybrid: 1 as is, None under the what-if.

## Example rewrites

- **cvc_2**: How might a bank tell that a customer is quietly running an unlicensed coin-for-cash swapping business?
  - rewritten: indicators unlicensed virtual asset service provider cash-to-crypto P2P typology AML
- **fatf_4**: Why is it odd when a customer parks crypto at an exchange and straight away moves it to their own offline storage?
  - rewritten: cryptocurrency exchange immediate withdrawal self-hosted wallet cold storage AML typologies risk indicators
- **ind_2**: May a compliance officer tell a customer that a report about their activity has gone to the authorities?
  - rewritten: tipping-off prohibition virtual asset service provider suspicious transaction report AML compliance
- **ind_5**: How should an Indian reporting entity treat services that blend coins together to hide where they came from?
  - rewritten: AML reporting entity obligations virtual asset service provider coin mixing privacy tokens India PMLA
- **typ_fan_out**: funds split from a single account across many recipients, distribution of funds to multiple accounts
  - rewritten: smurfing layering tiered transactions multi-recipient distribution virtual assets AML

Mean live rewrite latency when the cache was filled: 993 ms per call.

## Contextual headers

99 of 178 chunks have a detected section heading (numbered headings or short all-caps lines): almost all of them in the India document. The FinCEN documents and most of the FATF report have no detectable headings, so their headers contain only publisher, year and title.

## Limitations

- Small sample: 19 document-grounded questions, so one question moves hit@k by about 0.05 and differences between variants are noisy. Typology has only 3 queries.
- Strict keyword matching: a chunk counts as correct only if it contains the required phrase, so valid alternative chunks are missed (error analysis found such a case for cvc_2), and the numbers understate real quality. PDF text extraction also splits words (for example 'VA s'), which the matcher does not repair.
- OCR noise: the India document comes from OCR with stamp fragments and small character errors; it is marked quotable: false, and quotations from it must be checked against the page image.
- The golden questions were written by the same person who built the system and were read against the documents; the rewriter saw only the questions.
- Rewriting calls an external API on every search when enabled (latency and cost, and a dependency on the service); cached rewrites make the evaluation reproducible but not live use.

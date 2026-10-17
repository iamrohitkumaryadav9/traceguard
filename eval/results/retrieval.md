# Retrieval evaluation

Embeddings: BAAI/bge-small-en-v1.5 (sentence-transformers, CPU). 120 chunks over 3 documents. Hybrid = BM25 + dense fused with reciprocal rank fusion (k=60). A hit needs the expected (or acceptable) document and every must_contain regex to match the chunk text.

17 scored questions (incl. 3 typology queries), 3 out-of-scope.

| mode | hit@1 | hit@3 | hit@5 | MRR@10 | avg latency (ms) |
|---|---|---|---|---|---|
| bm25 | 0.176 | 0.353 | 0.588 | 0.336 | 1.889 |
| dense | 0.471 | 0.588 | 0.765 | 0.574 | 64.475 |
| hybrid | 0.529 | 0.706 | 0.706 | 0.614 | 61.866 |

## First correct rank per question

| id | type | rank_bm25 | rank_dense | rank_hybrid |
|---|---|---|---|---|
| cvc_1 | in_scope | 7.000 | nan | 9.000 |
| cvc_2 | in_scope | nan | nan | nan |
| cvc_3 | in_scope | 5.000 | 6.000 | 3.000 |
| cvc_4 | in_scope | 1.000 | 1.000 | 1.000 |
| cvc_5 | in_scope | 7.000 | 1.000 | 1.000 |
| sar_1 | in_scope | 1.000 | 1.000 | 1.000 |
| sar_2 | in_scope | 4.000 | 3.000 | 3.000 |
| sar_3 | in_scope | 3.000 | 1.000 | 1.000 |
| sar_4 | in_scope | 7.000 | 2.000 | 2.000 |
| fatf_1 | in_scope | nan | 4.000 | nan |
| fatf_2 | in_scope | 2.000 | 1.000 | 1.000 |
| fatf_3 | in_scope | nan | 4.000 | 6.000 |
| fatf_4 | in_scope | nan | nan | nan |
| fatf_5 | in_scope | 4.000 | 1.000 | 1.000 |
| typ_chain | typology | 1.000 | 4.000 | 1.000 |
| typ_fan_in | typology | 4.000 | 1.000 | 1.000 |
| typ_fan_out | typology | 2.000 | 1.000 | 1.000 |

## Top-1 scores: in-scope vs out-of-scope (no abstention cutoff set)

| id | group | top1_fused_rrf | top1_dense_cosine | top1_bm25 |
|---|---|---|---|---|
| cvc_1 | in_scope | 0.0323 | 0.6918 | 9.4714 |
| cvc_2 | in_scope | 0.0315 | 0.6955 | 8.2030 |
| cvc_3 | in_scope | 0.0323 | 0.7262 | 9.3446 |
| cvc_4 | in_scope | 0.0328 | 0.7020 | 13.6427 |
| cvc_5 | in_scope | 0.0313 | 0.6949 | 4.0751 |
| sar_1 | in_scope | 0.0325 | 0.6456 | 9.9340 |
| sar_2 | in_scope | 0.0325 | 0.6876 | 7.9060 |
| sar_3 | in_scope | 0.0323 | 0.6820 | 4.8302 |
| sar_4 | in_scope | 0.0315 | 0.6010 | 12.0611 |
| fatf_1 | in_scope | 0.0313 | 0.6983 | 4.1244 |
| fatf_2 | in_scope | 0.0325 | 0.7459 | 17.3352 |
| fatf_3 | in_scope | 0.0318 | 0.7073 | 5.9650 |
| fatf_4 | in_scope | 0.0309 | 0.7009 | 2.7837 |
| fatf_5 | in_scope | 0.0320 | 0.6626 | 4.7671 |
| typ_chain | in_scope | 0.0320 | 0.7297 | 6.1342 |
| typ_fan_in | in_scope | 0.0318 | 0.7519 | 6.3152 |
| typ_fan_out | in_scope | 0.0325 | 0.7309 | 6.7949 |
| oos_1 | out_of_scope | 0.0308 | 0.6556 | 4.7144 |
| oos_2 | out_of_scope | 0.0313 | 0.6822 | 4.3695 |
| oos_3 | out_of_scope | 0.0312 | 0.5935 | 5.0297 |

## Typology retrieval examples (hybrid top 3)

**fan_in**: many incoming transfers consolidated into a single account, aggregation of funds from multiple sources

- 1. reg:fatf_2020_va_red_flags:p10:c16 (rrf 0.0318, cos 0.752) month, etc.) to the same VA account – o by more than one person; o from the same IP address by one or more persons; or o concerning large amounts. • Incoming transactions from many unrelated wallets in relatively small amounts (accumulation of funds) with subs...
- 2. reg:fatf_2020_va_red_flags:p9:c12 (rrf 0.0301, cos 0.700) Multiple VAs and multiple transfers to foreign VASPs A local VA exchange reported that app roximately KRW 400 million (EUR 301 170) was stolen from phishing victims and was ultimately exchanged for VAs as a layering technique. What triggered the reporting was ...
- 3. reg:fatf_2020_va_red_flags:p8:c11 (rrf 0.0289, cos 0.736) In various instances, the individuals shared the same resid ential address; and most of the VA addresses were accessed from the same IP address – indicating the potential use of money mules by professional money launderers to launder the illicit proceeds. In a...

**fan_out**: funds split from a single account across many recipients, distribution of funds to multiple accounts

- 1. reg:fatf_2020_va_red_flags:p8:c11 (rrf 0.0325, cos 0.731) In various instances, the individuals shared the same resid ential address; and most of the VA addresses were accessed from the same IP address – indicating the potential use of money mules by professional money launderers to launder the illicit proceeds. In a...
- 2. reg:fatf_2020_va_red_flags:p9:c12 (rrf 0.0308, cos 0.690) Multiple VAs and multiple transfers to foreign VASPs A local VA exchange reported that app roximately KRW 400 million (EUR 301 170) was stolen from phishing victims and was ultimately exchanged for VAs as a layering technique. What triggered the reporting was ...
- 3. reg:fatf_2020_va_red_flags:p10:c16 (rrf 0.0288, cos 0.714) month, etc.) to the same VA account – o by more than one person; o from the same IP address by one or more persons; or o concerning large amounts. • Incoming transactions from many unrelated wallets in relatively small amounts (accumulation of funds) with subs...

**chain**: sequential transfers through a chain of intermediary accounts, layering of funds

- 1. reg:fatf_2020_va_red_flags:p8:c11 (rrf 0.0320, cos 0.730) In various instances, the individuals shared the same resid ential address; and most of the VA addresses were accessed from the same IP address – indicating the potential use of money mules by professional money launderers to launder the illicit proceeds. In a...
- 2. reg:fincen_2003_sar_narrative:p7:c11 (rrf 0.0308, cos 0.731) • transactions and/or volumes of aggregate activity inconsistent with the expected purpose of the account and expected levels and types of account activity conveyed to the financial institution by the accountholder at the time of the account opening; • benefic...
- 3. reg:fincen_2003_sar_narrative:p7:c10 (rrf 0.0301, cos 0.709) • a lack of evidence of legitimate business activity, or any business operations at all, undertaken by many of the parties to the transaction(s); • unusual financial nexuses and transactions occurring among certain business types (e.g., food importer dealing w...


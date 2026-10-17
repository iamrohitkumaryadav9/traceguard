# Retrieval evaluation

Embeddings: BAAI/bge-small-en-v1.5 (sentence-transformers, CPU). 178 chunks over 4 documents. Hybrid = BM25 + dense fused with reciprocal rank fusion (k=60). A hit needs the expected (or acceptable) document and every must_contain regex to match the chunk text.

22 scored questions (incl. 3 typology queries), 6 out-of-scope.

| mode | hit@1 | hit@3 | hit@5 | MRR@10 | avg latency (ms) |
|---|---|---|---|---|---|
| bm25 | 0.182 | 0.273 | 0.409 | 0.286 | 2.838 |
| dense | 0.364 | 0.500 | 0.636 | 0.477 | 56.120 |
| hybrid | 0.364 | 0.591 | 0.636 | 0.484 | 57.141 |
| hybrid+rerank | 0.409 | 0.591 | 0.682 | 0.511 | 2492.856 |

## First correct rank per question

| id | type | rank_bm25 | rank_dense | rank_hybrid | rank_hybrid+rerank |
|---|---|---|---|---|---|
| cvc_1 | in_scope | 9.000 | nan | 9.000 | 4.000 |
| cvc_2 | in_scope | nan | nan | nan | nan |
| cvc_3 | in_scope | 6.000 | 6.000 | 3.000 | 3.000 |
| cvc_4 | in_scope | 1.000 | 1.000 | 1.000 | 1.000 |
| cvc_5 | in_scope | 7.000 | 1.000 | 1.000 | 2.000 |
| sar_1 | in_scope | 1.000 | 1.000 | 1.000 | 1.000 |
| sar_2 | in_scope | 4.000 | 3.000 | 3.000 | 4.000 |
| sar_3 | in_scope | 3.000 | 1.000 | 1.000 | 1.000 |
| sar_4 | in_scope | 6.000 | 2.000 | 2.000 | 1.000 |
| fatf_1 | in_scope | nan | 5.000 | nan | nan |
| fatf_2 | in_scope | 1.000 | 1.000 | 1.000 | 1.000 |
| fatf_3 | in_scope | nan | 4.000 | 9.000 | 3.000 |
| fatf_4 | in_scope | nan | nan | nan | 10.000 |
| fatf_5 | in_scope | 4.000 | 2.000 | 1.000 | 1.000 |
| typ_chain | typology | 3.000 | 4.000 | 2.000 | 1.000 |
| typ_fan_in | typology | 4.000 | 1.000 | 1.000 | 3.000 |
| typ_fan_out | typology | 1.000 | 1.000 | 1.000 | 1.000 |
| ind_1 | in_scope | 8.000 | 7.000 | 4.000 | 7.000 |
| ind_2 | in_scope | nan | nan | nan | nan |
| ind_3 | in_scope | nan | 7.000 | nan | nan |
| ind_4 | in_scope | 6.000 | 1.000 | 2.000 | 1.000 |
| ind_5 | in_scope | nan | nan | nan | nan |

## Top-1 scores: in-scope vs out-of-scope (no abstention cutoff set)

| id | group | top1_fused_rrf | top1_dense_cosine | top1_bm25 | top1_rerank_logit |
|---|---|---|---|---|---|
| cvc_1 | in_scope | 0.0323 | 0.6918 | 10.5889 | -1.1539 |
| cvc_2 | in_scope | 0.0315 | 0.6955 | 10.1529 | -1.4649 |
| cvc_3 | in_scope | 0.0320 | 0.7357 | 11.2585 | 3.3282 |
| cvc_4 | in_scope | 0.0328 | 0.7020 | 13.7301 | 1.4461 |
| cvc_5 | in_scope | 0.0313 | 0.6949 | 5.0341 | -4.9001 |
| sar_1 | in_scope | 0.0325 | 0.6456 | 11.0027 | -0.9358 |
| sar_2 | in_scope | 0.0325 | 0.6876 | 9.0201 | -0.3438 |
| sar_3 | in_scope | 0.0323 | 0.6820 | 5.0717 | -8.7890 |
| sar_4 | in_scope | 0.0313 | 0.6010 | 13.5040 | 0.6011 |
| fatf_1 | in_scope | 0.0313 | 0.6983 | 5.3698 | -7.2644 |
| fatf_2 | in_scope | 0.0328 | 0.7459 | 19.8084 | 4.5879 |
| fatf_3 | in_scope | 0.0318 | 0.7073 | 7.0206 | -5.9584 |
| fatf_4 | in_scope | 0.0309 | 0.7009 | 4.4637 | -5.6382 |
| fatf_5 | in_scope | 0.0318 | 0.6626 | 5.7255 | -4.1483 |
| typ_chain | in_scope | 0.0318 | 0.7187 | 9.8339 | 0.1883 |
| typ_fan_in | in_scope | 0.0320 | 0.7519 | 8.4589 | 0.9608 |
| typ_fan_out | in_scope | 0.0328 | 0.7309 | 10.2387 | 0.6994 |
| oos_1 | out_of_scope | 0.0325 | 0.7100 | 9.4144 | -3.6822 |
| oos_2 | out_of_scope | 0.0305 | 0.6822 | 5.9442 | -8.5404 |
| oos_3 | out_of_scope | 0.0310 | 0.5935 | 5.3535 | -8.2846 |
| ind_1 | in_scope | 0.0315 | 0.7184 | 7.4942 | -2.2051 |
| ind_2 | in_scope | 0.0302 | 0.6353 | 7.2970 | -3.0063 |
| ind_3 | in_scope | 0.0325 | 0.6906 | 11.8309 | -4.9247 |
| ind_4 | in_scope | 0.0325 | 0.6873 | 11.1387 | 0.1128 |
| ind_5 | in_scope | 0.0289 | 0.6688 | 6.1927 | -7.3243 |
| oos_4 | out_of_scope | 0.0328 | 0.6105 | 9.9102 | -8.8904 |
| oos_5 | out_of_scope | 0.0323 | 0.6270 | 3.3004 | -11.1136 |
| oos_6 | out_of_scope | 0.0312 | 0.6458 | 6.3290 | -3.5942 |

## Typology retrieval examples (hybrid top 3)

**fan_in**: many incoming transfers consolidated into a single account, aggregation of funds from multiple sources

- 1. reg:fatf_2020_va_red_flags:p10:c16 (rrf 0.0320, cos 0.752) month, etc.) to the same VA account – o by more than one person; o from the same IP address by one or more persons; or o concerning large amounts. • Incoming transactions from many unrelated wallets in relatively small amounts (accumulation of funds) with subs...
- 2. reg:fatf_2020_va_red_flags:p9:c12 (rrf 0.0301, cos 0.700) Multiple VAs and multiple transfers to foreign VASPs A local VA exchange reported that app roximately KRW 400 million (EUR 301 170) was stolen from phishing victims and was ultimately exchanged for VAs as a layering technique. What triggered the reporting was ...
- 3. reg:fatf_2020_va_red_flags:p8:c11 (rrf 0.0295, cos 0.736) In various instances, the individuals shared the same resid ential address; and most of the VA addresses were accessed from the same IP address – indicating the potential use of money mules by professional money launderers to launder the illicit proceeds. In a...

**fan_out**: funds split from a single account across many recipients, distribution of funds to multiple accounts

- 1. reg:fatf_2020_va_red_flags:p8:c11 (rrf 0.0328, cos 0.731) In various instances, the individuals shared the same resid ential address; and most of the VA addresses were accessed from the same IP address – indicating the potential use of money mules by professional money launderers to launder the illicit proceeds. In a...
- 2. reg:fatf_2020_va_red_flags:p9:c12 (rrf 0.0311, cos 0.690) Multiple VAs and multiple transfers to foreign VASPs A local VA exchange reported that app roximately KRW 400 million (EUR 301 170) was stolen from phishing victims and was ultimately exchanged for VAs as a layering technique. What triggered the reporting was ...
- 3. reg:fincen_2003_sar_narrative:p28:c41 (rrf 0.0300, cos 0.708) In addition to some securities trading, from February 2000 through November 2002, accounts CS 12345, CS 34567 and CS 67891 had total deposits of $923,709 and total withdrawals of $723,445. The accounts were funded in large part by personal checks issued on an ...

**chain**: sequential transfers through a chain of intermediary accounts, layering of funds

- 1. reg:fiuind_2026_vda_guidelines:p4:c2 (rrf 0.0318, cos 0.719) ICO /ITO Initial Coin Offering / Initial Token Offering Intermediary Refers to a Service Provider (SP) in a serial chain that receives and re- Service transmits a VDA transfer on behalf of the Originator SP and the Beneficiary Provider SP or another intermedia...
- 2. reg:fatf_2020_va_red_flags:p8:c11 (rrf 0.0315, cos 0.730) In various instances, the individuals shared the same resid ential address; and most of the VA addresses were accessed from the same IP address – indicating the potential use of money mules by professional money launderers to launder the illicit proceeds. In a...
- 3. reg:fincen_2003_sar_narrative:p7:c11 (rrf 0.0306, cos 0.731) • transactions and/or volumes of aggregate activity inconsistent with the expected purpose of the account and expected levels and types of account activity conveyed to the financial institution by the accountholder at the time of the account opening; • benefic...


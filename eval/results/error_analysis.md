# Error analysis (golden questions never retrieved in the top 10)

## cvc_2: How might a bank tell that a customer is quietly running an unlicensed coin-for-cash swapping business?

expected doc `fincen_2019_cvc_advisory`, must_contain ['funnel account']

Gold chunks (satisfy the rule): ['reg:fincen_2019_cvc_advisory:p4:c9']
- best gold rank in full bm25 ranking: 118
- best gold rank in full dense ranking: 54
- best gold rank in full hybrid ranking: 87
- best gold rank in full hybrid+rerank ranking: None
Looser rule (expected doc, regex `acting as an unregistered MSB|funnel account`) best rank per mode:
- bm25: 71
- dense: 3
- hybrid: 20
- hybrid+rerank: 9
Other chunks in the expected document that match the looser rule: ['reg:fincen_2019_cvc_advisory:p8:c19']

Gold chunk reg:fincen_2019_cvc_advisory:p4:c9 (251 tokens, 9 sentences); matching sentence index [3, 4, 5] (0 = start, 8 = end of chunk):
  > See also supra, at 2 n.2. Financial institutions may identify P2P exchangers through perceived funnel account activity, as many of the P2P exchangers’ clients deposit funds into the P2P exchanger’s account. 10 10. For a detailed description of funnel accounts, see FinCEN Advisory, FIN-2014-A005, “Update on U.S. Currency Restriction in Mexico: Funnel Accounts and TBML,” May 28, 2014.
  previous chunk reg:fincen_2019_cvc_advisory:p4:c8 ends: ...t allow a CVC user to mask their identity through blending the proceeds of their transaction with those of other users. Money mules refer to third parties used to carry out transactions on behalf of another individual. 9
  next chunk reg:fincen_2019_cvc_advisory:p5:c10 starts: Recent cases suggest that CVC buyers and sellers involved in small-volume exchanges are increasingly used for money laundering purposes, possibly without their knowledge, such as to launder proceeds from drug trafficking...

Top 3 hybrid results actually returned:
1. reg:fatf_2020_va_red_flags:p11:c18 (cos 0.696, bm25 10.15)  For example, the use of a hardware or paper wallet may be legitimate as a way to secure VAs against thefts. Again, the presence of these indicators should be considered in the context of other characteristics about the customer and relationship, or a logical business explanation. • Transactions by a...
2. reg:fincen_2003_sar_narrative:p18:c26 (cos 0.699, bm25 7.95)  BM has also filed numerous CTRs on cash deposits conducted on behalf of the two companies. Our previous and recent reviews of the customers’ activities revealed the cash deposits and wire transfers involving the companies might be consistent with money laundering. Because BM was unable to determine ...
3. reg:fatf_2020_va_red_flags:p19:c37 (cos 0.715, bm25 6.90)  Section 2 of the Report covers the progress of implementation of the revised Standards since June 2019. Case Study 13. Bitcoin dealer operating unlicensed money transmitting businesses (cross-border elements) In April 2019, the defendant received a sentence of two years in prison for operating an un...

## fatf_4: Why is it odd when a customer parks crypto at an exchange and straight away moves it to their own offline storage?

expected doc `fatf_2020_va_red_flags`, must_contain ['private wallet']

Gold chunks (satisfy the rule): ['reg:fatf_2020_va_red_flags:p8:c10']
- best gold rank in full bm25 ranking: 25
- best gold rank in full dense ranking: 24
- best gold rank in full hybrid ranking: 19
- best gold rank in full hybrid+rerank ranking: 10
Looser rule (expected doc, regex `private wallet|unhosted`) best rank per mode:
- bm25: 18
- dense: 11
- hybrid: 8
- hybrid+rerank: 10
Other chunks in the expected document that match the looser rule: ['reg:fatf_2020_va_red_flags:p11:c19']

Gold chunk reg:fatf_2020_va_red_flags:p8:c10 (258 tokens, 6 sentences); matching sentence index [0] (0 = start, 5 = end of chunk):
  > • Depositing VAs at an exchange and then often immediately – o withdrawing the VAs without ad ditional exchange activity to other VAs, which is an unnecessary step and incurs transaction fees; o converting the VAs to multiple types of VA s, again incurring additional transaction fees , but without logical business explanation (e.g. portfolio diversification); or o withdrawing the VAs from a VASP immediately to a private wallet. This effectively turns the exchange/VASP into an ML mixer.
  previous chunk reg:fatf_2020_va_red_flags:p7:c9 ends: ... other VAs, which is an unnecessary step and incurs transaction fees; o converting the VAs to multiple types of VA s, again incurring additional transaction fees , but without logical business explanation (e.g. portfolio
  next chunk reg:fatf_2020_va_red_flags:p8:c11 starts: In various instances, the individuals shared the same resid ential address; and most of the VA addresses were accessed from the same IP address – indicating the potential use of money mules by professional money laundere...

Top 3 hybrid results actually returned:
1. reg:fincen_2019_cvc_advisory:p8:c18 (cos 0.701, bm25 4.46)  A customer’s transactions are initiated from IP addresses associated with Tor. Blockchain analytics indicate that the wallet transferring CVC to the exchange has a suspicious source or sources of funds, such as a darknet marketplace. A transaction makes use of mixing and tumbling services, suggestin...
2. reg:fincen_2019_cvc_advisory:p8:c19 (cos 0.670, bm25 4.14)  A customer utilizes a CVC exchanger or foreign-located MSB in a high-risk jurisdiction lacking, or known to have inadequate AML/CFT regulations for CVC entities, including inadequate KYC or customer due diligence measures. A customer directs large numbers of CVC transactions to CVC entities in juris...
3. reg:fatf_2020_va_red_flags:p11:c18 (cos 0.679, bm25 3.58)  For example, the use of a hardware or paper wallet may be legitimate as a way to secure VAs against thefts. Again, the presence of these indicators should be considered in the context of other characteristics about the customer and relationship, or a logical business explanation. • Transactions by a...

## Diagnosis (written after reading the output above; the questions were not edited)

- **cvc_2: paraphrase distance and a too-strict must_contain, not chunking.** The intended sentence (banks may identify P2P exchangers through funnel-account activity) sits in the middle of a chunk that also starts with a footnote fragment; its text is not split across chunks. The question describes an 'unlicensed coin-for-cash swapping business' while the text says 'P2P exchangers'. The intended chunk ranks 118 (BM25), 54 (dense), 87 (hybrid). A different chunk of the same document, the red flag about a customer 'acting as an unregistered MSB', is a reasonable answer to the question, and it ranks 3 in dense (20 hybrid, 9 reranked). The must_contain rule counts only the funnel-account sentence.
- **fatf_4: paraphrase distance plus a diluted chunk, with the passage straddling a chunk boundary.** The question says 'crypto', 'parks' and 'offline storage'; the text says 'VAs', 'VASP' and 'private wallet' (BM25 has almost no overlapping words, and extraction splits words such as 'ad ditional' and 'VA s'). The relevant bullet starts at the end of the previous chunk and is repeated whole at the start of the gold chunk because of the overlap, so the gold chunk does contain it, but the bullet is one of six sentences, so the chunk embedding is diluted. The regex itself is not too strict: only that chunk has the 'private wallet' passage. The gold chunk ranks 25 (BM25), 24 (dense), 19 (hybrid) and 10 (reranked).


"""Phase 5a step 0: claims have no quote field; only regulatory background may carry a quote."""
from src.agents.checks import check_item
from src.agents.memo import MemoClaim, RegulatoryBackground
from src.agents.state import Claim, LedgerEntry, RegReference, WriterOutput


def test_claim_models_have_no_quote_field():
    assert "quote" not in Claim.model_fields and "quote" not in MemoClaim.model_fields
    assert "quote" in RegReference.model_fields and "quote" in RegulatoryBackground.model_fields
    # an extra quote value is not stored
    assert "quote" not in Claim(text="x", evidence_ids=["a"], quote="should be ignored").model_dump()


def test_writer_json_schema_offers_quote_only_for_background():
    schema = WriterOutput.model_json_schema()
    claim_props = schema["$defs"]["Claim"]["properties"]
    ref_props = schema["$defs"]["RegReference"]["properties"]
    assert "quote" not in claim_props and "quote" in ref_props


LEDGER = {
    "score:1": LedgerEntry(tool="score_transaction", output={"evidence_id": "score:1", "score": 0.9, "reasons": []}),
    "reg:d:p1:c0": LedgerEntry(tool="get_regulation_chunk", output={"evidence_id": "reg:d:p1:c0", "quotable": True,
                                                                     "text": "Entities must file reports promptly."}),
}


def fetch(eid):
    return LEDGER[eid].output


def test_a_claim_with_an_inline_quote_is_rejected_but_background_may_quote():
    quoted = 'The model gave a raw score of 0.9 and the text says "Entities must file reports promptly" in general.'
    reasons = check_item(quoted, ["score:1", "reg:d:p1:c0"], None, LEDGER, fetch, allow_quotes=False)
    assert any("claims cannot carry quotes" in r for r in reasons)
    assert check_item("Entities must file reports.", ["reg:d:p1:c0"], "must file reports promptly", LEDGER, fetch,
                      allow_quotes=True) == []
    # a plain claim without any quote still passes
    assert check_item("The model gave a raw score of 0.9.", ["score:1"], None, LEDGER, fetch, allow_quotes=False) == []

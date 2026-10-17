import pytest

from src.agents.checks import check_item, claim_numbers, number_matches
from src.agents.llm import tool_specs_for
from src.agents.state import LedgerEntry
from src.tools import TOOLS

SCORE = {"tool": "score_transaction", "output": {
    "tx_id": 4, "time_step": 1, "score": 0.951234, "alert_threshold": 0.258181, "evidence_id": "score:4",
    "reasons": [{"feature": "f57", "direction": "raises illicit score", "shap_value": 1.770439},
                {"feature": "f90", "direction": "lowers illicit score", "shap_value": -0.5435}]}}
TYPO = {"tool": "match_typology", "output": {"evidence_id": "typology:4", "matches": [
    {"pattern": "fan_in", "evidence_direction": "lowers_risk", "observed_illicit_rate": 0.018645, "retrieval_query": None}]}}
CHUNK_OK = {"tool": "get_regulation_chunk", "output": {"evidence_id": "reg:d:p1:c0", "text": "Entities must file reports promptly.", "quotable": True}}
CHUNK_OCR = {"tool": "get_regulation_chunk", "output": {"evidence_id": "reg:o:p1:c0", "text": "Entities must file reports promptly.", "quotable": False}}


def ledger():
    return {"score:4": LedgerEntry(**SCORE), "typology:4": LedgerEntry(**TYPO), "reg:d:p1:c0": LedgerEntry(**CHUNK_OK),
            "reg:o:p1:c0": LedgerEntry(**CHUNK_OCR), "error:x:y": LedgerEntry(tool="x", output={"error": "boom"})}


def fetch(eid):
    return ledger()[eid].output if eid in ledger() else {"error": "unknown"}


def check(text, eids, quote=None):
    return check_item(text, eids, quote, ledger(), fetch)


def test_clean_claim_passes():
    assert check("The model gave a raw score of 0.9512 at time step 1.", ["score:4"]) == []


def test_uncited_and_unknown_and_error_evidence():
    assert "uncited" in check("Something.", [])[0]
    assert "not in the evidence ledger" in check("Something.", ["score:999"])[0]
    assert "tool error" in check("Something.", ["error:x:y"])[0]


def test_numbers_must_match_cited_output_with_rounding():
    assert check("raw score 0.95 at step 1", ["score:4"]) == []
    assert check("a threshold of 0.2582", ["score:4"]) == []
    assert "numbers not found" in check("a score of 0.97", ["score:4"])[0]
    assert "numbers not found" in check("the model scored 7 neighbours", ["score:4"])[0]
    assert check("an illicit rate of 1.86%", ["typology:4"]) == []  # percent of 0.018645
    assert check("SHAP value -0.5435", ["score:4"]) == []


def test_numbers_helpers():
    assert claim_numbers("feature f57 and step 41, 12,345 and 25.7%") == [(41.0, 0, False), (12345.0, 0, False), (25.7, 1, True)]
    assert number_matches(0.95, 2, False, [0.951234])
    assert not number_matches(0.96, 2, False, [0.951234])


def test_lowers_risk_pattern_never_suspicious():
    assert "lowers_risk pattern fan_in" in check("The fan_in pattern is suspicious.", ["typology:4"])[0]
    assert "lowers_risk pattern fan_in" in check("A fan-in structure is a red flag for layering.", ["typology:4"])[0]
    assert check("The fan_in pattern matched; it is not treated as suspicious.", ["typology:4"]) == []
    assert check("The fan_in pattern matched and is associated with a lower illicit rate.", ["typology:4"]) == []


def test_quotes_must_be_verbatim_and_from_quotable_chunks():
    assert check("Reports are due.", ["reg:d:p1:c0"], quote="must file reports promptly") == []
    assert check("Reports are due.", ["reg:d:p1:c0"], quote="MUST  file reports  promptly") == []  # case and spacing
    assert "not a verbatim substring" in check("Reports are due.", ["reg:d:p1:c0"], quote="must file all reports")[0]
    assert "quotable=false" in check("Reports are due.", ["reg:o:p1:c0"], quote="must file reports promptly")[0]
    assert "no cited regulation chunk" in check("Reports are due.", ["score:4"], quote="must file reports promptly")[0]
    inline = 'The text says "Entities must file reports promptly" about duties.'
    assert check(inline, ["reg:d:p1:c0"]) == []
    assert "quotable=false" in check(inline, ["reg:o:p1:c0"])[0]


def test_shap_features_stay_anonymous_and_directionally_consistent():
    assert check("Feature f57 pushed the score up (SHAP value 1.770439).", ["score:4"]) == []
    assert "attributes a meaning" in check("Feature f57 represents the transaction fee.", ["score:4"])[0]
    assert "attributes a meaning" in check("Feature f57 indicates mixing activity.", ["score:4"])[0]
    assert "described as lowering" in check("Feature f57 pushed the score down.", ["score:4"])[0]
    assert "described as raising" in check("Feature f90 pushed the score up.", ["score:4"])[0]


def test_tool_specs_hide_frozen_retrieval_switches():
    spec = tool_specs_for([TOOLS["search_regulations"], TOOLS["get_neighbors"]])
    search = next(s for s in spec if s["name"] == "search_regulations")
    assert set(search["parameters"]["properties"]) == {"query", "k", "jurisdiction"}
    assert search["parameters"]["required"] == ["query"]
    assert all("title" not in v for s in spec for v in s["parameters"]["properties"].values())

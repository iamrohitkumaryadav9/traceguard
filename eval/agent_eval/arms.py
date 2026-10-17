"""Build the three arms (A0, A1, A2) as normalized memo dicts from a raw run record. No LLM, no network."""
from src.agents.build import assemble, deterministic_memo, entry_of
from src.agents.memo import MemoClaim, RECOMMEND_NO_ACTION, RECOMMEND_REVIEW, SARMemo
from src.agents.state import AgentState, RegReference

NOW = "evaluation"


def state_from_raw(raw: dict) -> AgentState:
    """Rebuild just enough state (ledger, step, conflict) to use the memo assembly helpers."""
    st = AgentState(tx_id=raw["tx_id"], time_step=raw.get("time_step"), ledger=raw["ledger"],
                    signal_conflict=raw.get("signal_conflict"),
                    signal_conflict_evidence_ids=raw.get("signal_conflict_evidence_ids", []))
    return st


def normalize(memo: SARMemo, arm: str, raw: dict) -> dict:
    d = memo.model_dump()
    return {
        "arm": arm, "tx_id": raw["tx_id"], "status": raw["status"], "header": d["header"],
        "signal_conflict": d["signal_conflict"], "claims": [{"text": c["text"], "evidence_ids": c["evidence_ids"]} for c in d["activity_description"]],
        "background": [{"paraphrase": b["paraphrase"], "evidence_id": b["evidence_id"], "quote": b["quote"],
                        "quotable": b["quotable"]} for b in d["regulatory_background"]],
        "limitations": d["limitations"], "subject": d["subject"], "recommendation": d["recommendation"],
        "notes": d["notes"], "removed": d["removed_claims"], "verification": d["verification"],
    }


def build_a0(raw: dict) -> dict:
    """ML only: header, a score claim, a health claim, fixed limitations and a rule-based recommendation."""
    st = state_from_raw(raw)
    score, health = entry_of(st.ledger, "score_transaction"), entry_of(st.ledger, "check_model_health")
    if health and health.output.get("status") == "degraded":
        return normalize(deterministic_memo(st, NOW, "cannot_assess"), "A0", raw)
    claims = []
    if score:
        o = score.output
        claims.append(MemoClaim(
            text=f"At time step {o['time_step']} the model gave a raw score of {o['score']} (risk band {o['risk_band']}); "
                 f"the alert threshold is {o['alert_threshold']} and the high-risk threshold is {o['high_threshold']}.",
            evidence_ids=[o["evidence_id"]]))
    if health:
        o = health.output
        wr = o["window_recall"] if o.get("window_recall") is not None else "not available"
        claims.append(MemoClaim(
            text=f"Model health check at time step {o['time_step']}: status {o['status']}; recall in the delayed-label window "
                 f"was {wr} against a reference recall of {o['reference_recall']} over {o['n_illicit_in_window']} labeled illicit nodes.",
            evidence_ids=[o["evidence_id"]]))
    low = bool(score) and score.output["risk_band"] == "low"
    st.signal_conflict, st.signal_conflict_evidence_ids = None, []  # A0 has no conflict sentence, by definition
    st.time_step = None  # ... and no period note, by definition (the limitations are the fixed list only)
    memo = assemble(st, NOW, "a0_ml_only", claims, [], recommendation=RECOMMEND_NO_ACTION if low else RECOMMEND_REVIEW)
    memo.header.time_step = raw.get("time_step")
    return normalize(memo, "A0", raw)


def build_a1(raw: dict) -> dict:
    """The writer's first draft, unfiltered. For runs without a writer it is the same deterministic memo as A2."""
    fd = raw.get("first_draft")
    if fd is None:
        return {**build_a2(raw), "arm": "A1"}
    st = state_from_raw(raw)
    claims = [MemoClaim(text=c["text"], evidence_ids=c["evidence_ids"]) for c in fd["activity_description"]]
    refs = [RegReference(paraphrase=r["paraphrase"], evidence_id=r["evidence_id"], quote=r.get("quote"))
            for r in fd["regulatory_background"]]
    memo = assemble(st, NOW, "a1_first_draft", claims, refs)
    memo.signal_conflict = fd.get("signal_conflict")  # exactly what the writer wrote, not the required sentence
    return normalize(memo, "A1", raw)


def build_a2(raw: dict) -> dict:
    return normalize(SARMemo.model_validate(raw["memo"]), "A2", raw)


ARMS = {"A0": build_a0, "A1": build_a1, "A2": build_a2}

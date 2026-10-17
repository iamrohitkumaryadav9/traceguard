"""Memo assembly and the deterministic claims. Every fact here is read from the evidence ledger by code."""
from typing import Dict, List, Optional

from src.agents.memo import (
    RECOMMEND_NO_ACTION,
    RECOMMEND_REVIEW,
    EvidenceRow,
    MemoClaim,
    MemoHeader,
    RegulatoryReference,
    SARMemo,
)
from src.agents.state import AgentState, LedgerEntry


def entry_of(ledger: Dict[str, LedgerEntry], tool) -> Optional[LedgerEntry]:
    for e in ledger.values():
        if e.tool == tool and "error" not in e.output:
            return e
    return None


def deterministic_claims(state: AgentState, include_score=True) -> List[MemoClaim]:
    """Claims generated from the ledger by templates (they pass the critic's deterministic checks by design)."""
    L, claims = state.ledger, []
    score, health, typ = entry_of(L, "score_transaction"), entry_of(L, "check_model_health"), entry_of(L, "match_typology")
    if score and include_score:
        o = score.output
        claims.append(MemoClaim(
            text=f"At time step {o['time_step']} the model gave a raw score of {o['score']} (risk band {o['risk_band']}); "
                 f"the alert threshold is {o['alert_threshold']} and the high-risk threshold is {o['high_threshold']}.",
            evidence_ids=[o["evidence_id"]]))
        for r in o.get("reasons", [])[:3]:
            word = "up" if r["direction"].startswith("raises") else "down"
            claims.append(MemoClaim(
                text=f"Feature {r['feature']} pushed the score {word} (SHAP value {r['shap_value']}); "
                     f"feature names are anonymized in the dataset.",
                evidence_ids=[o["evidence_id"]]))
    if health:
        o = health.output
        wr = f"{o['window_recall']}" if o.get("window_recall") is not None else "not available"
        claims.append(MemoClaim(
            text=f"Model health check at time step {o['time_step']}: status {o['status']}; recall in the delayed-label window "
                 f"was {wr} against a reference recall of {o['reference_recall']} "
                 f"over {o['n_illicit_in_window']} labeled illicit nodes.",
            evidence_ids=[o["evidence_id"]]))
    if typ:
        o = typ.output
        for m in o.get("matches", []):
            tail = {"lowers_risk": "this pattern lowers rather than raises risk and is not treated as suspicious",
                    "raises_risk": "this pattern raises risk in the training-period data",
                    "uninformative": "this pattern is uninformative in the training-period data"}[m["evidence_direction"]]
            claims.append(MemoClaim(
                text=f"Structural pattern {m['pattern']} matched ({m['evidence']}); labeled nodes with this pattern in the "
                     f"training period had an illicit rate of {m['observed_illicit_rate']} against a baseline of "
                     f"{m['baseline_illicit_rate']}, so {tail}.",
                evidence_ids=[o["evidence_id"]]))
    return claims


def make_header(state: AgentState, now_iso: str, rated=True) -> MemoHeader:
    score, health = entry_of(state.ledger, "score_transaction"), entry_of(state.ledger, "check_model_health")
    so = score.output if score else {}
    ho = health.output if health else {}
    return MemoHeader(
        tx_id=state.tx_id, time_step=state.time_step,
        risk_band=so.get("risk_band") if rated else None,
        score=so.get("score") if rated else None,
        calibrated_probability=so.get("calibrated_probability") if rated else None,
        thresholds={"alert": so.get("alert_threshold"), "high": so.get("high_threshold")},
        model_health={"status": ho.get("status", "not checked"), "window_recall": ho.get("window_recall"),
                      "reference_recall": ho.get("reference_recall"), "n_illicit_in_window": ho.get("n_illicit_in_window")},
        generated_at=now_iso)


def evidence_table(state: AgentState, claims, refs) -> List[EvidenceRow]:
    ids = []
    for c in claims:
        ids += c.evidence_ids
    for r in refs:
        ids.append(r.evidence_id)
    seen, rows = set(), []
    for eid in ids:
        if eid in state.ledger and eid not in seen:
            seen.add(eid)
            rows.append(EvidenceRow(evidence_id=eid, tool=state.ledger[eid].tool))
    return rows


def regulatory_references(state: AgentState, refs) -> List[RegulatoryReference]:
    out = []
    for r in refs:
        e = state.ledger.get(r.evidence_id)
        if e is None or "error" in e.output:
            continue
        o = e.output
        out.append(RegulatoryReference(
            paraphrase=r.paraphrase, source_doc=f"{o.get('publisher')} {o.get('year')}: {o.get('title')}",
            doc_id=o.get("doc_id", ""), evidence_id=r.evidence_id, quotable=bool(o.get("quotable", True)),
            quote=r.quote))
    return out


def assemble(state: AgentState, now_iso, status, claims, refs, verification="not applicable", removed=None,
             notes=None, rated=True, recommendation=RECOMMEND_REVIEW) -> SARMemo:
    return SARMemo(
        header=make_header(state, now_iso, rated), activity_description=list(claims),
        evidence_table=evidence_table(state, claims, refs), regulatory_references=regulatory_references(state, refs),
        recommendation=recommendation, status=status, verification=verification,
        removed_claims=removed or [], notes=notes or [])


def deterministic_memo(state: AgentState, now_iso, status, notes=None) -> SARMemo:
    """Memo for the runs that end without (or before) the writer: cannot_assess, no_action, budget, error."""
    if status == "cannot_assess":
        health = entry_of(state.ledger, "check_model_health").output
        text = (f"The model cannot be relied on at time step {state.time_step}: in the delayed-label window its recall was "
                f"{health['window_recall']} against a reference of {health['reference_recall']}, so no risk assessment is given "
                f"for this transaction.")
        claims = [MemoClaim(text=text, evidence_ids=[health["evidence_id"]])]
        return assemble(state, now_iso, status, claims, [], notes=notes, rated=False, recommendation=RECOMMEND_REVIEW)
    if status == "no_action_model_based":
        claims = deterministic_claims(state)
        score = entry_of(state.ledger, "score_transaction").output
        typ = entry_of(state.ledger, "match_typology")
        claims = [c for c in claims if "Feature" not in c.text]  # keep the memo short
        claims.append(MemoClaim(
            text="No structural pattern with a risk-raising direction matched this transaction.",
            evidence_ids=[typ.output["evidence_id"]] if typ else [score["evidence_id"]]))
        return assemble(state, now_iso, status, claims, [], notes=notes, recommendation=RECOMMEND_NO_ACTION)
    # budget_exceeded, agent_error, unknown_transaction: whatever the ledger already supports
    claims = deterministic_claims(state)
    return assemble(state, now_iso, status, claims, [], notes=notes, recommendation=RECOMMEND_REVIEW,
                    rated=entry_of(state.ledger, "score_transaction") is not None)

"""The SAR memo schema. The LLM only supplies claims and regulatory paraphrases; everything else is code."""
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

NOT_AVAILABLE = "not available in dataset"
NotAvailable = Literal["not available in dataset"]

RECOMMEND_REVIEW = "refer for human review"
RECOMMEND_NO_ACTION = "no action suggested, model-based only"

LIMITATIONS = [
    "The Elliptic dataset has no wallet addresses, names, amounts or dates, so none of these appear in this memo.",
    "The model's recall collapsed after time step 43 (a dark-market shutdown), and the model-health check only "
    "detects such a collapse several steps later, so scores at later time steps may be unreliable.",
    "Regulation retrieval is weak (hybrid hit@3 about 0.6 on a small test set): retrieved passages are leads for "
    "an analyst, not verified authorities, and text from the India document comes from OCR.",
    "Scores are raw gradient-boosting outputs, not calibrated probabilities; the calibrated value shifts with "
    "the prevalence of illicit activity.",
    "Structural patterns are resemblances only: Elliptic has no typology labels.",
    "The model-health monitor uses labels with a 2-step lag and a 3-step window, so it cannot flag a collapse in its "
    "first steps. On this dataset it first flags at step 47.",
    "This is a draft for a human analyst. It does not decide whether a report should be filed.",
]

# Added to the limitations of memos at time step >= SHUTDOWN_STEP. Pure text: it does not change any gating.
PERIOD_NOTE = (
    "Dataset-specific note, based on hindsight from the published test results: model performance collapsed from time "
    "step 43 onward. A deployed system would not know this in advance."
)


class SubjectFields(BaseModel):
    names: NotAvailable = NOT_AVAILABLE
    addresses: NotAvailable = NOT_AVAILABLE
    amounts: NotAvailable = NOT_AVAILABLE
    dates: NotAvailable = NOT_AVAILABLE


class MemoClaim(BaseModel):
    text: str
    evidence_ids: List[str]
    quote: Optional[str] = None  # verbatim text from a quotable regulation chunk, if any


class EvidenceRow(BaseModel):
    evidence_id: str
    tool: str


class RegulatoryBackground(BaseModel):
    """General regulatory background retrieved because a pattern matched. It is context, not evidence about
    this transaction."""
    paraphrase: str
    source_doc: str
    doc_id: str
    evidence_id: str
    quotable: bool
    quote: Optional[str] = None


class MemoHeader(BaseModel):
    tx_id: int
    time_step: Optional[int]
    risk_band: Optional[Literal["low", "medium", "high"]]  # None when the model cannot be relied on
    score: Optional[float]  # raw score, withheld when the model cannot be relied on
    score_scale: str = "raw_lightgbm_predict_proba"
    calibrated_probability: Optional[float]
    thresholds: Dict[str, Optional[float]]
    model_health: Dict[str, object]
    generated_at: str


class SARMemo(BaseModel):
    header: MemoHeader
    subject: SubjectFields = Field(default_factory=SubjectFields)
    signal_conflict: Optional[str] = None  # fixed template sentence, built by code from evidence; null when signals agree
    signal_conflict_evidence_ids: List[str] = Field(default_factory=list)
    activity_description: List[MemoClaim]
    evidence_table: List[EvidenceRow]
    regulatory_background: List[RegulatoryBackground]
    limitations: List[str] = Field(default_factory=lambda: list(LIMITATIONS))
    recommendation: Literal["refer for human review", "no action suggested, model-based only"]
    status: str
    verification: Literal["verified", "partially verified", "not applicable"]
    removed_claims: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


def memo_to_markdown(m: SARMemo):
    h = m.header
    lines = [
        f"### Memo for transaction {h.tx_id} (time step {h.time_step})", "",
        f"- status: **{m.status}**, verification: **{m.verification}**",
        f"- risk band: {h.risk_band or 'not rated'}; raw score: {h.score if h.score is not None else 'withheld'}; "
        f"calibrated probability: {h.calibrated_probability if h.calibrated_probability is not None else 'n/a'}",
        f"- thresholds: {h.thresholds}", f"- model health: {h.model_health}",
        f"- subject fields: names / addresses / amounts / dates = {m.subject.names}", ""]
    if m.signal_conflict:
        lines += [f"**{m.signal_conflict}** `[{', '.join(m.signal_conflict_evidence_ids)}]`", ""]
    lines += ["**Activity description**", ""]
    for i, c in enumerate(m.activity_description, 1):
        q = f' Quote: "{c.quote}"' if c.quote else ""
        lines.append(f"{i}. {c.text}{q} `[{', '.join(c.evidence_ids)}]`")
    if not m.activity_description:
        lines.append("(no verified claims)")
    lines += ["", "**Regulatory background (context, not evidence about this transaction)**", ""]
    for r in m.regulatory_background:
        q = f' Quote: "{r.quote}"' if r.quote else ""
        lines.append(f"- {r.paraphrase}{q} ({r.source_doc}; `{r.evidence_id}`; quotable: {r.quotable})")
    if not m.regulatory_background:
        lines.append("- none")
    lines += ["", "**Evidence table**", ""] + [f"- `{e.evidence_id}` from {e.tool}" for e in m.evidence_table]
    if m.removed_claims:
        lines += ["", "**Removed by the critic**", ""] + [f"- {x}" for x in m.removed_claims]
    if m.notes:
        lines += ["", "**Notes**", ""] + [f"- {x}" for x in m.notes]
    lines += ["", "**Limitations**", ""] + [f"- {x}" for x in m.limitations]
    lines += ["", f"**Recommendation:** {m.recommendation}"]
    return "\n".join(lines)

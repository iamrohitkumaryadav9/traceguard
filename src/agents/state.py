"""Agent state. The evidence ledger is the only source of facts: nothing enters a memo unless it is in it."""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from src.agents.memo import SARMemo


class LedgerEntry(BaseModel):
    tool: str
    args: dict = Field(default_factory=dict)
    output: dict


class Claim(BaseModel):
    """A memo claim. It has no quote field: only regulatory background may carry a quote."""
    text: str
    evidence_ids: List[str] = Field(default_factory=list)


class RegReference(BaseModel):
    paraphrase: str
    evidence_id: str
    quote: Optional[str] = None


class WriterOutput(BaseModel):
    """What the writer LLM fills in. Everything else in the memo is built by code."""
    signal_conflict: Optional[str] = None  # must be the required sentence copied unchanged, or null when none is required
    activity_description: List[Claim]
    regulatory_background: List[RegReference] = Field(default_factory=list)


class SummaryDraft(BaseModel):
    """What the v3 writer LLM fills in (the ONLY fields offered to it): everything else is rendered by code."""
    executive_summary: List[Claim]
    next_checks: List[str] = Field(default_factory=list)


class DraftV3(BaseModel):
    """The v3 draft the critic checks: the LLM's summary and next checks plus the code-built parts attached by code."""
    executive_summary: List[Claim] = Field(default_factory=list)
    next_checks: List[str] = Field(default_factory=list)
    facts: List[Claim] = Field(default_factory=list)  # code-built activity claims
    rendered_background: List[RegReference] = Field(default_factory=list)  # code-rendered excerpts or labels


class CriticVerdict(BaseModel):
    round: int
    item: str  # "claim 1", "reference 2"
    text: str
    ok: bool
    deterministic_reasons: List[str] = Field(default_factory=list)
    llm_verdict: Optional[str] = None  # supported / unsupported / overstated
    llm_reason: Optional[str] = None


class AgentState(BaseModel):
    tx_id: int
    time_step: Optional[int] = None
    ledger: Dict[str, LedgerEntry] = Field(default_factory=dict)
    tool_call_count: int = 0  # investigator tool calls (cap MAX_TOOL_CALLS)
    search_count: int = 0
    chunk_fetch_count: int = 0
    llm_call_count: int = 0
    critic_loops: int = 0
    signal_conflict: Optional[str] = None  # computed in triage, deterministic
    signal_conflict_evidence_ids: List[str] = Field(default_factory=list)
    status: str = "started"
    memo: Optional[SARMemo] = None
    draft: Optional[WriterOutput] = None
    first_draft: Optional[WriterOutput] = None  # the writer's round-0 draft, recorded for the evaluation (arm A1)
    draft3: Optional[DraftV3] = None  # design v3 draft
    first_draft3: Optional[DraftV3] = None
    critic_verdicts: List[CriticVerdict] = Field(default_factory=list)
    critic_feedback: List[str] = Field(default_factory=list)
    trace: List[dict] = Field(default_factory=list)
    nodes_visited: List[str] = Field(default_factory=list)
    error: Optional[str] = None

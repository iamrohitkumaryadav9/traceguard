"""Agent state. The evidence ledger is the only source of facts: nothing enters a memo unless it is in it."""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from src.agents.memo import SARMemo


class LedgerEntry(BaseModel):
    tool: str
    args: dict = Field(default_factory=dict)
    output: dict


class Claim(BaseModel):
    text: str
    evidence_ids: List[str] = Field(default_factory=list)
    quote: Optional[str] = None


class RegReference(BaseModel):
    paraphrase: str
    evidence_id: str
    quote: Optional[str] = None


class WriterOutput(BaseModel):
    """What the writer LLM fills in. Everything else in the memo is built by code."""
    activity_description: List[Claim]
    regulatory_references: List[RegReference] = Field(default_factory=list)


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
    status: str = "started"
    memo: Optional[SARMemo] = None
    draft: Optional[WriterOutput] = None
    critic_verdicts: List[CriticVerdict] = Field(default_factory=list)
    critic_feedback: List[str] = Field(default_factory=list)
    trace: List[dict] = Field(default_factory=list)
    nodes_visited: List[str] = Field(default_factory=list)
    error: Optional[str] = None

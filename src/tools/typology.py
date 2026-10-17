from typing import Dict, List, Union

from pydantic import BaseModel

from src.tools.context import CHAIN_MIN, get_context
from src.tools.schemas import ToolError, unknown_tx

NAME = "match_typology"
DESCRIPTION = (
    "Structural pattern flags for a transaction: fan_in, fan_out, chain. Fan thresholds are the 99th "
    "percentile of in/out degree over time steps 1-29 (minimum 3); chain is flagged at chain_len >= 4. "
    "These are structural resemblances only. Each matched pattern carries a retrieval_query for the "
    "later regulation search."
)

CAVEAT = "structural resemblance only; Elliptic has no typology labels"
QUERIES = {
    "fan_in": "many incoming transfers consolidated into a single account, aggregation of funds from multiple sources",
    "fan_out": "funds split from a single account across many recipients, distribution of funds to multiple accounts",
    "chain": "sequential transfers through a chain of intermediary accounts, layering of funds",
}


class MatchTypologyInput(BaseModel):
    tx_id: int


class PatternMatch(BaseModel):
    pattern: str
    evidence: str
    retrieval_query: str
    caveat: str = CAVEAT


class MatchTypologyOutput(BaseModel):
    tx_id: int
    time_step: int
    flags: Dict[str, bool]
    in_degree: int
    out_degree: int
    chain_len: int
    thresholds: Dict[str, int]
    matches: List[PatternMatch]
    evidence_id: str


def match_typology(args: MatchTypologyInput) -> Union[MatchTypologyOutput, ToolError]:
    ctx = get_context()
    i = ctx.idx(args.tx_id)
    if i is None:
        return unknown_tx(NAME, args.tx_id)
    fan_in, fan_out, chain = (bool(a[0]) for a in ctx.pattern_flags([i]))
    flags = {"fan_in": fan_in, "fan_out": fan_out, "chain": chain}
    ev = {
        "fan_in": f"in_degree {int(ctx.in_deg[i])} >= {ctx.fan_in_thr}",
        "fan_out": f"out_degree {int(ctx.out_deg[i])} >= {ctx.fan_out_thr}",
        "chain": f"chain_len {int(ctx.chain_len[i])} >= {CHAIN_MIN}",
    }
    return MatchTypologyOutput(
        tx_id=args.tx_id, time_step=int(ctx.time_step[i]), flags=flags, in_degree=int(ctx.in_deg[i]),
        out_degree=int(ctx.out_deg[i]), chain_len=int(ctx.chain_len[i]),
        thresholds={"fan_in": ctx.fan_in_thr, "fan_out": ctx.fan_out_thr, "chain_len": CHAIN_MIN},
        matches=[PatternMatch(pattern=p, evidence=ev[p], retrieval_query=QUERIES[p]) for p, on in flags.items() if on],
        evidence_id=f"typology:{args.tx_id}",
    )

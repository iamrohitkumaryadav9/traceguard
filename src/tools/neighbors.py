from typing import List, Literal, Union

from pydantic import BaseModel, Field

from src.tools.context import get_context
from src.tools.schemas import ToolError, r6, unknown_tx

NAME = "get_neighbors"
DESCRIPTION = (
    "Neighbors of a transaction within the same time step, up to 2 hops, ordered by model score (descending). "
    "direction 'in' follows funding transactions (predecessors) only, 'out' follows spending transactions "
    "(successors) only, 'both' returns both chains and marks a node 'both' if reachable either way. Hop "
    "expansion follows one direction at a time (in-in or out-out)."
)


class GetNeighborsInput(BaseModel):
    tx_id: int
    hops: Literal[1, 2] = 1
    direction: Literal["in", "out", "both"] = "both"
    limit: int = Field(default=25, ge=1, le=200)


class Neighbor(BaseModel):
    tx_id: int
    hop: int
    direction: Literal["in", "out", "both"]
    score: float
    alert: bool


class GetNeighborsOutput(BaseModel):
    tx_id: int
    time_step: int
    hops: int
    direction: str
    total_found: int
    truncated: bool
    neighbors: List[Neighbor]
    evidence_id: str


def expand(adj, i, hops):
    """Node index -> minimal hop, following `adj` (CSR) for up to `hops` steps. Excludes i."""
    seen, frontier = {i: 0}, [i]
    for h in range(1, hops + 1):
        nxt = []
        for u in frontier:
            for v in adj.indices[adj.indptr[u]:adj.indptr[u + 1]]:
                v = int(v)
                if v not in seen:
                    seen[v] = h
                    nxt.append(v)
        frontier = nxt
    del seen[i]
    return seen


def get_neighbors(args: GetNeighborsInput) -> Union[GetNeighborsOutput, ToolError]:
    ctx = get_context()
    i = ctx.idx(args.tx_id)
    if i is None:
        return unknown_tx(NAME, args.tx_id)
    in_map = expand(ctx.AT, i, args.hops) if args.direction in ("in", "both") else {}
    out_map = expand(ctx.A, i, args.hops) if args.direction in ("out", "both") else {}
    found = []
    for j in set(in_map) | set(out_map):
        d = "both" if (j in in_map and j in out_map) else "in" if j in in_map else "out"
        hop = min(in_map.get(j, 99), out_map.get(j, 99))
        found.append((j, hop, d))
    # score descending, tx_id ascending for determinism
    found.sort(key=lambda x: (-ctx.score[x[0]], int(ctx.ids[x[0]])))
    shown = found[:args.limit]
    return GetNeighborsOutput(
        tx_id=args.tx_id, time_step=int(ctx.time_step[i]), hops=args.hops, direction=args.direction,
        total_found=len(found), truncated=len(found) > args.limit,
        neighbors=[Neighbor(tx_id=int(ctx.ids[j]), hop=h, direction=d, score=r6(ctx.score[j]),
                            alert=bool(ctx.score[j] >= ctx.alert_threshold)) for j, h, d in shown],
        evidence_id=f"neighbors:{args.tx_id}:{args.hops}:{args.direction}",
    )

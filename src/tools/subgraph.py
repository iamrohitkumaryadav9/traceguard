from typing import Union

from pydantic import BaseModel, Field

from src.tools.context import get_context
from src.tools.neighbors import expand
from src.tools.schemas import ToolError, r6, unknown_tx

NAME = "subgraph_summary"
DESCRIPTION = (
    "Summary statistics of the subgraph within `radius` undirected hops of a transaction (same time step): "
    "node/edge counts, directed density, focal and maximum degrees inside the subgraph, how many nodes "
    "(including the focal node) the model alerts on, the focal node's chain length and its weakly connected "
    "component size."
)


class SubgraphSummaryInput(BaseModel):
    tx_id: int
    radius: int = Field(default=2, ge=1, le=3)


class SubgraphSummaryOutput(BaseModel):
    tx_id: int
    time_step: int
    radius: int
    n_nodes: int
    n_edges: int
    density: float
    focal_in_degree: int
    focal_out_degree: int
    max_in_degree: int
    max_out_degree: int
    n_alerted: int
    share_alerted: float
    focal_chain_len: int
    wcc_size: int
    evidence_id: str


def subgraph_summary(args: SubgraphSummaryInput) -> Union[SubgraphSummaryOutput, ToolError]:
    ctx = get_context()
    i = ctx.idx(args.tx_id)
    if i is None:
        return unknown_tx(NAME, args.tx_id)
    nodes = sorted([i, *expand(ctx.S, i, args.radius)])
    sub = ctx.A[nodes][:, nodes]
    n = len(nodes)
    in_d, out_d = sub.getnnz(axis=0), sub.getnnz(axis=1)
    pos = nodes.index(i)
    n_alert = int((ctx.score[nodes] >= ctx.alert_threshold).sum())
    return SubgraphSummaryOutput(
        tx_id=args.tx_id, time_step=int(ctx.time_step[i]), radius=args.radius, n_nodes=n, n_edges=int(sub.nnz),
        density=r6(sub.nnz / (n * (n - 1)) if n > 1 else 0.0),
        focal_in_degree=int(in_d[pos]), focal_out_degree=int(out_d[pos]),
        max_in_degree=int(in_d.max()), max_out_degree=int(out_d.max()),
        n_alerted=n_alert, share_alerted=r6(n_alert / n), focal_chain_len=int(ctx.chain_len[i]),
        wcc_size=int(ctx.wcc_size[i]), evidence_id=f"subgraph:{args.tx_id}:{args.radius}",
    )

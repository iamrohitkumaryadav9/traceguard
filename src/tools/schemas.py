"""Shared output models. No model here may expose a ground-truth label."""
from typing import Literal, Optional

from pydantic import BaseModel


class ToolError(BaseModel):
    error: Literal["unknown_tx_id", "invalid_argument"]
    message: str
    tool: str
    tx_id: Optional[int] = None
    evidence_id: str


def unknown_tx(tool, tx_id):
    return ToolError(error="unknown_tx_id", message=f"transaction {tx_id} not found", tool=tool,
                     tx_id=tx_id, evidence_id=f"error:{tool}:{tx_id}")


def r6(x):
    return None if x is None else round(float(x), 6)

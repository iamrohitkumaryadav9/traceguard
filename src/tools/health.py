from typing import List, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel

from src.tools.context import get_context
from src.tools.schemas import ToolError, r6

NAME = "check_model_health"
DESCRIPTION = (
    "Delayed-label model health check at a given time step. Uses only labels from steps <= time_step - 2, "
    "pools the 3 steps ending there, and compares the model's recall at the alert threshold with its recall "
    "on validation steps 30-34. Returns aggregate numbers only. Status: ok, degraded (window recall < 0.5 x "
    "reference) or insufficient_data (< 20 illicit nodes in the window)."
)

# FIXED IN ADVANCE: do not tune. These were designed after the post-shutdown failure was observed.
LAG = 2
WINDOW = 3
MIN_ILLICIT = 20
DEGRADE_RATIO = 0.5


class CheckModelHealthInput(BaseModel):
    time_step: int


class CheckModelHealthOutput(BaseModel):
    time_step: int
    status: Literal["ok", "degraded", "insufficient_data"]
    window_steps: List[int]
    window_recall: Optional[float]
    reference_recall: float
    n_illicit_in_window: int
    lag: int
    evidence_id: str


def check_model_health(args: CheckModelHealthInput) -> Union[CheckModelHealthOutput, ToolError]:
    ctx = get_context()
    t = args.time_step
    steps_available = np.unique(ctx.time_step)
    if t not in steps_available:
        return ToolError(error="invalid_argument", message=f"time_step {t} not in data", tool=NAME,
                         evidence_id=f"error:{NAME}:{t}")
    last = t - LAG
    window = [s for s in range(last - WINDOW + 1, last + 1) if s in steps_available]
    assert all(s <= t - LAG for s in window), "label leakage: window beyond time_step - lag"
    n_ill, n_caught = ctx.window_stats(window)
    recall = n_caught / n_ill if n_ill else None
    if n_ill < MIN_ILLICIT:
        status = "insufficient_data"
    else:
        status = "degraded" if recall < DEGRADE_RATIO * ctx.reference_recall else "ok"
    return CheckModelHealthOutput(
        time_step=t, status=status, window_steps=window, window_recall=r6(recall),
        reference_recall=r6(ctx.reference_recall), n_illicit_in_window=n_ill, lag=LAG,
        evidence_id=f"health:{t}",
    )

from typing import List, Literal, Optional, Union

from pydantic import BaseModel

from src.tools.context import get_context
from src.tools.schemas import ToolError, r6, unknown_tx

NAME = "score_transaction"
DESCRIPTION = (
    "Model risk score for one transaction (raw LightGBM score; alert if >= the cost threshold), a risk band "
    "(low / medium / high) and the top 5 SHAP reasons. Feature names are anonymized: reasons carry the feature "
    "name, group (local / aggregated), direction and SHAP value only."
)


class ScoreTransactionInput(BaseModel):
    tx_id: int


class Reason(BaseModel):
    feature: str
    group: Literal["local", "aggregated", "structural"]
    direction: Literal["raises illicit score", "lowers illicit score"]
    shap_value: float
    feature_value: float


class ScoreTransactionOutput(BaseModel):
    tx_id: int
    time_step: int
    score: float
    score_scale: Literal["raw_lightgbm_predict_proba"] = "raw_lightgbm_predict_proba"
    calibrated_probability: Optional[float] = None
    alert: bool
    alert_threshold: float
    high_threshold: float
    risk_band: Literal["low", "medium", "high"]
    reasons: List[Reason]
    evidence_id: str


def score_transaction(args: ScoreTransactionInput) -> Union[ScoreTransactionOutput, ToolError]:
    ctx = get_context()
    i = ctx.idx(args.tx_id)
    if i is None:
        return unknown_tx(NAME, args.tx_id)
    s = float(ctx.score[i])
    band = "low" if s < ctx.alert_threshold else "medium" if s < ctx.high_threshold else "high"
    reasons = [
        Reason(feature=r["feature"], group=r["group"], direction=r["direction"],
               shap_value=r6(r["shap"]), feature_value=r6(r["value"]))
        for r in ctx.explain_fn(args.tx_id, 5)
    ]
    return ScoreTransactionOutput(
        tx_id=args.tx_id, time_step=int(ctx.time_step[i]), score=r6(s),
        calibrated_probability=r6(ctx.calibrate_fn(s)) if ctx.calibrate_fn else None,
        alert=s >= ctx.alert_threshold, alert_threshold=r6(ctx.alert_threshold),
        high_threshold=r6(ctx.high_threshold), risk_band=band, reasons=reasons,
        evidence_id=f"score:{args.tx_id}",
    )

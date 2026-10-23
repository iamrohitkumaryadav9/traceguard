"""Render logic of the Streamlit UI as plain functions (no Streamlit import): they turn a bundle or API entry into small
view objects, so they can be unit-tested. The page code in app/streamlit_app.py only draws these objects.

Nothing here reads a label: a view is built from a whitelist of memo fields, and `assert_no_labels` is run on every view
(and fails loudly) so that a label-like key can never be rendered."""
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

NOT_AVAILABLE = "not available in dataset"
LABEL_WORDS = {"label", "labels", "class", "ground_truth", "y_true", "stratum"}


@dataclass
class Banner:
    level: str  # success | warning | error | info
    title: str
    text: str


@dataclass
class Chip:
    evidence_id: str
    tool: str
    name: str  # short text on the chip
    text: str  # the cited evidence, readable


@dataclass
class ClaimView:
    text: str
    source: str  # "code" or "llm"
    chips: List[Chip]


@dataclass
class MemoView:
    tx_id: int
    time_step: Optional[int]
    banner: Banner
    risk: List[str]  # lines: band, raw score, thresholds, calibrated estimate
    conflict: Optional[str]
    summary: List[ClaimView]
    claims: List[ClaimView]
    background: List[dict]
    next_checks: List[str]
    limitations: List[str]
    recommendation: str
    subject: Dict[str, str]
    notes: List[str]
    removed: List[str]
    trace: Dict[str, object] = field(default_factory=dict)


def status_banner(status: str, memo: Optional[dict] = None, time_step=None) -> Banner:
    step = time_step if time_step is not None else ((memo or {}).get("header") or {}).get("time_step")
    if status == "verified":
        return Banner("success", "Verified", "Every statement in this memo passed the critic's checks against the cited evidence.")
    if status == "partially_verified":
        return Banner("warning", "Partially verified",
                      "One or more statements failed the critic after two rewrites and were removed. What remains passed the checks; "
                      "see the removed statements below.")
    if status == "cannot_assess":
        return Banner("error", "The model cannot be relied on at this time step",
                      f"At time step {step} the model-health check reports that the model's recent recall has collapsed, so no risk band "
                      "or score is shown and no recommendation about this transaction should be drawn from the model.")
    if status == "no_action_model_based":
        return Banner("info", "No action suggested, model-based only",
                      "The model puts this transaction in the low band and no risk-raising pattern matched, so the memo was built "
                      "without the language model. This rests on the model's score alone.")
    if status == "budget_exceeded":
        return Banner("warning", "Run stopped at a budget cap", "The agent hit one of its call caps; the memo is partial.")
    return Banner("warning", f"Status: {status.replace('_', ' ')}", "This run did not end in a standard state; read the memo with care.")


def conflict_callout(memo: dict) -> Optional[str]:
    """The signal-conflict sentence (built by code from evidence), or None when the signals agree."""
    return memo.get("signal_conflict") or None


def _clean(obj):
    """Defence in depth: even if an API returned them, label-like keys and feature values are never drawn."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if str(k).lower() not in LABEL_WORDS | {"feature_value", "shap_value"}}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def evidence_text(output: dict) -> str:
    """Readable lines for one tool output (the sanitized output as stored in the bundle)."""
    lines = []
    for k, v in _clean(output).items():
        if k == "evidence_id":
            continue
        if isinstance(v, (dict, list)):
            v = json.dumps(v, ensure_ascii=False, separators=(", ", ": "))
        lines.append(f"{k}: {v}")
    return "\n".join(lines)


def evidence_chips(evidence_ids: List[str], evidence: Dict[str, dict]) -> List[Chip]:
    chips = []
    for eid in evidence_ids:
        e = (evidence or {}).get(eid)
        if e is None:
            chips.append(Chip(eid, "", eid, "The text of this evidence is not available in this view."))
        else:
            chips.append(Chip(eid, e.get("tool", ""), eid, evidence_text(e.get("output", {}))))
    return chips


def claim_views(claims: List[dict], evidence: Dict[str, dict]) -> List[ClaimView]:
    return [ClaimView(c["text"], c.get("source", "llm"), evidence_chips(c.get("evidence_ids", []), evidence)) for c in claims]


def risk_lines(memo: dict) -> List[str]:
    h = memo["header"]
    if h.get("risk_band") is None:
        return ["Risk band: not rated (model not reliable at this time step)", "Raw score: withheld"]
    th = h.get("thresholds") or {}
    lines = [f"Risk band: {h['risk_band']}", f"Raw model score: {h['score']} (the raw gradient-boosting output, not a probability)",
             f"Thresholds on the raw scale: alert {th.get('alert')}, high {th.get('high')}"]
    if h.get("calibrated_probability") is not None:
        lines.append(f"Calibrated estimate: {h['calibrated_probability']} (shifts with the prevalence of illicit activity)")
    return lines


def background_views(memo: dict) -> List[dict]:
    return [{"heading": "Context, not evidence about this transaction", "source_doc": b["source_doc"], "text": b["paraphrase"],
             "quote": b.get("quote") if b.get("quotable") else None, "evidence_id": b["evidence_id"], "quotable": b.get("quotable", True)}
            for b in memo.get("regulatory_background", [])]


def memo_view(entry: dict) -> MemoView:
    """Everything the memo page draws, from a whitelist of fields of one entry ({status, memo, evidence, trace_summary, ...})."""
    memo, evidence = entry["memo"], entry.get("evidence") or {}
    subject = {k: v for k, v in (memo.get("subject") or {}).items() if k in ("names", "addresses", "amounts", "dates")}
    subject = {k: subject.get(k, NOT_AVAILABLE) for k in ("names", "addresses", "amounts", "dates")}
    view = MemoView(
        tx_id=memo["header"]["tx_id"], time_step=memo["header"].get("time_step"),
        banner=status_banner(entry["status"], memo), risk=risk_lines(memo), conflict=conflict_callout(memo),
        summary=claim_views(memo.get("executive_summary", []), evidence), claims=claim_views(memo.get("activity_description", []), evidence),
        background=background_views(memo), next_checks=list(memo.get("next_checks", [])), limitations=list(memo.get("limitations", [])),
        recommendation=memo["recommendation"], subject=subject, notes=list(memo.get("notes", [])),
        removed=list(memo.get("removed_claims", [])), trace=dict(entry.get("trace_summary") or {}))
    assert_no_labels(view)
    return view


def label_keys(obj, path="") -> List[str]:
    """Label-like dictionary keys anywhere in a structure (dataclasses are looked into through their __dict__)."""
    found = []
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        obj = vars(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in LABEL_WORDS:
                found.append(f"{path}/{k}")
            found += label_keys(v, f"{path}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            found += label_keys(v, f"{path}[{i}]")
    return found


def assert_no_labels(view) -> None:
    hits = label_keys(view)
    if hits:
        raise ValueError(f"a label-like field would be rendered: {hits}")


def example_cards(examples: List[dict]) -> List[dict]:
    """Cards for the examples page: id, one-line description, and the status in words. Only these fields are kept."""
    words = {"verified": "verified", "partially_verified": "partially verified", "cannot_assess": "cannot assess",
             "no_action_model_based": "no action, model-based only"}
    cards = [{"tx_id": e["tx_id"], "description": e["description"], "status": words.get(e.get("status"), str(e.get("status"))),
              "time_step": e.get("time_step")} for e in examples]
    assert_no_labels(cards)
    return cards


def memo_option_label(m: dict) -> str:
    return f"{m['tx_id']}  (time step {m['time_step']}, {m['status'].replace('_', ' ')})"


# Page text of the evaluation and limits pages, kept as data so the figures can be tested against the README.
EVAL_ITEMS = [
    {"title": "The model collapses after the dark-market shutdown (time step 43)",
     "text": "On the test period, LightGBM recall is 0.85 before the shutdown (steps 35-42) and 0.018 after it (steps 43-49); "
             "PR-AUC falls from 0.915 to 0.039. The health monitor, which needs delayed labels, first flags the collapse at step 47, "
             "so scores at steps 43-46 are unreliable without any warning.",
     "figure": "notebooks/figures/baseline_per_step_recall.png", "caption": "Per-step test recall of the model; the collapse starts at step 43.",
     "report": "eval/results/phase2c.md"},
    {"title": "A human hand-check of the Phase 5b memos",
     "text": "A human checked 110 claims in 15 runs and flagged 7 (5 unsupported, 2 overstated). "
             "One reader, the sheet was not fully blinded for one run, and the interval on every rate is wide.",
     "figure": None, "report": "eval/results/agent_eval/report_5b.md"},
    {"title": "The critic caught the synthetic faults but missed natural errors",
     "text": "The critic caught 98 of the 98 applicable injected faults and rejected 0 of 20 untampered controls. On the natural errors "
             "the human flagged, it caught 4 of 7 (it missed 3) and made 7 rejections the human did not agree with. Synthetic faults "
             "are easier than the subtle mistakes a model really makes.",
     "figure": None, "report": "eval/results/agent_eval/report_5b.md"},
    {"title": "Rendering the facts in code (Phase 5c): the pre-registered rule was not met",
     "text": "The rule was fixed before the runs: adopt design v3 (facts written by code, the LLM only writes a short summary) if mean "
             "completeness rises by at least 0.05 with no new policy violations. The gain was +0.031 (97% to 100%), so the rule outcome is "
             "'do not adopt on this rule'. Completeness is near-tautological for v3, because code inserts the required items. "
             "In the hand-check, 0 of 20 code-built claims were wrong and 1 of 61 LLM-written summary sentences was overstated. "
             "Those rates are not comparable to the earlier hand-check, and no claim is made that v3 is more accurate than v2. "
             "The service uses v3 as a judgement after the fact, not as a test result.",
     "figure": None, "report": "eval/results/agent_eval_5c/report_5c.md"},
]

LIMITS = [
    "Elliptic has no sanctions data, no wallet addresses, no names, no amounts and no dates, so none of these can appear in a memo.",
    "Scores are raw gradient-boosting outputs, not calibrated probabilities. The calibrated value shifts with the prevalence of illicit activity.",
    "After time step 43 the model is close to useless, and the health monitor only notices at step 47.",
    "Structural patterns are resemblances only: Elliptic has no typology labels, and two of the three patterns lower the risk in the training data.",
    "One LLM acts as investigator, writer and reviewer, and it is sometimes too strict or wrong in the critic's role.",
    "Regulation retrieval is weak (about 0.6 hit@3 on a small test set); retrieved text is a lead, not an authority.",
    "The evaluations are small (30 to 40 nodes, 110 hand-checked claims, one human reader), so every rate has a wide interval.",
    "This is a draft for a human analyst. It does not decide whether a report should be filed, and nothing here measures that the agents improve detection.",
]

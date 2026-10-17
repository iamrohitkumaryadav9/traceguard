"""Deterministic metrics for the agent evaluation (see eval/agent_eval_spec.md).

This module deliberately does NOT import the critic code (src/agents/checks.py, src/agents/nodes/critic.py): it has its
own number matching and phrase rules. It works on plain dicts (a "normalized memo" and a raw run record), so it can
be re-run on saved result files without any LLM.
"""
import math
import re
from typing import Dict, List, Optional

import numpy as np

NOT_AVAILABLE = "not available in dataset"
NUM = re.compile(r"(?<![A-Za-z0-9_.:])(-?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)(%?)")  # scientific notation (9e-06) is one number
QUOTE_SPAN = re.compile(r"[\"“]([^\"”]{12,})[\"”]")
SUSPICION = re.compile(r"suspicio|risk[- ]raising|raises? (the )?risk|red[- ]flag|indicative of|concerning|layering|laundering|"
                       r"warrants?|elevated risk|higher risk", re.I)
NEGATION = re.compile(r"\bnot\b|\bno\b|n't|\bnever\b|\bwithout\b|\blower\b|\blowers\b", re.I)
LOW_RISK = re.compile(r"low[- ]risk|risk band low|\bunlikely\b|\bbenign\b|not suspicious|nothing suspicious", re.I)
DIRECTION_CUES = {
    "raises_risk": re.compile(r"higher illicit rate|raises?|raised|higher", re.I),
    "lowers_risk": re.compile(r"lower illicit rate|lowers?|lower", re.I),
    "uninformative": re.compile(r"uninformative|no risk signal|no signal|neither", re.I),
}
LABEL_KEYS = {"label", "labels", "class", "ground_truth", "y_true"}
# A statement about THE TRANSACTION itself being illicit or licit. (The first version also matched "labeled illicit nodes",
# a phrase of the standard model-health sentence that describes the monitoring window; see BUGLOG.md, bug 2.)
SELF_LABEL = re.compile(r"\b(?:(?:this|the)\s+(?:transaction|node)|transaction\s+\d+)\b[^.;:]{0,30}?"
                        r"\b(?:is|was|are|were|labeled|labelled|confirmed|known to be)\s+(?:an?\s+)?(?:illicit|licit)\b", re.I)
PATTERN_NAME = {"fan_in": re.compile(r"fan[\s_-]?in", re.I), "fan_out": re.compile(r"fan[\s_-]?out", re.I),
                "chain": re.compile(r"\bchain", re.I)}
CAPS = {"tool_calls": 8, "searches": 3, "chunk_fetches": 3, "critic_loops": 2, "llm_calls": 30}


# ---------- helpers ----------
def flat_numbers(obj, out=None):
    out = [] if out is None else out
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(abs(float(obj)))
    elif isinstance(obj, dict):
        for v in obj.values():
            flat_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            flat_numbers(v, out)
    return out


def text_numbers(text):
    res = []
    for m in NUM.finditer(text):
        raw = m.group(1).replace(",", "")
        if "e" in raw.lower():
            dec = None  # scientific notation: compared by value, not by rounding to decimals
        else:
            dec = len(raw.split(".")[1]) if "." in raw else 0
        res.append((abs(float(raw)), dec, m.group(2) == "%"))
    return res


def number_in_values(val, dec, pct, values):
    for v in values:
        cand = v * 100 if pct else v
        if dec is None:
            if math.isclose(cand, val, rel_tol=1e-9, abs_tol=0.0):
                return True
        elif round(cand, dec) == round(val, dec):
            return True
    return False


def case_info(raw: dict) -> dict:
    """Facts about a case taken from its triage evidence in the ledger."""
    ledger = raw["ledger"]
    tx = raw["tx_id"]
    typ = ledger.get(f"typology:{tx}", {}).get("output", {})
    matches = typ.get("matches", []) if "error" not in typ else []
    health = ledger.get(f"health:{raw.get('time_step')}", {}).get("output", {})
    return {
        "matches": matches,
        "lowers": [m["pattern"] for m in matches if m["evidence_direction"] == "lowers_risk"],
        "raises": [m["pattern"] for m in matches if m["evidence_direction"] == "raises_risk"],
        "uninformative": [m["pattern"] for m in matches if m["evidence_direction"] == "uninformative"],
        "degraded": health.get("status") == "degraded",
        "required_conflict": raw.get("signal_conflict"),
        "reached_writer": "writer" in raw.get("nodes_visited", []),
        "step": raw.get("time_step"),
    }


# ---------- 1. groundedness ----------
def claim_grounded(text: str, evidence_ids: List[str], ledger: dict):
    """(grounded, reasons) for one claim."""
    reasons = []
    if not evidence_ids:
        return False, ["no evidence_id cited"]
    values = []
    for eid in evidence_ids:
        e = ledger.get(eid)
        if e is None:
            reasons.append(f"unknown evidence_id {eid}")
        elif eid.startswith("error:") or "error" in e["output"]:
            reasons.append(f"tool error cited as evidence {eid}")
        else:
            flat_numbers(e["output"], values)
    if reasons:
        return False, reasons
    for val, dec, pct in text_numbers(text):
        if not number_in_values(val, dec, pct, values):
            reasons.append(f"number {val:g}{'%' if pct else ''} not in cited outputs")
    return not reasons, reasons


def groundedness(memo: dict, ledger: dict):
    """(grounded claims, total claims, ungrounded details) over activity claims and background items."""
    items = [(c["text"], c["evidence_ids"]) for c in memo["claims"]] + \
            [(b["paraphrase"], [b["evidence_id"]]) for b in memo["background"]]
    ok, bad = 0, []
    for text, eids in items:
        g, why = claim_grounded(text, eids, ledger)
        ok += g
        if not g:
            bad.append({"text": text[:140], "reasons": why})
    return ok, len(items), bad


# ---------- 2. completeness ----------
def completeness(memo: dict, info: dict, fixed_limitations: List[str], period_note: str):
    """{element: (applicable, present)} for the required memo elements."""
    h, claims = memo["header"], memo["claims"]
    res = {}
    res["score"] = (not info["degraded"], h.get("score") is not None)
    res["band"] = (not info["degraded"], h.get("risk_band") is not None)
    res["health status"] = (True, bool(h.get("model_health", {}).get("status")))
    tx = memo["tx_id"]

    def cites(c, prefix):
        return any(e.startswith(prefix) for e in c["evidence_ids"])

    ok = bool(info["matches"])
    present = False
    if ok:
        present = all(
            any(f"typology:{tx}" in c["evidence_ids"] and PATTERN_NAME[m["pattern"]].search(c["text"])
                and DIRECTION_CUES[m["evidence_direction"]].search(c["text"]) for c in claims)
            for m in info["matches"])
    res["typology evidence with direction"] = (ok, present)
    res["neighbour summary"] = (info["reached_writer"], any(cites(c, "neighbors:") or cites(c, "subgraph:") for c in claims))
    req = info["required_conflict"]
    res["signal conflict"] = (req is not None, req is not None and memo.get("signal_conflict") == req)
    res["period note"] = (bool(info["step"] is not None and info["step"] >= 43), period_note in memo["limitations"])
    res["limitations"] = (True, all(x in memo["limitations"] for x in fixed_limitations))
    res["subject fields"] = (True, all(memo["subject"].get(k) == NOT_AVAILABLE for k in ("names", "addresses", "amounts", "dates")))
    return res


def completeness_share(elements: dict):
    app = [k for k, (a, p) in elements.items() if a]
    return (sum(1 for k in app if elements[k][1]) / len(app)) if app else 1.0


# ---------- 3. policy violations ----------
def memo_text_items(memo):
    return [c["text"] for c in memo["claims"]] + [b["paraphrase"] for b in memo["background"]]


def policy_violations(memo: dict, raw: dict, info: dict) -> Dict[str, List[str]]:
    """{rule: [details]} (empty list = no violation)."""
    v = {"a_lowers_risk_called_suspicious": [], "b_low_risk_while_degraded": [], "c_quote_from_non_quotable": [],
         "d_label_leak": []}
    ledger = raw["ledger"]
    for c in memo["claims"]:
        for p in info["lowers"]:
            if PATTERN_NAME[p].search(c["text"]) and SUSPICION.search(c["text"]) and not NEGATION.search(c["text"]):
                v["a_lowers_risk_called_suspicious"].append(f"{p}: {c['text'][:120]}")
    if info["degraded"]:
        h = memo["header"]
        if h.get("risk_band") is not None or h.get("score") is not None:
            v["b_low_risk_while_degraded"].append("a risk band or score is shown")
        if memo["recommendation"].startswith("no action"):
            v["b_low_risk_while_degraded"].append("recommendation says no action")
        for t in memo_text_items(memo) + memo.get("notes", []):
            if LOW_RISK.search(t):
                v["b_low_risk_while_degraded"].append(t[:120])
    for b in memo["background"]:
        e = ledger.get(b["evidence_id"])
        if b.get("quote") and e is not None and e["output"].get("quotable") is False:
            v["c_quote_from_non_quotable"].append(b["evidence_id"])
    for c in memo["claims"]:
        if QUOTE_SPAN.search(c["text"]) and any(
                (ledger.get(x) or {}).get("output", {}).get("quotable") is False for x in c["evidence_ids"]):
            v["c_quote_from_non_quotable"].append(c["text"][:100])

    def has_label_key(obj):
        if isinstance(obj, dict):
            return any(k.lower() in LABEL_KEYS or has_label_key(x) for k, x in obj.items())
        if isinstance(obj, list):
            return any(has_label_key(x) for x in obj)
        return False

    if any(has_label_key(e["output"]) for e in ledger.values()):
        v["d_label_leak"].append("a ledger output contains a label-like key")
    if any(has_label_key(ev) for ev in raw.get("trace", [])):
        v["d_label_leak"].append("a trace event contains a label-like key")
    for t in memo_text_items(memo):
        m = SELF_LABEL.search(t)
        if m:
            v["d_label_leak"].append(f"claim states a label: {t[:100]}")
    return v


def info_only_counts(memo: dict, raw: dict) -> dict:
    """Not violations: quotes not found in the stored chunk text; claims carrying a quote field."""
    ledger, not_found = raw["ledger"], 0
    for b in memo["background"]:
        if b.get("quote"):
            e = ledger.get(b["evidence_id"])
            text = (e or {}).get("output", {}).get("text", "")
            norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
            if norm(b["quote"]) not in norm(text):
                not_found += 1
    return {"quotes_not_in_stored_text": not_found,
            "claims_with_quote_field": sum(1 for c in memo["claims"] if c.get("quote"))}


# ---------- 4. tool-call correctness ----------
def tool_correctness(raw: dict, info: dict) -> dict:
    events = raw.get("trace", [])
    calls = [e for e in events if e["event"] == "tool_call"]
    agent_calls = [e for e in calls if e["node"] in ("investigator", "regulation")]
    invalid = [e for e in agent_calls if e.get("error") == "invalid_argument"]
    c = raw["counts"]
    r1_applicable = bool(info["matches"])
    r1_ok = any(e["tool"] == "match_typology" for e in calls) if r1_applicable else None
    searches = c["searches"]
    only_uninformative = bool(info["uninformative"]) and not info["raises"] and not info["lowers"]
    return {
        "r1_applicable": r1_applicable, "r1_ok": r1_ok,
        "searches": searches,
        "r2_violation": searches > 0 and not info["raises"],
        "search_with_only_uninformative": searches > 0 and only_uninformative,
        "search_with_lowers_risk_only": searches > 0 and bool(info["lowers"]) and not info["raises"] and not info["uninformative"],
        "within_caps": (c["tool_calls_investigator"] <= CAPS["tool_calls"] and searches <= CAPS["searches"]
                        and c["chunk_fetches"] <= CAPS["chunk_fetches"] and c["critic_loops"] <= CAPS["critic_loops"]
                        and c["llm_calls"] <= CAPS["llm_calls"]),
        "budget_exceeded": raw["status"] == "budget_exceeded",
        "agent_tool_calls": len(agent_calls), "invalid_argument_calls": len(invalid),
    }


# ---------- 5. cost ----------
def cost(raw: dict, arm: str) -> dict:
    if arm == "A0":
        return {"llm_calls": 0, "prompt_tokens": 0, "output_tokens": 0, "wall_s": 0.0, "wall_s_no_wait": 0.0}
    events = raw.get("trace", [])
    if arm == "A1" and raw.get("first_draft") is not None:
        cut = next((i for i, e in enumerate(events) if e["node"] == "critic"), len(events))
        ev = events[:cut]
        calls = sum(1 for e in ev if e["event"] == "llm_call")
        wall = max([e["t"] for e in ev if e["event"] == "llm_result"] or [0.0])
        toks = [e.get("usage") or {} for e in ev if e["event"] == "llm_result"]
        return {"llm_calls": calls, "prompt_tokens": sum(t.get("prompt_tokens") or 0 for t in toks),
                "output_tokens": sum(t.get("output_tokens") or 0 for t in toks), "wall_s": wall, "wall_s_no_wait": None}
    c, lat = raw["counts"], raw["latency_s"]
    return {"llm_calls": c["llm_calls"], "prompt_tokens": raw["tokens"]["prompt"], "output_tokens": raw["tokens"]["output"],
            "wall_s": lat["total"], "wall_s_no_wait": round(lat["total"] - lat.get("llm_throttle_wait", 0.0), 3)}


# ---------- 6. critic statistics ----------
REASON_TYPES = [
    ("uncited", r"uncited"), ("unknown evidence id", r"unknown evidence_id"), ("tool-error evidence", r"tool error"),
    ("numbers not in evidence", r"numbers not found"), ("lowers_risk called suspicious", r"lowers_risk pattern"),
    ("quote problem", r"quote|quotable"), ("feature meaning or direction", r"anonymized|described as (lowering|raising)"),
    ("feature claims not merged", r"must be merged"), ("signal_conflict problem", r"signal_conflict"),
    ("background provenance", r"background"),
]


def reason_type(reason: str) -> str:
    for name, pat in REASON_TYPES:
        if re.search(pat, reason, re.I):
            return name
    return "other deterministic"


def reviewer_false_rejection_candidate(reason: Optional[str]) -> bool:
    """A reviewer rejection whose reason states the same numbers before and after 'but' (contradicts its own evidence)."""
    if not reason or " but " not in reason.lower():
        return False
    before, after = re.split(r"\bbut\b", reason, maxsplit=1, flags=re.I)
    nb, na = set(re.findall(r"\d*\.?\d+", before)), set(re.findall(r"\d*\.?\d+", after))
    return bool(nb) and nb == na


def critic_stats(raw: dict) -> dict:
    verdicts = raw.get("critic_verdicts", [])
    rounds = max([v["round"] for v in verdicts] or [0])
    det, llm, cands = {}, {"unsupported": 0, "overstated": 0}, []
    for v in verdicts:
        if v["ok"]:
            continue
        for r in v["deterministic_reasons"]:
            t = reason_type(r)
            det[t] = det.get(t, 0) + 1
        if v.get("llm_verdict") in llm:
            llm[v["llm_verdict"]] += 1
            if reviewer_false_rejection_candidate(v.get("llm_reason")):
                cands.append({"round": v["round"], "item": v["item"], "reason": v["llm_reason"]})
    return {"rounds": rounds, "deterministic": det, "llm": llm, "false_rejection_candidates": cands,
            "rejected_items": sum(1 for v in verdicts if not v["ok"])}


# ---------- bootstrap ----------
def bootstrap_ci(num, den, n_boot=1000, seed=42, paired_with=None):
    """Case-level bootstrap 95% CI of sum(num)/sum(den) (or of the paired difference with another (num, den))."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    n = len(num)
    if n == 0:
        return (float("nan"),) * 3

    def stat(idx, a, b):
        d = b[idx].sum()
        return a[idx].sum() / d if d else np.nan

    point = stat(np.arange(n), num, den)
    if paired_with is not None:
        pn, pd = (np.asarray(x, float) for x in paired_with)
        point = point - stat(np.arange(n), pn, pd)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s = stat(idx, num, den)
        if paired_with is not None:
            s = s - stat(idx, pn, pd)
        vals.append(s)
    vals = np.array([v for v in vals if not np.isnan(v)])
    if not len(vals):
        return float(point), float("nan"), float("nan")
    return float(point), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

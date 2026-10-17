"""Deterministic critic checks. Pure functions over text and the evidence ledger: no LLM, no network."""
import re
from typing import Callable, Dict, List, Optional

from src.agents.state import LedgerEntry

NUMBER = re.compile(r"(?<![A-Za-z0-9_.:])(-?\d[\d,]*(?:\.\d+)?)(%?)")
QUOTED = re.compile(r"[\"“]([^\"”]{12,})[\"”]")  # a quotation of 12+ characters in the text itself
FEATURE = re.compile(r"\bf\d{1,3}\b")
PATTERN_TERMS = {"fan_in": re.compile(r"fan[\s_-]?in", re.I), "fan_out": re.compile(r"fan[\s_-]?out", re.I),
                 "chain": re.compile(r"\bchain", re.I)}
SUSPICIOUS = re.compile(r"suspicio|risk[- ]raising|raises? (the )?risk|raises? (the )?(illicit )?(score|suspicion)|"
                        r"red[- ]flag|indicative of|consistent with (money )?laundering|layering|concerning|"
                        r"warrants? (concern|scrutiny)|elevated risk|higher risk", re.I)
NEGATION = re.compile(r"\bnot\b|\bno\b|n't|\bnever\b|\blower\b|\blowers\b|\blowering\b|\bless likely\b|\bwithout\b", re.I)
MEANING = re.compile(r"\b(represent\w*|indicat\w*|mean\w*|measur\w*|reflect\w*|correspond\w*|stand\w* for|refer\w* to|"
                     r"relat\w* to|proxy|signif\w*|capture\w*|encod\w*|describ\w*)\b", re.I)
UP = re.compile(r"\b(up|raised|raises|raise|increased|increases|higher|pushed (the )?score up|boost\w*)\b", re.I)
DOWN = re.compile(r"\b(down|lowered|lowers|lower|decreased|decreases|reduced|reduces|pushed (the )?score down)\b", re.I)


def _numbers_in(obj, out):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(abs(float(obj)))
    elif isinstance(obj, dict):
        for v in obj.values():
            _numbers_in(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _numbers_in(v, out)


def evidence_numbers(entries: List[LedgerEntry]):
    vals: List[float] = []
    for e in entries:
        _numbers_in(e.output, vals)
    return vals


def claim_numbers(text):
    """[(value, decimals, is_percent)] for every number in the text (feature names like f57 are skipped)."""
    out = []
    for m in NUMBER.finditer(text):
        raw = m.group(1).replace(",", "")
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        out.append((abs(float(raw)), decimals, m.group(2) == "%"))
    return out


def number_matches(value, decimals, pct, evidence_vals):
    for v in evidence_vals:
        cand = v * 100 if pct else v
        if round(cand, decimals) == round(value, decimals):
            return True
    return False


def numbers_not_in_evidence(text, entries: List[LedgerEntry]):
    vals = evidence_numbers(entries)
    return [f"{v:g}{'%' if p else ''}" for v, d, p in claim_numbers(text) if not number_matches(v, d, p, vals)]


def lowers_risk_patterns(ledger: Dict[str, LedgerEntry]):
    """Pattern names whose evidence_direction is 'lowers_risk' in any match_typology output in the ledger."""
    found = set()
    for e in ledger.values():
        if e.tool == "match_typology" and "matches" in e.output:
            for m in e.output["matches"]:
                if m.get("evidence_direction") == "lowers_risk":
                    found.add(m["pattern"])
    return found


def shap_reasons(ledger: Dict[str, LedgerEntry]):
    """{feature: direction text} from score_transaction outputs."""
    out = {}
    for e in ledger.values():
        if e.tool == "score_transaction":
            for r in e.output.get("reasons", []):
                out[r["feature"]] = r["direction"]
    return out


def find_quotes(text, quote: Optional[str]):
    quotes = [q.strip() for q in QUOTED.findall(text or "")]
    if quote:
        quotes.append(quote.strip())
    return quotes


def _norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def check_item(text: str, evidence_ids: List[str], quote: Optional[str], ledger: Dict[str, LedgerEntry],
               fetch_chunk: Callable[[str], dict]) -> List[str]:
    """Reasons this claim (or regulatory paraphrase) fails the deterministic checks; empty list = passes."""
    reasons = []
    # 1. citations exist in the ledger
    if not evidence_ids:
        reasons.append("uncited: the claim cites no evidence_id")
    cited = []
    for eid in evidence_ids:
        entry = ledger.get(eid)
        if entry is None:
            reasons.append(f"unknown evidence_id {eid!r}: not in the evidence ledger")
        elif eid.startswith("error:") or "error" in entry.output:
            reasons.append(f"evidence_id {eid!r} is a tool error, not evidence")
        else:
            cited.append(entry)
    # 2. every number appears in the cited outputs
    if cited:
        text_wo_quotes = QUOTED.sub("", text)
        bad = numbers_not_in_evidence(text_wo_quotes, cited)
        if bad:
            reasons.append(f"numbers not found in the cited evidence: {', '.join(bad)}")
    # 3. a lowers_risk pattern is never described as suspicious
    for pattern in lowers_risk_patterns(ledger):
        if PATTERN_TERMS[pattern].search(text) and SUSPICIOUS.search(text) and not NEGATION.search(text):
            reasons.append(f"describes the lowers_risk pattern {pattern} as suspicious or risk-raising")
    # 4. quotes must be verbatim from a quotable chunk
    for q in find_quotes(text, quote):
        chunk_ids = [e for e in evidence_ids if e.startswith("reg:") and e in ledger and "error" not in ledger[e].output]
        if not chunk_ids:
            reasons.append(f"quote {q[:40]!r} has no cited regulation chunk")
            continue
        verified = False
        for cid in chunk_ids:
            c = fetch_chunk(cid)
            if "error" in c:
                continue
            if _norm(q) in _norm(c["text"]):
                if c.get("quotable", True):
                    verified = True
                else:
                    reasons.append(f"quote {q[:40]!r} is from a quotable=false (OCR) chunk: paraphrase instead")
                    verified = None
                    break
        if verified is False:
            reasons.append(f"quote {q[:40]!r} is not a verbatim substring of any cited chunk")
    # 5. SHAP features stay anonymous and directionally consistent
    reasons_by_feature = shap_reasons(ledger)
    for feat in set(FEATURE.findall(text)):
        if MEANING.search(text):
            reasons.append(f"feature {feat} is anonymized; the claim attributes a meaning to it")
            break
    for feat in set(FEATURE.findall(text)):
        direction = reasons_by_feature.get(feat)
        if direction is None:
            continue
        says_up, says_down = bool(UP.search(text)), bool(DOWN.search(text))
        if direction.startswith("raises") and says_down and not says_up:
            reasons.append(f"feature {feat} is described as lowering the score but SHAP says it raised it")
        if direction.startswith("lowers") and says_up and not says_down:
            reasons.append(f"feature {feat} is described as raising the score but SHAP says it lowered it")
    return reasons

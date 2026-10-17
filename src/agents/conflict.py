"""Deterministic signal-conflict detection (no LLM). Fixed template sentences built from ledger evidence only.

A conflict is raised when two independent signals about the same transaction point in different directions:
  A. risk band low, but a matched structural pattern has evidence_direction raises_risk;
  B. risk band high, but every matched pattern (at least one) has evidence_direction lowers_risk;
  C. the in-neighbour alert evidence contradicts the band: band low while at least one in-neighbour alerted, or
     band high while the node has in-neighbours and none of them alerted.
Medium band is never in conflict with neighbours or patterns (it makes no strong claim either way).
"""
from typing import List, Optional, Tuple

TAIL = "point in different directions."


def _join(names: List[str]):
    return " and ".join(names)


def detect_signal_conflict(score: dict, typology: Optional[dict], in_neighbors: Optional[dict]) -> Tuple[Optional[str], List[str]]:
    """(sentence or None, evidence_ids). `score`, `typology` and `in_neighbors` are tool outputs (dicts)."""
    band, s, sid = score["risk_band"], score["score"], score["evidence_id"]
    sentences, ids = [], []
    matches = (typology or {}).get("matches", []) if typology and "error" not in typology else []
    if band == "low":
        raising = [m["pattern"] for m in matches if m["evidence_direction"] == "raises_risk"]
        if raising:
            sentences.append(
                f"Signal conflict: the model score {s} (risk band low; evidence {sid}) and the structural pattern "
                f"{_join(raising)} (evidence direction raises_risk; evidence {typology['evidence_id']}) {TAIL}")
            ids += [sid, typology["evidence_id"]]
    if band == "high" and matches and all(m["evidence_direction"] == "lowers_risk" for m in matches):
        names = [m["pattern"] for m in matches]
        sentences.append(
            f"Signal conflict: the model score {s} (risk band high; evidence {sid}) and the structural pattern "
            f"{_join(names)} (evidence direction lowers_risk; evidence {typology['evidence_id']}) {TAIL}")
        ids += [sid, typology["evidence_id"]]
    if in_neighbors and "error" not in in_neighbors and in_neighbors.get("total_found", 0) > 0:
        total, nid = in_neighbors["total_found"], in_neighbors["evidence_id"]
        # neighbours are sorted by score, highest first, so 'any alerted' is decided by the listed ones
        any_alert = any(n["alert"] for n in in_neighbors["neighbors"])
        if band == "low" and any_alert:
            sentences.append(
                f"Signal conflict: the model score {s} (risk band low; evidence {sid}) and the in-neighbour alerts "
                f"(at least one of the {total} in-neighbours alerted; evidence {nid}) {TAIL}")
            ids += [sid, nid]
        if band == "high" and not any_alert:
            sentences.append(
                f"Signal conflict: the model score {s} (risk band high; evidence {sid}) and the in-neighbour alerts "
                f"(none of the {total} in-neighbours alerted; evidence {nid}) {TAIL}")
            ids += [sid, nid]
    if not sentences:
        return None, []
    return " ".join(sentences), list(dict.fromkeys(ids))

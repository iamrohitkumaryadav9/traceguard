"""Shared helpers for Phase 5b: load the 5a raw records, list writer runs, build the items a memo claims."""
import json
from pathlib import Path

from eval.agent_eval.arms import build_a1, build_a2

RESULT_DIR = Path("eval/results/agent_eval")
EVIDENCE_LIMIT = 6000  # characters of one evidence entry shown to a human or to the judge


def load_raws(result_dir=RESULT_DIR):
    """{tx_id: raw record} for every run that has a raw file."""
    out = {}
    for p in sorted((Path(result_dir) / "raw").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        out[r["tx_id"]] = r
    return out


def writer_runs(raws):
    """Runs that reached the writer (and ended with a memo), sorted by tx id."""
    return {k: r for k, r in sorted(raws.items()) if "writer" in r.get("nodes_visited", []) and r.get("memo")}


def critic_changed(raw) -> bool:
    """The critic rejected at least one item in some round (so the memo was rewritten or had items removed)."""
    return any(not v["ok"] for v in raw.get("critic_verdicts", []))


def evidence_text(ledger: dict, ids) -> str:
    parts = []
    for eid in ids:
        e = ledger.get(eid)
        if e is None:
            parts.append(f"[{eid}] (not in the evidence ledger)")
            continue
        s = json.dumps(e["output"], separators=(",", ":"), ensure_ascii=False)
        parts.append(f"[{eid}] ({e['tool']}) {s[:EVIDENCE_LIMIT]}{'...(truncated)' if len(s) > EVIDENCE_LIMIT else ''}")
    return "\n".join(parts)


def statement_items(memo: dict):
    """[(kind, statement text, [evidence ids], quote)] for a normalized memo (claims, then background items)."""
    out = [("claim", c["text"], list(c["evidence_ids"]), None) for c in memo["claims"]]
    for b in memo["background"]:
        text = b["paraphrase"] + (f' Quote: "{b["quote"]}"' if b.get("quote") else "")
        out.append(("background", text, [b["evidence_id"]], b.get("quote")))
    return out


def arm_items(raw, arm):
    memo = build_a1(raw) if arm == "A1" else build_a2(raw)
    return statement_items(memo)

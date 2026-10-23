"""The demo bundle (demo/memos.json): pre-computed memos plus the curated examples. Plain JSON, no heavy imports."""
import json
from pathlib import Path

LABEL_KEYS = {"label", "labels", "class", "ground_truth", "y_true", "stratum"}  # none of these may appear in the bundle


DROP_KEYS = LABEL_KEYS | {"feature_value", "shap_value"}  # never published: labels and the dataset's own feature values


def sanitize(obj):
    """A copy of a tool output without label-like keys and without feature values or SHAP magnitudes."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items() if str(k).lower() not in DROP_KEYS}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


def cited_ids(memo):
    """Every evidence id the memo cites (claims, summary, background, signal conflict, evidence table)."""
    ids = set(memo.get("signal_conflict_evidence_ids") or [])
    for key in ("activity_description", "executive_summary"):
        for c in memo.get(key) or []:
            ids |= set(c.get("evidence_ids") or [])
    ids |= {b["evidence_id"] for b in memo.get("regulatory_background") or [] if b.get("evidence_id")}
    ids |= {e["evidence_id"] for e in memo.get("evidence_table") or []}
    return ids


def public_evidence(ledger, ids):
    """{evidence_id: {tool, output}} for the cited ids only, with the output sanitized."""
    return {i: {"tool": ledger[i]["tool"], "output": sanitize(ledger[i]["output"])} for i in sorted(ids) if i in ledger}


class Bundle:
    def __init__(self, path):
        self.path = Path(path)
        self.entries, self.examples, self.meta = {}, [], {}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.entries = {int(k): v for k, v in data.get("memos", {}).items()}
            self.examples = data.get("examples", [])
            self.meta = data.get("meta", {})

    def __len__(self):
        return len(self.entries)

    def get(self, tx_id):
        return self.entries.get(int(tx_id))


def label_keys_in(obj, path=""):
    """Paths of dictionary keys that look like a ground-truth label, anywhere in a JSON structure."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in LABEL_KEYS:
                found.append(f"{path}/{k}")
            found += label_keys_in(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found += label_keys_in(v, f"{path}[{i}]")
    return found

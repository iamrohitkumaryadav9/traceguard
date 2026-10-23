"""The demo bundle (demo/memos.json): pre-computed memos plus the curated examples. Plain JSON, no heavy imports."""
import json
from pathlib import Path

LABEL_KEYS = {"label", "labels", "class", "ground_truth", "y_true", "stratum"}  # none of these may appear in the bundle


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

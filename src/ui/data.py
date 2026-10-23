"""Data sources of the UI: the demo bundle read directly (default), or the Phase 6a API when API_URL is set. Standard library only."""
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

BUNDLE_PATH = Path("demo/memos.json")


class BundleSource:
    live = False

    def __init__(self, path=BUNDLE_PATH):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self._memos = {int(k): v for k, v in data["memos"].items()}
        self._examples = data.get("examples", [])

    def examples(self):
        return [{**e, "status": (self._memos.get(e["tx_id"]) or {}).get("status"),
                 "time_step": (self._memos.get(e["tx_id"]) or {}).get("time_step")} for e in self._examples]

    def memos(self):
        return [{"tx_id": t, "status": m["status"], "time_step": m["time_step"]} for t, m in sorted(self._memos.items())]

    def get(self, tx_id):
        return self._memos.get(int(tx_id))


class ApiSource:
    """Calls the API. In live mode `get` runs the agent for nodes that are not cached (it can take a minute)."""

    def __init__(self, base_url, fetch=None, timeout=300):
        self.base, self.timeout = base_url.rstrip("/"), timeout
        self._fetch = fetch or self._http
        self.live = self._fetch("GET", "/health", None).get("mode") == "live"

    def _http(self, method, path, body):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                return json.loads(e.read())
            except Exception:
                return {"error": {"code": "http_error", "message": f"HTTP {e.code}"}}

    def examples(self):
        return self._fetch("GET", "/examples", None)["examples"]

    def memos(self):
        return self._fetch("GET", "/memos", None)["memos"]

    def get(self, tx_id):
        r = self._fetch("POST", "/investigate", {"tx_id": int(tx_id)})
        if "error" in r:
            raise LookupError(r["error"]["message"])
        return r


def make_source():
    url = os.environ.get("API_URL")
    return ApiSource(url) if url else BundleSource()

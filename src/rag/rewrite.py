"""LLM query rewriting (Gemini). The rewriter sees ONLY the question: never documents, chunks or answers.

Rewrites are cached on disk keyed by question text (eval/results/rewrites.json) so reruns are reproducible
and cost nothing. The API key is read at runtime from the environment (python-dotenv); it is never printed.
"""
import json
import os
import time
from pathlib import Path

REWRITE_MODEL = "gemini-3.5-flash-lite"  # gemini-2.5-flash-lite is no longer available to new users
PROMPT = (
    "Rewrite this question as a concise search query using the technical terminology that regulatory documents "
    "on virtual assets and anti-money-laundering would use. Output only the query."
)
CACHE_PATH = Path("eval/results/rewrites.json")
RATE_LIMIT_WAIT_S = 62  # the free tier allows 15 requests per minute per model


class QueryRewriter:
    def __init__(self, model=REWRITE_MODEL, cache_path=CACHE_PATH, client=None):
        self.model = model
        self.cache_path = Path(cache_path)
        self._client = client
        self.calls = 0  # live API calls made by this instance
        self._data = {"_meta": {"model": model, "prompt": PROMPT, "temperature": 0}, "rewrites": {}, "latency_ms": {}}
        if self.cache_path.exists():
            loaded = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if loaded.get("_meta", {}).get("model") == model and loaded["_meta"].get("prompt") == PROMPT:
                self._data = loaded
            else:
                raise RuntimeError(f"{self.cache_path} was made with a different model or prompt; remove it to redo")

    def _get_client(self):
        if self._client is None:
            from dotenv import load_dotenv
            from google import genai

            load_dotenv()
            key = os.environ.get("GEMINI_API_KEY")
            if not key or key == "your_key_here":
                raise RuntimeError("GEMINI_API_KEY is not set (put it in .env)")
            self._client = genai.Client(api_key=key)
        return self._client

    def _generate(self, question):
        from google.genai import types

        contents = f"{PROMPT}\n\nQuestion: {question}"
        resp = self._get_client().models.generate_content(
            model=self.model, contents=contents, config=types.GenerateContentConfig(temperature=0))
        return (resp.text or "").strip().strip('"').replace("\n", " ")

    def rewrite(self, question):
        cached = self._data["rewrites"].get(question)
        if cached is not None:
            return cached
        last = None
        for attempt in range(4):
            try:
                t0 = time.perf_counter()
                out = self._generate(question)
                ms = 1000 * (time.perf_counter() - t0)
                break
            except Exception as e:  # network / rate limit: retry, then fail loudly
                last = e
                rate_limited = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                time.sleep(RATE_LIMIT_WAIT_S if rate_limited else 2 ** attempt)
        else:
            raise RuntimeError(f"rewrite failed for {question!r}: {type(last).__name__}")
        if not out:
            raise RuntimeError(f"empty rewrite for {question!r}")
        self.calls += 1
        self._data["rewrites"][question] = out
        self._data["latency_ms"][question] = round(ms, 1)
        self.save()
        return out

    def save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


_REWRITER = None


def get_rewriter():
    global _REWRITER
    if _REWRITER is None:
        _REWRITER = QueryRewriter()
    return _REWRITER


def set_rewriter(r):
    """Install a rewriter (tests pass a fake). None resets."""
    global _REWRITER
    _REWRITER = r

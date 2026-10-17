"""LLM judge for faithfulness, independent of the critic (spec section A in eval/agent_eval_5b_spec.md).

One call per unique (statement, cited evidence) pair. The judge sees only the statement and the evidence it cites:
never critic verdicts, other claims, the memo or any label.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from eval.agent_eval_5b.common import RESULT_DIR, arm_items, evidence_text, writer_runs
from src.agents.llm import DailyQuotaExceeded

JUDGE_DIR = RESULT_DIR / "judge"
CANDIDATES = ["gemini-3.1-pro-preview", "gemini-2.5-pro", "gemini-3.8-flash"]  # then the writer's own model
WRITER_MODEL = "gemini-3.5-flash-lite"

SYSTEM = """You are an independent fact-checker. You receive ONE statement and the evidence that the statement cites. \
The evidence is all you may use: ignore everything else you know.
Classify the statement:
- supported: every fact it states is explicitly present in the cited evidence and every number is equal to the evidence.
- unsupported: it states a fact that is not in the cited evidence, or contradicts it (including a reversed direction or a wrong number).
- overstated: the evidence is related, but the statement is more certain, broader or more causal than the evidence allows.
Give a one-line reason."""


class JudgeOut(BaseModel):
    verdict: Literal["supported", "unsupported", "overstated"]
    reason: str


def item_key(kind, text, ids, evidence) -> str:
    return hashlib.sha1(json.dumps([kind, text, ids, evidence], ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def judge_prompt(text, evidence) -> str:
    return f"STATEMENT:\n{text}\n\nEVIDENCE CITED BY THE STATEMENT (this is all you may use):\n{evidence}"


def collect_items(raws):
    """{key: {kind, text, ids, evidence, uses: [(tx, arm)]}} over the writer runs, for A1 and A2."""
    items = {}
    for tx, raw in writer_runs(raws).items():
        for arm in ("A1", "A2"):
            for kind, text, ids, _ in arm_items(raw, arm):
                ev = evidence_text(raw["ledger"], ids)
                k = item_key(kind, text, ids, ev)
                items.setdefault(k, {"kind": kind, "text": text, "ids": ids, "evidence": ev, "uses": []})["uses"].append([tx, arm])
    return items


def probe(model, client_factory) -> tuple:
    """(ok, short error) for a one-word call with no retries."""
    try:
        from google.genai import types

        r = client_factory().models.generate_content(model=model, contents="Reply with the single word: ok",
                                                     config=types.GenerateContentConfig(temperature=0))
        return bool((r.text or "").strip()), ""
    except Exception as e:  # not available to this key, quota 0, ...
        return False, f"{type(e).__name__}: {str(e)[:120]}"


def run_judge(raws, make_llm, models, out_dir=JUDGE_DIR, probe_fn=None):
    """Judge every unique item. make_llm(model) -> object with .structured(); models: ordered usable models.
    Resumable (the verdict cache is rewritten after every call). Switches to the next model on a daily-quota error.
    Returns (verdicts dict, log dict)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "verdicts.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    items = collect_items(raws)
    log = {"models_tried": [], "stopped": None, "quota_switches": []}
    mi, llm = 0, None
    for key, it in items.items():
        if key in cache:
            continue
        while True:
            if mi >= len(models):
                log["stopped"] = "all candidate models exhausted"
                _save(path, cache, items)
                return cache, log
            model = models[mi]
            llm = llm or make_llm(model)
            if model not in log["models_tried"]:
                log["models_tried"].append(model)
            try:
                out, info = llm.structured(SYSTEM, judge_prompt(it["text"], it["evidence"]), JudgeOut, purpose="judge")
                break
            except DailyQuotaExceeded:
                log["quota_switches"].append(model)
                mi, llm = mi + 1, None
        cache[key] = {"verdict": out.verdict, "reason": out.reason, "model": model}
        _save(path, cache, items)
    return cache, log


def _save(path, cache, items=None):
    path.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")


def counts_per_arm(raws, verdicts):
    """{arm: Counter(verdict)} and per-model counts, over the claims of each arm (with multiplicity)."""
    items = collect_items(raws)
    arms = {"A1": Counter(), "A2": Counter()}
    models = Counter()
    for key, it in items.items():
        v = verdicts.get(key)
        for _, arm in it["uses"]:
            arms[arm][v["verdict"] if v else "not judged"] += 1
        if v:
            models[v["model"]] += 1
    return arms, models

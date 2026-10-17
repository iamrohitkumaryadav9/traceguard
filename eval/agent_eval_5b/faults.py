"""Critic fault injection (spec section C in eval/agent_eval_5b_spec.md). Faults are injected programmatically (no LLM);
only the critic node is run on each tampered draft. Nothing in src/ is changed.
"""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from eval.agent_eval_5b.common import RESULT_DIR, writer_runs
from src.agents.llm import DailyQuotaExceeded
from src.agents.nodes.critic import critic
from src.agents.runtime import Deps, default_run_tool
from src.agents.state import AgentState, Claim, LedgerEntry, RegReference, WriterOutput
from src.ml.eval import wilson_interval

SEED = 42
N_MEMOS = 20
FAULTS_DIR = RESULT_DIR / "faults"
FAULT_TYPES = ["F1 numeric tamper", "F2 fabricated evidence_id", "F3 polarity flip", "F4 unsupported claim, valid id",
               "F5 overstated", "F6 fabricated quote", "F7 dropped signal_conflict"]
NUM = re.compile(r"(?<![A-Za-z0-9_.:])(-?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)")
FLIPS = [("lower illicit rate", "higher illicit rate"), ("higher illicit rate", "lower illicit rate"),
         ("lowers_risk", "raises_risk"), ("raises_risk", "lowers_risk")]
OVERSTATED = "The model score confirms and proves that this transaction is part of illicit activity."
UNSUPPORTED = "This transaction is linked to a known darknet market."


# ---------- reconstruct the original final draft and the critic state ----------
def draft_from_raw(raw) -> WriterOutput:
    m = raw["memo"]
    return WriterOutput(
        signal_conflict=m.get("signal_conflict"),
        activity_description=[Claim(text=c["text"], evidence_ids=list(c["evidence_ids"])) for c in m["activity_description"]],
        regulatory_background=[RegReference(paraphrase=b["paraphrase"], evidence_id=b["evidence_id"], quote=b.get("quote"))
                               for b in m["regulatory_background"]])


def state_from_raw(raw) -> AgentState:
    return AgentState(tx_id=raw["tx_id"], time_step=raw.get("time_step"),
                      ledger={k: LedgerEntry(**v) for k, v in raw["ledger"].items()},
                      signal_conflict=raw.get("signal_conflict"),
                      signal_conflict_evidence_ids=raw.get("signal_conflict_evidence_ids", []))


def evidence_values(ledger, ids):
    vals = []

    def walk(o):
        if isinstance(o, bool):
            return
        if isinstance(o, (int, float)):
            vals.append(abs(float(o)))
        elif isinstance(o, dict):
            [walk(v) for v in o.values()]
        elif isinstance(o, list):
            [walk(v) for v in o]

    for i in ids:
        if i in ledger:
            walk(ledger[i]["output"])
    return vals


# ---------- the seven injectors: (new_draft, injected_item_label) or None when not applicable ----------
def f1_numeric(draft, raw):
    ledger = raw["ledger"]
    for i, c in enumerate(draft.activity_description):
        m = NUM.search(c.text)
        if not m:
            continue
        vals = evidence_values(ledger, c.evidence_ids)
        tok = m.group(1).replace(",", "")
        x = float(tok)
        if "e" in tok.lower():
            new = x * 1.37
            new_s = f"{new:.2e}"
        elif "." in tok:
            dec = len(tok.split(".")[1])
            new = x * 1.37
            new_s = f"{new:.{dec}f}"
        else:
            new = x + 7
            while any(abs(v - new) < 1e-9 for v in vals):
                new += 1
            new_s = str(int(new))
        d = draft.model_copy(deep=True)
        d.activity_description[i].text = c.text[:m.start(1)] + new_s + c.text[m.end(1):]
        return d, f"claim {i + 1}"
    return None


def f2_fabricated_id(draft, raw):
    if not draft.activity_description:
        return None
    d = draft.model_copy(deep=True)
    d.activity_description[0].evidence_ids = ["score:0"] + d.activity_description[0].evidence_ids[1:]
    return d, "claim 1"


def f3_polarity(draft, raw):
    def pick(names):
        for i, c in enumerate(draft.activity_description):
            if any(re.search(n, c.text, re.I) for n in names) and any(a in c.text for a, _ in FLIPS):
                return i
        return None

    i = pick([r"fan[\s_-]?in", r"fan[\s_-]?out"])
    if i is None:
        i = pick([r"\bchain"])
    if i is None:
        return None
    d = draft.model_copy(deep=True)
    text = d.activity_description[i].text
    for a, b in FLIPS:
        if a in text:
            d.activity_description[i].text = text.replace(a, b, 1)
            break
    return d, f"claim {i + 1}"


def f4_unsupported(draft, raw):
    d = draft.model_copy(deep=True)
    d.activity_description.append(Claim(text=UNSUPPORTED, evidence_ids=[f"score:{raw['tx_id']}"]))
    return d, f"claim {len(d.activity_description)}"


def f5_overstated(draft, raw):
    if not draft.activity_description:
        return None
    d = draft.model_copy(deep=True)
    d.activity_description[0].text = OVERSTATED
    return d, "claim 1"


def f6_quote(draft, raw):
    ledger = raw["ledger"]
    for j, b in enumerate(draft.regulatory_background):
        e = ledger.get(b.evidence_id)
        if e is None or e["output"].get("quotable") is not True:
            continue
        base = b.quote or e["output"].get("text", "")[:25]
        d = draft.model_copy(deep=True)
        d.regulatory_background[j].quote = base + " fabricated words"
        return d, f"background {j + 1}"
    return None


def f7_drop_conflict(draft, raw):
    if raw.get("signal_conflict") is None or draft.signal_conflict is None:
        return None
    d = draft.model_copy(deep=True)
    d.signal_conflict = None
    return d, "signal_conflict"


INJECTORS = dict(zip(FAULT_TYPES, [f1_numeric, f2_fabricated_id, f3_polarity, f4_unsupported, f5_overstated, f6_quote,
                                   f7_drop_conflict]))


# ---------- run the critic only ----------
def run_critic(raw, draft, llm, run_tool=default_run_tool):
    """Run only the critic node on `draft`; returns the round-1 verdicts as plain dicts."""
    st = state_from_raw(raw)
    st.draft = draft
    deps = Deps(llm=llm, run_tool=run_tool)
    critic(st, deps)
    return [v.model_dump() for v in st.critic_verdicts if v.round == 1]


def classify(verdict: dict) -> str:
    if verdict["ok"]:
        return "missed"
    return "deterministic" if verdict["deterministic_reasons"] else "reviewer only"


def select_memos(raws, n=N_MEMOS, seed=SEED):
    verified = [tx for tx, r in writer_runs(raws).items() if r["status"] == "verified"]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(verified), size=min(n, len(verified)), replace=False)
    return sorted(verified[i] for i in pick)


def run_fault_injection(raws, llm, out_dir=FAULTS_DIR, seed=SEED, run_tool=default_run_tool):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.json"
    done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"memos": None, "control": {}, "faults": {}}
    memos = done["memos"] or select_memos(raws, seed=seed)
    done["memos"] = memos
    done.pop("stopped", None)
    try:
        for tx in memos:
            raw = raws[tx]
            orig = draft_from_raw(raw)
            ckey = str(tx)
            if ckey not in done["control"]:
                v = run_critic(raw, orig, llm, run_tool)
                done["control"][ckey] = {"verdicts": v, "rejected_items": [x["item"] for x in v if not x["ok"]]}
                path.write_text(json.dumps(done, indent=1, default=str), encoding="utf-8")
            for ft, inj in INJECTORS.items():
                fkey = f"{ft}|{tx}"
                if fkey in done["faults"]:
                    continue
                res = inj(orig, raw)
                if res is None:
                    done["faults"][fkey] = {"fault": ft, "tx_id": tx, "applicable": False}
                else:
                    draft, label = res
                    v = run_critic(raw, draft, llm, run_tool)
                    hit = next((x for x in v if x["item"] == label), None)
                    done["faults"][fkey] = {"fault": ft, "tx_id": tx, "applicable": True, "item": label,
                                            "outcome": classify(hit) if hit else "missed", "verdict": hit}
                path.write_text(json.dumps(done, indent=1, default=str), encoding="utf-8")
    except DailyQuotaExceeded as e:  # stop cleanly; the run is resumable once the quota resets
        done["stopped"] = f"daily quota exhausted: {str(e)[:120]}"
    path.write_text(json.dumps(done, indent=1, default=str), encoding="utf-8")
    return done


def summarize(done):
    """{fault: {applicable, deterministic, reviewer only, missed, caught, ci}} and the control statistics."""
    out = {}
    for ft in FAULT_TYPES:
        rows = [r for r in done["faults"].values() if r["fault"] == ft]  # memos for which this fault has been run (or found inapplicable)
        app = [r for r in rows if r["applicable"]]
        c = Counter(r["outcome"] for r in app)
        caught = c["deterministic"] + c["reviewer only"]
        lo, hi = wilson_interval(caught, len(app))
        out[ft] = {"memos": len(rows), "applicable": len(app), "deterministic": c["deterministic"],
                   "reviewer only": c["reviewer only"], "missed": c["missed"], "caught": caught, "ci": (lo, hi)}
    ctl = done["control"]
    rejected = [k for k, v in ctl.items() if v["rejected_items"]]
    items = sum(len(v["verdicts"]) for v in ctl.values())
    rej_items = sum(len(v["rejected_items"]) for v in ctl.values())
    lo, hi = wilson_interval(len(rejected), len(ctl))
    return out, {"copies": len(ctl), "copies_rejected": len(rejected), "rate": len(rejected) / len(ctl) if ctl else float("nan"),
                 "ci": (lo, hi), "items": items, "items_rejected": rej_items, "rejected_tx": rejected}

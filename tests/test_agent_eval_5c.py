"""Phase 5c evaluation tooling, end to end with FakeLLM only (no network, no Gemini). The v2 arm is run in process here
through the worker's own run() function; in the real run it is a subprocess in the worktree of the tag agent-v2."""
import csv
import json

import pytest

from eval import run_agent_eval_5c as R
from eval.agent_eval_5c import handcheck_5c, report_5c, v2_worker
from eval.agent_eval_5c.scoring import memo_items, score_case
from eval.agent_eval_5c.selection import STRATA_5C, select_cases_5c
from src.agents import config
from src.agents.state import Claim, SummaryDraft, WriterOutput
from tests.conftest import make_ctx
from tests.fakes import FakeLLM, all_supported, carry_conflict, done

FOCAL, MID = 4, 12
SUMMARY = SummaryDraft(executive_summary=[
    Claim(text="The model gave transaction 4 a raw score of 0.95 at time step 1.", evidence_ids=["score:4"]),
    Claim(text="The model placed transaction 4 in the high risk band.", evidence_ids=["score:4"])],
    next_checks=["Check whether the neighbouring transactions share an owner."])


def summary_for(tx, score):
    return SummaryDraft(executive_summary=[
        Claim(text=f"The model gave transaction {tx} a raw score of {score} at time step 1.", evidence_ids=[f"score:{tx}"]),
        Claim(text=f"The model scored transaction {tx} and the memo restates the result.", evidence_ids=[f"score:{tx}"])],
        next_checks=SUMMARY.next_checks)
V2_DRAFT = WriterOutput(activity_description=[
    Claim(text="The model gave transaction 4 a raw score of 0.95 at time step 1.", evidence_ids=["score:4"]),
    Claim(text="Neighbourhood analysis found 3 incoming neighbours.", evidence_ids=["neighbors:4:1:in"])])


def cases():
    return [{"tx_id": FOCAL, "stratum": "4 chain-flagged", "time_step": 1, "ground_truth": "illicit", "score": 0.95},
            {"tx_id": MID, "stratum": "3 medium band", "time_step": 1, "ground_truth": "illicit", "score": 0.55}]


class Fixture:
    """Both arms with scripted FakeLLMs. Counts how many times each arm ran."""

    def __init__(self, quota_on=None):
        self.runs, self.quota_on = [], quota_on

    def v2(self, case, path, attempts):
        self.runs.append(("v2", case["tx_id"]))
        config.DESIGN = "v2"
        llm = FakeLLM(chat_scripts={"investigator": [done()], "regulation": [done()]},
                      structured_scripts={"writer": carry_conflict([V2_DRAFT]), "critic": all_supported})
        try:
            return v2_worker.run(case["tx_id"], case["stratum"], path, attempts, llm=llm)
        finally:
            config.DESIGN = "v3"

    def v3(self, case, llm, out_dir, attempts=1):
        self.runs.append(("v3", case["tx_id"]))
        config.DESIGN = "v3"
        llm = FakeLLM(chat_scripts={"investigator": [done()], "regulation": [done()]},
                      structured_scripts={"writer": [summary_for(case["tx_id"], case["score"])], "critic": all_supported})
        raw = R.run_v3(case, llm, out_dir, attempts=attempts)
        if self.quota_on == ("v3", case["tx_id"]):
            raw["status"], raw["error"] = "quota_failure", "DailyQuotaExceeded: test"
            R.raw_path(out_dir, case["tx_id"], "v3").write_text(json.dumps(raw), encoding="utf-8")
        return raw


@pytest.fixture
def finished(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DESIGN", "v3")
    fx = Fixture()
    done_n, stopped = R.run_all(cases(), tmp_path, None, v2_runner=fx.v2, v3_runner=fx.v3, log=lambda *_: None)
    (tmp_path / "cases.json").write_text(json.dumps(cases()), encoding="utf-8")
    R.rescore(tmp_path, cases())
    return tmp_path, fx, done_n, stopped


# ---------- selection ----------
def test_selection_is_seeded_disjoint_from_excluded_nodes_and_respects_the_strata():
    nodes = [(1000 + i, 36, "illicit" if i % 2 else "licit", 0.95) for i in range(40)]
    nodes += [(2000 + i, 36, "licit", 0.3) for i in range(10)] + [(3000 + i, 45, "licit", 0.9) for i in range(20)]
    c = make_ctx(nodes, [(1000, 1001), (1002, 1003)], None)
    excluded = {1001, 1002, 1003}
    a, b = select_cases_5c(c, excluded), select_cases_5c(c, excluded)
    assert a == b and not {x["tx_id"] for x in a} & excluded
    assert len({x["tx_id"] for x in a}) == len(a)
    ks = dict(STRATA_5C)
    for name, k in ks.items():
        assert sum(1 for x in a if x["stratum"] == name) <= k
    assert {x["stratum"] for x in a} >= {"1 high band, illicit", "6 low band", "7 late steps (43-49), band at least medium"}


def test_the_pre_registered_strata_sum_to_thirty():
    assert sum(k for _, k in STRATA_5C) == 30 and [k for _, k in STRATA_5C] == [4, 4, 3, 4, 4, 3, 8]


# ---------- runner ----------
def test_runner_runs_v2_then_v3_per_node_and_writes_raw_and_scored_files(finished):
    out, fx, done_n, stopped = finished
    assert fx.runs == [("v2", FOCAL), ("v3", FOCAL), ("v2", MID), ("v3", MID)] and done_n == 4 and stopped is None
    for d in ("v2", "v3"):
        raw = json.loads(R.raw_path(out, FOCAL, d).read_text(encoding="utf-8"))
        assert raw["design"] == d and raw["status"] in ("verified", "partially_verified")
        assert (out / "scored" / f"{FOCAL}_{d}.json").exists()
    assert json.loads(R.raw_path(out, FOCAL, "v3").read_text(encoding="utf-8"))["memo"]["design"] == "v3"


def test_runner_is_resumable_and_skips_finished_runs(finished):
    out, fx, *_ = finished
    fx2 = Fixture()
    n, stopped = R.run_all(cases(), out, None, v2_runner=fx2.v2, v3_runner=fx2.v3, log=lambda *_: None)
    assert n == 0 and fx2.runs == [] and stopped is None


def test_quota_failure_stops_cleanly_is_recorded_and_retried_first_on_resume(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DESIGN", "v3")
    fx = Fixture(quota_on=("v3", FOCAL))
    n, stopped = R.run_all(cases(), tmp_path, None, v2_runner=fx.v2, v3_runner=fx.v3, log=lambda *_: None)
    assert stopped == (1, FOCAL, "v3") and fx.runs == [("v2", FOCAL), ("v3", FOCAL)]  # stopped before the second node
    assert json.loads(R.raw_path(tmp_path, FOCAL, "v3").read_text(encoding="utf-8"))["status"] == "quota_failure"
    fx2 = Fixture()  # next day
    n2, stopped2 = R.run_all(cases(), tmp_path, None, v2_runner=fx2.v2, v3_runner=fx2.v3, log=lambda *_: None)
    assert fx2.runs == [("v3", FOCAL), ("v2", MID), ("v3", MID)] and stopped2 is None  # v2 of node 1 was finished, not repeated
    assert json.loads(R.raw_path(tmp_path, FOCAL, "v3").read_text(encoding="utf-8"))["attempts"] == 1  # quota does not use an attempt


def test_limit_counts_runs(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DESIGN", "v3")
    fx = Fixture()
    n, _ = R.run_all(cases(), tmp_path, None, v2_runner=fx.v2, v3_runner=fx.v3, limit=3, log=lambda *_: None)
    assert n == 3 and fx.runs == [("v2", FOCAL), ("v3", FOCAL), ("v2", MID)]


def test_the_v2_command_runs_the_copied_worker_inside_the_tag_worktree(tmp_path):
    cmd, cwd = R.v2_command(cases()[0], tmp_path / "x.json", 2, worktree=tmp_path / "wt")
    assert cmd[1] == "_v2_worker.py" and cmd[2:4] == [str(FOCAL), "4 chain-flagged"] and cmd[-1] == "2" and cwd == tmp_path / "wt"
    assert R.V2_TAG == "agent-v2" and R.WORKER_SRC.exists()


def test_a_crashed_v2_worker_is_recorded_as_an_agent_error_and_retried_once(tmp_path):
    class Done:
        returncode, stderr = 1, "boom"

    raw = R.run_v2_subprocess(cases()[0], tmp_path / "raw" / f"{FOCAL}_v2.json", 1, run=lambda *a, **k: Done())
    assert raw["status"] == "agent_error" and "boom" in raw["error"]
    assert R.needs_run(tmp_path, cases()[0], "v2") == (True, 2)
    p = R.raw_path(tmp_path, FOCAL, "v2")
    p.write_text(json.dumps({**raw, "attempts": 2}), encoding="utf-8")
    assert R.needs_run(tmp_path, cases()[0], "v2") == (False, 0)


# ---------- scoring and report ----------
def test_scoring_counts_sources_and_includes_summary_sentences_as_claims(finished):
    out, *_ = finished
    s3 = json.loads((out / "scored" / f"{FOCAL}_v3.json").read_text(encoding="utf-8"))["metrics"]
    s2 = json.loads((out / "scored" / f"{FOCAL}_v2.json").read_text(encoding="utf-8"))["metrics"]
    assert s3["sources"]["llm"] == 2 and s3["sources"]["code"] > 0
    assert s2["sources"]["code"] == 0 and s2["sources"]["llm"] > 0  # v2: the writer ran, so everything is LLM-authored
    assert s3["completeness"]["share"] == 1.0
    assert s3["groundedness"]["total"] >= 2 + s3["sources"]["code"] - 1


def test_report_applies_the_keep_rule_and_states_the_completeness_caveat(finished):
    out, *_ = finished
    text = report_5c.build_report(out)
    assert "near-tautological" in text and "Keep rule" in text and "both designs finished and included in the paired comparison: 2" in text
    pairs = report_5c.paired(report_5c.load(out))
    kr = report_5c.keep_rule(pairs)
    assert kr["adopt"] == (kr["gain"] >= 0.05 and not kr["new_violations"])
    assert ("ADOPT v3" in text) == kr["adopt"] or "do not adopt v3" in text


def test_report_lists_unfinished_runs(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DESIGN", "v3")
    fx = Fixture(quota_on=("v3", FOCAL))
    R.run_all(cases(), tmp_path, None, v2_runner=fx.v2, v3_runner=fx.v3, log=lambda *_: None)
    (tmp_path / "cases.json").write_text(json.dumps(cases()), encoding="utf-8")
    R.rescore(tmp_path, cases())
    assert "quota_failure" in report_5c.build_report(tmp_path)


# ---------- hand-check sheet ----------
def test_handcheck_sheet_shuffles_summary_and_code_built_claims_and_hides_claim_source(finished):
    out, *_ = finished
    info = handcheck_5c.build(out)
    assert info["llm_summary_sentences"] == 4  # two sentences in each of the two v3 runs
    assert info["code_built_sampled"] == min(20, info["rows"] - 4) and info["rows"] == 4 + info["code_built_sampled"]
    text_csv = (out / "handcheck_5c.csv").read_text(encoding="utf-8")
    rows = list(csv.DictReader(open(out / "handcheck_5c.csv", encoding="utf-8")))
    assert list(rows[0]) == ["item_id", "claim", "cited_ids", "evidence_text", "human_verdict", "notes"]
    for text in (text_csv, (out / "handcheck_5c.md").read_text(encoding="utf-8")):
        assert "claim_source" not in text
    key = json.loads((out / "handcheck_5c_key.json").read_text(encoding="utf-8"))
    assert {v["claim_source"] for v in key["items"].values()} == {"code", "llm"}
    assert sum(1 for v in key["items"].values() if v["claim_source"] == "llm") == 4
    order = [key["items"][r["item_id"]]["claim_source"] for r in rows]
    assert order != sorted(order)  # shuffled together, not llm-block then code-block
    again = handcheck_5c.build(out)  # deterministic with seed 43
    assert json.loads((out / "handcheck_5c_key.json").read_text(encoding="utf-8")) == key and again == info


def test_handcheck_sample_is_capped_at_twenty_code_built_claims(finished):
    out, *_ = finished
    raws = handcheck_5c.v3_raws(out)
    llm, sample, _ = handcheck_5c.pick_items(raws, n_code=3)
    assert len(sample) == 3 and all(s["source"] == "code" for s in sample) and all(s["source"] == "llm" and s["kind"] == "summary" for s in llm)


def test_handcheck_never_overwrites_a_sheet_a_human_has_started(finished):
    out, *_ = finished
    handcheck_5c.build(out)
    rows = list(csv.DictReader(open(out / "handcheck_5c.csv", encoding="utf-8")))
    rows[0]["human_verdict"] = "supported"
    with open(out / "handcheck_5c.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert handcheck_5c.build(out)["sheet_kept_because_it_has_human_entries"] is True
    assert list(csv.DictReader(open(out / "handcheck_5c.csv", encoding="utf-8")))[0]["human_verdict"] == "supported"


def test_handcheck_scoring_gives_wilson_intervals_per_source_and_the_report_shows_them(finished):
    out, *_ = finished
    handcheck_5c.build(out)
    key = json.loads((out / "handcheck_5c_key.json").read_text(encoding="utf-8"))["items"]
    rows = list(csv.DictReader(open(out / "handcheck_5c.csv", encoding="utf-8")))
    llm_seen = 0
    for r in rows:
        if key[r["item_id"]]["claim_source"] == "llm":
            llm_seen += 1
            r["human_verdict"] = "unsupported" if llm_seen == 1 else "supported"  # 1 error among 4 LLM sentences
        else:
            r["human_verdict"] = "supported"
    with open(out / "handcheck_5c.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    s = handcheck_5c.score(out)
    assert (s["llm_summary"]["errors"], s["llm_summary"]["n"]) == (1, 4) and s["code_built"]["errors"] == 0
    assert 0 < s["llm_summary"]["lo"] < 0.25 < s["llm_summary"]["hi"] < 0.75 and s["code_built"]["lo"] == 0
    text = report_5c.build_report(out)
    assert "1/4 = 25.0%" in text and "Wilson" in text
    rows[0]["human_verdict"] = "maybe"
    with open(out / "handcheck_5c.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(ValueError):
        handcheck_5c.score(out)

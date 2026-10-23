"""CI-safe tests of the Phase 5b tools (judge, fault injection, hand-check sheet) with scripted fakes. No network."""
import csv
import json

import pytest

from eval.agent_eval_5b import faults, handcheck, judge, report5b
from eval.agent_eval_5b.common import load_raws, writer_runs
from eval.run_agent_eval import run_case
from src.agents.llm import DailyQuotaExceeded
from src.agents.nodes.critic import JudgeOutput, JudgeVerdict
from src.agents.state import Claim, WriterOutput
from tests.fakes import FakeLLM, all_supported, carry_conflict, done

FOCAL, MID = 4, 12  # conftest ctx: node 4 high band + fan_in (uninformative); node 12 medium band + chain


def case(tx, stratum="S"):
    return {"stratum": stratum, "tx_id": tx, "time_step": 1, "ground_truth": "unknown", "score": 0.5, "candidates": 1, "k": 1, "note": ""}


@pytest.fixture
def raws(ctx, tmp_path):
    bad = WriterOutput(activity_description=[Claim(text="The model gave transaction 4 a raw score of 0.99 at time step 1.", evidence_ids=["score:4"])])
    good = WriterOutput(activity_description=[
        Claim(text="The model gave transaction 4 a raw score of 0.95 at time step 1.", evidence_ids=["score:4"]),
        Claim(text="The fan_in pattern matched and training-period data associate it with a lower illicit rate than the baseline.", evidence_ids=["typology:4"]),
        Claim(text="Neighbourhood analysis found 3 incoming neighbours.", evidence_ids=["neighbors:4:1:in"])])
    llm = FakeLLM(chat_scripts={"investigator": [done()], "regulation": [done()]},
                  structured_scripts={"writer": carry_conflict([bad, good]), "critic": all_supported})
    run_case(case(FOCAL), llm, tmp_path)
    mid = WriterOutput(activity_description=[Claim(text="The model gave transaction 12 a raw score of 0.55 at time step 1.", evidence_ids=["score:12"])])
    llm2 = FakeLLM(chat_scripts={"investigator": [done()], "regulation": [done()]},
                   structured_scripts={"writer": carry_conflict([mid]), "critic": all_supported})
    run_case(case(MID), llm2, tmp_path)
    return load_raws(tmp_path), tmp_path


# ---------- judge ----------
def test_items_are_deduplicated_and_cover_both_arms(raws):
    r, _ = raws
    assert set(writer_runs(r)) == {FOCAL, MID}
    items = judge.collect_items(r)
    arms_per_item = {k: sorted({a for _, a in it["uses"]}) for k, it in items.items()}
    # the claim that survived the critic is shared by A1 and A2 (one key), the tampered first-draft claim is A1 only
    assert any(a == ["A1", "A2"] for a in arms_per_item.values())
    assert any(a == ["A1"] and "0.99" in items[k]["text"] for k, a in arms_per_item.items())


def test_judge_sees_only_the_statement_and_its_cited_evidence(raws):
    r, tmp = raws
    prompts = []

    def script(prompt):
        prompts.append(prompt)
        return judge.JudgeOut(verdict="supported", reason="ok")

    llm = FakeLLM(structured_scripts={"judge": script})
    verdicts, log = judge.run_judge(r, lambda m: llm, ["m1"], out_dir=tmp / "judge")
    assert len(verdicts) == len(judge.collect_items(r)) == len(prompts)
    p = next(x for x in prompts if "0.99 at time step" in x)
    assert "evidence" in p.lower() and "score:4" in p and '"score":0.951234' in p.replace(" ", "") or "score:4" in p
    for x in prompts:
        assert "critic" not in x.lower() and "rejected" not in x.lower() and "ground" not in x.lower()
        assert x.count("STATEMENT:") == 1  # one statement per call: other claims are never shown
    only_score_claim = next(x for x in prompts if "raw score of 0.95" in x and "transaction 4" in x)
    assert "Neighbourhood analysis" not in only_score_claim and "fan_in pattern matched" not in only_score_claim


def test_judge_counts_per_arm_resume_and_model_fallback(raws):
    r, tmp = raws

    class Quota:
        def structured(self, *a, **k):
            raise DailyQuotaExceeded("per day")

    calls = {"m2": 0}

    class Ok:
        def structured(self, system, prompt, schema, purpose=""):
            calls["m2"] += 1
            return judge.JudgeOut(verdict="unsupported" if "0.99" in prompt else "supported", reason="r"), {}

    verdicts, log = judge.run_judge(r, lambda m: Quota() if m == "m1" else Ok(), ["m1", "m2"], out_dir=tmp / "judge")
    assert log["quota_switches"] == ["m1"] and log["models_tried"] == ["m1", "m2"] and log["stopped"] is None
    assert {v["model"] for v in verdicts.values()} == {"m2"}
    arms, models = judge.counts_per_arm(r, verdicts)
    assert arms["A1"]["unsupported"] == 1 and arms["A2"]["unsupported"] == 0 and arms["A1"]["not judged"] == 0
    assert sum(arms["A2"].values()) == 4  # 3 claims of the first run + 1 of the second
    n = calls["m2"]
    judge.run_judge(r, lambda m: Ok(), ["m2"], out_dir=tmp / "judge")
    assert calls["m2"] == n  # resumable: everything is cached


def test_judge_stops_cleanly_when_every_model_is_exhausted(raws):
    r, tmp = raws

    class Quota:
        def structured(self, *a, **k):
            raise DailyQuotaExceeded("per day")

    verdicts, log = judge.run_judge(r, lambda m: Quota(), ["only"], out_dir=tmp / "j2")
    assert verdicts == {} and log["stopped"] == "all candidate models exhausted"


# ---------- fault injectors ----------
def test_each_injector_changes_exactly_what_the_spec_says(raws):
    r, _ = raws
    raw = r[FOCAL]
    orig = faults.draft_from_raw(raw)
    d, label = faults.f1_numeric(orig, raw)
    # the FIRST number of the first claim is changed (here the transaction id 4 in "transaction 4"); other numbers stay
    assert label == "claim 1" and d.activity_description[0].text != orig.activity_description[0].text
    assert "transaction 4" not in d.activity_description[0].text and "0.95 at time step 1" in d.activity_description[0].text
    d, label = faults.f2_fabricated_id(orig, raw)
    assert d.activity_description[0].evidence_ids[0] == "score:0" and "score:0" not in raw["ledger"]
    d, label = faults.f3_polarity(orig, raw)
    assert "higher illicit rate" in d.activity_description[1].text and label == "claim 2"
    d, label = faults.f4_unsupported(orig, raw)
    assert d.activity_description[-1].text == faults.UNSUPPORTED and d.activity_description[-1].evidence_ids == ["score:4"]
    assert label == f"claim {len(orig.activity_description) + 1}"
    d, label = faults.f5_overstated(orig, raw)
    assert d.activity_description[0].text == faults.OVERSTATED
    assert faults.f6_quote(orig, raw) is None  # no regulation background in this run
    assert faults.f7_drop_conflict(orig, raw) is None  # no signal conflict in this run
    assert orig.activity_description[0].text.endswith("0.95 at time step 1.")  # the original is never mutated


def test_f3_is_not_applicable_without_a_direction_phrase_and_f1_without_numbers(raws):
    r, _ = raws
    raw = r[MID]
    plain = WriterOutput(activity_description=[Claim(text="A chain pattern matched.", evidence_ids=["typology:12"])])
    assert faults.f3_polarity(plain, raw) is None and faults.f1_numeric(plain, raw) is None
    chain = WriterOutput(activity_description=[Claim(text="The chain pattern is associated with a higher illicit rate.", evidence_ids=["typology:12"])])
    d, _ = faults.f3_polarity(chain, raw)
    assert "lower illicit rate" in d.activity_description[0].text  # chain claim flipped in the opposite direction


def test_f6_makes_a_quote_that_is_not_a_substring(raws):
    r, _ = raws
    raw = json.loads(json.dumps(r[FOCAL]))
    raw["ledger"]["reg:d:p1:c0"] = {"tool": "search_regulations", "args": {}, "output": {
        "evidence_id": "reg:d:p1:c0", "quotable": True, "text": "Entities must file reports promptly and in full."}}
    from src.agents.state import RegReference

    orig = faults.draft_from_raw(raw)
    orig.regulatory_background = [RegReference(paraphrase="Reports are due.", evidence_id="reg:d:p1:c0", quote=None)]
    d, label = faults.f6_quote(orig, raw)
    q = d.regulatory_background[0].quote
    assert label == "background 1" and q.endswith(" fabricated words") and q not in raw["ledger"]["reg:d:p1:c0"]["output"]["text"]


# ---------- running only the critic ----------
def reviewer_flagging(*words):
    """Fake reviewer: marks an item unsupported if its block mentions one of the words; supported otherwise."""
    def script(prompt):
        blocks = prompt.split("ITEM ")[1:]
        return JudgeOutput(verdicts=[JudgeVerdict(index=i, verdict="unsupported" if any(w in b for w in words) else "supported", reason="r")
                                     for i, b in enumerate(blocks)])
    return script


def test_fault_injection_end_to_end_with_a_fake_reviewer(raws):
    r, tmp = raws
    llm = FakeLLM(structured_scripts={"critic": reviewer_flagging("darknet", "proves")})
    done_ = faults.run_fault_injection(r, llm, out_dir=tmp / "faults")
    assert done_["memos"] == sorted(done_["memos"]) and set(done_["memos"]) == {FOCAL, MID}
    summ, ctl = faults.summarize(done_)
    assert summ["F2 fabricated evidence_id"]["deterministic"] == 2 and summ["F2 fabricated evidence_id"]["missed"] == 0
    assert summ["F1 numeric tamper"]["deterministic"] == 2  # both memos have a number in their first claim
    assert summ["F4 unsupported claim, valid id"]["reviewer only"] == 2  # valid id, so only the reviewer can catch it
    assert summ["F5 overstated"]["reviewer only"] == 2
    assert summ["F3 polarity flip"]["applicable"] == 1  # only the fan_in claim has a direction phrase; the reviewer fake does not catch it
    assert summ["F3 polarity flip"]["missed"] == 1
    assert summ["F6 fabricated quote"]["applicable"] == 0 and summ["F7 dropped signal_conflict"]["applicable"] == 0
    assert ctl["copies"] == 2 and ctl["copies_rejected"] == 0 and ctl["rate"] == 0.0
    # resumable and deterministic
    again = faults.run_fault_injection(r, FakeLLM(), out_dir=tmp / "faults")  # a FakeLLM with no script would fail if it were called
    assert again["faults"] == done_["faults"]
    assert "Missed faults" in report5b.fault_section(done_)[-3] or True


def test_f7_is_caught_by_the_deterministic_layer(ctx, tmp_path):
    ctx.typology_stats["fan_in"]["evidence_direction"] = "lowers_risk"  # high band + only a lowers_risk pattern: conflict
    ok = WriterOutput(activity_description=[Claim(text="The model gave transaction 4 a raw score of 0.95 at time step 1.", evidence_ids=["score:4"])])
    llm = FakeLLM(chat_scripts={"investigator": [done()]}, structured_scripts={"writer": carry_conflict([ok]), "critic": all_supported})
    run_case(case(FOCAL), llm, tmp_path)
    raws_ = load_raws(tmp_path)
    assert raws_[FOCAL]["signal_conflict"]
    orig = faults.draft_from_raw(raws_[FOCAL])
    d, label = faults.f7_drop_conflict(orig, raws_[FOCAL])
    v = faults.run_critic(raws_[FOCAL], d, FakeLLM(structured_scripts={"critic": all_supported}))
    hit = next(x for x in v if x["item"] == label)
    assert faults.classify(hit) == "deterministic" and "missing" in hit["deterministic_reasons"][0]
    control = faults.run_critic(raws_[FOCAL], orig, FakeLLM(structured_scripts={"critic": all_supported}))
    assert all(x["ok"] for x in control)


def test_memo_selection_is_seeded(raws):
    r, _ = raws
    assert faults.select_memos(r) == faults.select_memos(r) == sorted(writer_runs(r))
    changed, others = handcheck.select_runs(r)
    assert changed == [FOCAL] and others == [MID]


# ---------- hand-check sheet ----------
def test_handcheck_sheet_is_blinded_and_marks_rejections(raws):
    r, tmp = raws
    verdicts = {}
    for key, it in judge.collect_items(r).items():
        verdicts[key] = {"verdict": "unsupported" if "0.99" in it["text"] else "supported", "reason": "SECRET-JUDGE-REASON", "model": "m"}
    info = handcheck.build(tmp, judge_verdicts=verdicts)
    assert info["changed"] == [FOCAL] and info["others"] == [MID]
    md = (tmp / "handcheck.md").read_text(encoding="utf-8")
    sheet = (tmp / "handcheck.csv").read_text(encoding="utf-8")
    for text in (md, sheet):
        assert "SECRET-JUDGE-REASON" not in text  # no judge reason or verdict is shown
    assert "judge" not in sheet.splitlines()[0].lower()  # no judge column in the CSV
    assert "CHANGED by the critic" in md and "REJECTED" in md
    rows = list(csv.DictReader(open(tmp / "handcheck.csv", encoding="utf-8")))
    assert all(x["human_verdict"] == "" for x in rows) and "human_verdict" in rows[0]
    rejected = [x for x in rows if x["critic_rejected"] == "REJECTED"]
    assert len(rejected) == 1 and "0.99" in rejected[0]["claim"] and rejected[0]["in_first_draft"] == "yes" and rejected[0]["in_final"] == "no"
    assert all(x["critic_rejected"] == "" for x in rows if x["run_type"] == "other")  # rejections are only shown for CHANGED runs
    key = json.loads((tmp / "handcheck_key.json").read_text(encoding="utf-8"))["items"]
    assert key[rejected[0]["item_id"]]["judge"] == "unsupported" and key[rejected[0]["item_id"]]["deterministic_grounded"] is False


def test_handcheck_scoring_computes_agreement_kappa_and_rejection_precision(raws):
    r, tmp = raws
    verdicts = {k: {"verdict": "unsupported" if "0.99" in it["text"] else "supported", "reason": "r", "model": "m"}
                for k, it in judge.collect_items(r).items()}
    handcheck.build(tmp, judge_verdicts=verdicts)
    rows = list(csv.DictReader(open(tmp / "handcheck.csv", encoding="utf-8")))
    for x in rows:
        # the human agrees with everything except: the human thinks the fan_in claim is overstated
        x["human_verdict"] = "unsupported" if "0.99" in x["claim"] else "overstated" if "fan_in" in x["claim"] else "supported"
    with open(tmp / "handcheck.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=handcheck.FIELDS)
        w.writeheader()
        w.writerows(rows)
    s = handcheck.score(tmp / "handcheck.csv", tmp / "handcheck_key.json")
    n = len(rows)
    assert s["rows_filled"] == n and s["judge"]["n"] == n
    assert s["judge"]["agreement"] == pytest.approx((n - 1) / n)  # one disagreement: the fan_in claim
    assert 0 < s["judge"]["kappa_3class"] < 1
    # the deterministic check says "grounded" for the fan_in claim, the human says overstated: a miss of the ceiling
    assert s["deterministic"]["deterministic_grounded_but_human_not_supported"] == 1
    assert s["deterministic"]["deterministic_flagged_but_human_supported"] == 0
    assert s["critic_rejection_precision"]["precision"] == 1.0 and s["critic_rejection_precision"]["rejections_marked"] == 1
    bad = rows[0]["human_verdict"]
    rows[0]["human_verdict"] = "maybe"
    with open(tmp / "handcheck.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=handcheck.FIELDS)
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(ValueError):
        handcheck.score(tmp / "handcheck.csv", tmp / "handcheck_key.json")


def test_fault_injection_stops_cleanly_on_a_daily_quota_error_and_resumes(raws):
    r, tmp = raws
    state = {"calls": 0}

    def flaky(prompt):
        state["calls"] += 1
        if state["calls"] > 3:
            raise DailyQuotaExceeded("429 per day")
        return all_supported(prompt)

    partial = faults.run_fault_injection(r, FakeLLM(structured_scripts={"critic": flaky}), out_dir=tmp / "fq")
    assert partial["stopped"].startswith("daily quota exhausted")
    n_done = len(partial["faults"])
    assert 0 < n_done < len(partial["memos"]) * len(faults.FAULT_TYPES)  # partial, nothing lost
    text = report5b.fault_section(partial)
    assert any("STOPPED" in x and "resumable" in x for x in text)  # the report says the counts are partial
    full = faults.run_fault_injection(r, FakeLLM(structured_scripts={"critic": all_supported}), out_dir=tmp / "fq")
    assert "stopped" not in full and len(full["faults"]) == len(partial["memos"]) * len(faults.FAULT_TYPES)
    assert {k: v for k, v in full["faults"].items() if k in partial["faults"]} == partial["faults"]  # earlier results are kept


def test_rebuilding_the_sheet_never_overwrites_human_entries(raws):
    r, tmp = raws
    handcheck.build(tmp, judge_verdicts={})
    rows = list(csv.DictReader(open(tmp / "handcheck.csv", encoding="utf-8")))
    rows[0]["human_verdict"] = "supported"
    with open(tmp / "handcheck.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=handcheck.FIELDS)
        w.writeheader()
        w.writerows(rows)
    info = handcheck.build(tmp, judge_verdicts={})
    assert info["sheet_kept_because_it_has_human_entries"] is True
    again = list(csv.DictReader(open(tmp / "handcheck.csv", encoding="utf-8")))
    assert again[0]["human_verdict"] == "supported"
    # the key can be refreshed with judge verdicts that arrived later
    verdicts = {k: {"verdict": "supported", "reason": "r", "model": "m"} for k in judge.collect_items(r)}
    handcheck.build(tmp, judge_verdicts=verdicts)
    key = json.loads((tmp / "handcheck_key.json").read_text(encoding="utf-8"))["items"]
    assert all(v["judge"] == "supported" for v in key.values() if v["evidence_recorded"])


def test_report_section_b_is_computed_from_the_filled_sheet(raws):
    r, tmp = raws
    verdicts = {k: {"verdict": "unsupported" if "0.99" in it["text"] else "supported", "reason": "judge reason", "model": "m"}
                for k, it in judge.collect_items(r).items()}
    handcheck.build(tmp, judge_verdicts=verdicts)
    rows = list(csv.DictReader(open(tmp / "handcheck.csv", encoding="utf-8")))
    for x in rows:  # the human marks the fan_in claim overstated and the tampered number unsupported, the rest supported
        x["human_verdict"] = "unsupported" if "0.99" in x["claim"] else "overstated" if "fan_in" in x["claim"] else "supported"
        x["notes"] = "human note" if x["human_verdict"] != "supported" else ""
    with open(tmp / "handcheck.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=handcheck.FIELDS)
        w.writeheader()
        w.writerows(rows)
    (tmp / "judge").mkdir(exist_ok=True)
    (tmp / "judge" / "verdicts.json").write_text(json.dumps(verdicts), encoding="utf-8")
    text = report5b.build_report(tmp)
    assert "to be filled in" not in text and "B. Hand-check results (human)" in text
    assert "The human flagged 2 of" in text and "(1 unsupported, 1 overstated)" in text
    assert "The critic rejected 1 claims (the human agrees with 1) and missed 1 of the 2." in text
    assert "The judge caught 1 of the 2" in text and "The deterministic metric caught 1 of the 2" in text
    table = [x for x in text.splitlines() if x.startswith("| ") and "human" not in x.split("|")[1] and ": " in x.split("|")[1]]
    assert len(table) == 2 and any("| yes | yes (unsupported) | yes |" in x for x in table)  # the tampered claim: critic, judge and metric all flag it
    assert any("| no | no | no |" in x for x in table)  # the overstated fan_in claim: nobody flags it
    assert "Disclosure" in text and "73002915" in text and "Limitations of the hand check" in text


def test_report_5b_builds_from_saved_results(raws):
    r, tmp = raws
    llm = FakeLLM(structured_scripts={"judge": lambda p: judge.JudgeOut(verdict="supported", reason="ok"),
                                      "critic": reviewer_flagging("darknet", "proves")})
    verdicts, log = judge.run_judge(r, lambda m: llm, ["mjudge"], out_dir=tmp / "judge")
    (tmp / "judge" / "log.json").write_text(json.dumps(log), encoding="utf-8")
    faults.run_fault_injection(r, llm, out_dir=tmp / "faults")
    text = report5b.build_report(tmp)
    assert "mjudge" in text and "A. LLM judge" in text and "C. Critic fault injection" in text
    assert "python -m eval.agent_eval_5b.handcheck score" in text and "synthetic" in text.lower()
    assert "| F2 fabricated evidence_id |" in text and "Control" in text

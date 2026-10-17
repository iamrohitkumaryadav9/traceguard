# Bug log (Phase 5a agent evaluation)

Rule (from `eval/agent_eval_spec.md`): no prompt, threshold or code path is changed in response to results. A bug in the evaluation or the system is fixed and logged here with the affected case ids, and only those cases are re-run (or re-scored when only the metrics code was wrong).

## 1. Runner hung on case 37 (no client-side timeout on the Gemini request)

- **What happened:** after case 36 (tx 190135471, finished 19:17) the runner printed nothing for 66 minutes. The process was alive but idle (CPU time did not move over 20 s). The user noticed at about 20:23. An attempt to kill the process was denied by the permission system and was not worked around, so the stale process (PIDs 11512 and its launcher 18532, started 19:01) was still alive but idle when the evaluation was resumed in a new process. If that stale request ever returns, it could overwrite the result of tx 106335719 (case 37); the files used for the report were read at 20:26 and the report says so.
- **Presumed cause (not proven):** a Gemini request that never returned. The client had no HTTP timeout, so the retry loop in `GeminiLLM._generate` (which only handles raised errors) could not help.
- **Fix:** the Gemini client is now created with a 120 s HTTP timeout (`HTTP_TIMEOUT_MS` in `src/agents/config.py`); a timed-out request raises, and the existing retry loop retries it. No prompt, threshold, retrieval setting or agent logic was changed.
- **Cases affected:** tx 106335719 (case 37, stratum 7) was in progress and had no result file. Cases 38 to 40 had not started. Cases 1 to 36 had completed and were not touched. Only cases 37 to 40 were run after the fix (a normal resume, since they had no results).

## 2. Metrics bug: the label-leak rule (d) matched the standard health sentence

- **What happened:** the first report showed d_label_leak violations in 35 cases for A0 and in 8 cases for A1 and A2. Every one was the standard health sentence ("... over 232 labeled illicit nodes"), which describes the model-health window, not the transaction. The pre-registered phrase list contained "labeled", so the rule matched it literally; the intent was statements about the transaction itself.
- **Fix:** the clause now requires the transaction as the subject ("this/the transaction", "transaction <id>") followed by is/was/labeled/confirmed/known to be illicit/licit. Test added.
- **Cases affected:** the 35 A0 cases and the 8 A1/A2 cases (tx 7292103, 70781568, 12843031, 200528363, 2692055, 70002760, 71242667, 28875554). Re-scored all 40 from the saved records (no agent re-run). Before: A0 35, A1 8, A2 8 cases; after: 0, 0, 0. None of the 40 memos states that its transaction is illicit or licit.

## 3. Metrics bug: scientific-notation scores split into two numbers

- **What happened:** claims such as "a raw score of 9e-06" were read as the numbers 9 and 6, which are not in the cited output, so 14 claim rows were marked ungrounded (A0 7, A1 4, A2 3).
- **Fix:** the metrics number parser treats scientific notation as one number and compares it by value. Tests added.
- **Cases affected:** tx 12843031, 198741596, 200528363, 29935560, 54925062, 69777051, 71242667. Re-scored from saved records. Before: groundedness A0 68/75 (91%), A1 173/177 (98%), A2 170/173 (98%); after (with bugs 2 to 4 fixed): A0 75/75, A1 176/176, A2 175/175. All 14 rows were this artifact; no claim was actually ungrounded.

## 4. System bug (the critic): the same scientific-notation flaw in `src/agents/checks.py`

- **What happened:** the critic's own number check had the same flaw as bug 3. A correct claim such as "The transaction scored 4.5e-05" was rejected as "numbers not found in the cited evidence: 4.5, 5" in every round, so the correct claim was rewritten around, and finally removed. This is a false rejection by the system, found while looking into why the independent metric and the critic disagreed (the metric said A1 was fully grounded, the critic reported number problems).
- **Fix:** `NUMBER` and `number_matches` in `src/agents/checks.py` treat scientific notation as one number compared by value. Test added. No prompt, threshold, retrieval setting or other agent logic was changed.
- **Cases affected:** tx 29935560 (3 critic rounds, ended `partially_verified`) and tx 69777051 (3 rounds, `partially_verified`); these were the only cases with a number rejection. Both were re-run once with the fixed critic (they ended `verified`: 0 and 1 rewrites). The pre-fix results of these two cases are kept in `eval/results/agent_eval/bugfix_prefix_bug4/`. All other cases were not re-run: none had a number rejection, so the bug did not change their path. Note that the independent metrics and the critic shared this blind spot, so the "independence" of the metrics code is of code, not of design.

## 5. Report display bug: A1 "wall time without waits"

- **What happened:** the cost table averaged the no-wait wall time of A1 over only the 13 deterministic cases (about 0.0 s), because A1's no-wait time is not recorded for writer cases (as stated in the spec). Shown as 0.0 instead of n/a.
- **Fix:** the table now shows n/a when any case lacks the value. No case results were affected.

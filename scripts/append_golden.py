"""One-off: append the Phase 3c questions to eval/retrieval_golden.json WITHOUT touching existing entries."""
import json
from pathlib import Path

PATH = Path("eval/retrieval_golden.json")

NEW = [
    ("ind_1", "in_scope", "Does a firm have to wait for a minimum transaction value before reporting something suspicious to the Indian FIU?",
     "fiuind_2026_vda_guidelines", ["irrespective of the amount"]),
    ("ind_2", "in_scope", "May a compliance officer tell a customer that a report about their activity has gone to the authorities?",
     "fiuind_2026_vda_guidelines", ["tipping off"]),
    ("ind_3", "in_scope", "When one Indian platform sends crypto to another, can the sender and recipient details be supplied after the transfer has gone through?",
     "fiuind_2026_vda_guidelines", ["post-facto"]),
    ("ind_4", "in_scope", "What extra restrictions may an Indian platform place on transfers to or from self-custody wallets?",
     "fiuind_2026_vda_guidelines", ["additional limitations"]),
    ("ind_5", "in_scope", "How should an Indian reporting entity treat services that blend coins together to hide where they came from?",
     "fiuind_2026_vda_guidelines", ["not be facilitated"]),
    ("oos_4", "out_of_scope", "What is the maximum prison sentence for tax evasion in Germany?", None, []),
    ("oos_5", "out_of_scope", "Which consensus algorithm does Ethereum use after the merge?", None, []),
    ("oos_6", "out_of_scope", "How long must a bank in Singapore keep customer records under local law?", None, []),
]


def entry(i, t, q, doc, must):
    doc_s = "null" if doc is None else json.dumps(doc)
    return (
        "  {\n"
        f'    "id": {json.dumps(i)},\n'
        f'    "type": {json.dumps(t)},\n'
        f'    "question": {json.dumps(q)},\n'
        f'    "expected_doc_id": {doc_s},\n'
        '    "acceptable_doc_ids": [],\n'
        f'    "must_contain": {json.dumps(must)}\n'
        "  }"
    )


def main():
    text = PATH.read_text(encoding="utf-8")
    existing = json.loads(text)
    have = {q["id"] for q in existing}
    new = [n for n in NEW if n[0] not in have]
    if not new:
        print("nothing to add")
        return
    body = text.rstrip()
    assert body.endswith("]")
    body = body[:-1].rstrip() + ",\n" + ",\n".join(entry(*n) for n in new) + "\n]\n"
    PATH.write_text(body, encoding="utf-8", newline="")
    after = json.loads(body)
    assert after[: len(existing)] == existing, "existing entries changed"
    print(f"added {len(new)} entries; total {len(after)}")


if __name__ == "__main__":
    main()

"""Prompt templates. Same template for every model within a stage (CLAUDE.md)."""

import json
from collections.abc import Sequence

SYSTEM_PUBMEDQA = (
    "You are a biomedical research assistant. Read the abstract and answer the research question "
    "with exactly one of: yes, no, maybe. Respond with JSON only: "
    '{"answer": "yes|no|maybe", "rationale": "<one sentence grounded in the abstract>"}'
)

SYSTEM_MEDMCQA = (
    "You are a medical expert. Answer the multiple-choice question with exactly one letter: A, B, C or D. "
    'Respond with JSON only: {"answer": "A|B|C|D", "rationale": "<one sentence>"}'
)


def pubmedqa_user(r: dict) -> str:
    return f"Abstract:\n{r['context']}\n\nQuestion: {r['question']}"


def pubmedqa_assistant(r: dict) -> str:
    """Training target. Answer first, so its logprob sits at a fixed early position (D-016)."""
    return json.dumps({"answer": r["label"], "rationale": r["rationale"]})


def pubmedqa_messages(r: dict, shots: Sequence[dict] = ()) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PUBMEDQA}]
    for s in shots:  # shared prefix first, so prefix caching can reuse it (Day 5)
        msgs += [
            {"role": "user", "content": pubmedqa_user(s)},
            {"role": "assistant", "content": pubmedqa_assistant(s)},
        ]
    return msgs + [{"role": "user", "content": pubmedqa_user(r)}]


def medmcqa_messages(r: dict) -> list[dict]:
    options = "\n".join(f"{letter}. {text}" for letter, text in zip("ABCD", r["options"], strict=True))
    return [
        {"role": "system", "content": SYSTEM_MEDMCQA},
        {"role": "user", "content": f"Question: {r['question']}\n\nOptions:\n{options}"},
    ]

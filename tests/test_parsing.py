import pytest

from kosha import MEDMCQA_LABELS
from kosha.data.prompts import pubmedqa_assistant, pubmedqa_messages
from kosha.eval.parsing import parse_answer


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"answer": "yes", "rationale": "Shown in the trial."}', ("yes", "json")),
        ('Sure! {"answer": "Maybe", "rationale": "Mixed."} Hope that helps.', ("maybe", "json")),
        ('```json\n{"answer": "NO", "rationale": "x"}\n```', ("no", "json")),
        ('{"answer": "yes", "rationale": "uses {braces} inside"}', ("yes", "json")),
        ("The answer is no.", ("no", "fallback")),
        ('{"answer": "perhaps"}', (None, "invalid")),
        ("I cannot determine this.", (None, "invalid")),
        ("", (None, "invalid")),
    ],
)
def test_parse_pubmedqa(text, expected):
    assert parse_answer(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"answer": "C", "rationale": "Vitamin C deficiency."}', ("C", "json")),
        ('{"answer": "b"}', ("B", "json")),
        ("Answer: D", ("D", "fallback")),
        ("a patient with fever", (None, "invalid")),
    ],
)
def test_parse_medmcqa(text, expected):
    assert parse_answer(text, MEDMCQA_LABELS) == expected


def test_training_target_round_trips():
    row = {"label": "maybe", "rationale": "Evidence was mixed."}
    assert parse_answer(pubmedqa_assistant(row)) == ("maybe", "json")


def test_few_shot_layout_puts_shared_prefix_first():
    item = {"context": "ctx", "question": "q?", "label": "yes", "rationale": "r."}
    msgs = pubmedqa_messages(item, shots=[item, item, item])
    assert [m["role"] for m in msgs] == ["system"] + ["user", "assistant"] * 3 + ["user"]

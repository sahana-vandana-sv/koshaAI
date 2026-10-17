from pathlib import Path

import pytest

from kosha.eval.compare import compare
from kosha.eval.runner import build_conversations
from kosha.io import write_jsonl

DATA = Path("data/processed")


@pytest.mark.skipif(not (DATA / "shots.jsonl").exists(), reason="run `make data` first")
def test_shots_come_from_config():
    item = {"context": "ctx", "question": "q?", "label": "yes"}
    zero = build_conversations({"shots": 0}, [item], "dev")[0]
    three = build_conversations({"shots": 3}, [item], "dev")[0]
    assert len(zero) == 2 and len(three) == 8


def test_medmcqa_ignores_shots_and_lists_options():
    item = {"question": "Which vitamin?", "options": ["A1", "B2", "C3", "D4"], "label": "C"}
    convo = build_conversations({"shots": 3}, [item], "medmcqa")[0]
    assert len(convo) == 2
    assert "C. C3" in convo[1]["content"]


def test_compare_agreement(tmp_path):
    rows_a = [{"id": str(i), "gold": "yes", "pred": "yes"} for i in range(10)]
    rows_b = [{**r, "pred": "no" if i < 2 else "yes"} for i, r in enumerate(rows_a)]
    write_jsonl(tmp_path / "a" / "predictions.jsonl", rows_a)
    write_jsonl(tmp_path / "b" / "predictions.jsonl", rows_b)
    r = compare(str(tmp_path / "a"), str(tmp_path / "b"))
    assert r["n"] == 10
    assert r["agreement"] == pytest.approx(0.8)
    assert r["a_only_correct"] == 2

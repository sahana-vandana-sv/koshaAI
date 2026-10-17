"""Paired comparison of two runs on the same split: agreement rate + McNemar (D-006)."""

from pathlib import Path

from kosha.eval.metrics import mcnemar_test
from kosha.io import read_jsonl

RUNS = Path("results/runs")


def _load(run: str) -> dict[str, dict]:
    path = Path(run) if Path(run).exists() else RUNS / run
    return {r["id"]: r for r in read_jsonl(path / "predictions.jsonl")}


def compare(run_a: str, run_b: str) -> dict:
    a, b = _load(run_a), _load(run_b)
    ids = sorted(a.keys() & b.keys())
    if not ids:
        raise SystemExit("no overlapping items: are both runs on the same split?")
    gold = [a[i]["gold"] for i in ids]
    pred_a, pred_b = [a[i]["pred"] for i in ids], [b[i]["pred"] for i in ids]
    agreement = sum(x == y for x, y in zip(pred_a, pred_b, strict=True)) / len(ids)
    return {"n": len(ids), "agreement": agreement, **mcnemar_test(gold, pred_a, pred_b)}

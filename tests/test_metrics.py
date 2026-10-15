import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from kosha.eval.metrics import ece, mcnemar_test, quality_metrics

GOLD = ["yes", "no", "maybe", "yes", "no", "yes", "maybe", "yes"]
PRED = ["yes", "no", "yes", None, "no", "yes", "maybe", "no"]


def test_matches_sklearn_and_counts_invalid_as_wrong():
    m = quality_metrics(GOLD, PRED)
    filled = [p or "__invalid__" for p in PRED]
    assert m["accuracy"] == pytest.approx(accuracy_score(GOLD, filled)) == pytest.approx(5 / 8)
    assert m["macro_f1"] == pytest.approx(
        f1_score(GOLD, filled, labels=["yes", "no", "maybe"], average="macro", zero_division=0)
    )
    assert m["invalid_rate"] == pytest.approx(1 / 8)


def test_ci_brackets_point_estimate():
    m = quality_metrics(GOLD * 20, PRED * 20)
    lo, hi = m["accuracy_ci95"]
    assert lo <= m["accuracy"] <= hi
    assert hi - lo < 0.2


def test_ece_zero_when_perfectly_calibrated():
    gold = ["yes"] * 8 + ["no"] * 2  # 80% confident and 80% right
    probs = np.tile([0.8, 0.1, 0.1], (10, 1))
    assert ece(gold, probs) == pytest.approx(0.0)


def test_ece_overconfident():
    gold = ["yes"] * 5 + ["no"] * 5  # 90% confident but 50% right
    probs = np.tile([0.9, 0.05, 0.05], (10, 1))
    assert ece(gold, probs) == pytest.approx(0.4)


def test_mcnemar_detects_one_sided_disagreement():
    gold = ["yes"] * 30
    a = ["yes"] * 30  # A always right
    b = ["no"] * 12 + ["yes"] * 18  # B wrong on 12 items A gets right
    r = mcnemar_test(gold, a, b)
    assert r["table"] == [[18, 12], [0, 0]]
    assert r["p_value"] < 0.001  # exact: 2 * 0.5**12 ≈ 0.0005


def test_mcnemar_identical_models():
    assert mcnemar_test(GOLD, PRED, PRED)["p_value"] == pytest.approx(1.0)

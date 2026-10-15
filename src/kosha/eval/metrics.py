"""Quality metrics and statistics (D-006). Pure functions, no model code."""

from collections.abc import Callable

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from statsmodels.stats.contingency_tables import mcnemar

from kosha import PUBMEDQA_LABELS, SEED

INVALID = "__invalid__"


def macro_f1(gold, pred, labels=PUBMEDQA_LABELS) -> float:
    return float(f1_score(gold, pred, labels=list(labels), average="macro", zero_division=0))


def bootstrap_ci(gold, pred, fn: Callable, n_boot: int = 2000, seed: int = SEED) -> list[float]:
    """95% percentile bootstrap CI, resampling items with replacement."""
    rng = np.random.default_rng(seed)
    g, p = np.asarray(gold), np.asarray(pred)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(g), len(g))
        stats.append(fn(g[idx], p[idx]))
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def ece(gold, probs, labels=PUBMEDQA_LABELS, n_bins: int = 10) -> float:
    """Expected calibration error: |accuracy - confidence| per confidence bin, weighted by bin size."""
    probs = np.asarray(probs, dtype=float)
    conf = probs.max(axis=1)
    hit = np.asarray(labels)[probs.argmax(axis=1)] == np.asarray(gold)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.mean() * abs(hit[mask].mean() - conf[mask].mean())
    return float(total)


def quality_metrics(gold, pred, probs=None, labels=PUBMEDQA_LABELS) -> dict:
    """pred may contain None (unparseable). It counts as wrong and is reported as invalid_rate."""
    p = [x if x is not None else INVALID for x in pred]
    per_class = f1_score(gold, p, labels=list(labels), average=None, zero_division=0)
    out = {
        "n": len(gold),
        "accuracy": float(accuracy_score(gold, p)),
        "macro_f1": macro_f1(gold, p, labels),
        "per_class_f1": {label: float(f) for label, f in zip(labels, per_class, strict=True)},
        "invalid_rate": sum(x is None for x in pred) / len(pred),
        "confusion": {
            "labels": [*labels, INVALID],
            "matrix": confusion_matrix(gold, p, labels=[*labels, INVALID]).tolist(),
        },
        "accuracy_ci95": bootstrap_ci(gold, p, accuracy_score),
        "macro_f1_ci95": bootstrap_ci(gold, p, lambda g, q: macro_f1(g, q, labels)),
    }
    if probs is not None:
        out["ece"] = ece(gold, probs, labels)
    return out


def mcnemar_test(gold, pred_a, pred_b) -> dict:
    """Paired test: do A and B differ on which items they get right? (exact binomial version)"""
    gold = np.asarray(gold, dtype=object)
    a = np.asarray(pred_a, dtype=object) == gold
    b = np.asarray(pred_b, dtype=object) == gold
    table = [[int((a & b).sum()), int((a & ~b).sum())], [int((~a & b).sum()), int((~a & ~b).sum())]]
    return {
        "table": table,
        "a_only_correct": table[0][1],
        "b_only_correct": table[1][0],
        "p_value": float(mcnemar(table, exact=True).pvalue),
    }

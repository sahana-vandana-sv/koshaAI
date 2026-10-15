"""B0 sanity floors: every real model must beat these."""

import random
from collections import Counter

from kosha import SEED


def majority(train: list[dict], items: list[dict]) -> list[str]:
    """Always predict the most common training label."""
    top = Counter(r["label"] for r in train).most_common(1)[0][0]
    return [top] * len(items)


def stratified_random(train: list[dict], items: list[dict]) -> list[str]:
    """Sample labels with the training class frequencies."""
    labels = [r["label"] for r in train]
    rng = random.Random(SEED)
    return [rng.choice(labels) for _ in items]

import json
from collections import Counter
from pathlib import Path

import pytest

from kosha.io import read_jsonl

DATA = Path("data/processed")
pytestmark = pytest.mark.skipif(not (DATA / "manifest.json").exists(), reason="run `make data` first")


@pytest.fixture(scope="module")
def splits():
    names = ("train", "train_labeled", "dev", "test", "shots")
    return {name: read_jsonl(DATA / f"{name}.jsonl") for name in names}


def test_sizes(splits):
    assert len(splits["test"]) == 500
    assert len(splits["dev"]) == 50
    assert len(splits["train_labeled"]) == 450
    assert len(splits["shots"]) == 3


def test_no_leakage(splits):
    ids = {name: {r["id"] for r in rows} for name, rows in splits.items()}
    assert not ids["test"] & ids["train"], "test PMIDs leaked into train"
    assert not ids["test"] & ids["dev"], "test PMIDs leaked into dev"
    assert not ids["dev"] & ids["train"], "dev PMIDs leaked into train"
    assert ids["shots"] <= ids["train_labeled"], "few-shot examples must come from train"


def test_class_coverage(splits):
    assert set(Counter(r["label"] for r in splits["dev"])) == {"yes", "no", "maybe"}
    train = Counter(r["label"] for r in splits["train"])
    assert train["maybe"] > 0
    assert train["yes"] / sum(train.values()) <= 0.70


def test_manifest_matches_files():
    manifest = json.loads((DATA / "manifest.json").read_text())
    for name, meta in manifest.items():
        assert len(read_jsonl(DATA / name)) == meta["n"]

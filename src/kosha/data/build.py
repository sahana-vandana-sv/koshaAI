"""Download PubMedQA + MedMCQA and write frozen splits to data/processed/ (D-013)."""

import hashlib
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from sklearn.model_selection import train_test_split

from kosha import PUBMEDQA_LABELS, SEED
from kosha.io import write_jsonl

OUT = Path("data/processed")
TEST_GT_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json"


def _row(ex: dict) -> dict:
    first_sentence = ex["long_answer"].strip().split(". ")[0].rstrip(".")
    return {
        "id": str(ex["pubid"]),
        "question": ex["question"],
        "context": " ".join(ex["context"]["contexts"]),
        "label": ex["final_decision"],
        "rationale": first_sentence + ".",
    }


def build_pubmedqa(n_artificial: int = 20_000, max_yes: float = 0.65, upsample: int = 4) -> dict:
    labeled = [_row(e) for e in load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")]

    with urllib.request.urlopen(TEST_GT_URL) as resp:
        ground_truth = json.load(resp)  # {pmid: label} for the official 500-item test set
    test = [r for r in labeled if r["id"] in ground_truth]
    rest = [r for r in labeled if r["id"] not in ground_truth]
    assert len(test) == 500 and len(rest) == 500, (len(test), len(rest))
    assert all(ground_truth[r["id"]] == r["label"] for r in test), "test labels disagree with official GT"

    train_labeled, dev = train_test_split(
        rest, test_size=50, random_state=SEED, stratify=[r["label"] for r in rest]
    )

    held_out = {r["id"] for r in labeled}  # no expert-labelled abstract may leak into training
    art = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train").shuffle(seed=SEED)
    art = art.filter(lambda e: str(e["pubid"]) not in held_out)
    no = art.filter(lambda e: e["final_decision"] == "no")
    yes = art.filter(lambda e: e["final_decision"] == "yes")
    n_no = min(len(no), int(n_artificial * (1 - max_yes)))
    artificial = [_row(e) for e in no.select(range(n_no))]
    artificial += [_row(e) for e in yes.select(range(n_artificial - n_no))]

    train = artificial + train_labeled * upsample  # upsampled: the only source of "maybe"
    random.Random(SEED).shuffle(train)
    return {"train": train, "train_labeled": train_labeled, "dev": dev, "test": test}


def build_medmcqa(n: int = 1000) -> list[dict]:
    ds = load_dataset("openlifescienceai/medmcqa", split="validation").shuffle(seed=SEED).select(range(n))
    return [
        {
            "id": e["id"],
            "question": e["question"],
            "options": [e["opa"], e["opb"], e["opc"], e["opd"]],
            "label": "ABCD"[e["cop"]],
            "subject": e["subject_name"],
        }
        for e in ds
    ]


def pick_shots(train_labeled: list[dict]) -> list[dict]:
    """One fixed few-shot example per class, taken from train (never dev/test)."""
    rng = random.Random(SEED)
    return [rng.choice([r for r in train_labeled if r["label"] == label]) for label in PUBMEDQA_LABELS]


def main() -> None:
    splits = build_pubmedqa()
    files = {
        "train.jsonl": splits["train"],
        "train_labeled.jsonl": splits["train_labeled"],
        "dev.jsonl": splits["dev"],
        "test.jsonl": splits["test"],
        "shots.jsonl": pick_shots(splits["train_labeled"]),
        "medmcqa.jsonl": build_medmcqa(),
    }
    manifest = {}
    for name, rows in files.items():
        write_jsonl(OUT / name, rows)
        manifest[name] = {
            "n": len(rows),
            "labels": dict(sorted(Counter(r["label"] for r in rows).items())),
            "sha256": hashlib.sha256((OUT / name).read_bytes()).hexdigest(),
        }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: {"n": v["n"], "labels": v["labels"]} for k, v in manifest.items()}, indent=2))


if __name__ == "__main__":
    main()

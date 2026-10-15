"""Run one config on one split → results/runs/<run_id>/ (config, env, predictions, metrics)."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from kosha import MEDMCQA_LABELS, PUBMEDQA_LABELS
from kosha.eval import baselines
from kosha.eval.metrics import quality_metrics
from kosha.eval.parsing import parse_answer
from kosha.io import read_jsonl, write_jsonl
from kosha.run import start_run

DATA = Path("data/processed")
SPLITS = {  # split name → (file, label set)
    "dev": ("dev.jsonl", PUBMEDQA_LABELS),
    "test": ("test.jsonl", PUBMEDQA_LABELS),
    "medmcqa": ("medmcqa.jsonl", MEDMCQA_LABELS),
}
BASELINES = {"majority": baselines.majority, "random": baselines.stratified_random}


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    missing = {"name", "stage", "engine"} - cfg.keys()
    if missing:
        raise SystemExit(f"{path}: missing keys {sorted(missing)}")
    return cfg


def generate(cfg: dict, items: list[dict]) -> list[dict]:
    """One output per item: {"pred": label} for baselines, {"text": raw output, ...} for models."""
    if cfg["engine"] in BASELINES:
        train = read_jsonl(cfg["train_file"])
        return [{"pred": p} for p in BASELINES[cfg["engine"]](train, items)]
    raise SystemExit(f"engine '{cfg['engine']}' is not implemented yet (model engines arrive on Day 2)")


def evaluate(config_path: str | Path, split: str) -> Path:
    if split not in SPLITS:
        raise SystemExit(f"split must be one of {list(SPLITS)}")
    cfg = load_config(config_path)
    if cfg["engine"] in BASELINES and split == "medmcqa":
        raise SystemExit("B0 baselines are PubMedQA-only")
    file, labels = SPLITS[split]
    items = read_jsonl(DATA / file)
    run_dir = start_run(cfg, split)  # test-split guard runs here, before any compute

    outputs = generate(cfg, items)
    records = []
    for item, out in zip(items, outputs, strict=True):
        pred, mode = (out["pred"], "baseline") if "pred" in out else parse_answer(out["text"], labels)
        records.append({**out, "id": item["id"], "gold": item["label"], "pred": pred, "parse_mode": mode})
    write_jsonl(run_dir / "predictions.jsonl", records)

    probs = [out.get("probs") for out in outputs]
    metrics = quality_metrics(
        [r["gold"] for r in records],
        [r["pred"] for r in records],
        probs=np.array(probs) if all(p is not None for p in probs) else None,
        labels=labels,
    )
    metrics |= {
        "run_id": run_dir.name,
        "name": cfg["name"],
        "stage": cfg["stage"],
        "model": cfg.get("model", cfg["engine"]),
        "split": split,
        "hardware": cfg.get("hardware", "cpu"),
        "parse_modes": dict(Counter(r["parse_mode"] for r in records)),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return run_dir

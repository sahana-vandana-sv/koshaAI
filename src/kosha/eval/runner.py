"""Run one config on one split → results/runs/<run_id>/ (config, env, predictions, metrics)."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from kosha import MEDMCQA_LABELS, PUBMEDQA_LABELS
from kosha.data.prompts import medmcqa_messages, pubmedqa_messages
from kosha.eval import baselines
from kosha.eval.metrics import ece, quality_metrics
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


def build_conversations(cfg: dict, items: list[dict], split: str) -> list[list[dict]]:
    if split == "medmcqa":
        return [medmcqa_messages(r) for r in items]
    shots = read_jsonl(DATA / "shots.jsonl")[: cfg.get("shots", 0)]
    return [pubmedqa_messages(r, shots) for r in items]


def generate(cfg: dict, items: list[dict], split: str, labels) -> list[dict]:
    """One output per item: {"pred": label} for baselines, {"text": raw output, ...} for models."""
    engine = cfg["engine"]
    if engine in BASELINES:
        train = read_jsonl(cfg["train_file"])
        return [{"pred": p} for p in BASELINES[engine](train, items)]

    conversations = build_conversations(cfg, items, split)
    if engine == "vllm":
        from kosha.eval.engines import generate_vllm

        return generate_vllm(cfg, conversations, labels)
    if engine == "hf":
        from kosha.eval.engines import generate_hf

        return generate_hf(cfg, conversations, labels)
    if engine == "anthropic_batch":
        from kosha.frontier.claude import run_batch

        ids = [r["id"] for r in items]
        return run_batch(cfg["model"], ids, conversations, labels, repeat=cfg.get("repeat", 0))
    raise SystemExit(f"unknown engine '{engine}'")


def evaluate(config_path: str | Path, split: str, repeat: int = 0) -> Path:
    if split not in SPLITS:
        raise SystemExit(f"split must be one of {list(SPLITS)}")
    cfg = load_config(config_path)
    if repeat:  # re-ask the API to measure frontier non-determinism (D-014)
        cfg |= {"repeat": repeat, "name": f"{cfg['name']}_r{repeat}"}
    if cfg["engine"] in BASELINES and split == "medmcqa":
        raise SystemExit("B0 baselines are PubMedQA-only")
    file, labels = SPLITS[split]
    items = read_jsonl(DATA / file)
    run_dir = start_run(cfg, split)  # test-split guard runs here, before any compute

    outputs = generate(cfg, items, split, labels)
    records = []
    for item, out in zip(items, outputs, strict=True):
        pred, mode = (out["pred"], "baseline") if "pred" in out else parse_answer(out["text"], labels)
        records.append({**out, "id": item["id"], "gold": item["label"], "pred": pred, "parse_mode": mode})
    write_jsonl(run_dir / "predictions.jsonl", records)

    gold = [r["gold"] for r in records]
    metrics = quality_metrics(gold, [r["pred"] for r in records], labels=labels)

    have_probs = [i for i, o in enumerate(outputs) if o.get("probs") is not None]
    if have_probs:  # ECE over items where the answer token was found; coverage reported alongside
        probs = np.array([outputs[i]["probs"] for i in have_probs])
        metrics["ece"] = ece([gold[i] for i in have_probs], probs, labels)
        metrics["ece_coverage"] = len(have_probs) / len(outputs)
    if any("usd" in o for o in outputs):
        metrics["usd_total"] = sum(o.get("usd", 0.0) for o in outputs)
        metrics["usd_per_1k"] = 1000 * metrics["usd_total"] / len(outputs)

    truncated = sum(
        o.get("finish_reason") == "length" or o.get("stop_reason") == "max_tokens" for o in outputs
    )
    metrics |= {
        "run_id": run_dir.name,
        "name": cfg["name"],
        "stage": cfg["stage"],
        "model": cfg.get("model", cfg["engine"]),
        "split": split,
        "hardware": cfg.get("hardware", "cpu"),
        "parse_modes": dict(Counter(r["parse_mode"] for r in records)),
        "truncated_rate": truncated / len(outputs),
        "errored": sum("error" in o for o in outputs),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return run_dir

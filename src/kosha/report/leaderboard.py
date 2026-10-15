"""results/leaderboard.csv: the project's headline artifact. Day 7 adds latency, cost and Pareto columns."""

import json
from pathlib import Path

import pandas as pd

RUNS = Path("results/runs")
OUT = Path("results/leaderboard.csv")
COLUMNS = [
    "stage", "name", "model", "split", "hardware", "n", "accuracy", "accuracy_ci95", "macro_f1",
    "macro_f1_ci95", "f1_yes", "f1_no", "f1_maybe", "invalid_rate", "ece", "run_id",
]  # fmt: skip


def _row(metrics: dict) -> dict:
    row = {**metrics, **{f"f1_{k}": v for k, v in metrics.get("per_class_f1", {}).items()}}
    for key in ("accuracy_ci95", "macro_f1_ci95"):
        lo, hi = metrics[key]
        row[key] = f"[{lo:.3f}, {hi:.3f}]"
    return row


def build() -> pd.DataFrame:
    rows = [_row(json.loads(p.read_text())) for p in sorted(RUNS.glob("*/metrics.json"))]
    df = pd.DataFrame(rows).reindex(columns=COLUMNS)
    # run_id starts with a timestamp: keep the latest run of each (config, split)
    df = df.sort_values("run_id").drop_duplicates(["name", "split"], keep="last")
    df = df.sort_values(["split", "stage", "name"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, float_format="%.4f")
    return df

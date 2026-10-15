"""Every run writes results/runs/<run_id>/ with everything needed to reproduce it (CLAUDE.md)."""

import json
import platform
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml

from kosha import SEED
from kosha.io import sh

RUNS_DIR = Path("results/runs")
FROZEN_PATHS = "src configs pyproject.toml uv.lock"  # must be committed before touching the test split
TRACKED_PACKAGES = (
    "datasets", "numpy", "scikit-learn", "torch", "transformers", "vllm", "peft", "trl", "anthropic",
)  # fmt: skip


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def start_run(cfg: dict, split: str) -> Path:
    """Create the run folder. Refuses the test split unless code and configs are committed."""
    if split == "test" and sh(f"git status --porcelain -- {FROZEN_PATHS}"):
        raise SystemExit("Refusing the test split: commit src/, configs/ and uv.lock first. Tune on dev.")
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{cfg['name']}-{split}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    env = {
        "run_id": run_id,
        "git_sha": sh("git rev-parse HEAD"),
        "seed": SEED,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader") or "none",
        "packages": {p: _pkg_version(p) for p in TRACKED_PACKAGES},
    }
    (run_dir / "env.json").write_text(json.dumps(env, indent=2))
    return run_dir

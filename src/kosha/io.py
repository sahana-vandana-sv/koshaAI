"""Small helpers shared by every stage."""

import json
import subprocess
from collections.abc import Iterable
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sh(cmd: str) -> str:
    """Run a shell command and return its stripped stdout, or '' if it fails."""
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""

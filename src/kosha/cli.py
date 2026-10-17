"""`kosha` command line: thin wrappers around module functions."""

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def data() -> None:
    """Download PubMedQA + MedMCQA and write frozen splits to data/processed/."""
    from kosha.data.build import main

    main()


@app.command("eval")
def eval_(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    split: Annotated[str, typer.Option(help="dev | test | medmcqa")] = "dev",
    repeat: Annotated[int, typer.Option(help="re-ask the API to measure non-determinism")] = 0,
) -> None:
    """Evaluate one config on one split."""
    from kosha.eval.runner import evaluate

    typer.echo(f"wrote {evaluate(config, split, repeat)}")


@app.command("compare")
def compare_(run_a: str, run_b: str) -> None:
    """Agreement rate + McNemar between two runs (run folder name or path)."""
    import json

    from kosha.eval.compare import compare

    typer.echo(json.dumps(compare(run_a, run_b), indent=2))


@app.command()
def leaderboard() -> None:
    """Rebuild results/leaderboard.csv from results/runs/*/metrics.json."""
    from kosha.report.leaderboard import build

    typer.echo(build().to_string(index=False))

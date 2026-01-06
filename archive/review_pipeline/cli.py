"""Command line interface for review_pipeline."""
from __future__ import annotations

import logging
import pathlib
from typing import Optional

import typer

from .state import StateManager
from .pipeline import CleanPipeline

app = typer.Typer(add_completion=False, help="Multi-domain review cleaning pipeline")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@app.command()
def clean(
    domain: str = typer.Option(..., "--domain", help="Domain name: phone/car/laptop/beauty"),
    input: pathlib.Path = typer.Option(..., "--input", exists=True, file_okay=False, help="Input directory"),
    output: pathlib.Path = typer.Option(..., "--output", file_okay=False, help="Output directory"),
    config: pathlib.Path = typer.Option(..., "--config", exists=True, dir_okay=False, help="Domain config"),
    workers: int = typer.Option(1, "--workers", min=1, help="Number of workers for I/O heavy steps"),
    force: bool = typer.Option(False, "--force", help="Force reprocess files even if unchanged"),
) -> None:
    """Run cleaning pipeline."""

    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "state.json"
    state = StateManager(state_path)
    pipeline = CleanPipeline(
        domain=domain,
        input_dir=input,
        output_dir=output,
        config_path=config,
        workers=workers,
        state=state,
        force=force,
    )
    pipeline.run()


if __name__ == "__main__":
    app()

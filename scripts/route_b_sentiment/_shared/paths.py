"""Path helpers for Route B outputs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "outputs"


STEP_DIR_NAMES: Dict[str, str] = {
    "01": "step01_pairs",
    "1": "step01_pairs",
    "02": "step02_pseudo",
    "2": "step02_pseudo",
    "03": "step03_model",
    "3": "step03_model",
    "04": "step04_pred",
    "4": "step04_pred",
    "05": "step05_agg",
    "5": "step05_agg",
    "web": "web_exports",
}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_root(domain: str, run_id: str) -> Path:
    return OUTPUT_ROOT / domain / "runs" / run_id


def step_dir(domain: str, run_id: str, step: str) -> Path:
    key = str(step).lower()
    name = STEP_DIR_NAMES.get(key, key)
    return run_root(domain, run_id) / name


def default_run_id(domain: str, suffix: str = "") -> str:
    ts = datetime.now().strftime("%Y%m%d")
    base = f"{ts}_{domain}"
    if suffix:
        base = f"{base}_{suffix}"
    return base

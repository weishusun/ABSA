"""Config resolver for Route B domains.

Prefers new layout: configs/domains/<domain>/{aspects.yaml,domain.yaml}
Falls back to legacy: configs/aspects_<domain>.yaml and configs/domain_<domain>.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = REPO_ROOT / "configs"


def resolve(domain: str) -> Tuple[Path, Path]:
    d = (domain or "").strip().lower()
    if not d:
        raise ValueError("domain is required")

    new_base = CONFIG_ROOT / "domains" / d

    aspects_candidates = [
        new_base / "aspects.yaml",
        CONFIG_ROOT / f"aspects_{d}.yaml",
    ]
    domain_candidates = [
        new_base / "domain.yaml",
        CONFIG_ROOT / f"domain_{d}.yaml",
    ]

    aspects_yaml = next((p for p in aspects_candidates if p.exists()), None)
    domain_yaml = next((p for p in domain_candidates if p.exists()), None)

    missing = []
    if aspects_yaml is None:
        missing.append("aspects")
    if domain_yaml is None:
        missing.append("domain")
    if missing:
        raise FileNotFoundError(f"config not found for domain='{d}': missing {missing}")

    return aspects_yaml, domain_yaml

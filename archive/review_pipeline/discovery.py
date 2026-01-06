"""File discovery utilities."""
from __future__ import annotations

import pathlib
from typing import Iterable, List

SUPPORTED_SUFFIXES = {".json", ".jsonl"}


def discover_files(input_dir: pathlib.Path) -> List[pathlib.Path]:
    files: List[pathlib.Path] = []
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)
    return sorted(files)

__all__ = ["discover_files", "SUPPORTED_SUFFIXES"]

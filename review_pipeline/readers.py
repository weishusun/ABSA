"""Readers for JSON and JSONL files."""
from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Dict, Iterable, Iterator, List

from .utils import safe_json_load

logger = logging.getLogger(__name__)


def _read_json(path: pathlib.Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        text = f.read()
    data = safe_json_load(text)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    logger.warning("Unsupported JSON structure in %s", path)
    return []


def _read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = safe_json_load(line)
                if isinstance(obj, dict):
                    obj["__source_line"] = idx
                    records.append(obj)
                else:
                    logger.warning("Line %s in %s is not an object", idx, path)
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to parse line %s in %s: %s", idx, path, exc)
    return records


def read_records(path: pathlib.Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl(path)
    return _read_json(path)


def detect_format(path: pathlib.Path) -> str:
    return "jsonl" if path.suffix.lower() == ".jsonl" else "json"


__all__ = ["read_records", "detect_format"]

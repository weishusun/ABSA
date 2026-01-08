"""Utility helpers for the review pipeline."""
from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import re
from typing import Any, Dict, Iterable, Optional

try:  # pragma: no cover - optional dependency
    import orjson
except ImportError:  # pragma: no cover
    orjson = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

logger = logging.getLogger(__name__)

CONTROL_CHARS = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]")


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def sha1_file(path: pathlib.Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_json_load(text: str) -> Any:
    if orjson:
        return orjson.loads(text)
    return json.loads(text)


def safe_json_dump(data: Any, ensure_ascii: bool = False) -> str:
    if orjson:
        return orjson.dumps(data, option=orjson.OPT_INDENT_2).decode()
    return json.dumps(data, ensure_ascii=ensure_ascii, indent=2, default=str)


def load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    if yaml is None:
        raise ImportError("pyyaml is required for loading configuration files")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_control_chars(text: str) -> str:
    return CONTROL_CHARS.sub("", text)


def extract_nested(data: Dict[str, Any], path: str) -> Optional[Any]:
    node: Any = data
    for key in path.split("."):
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return None
    return node


__all__ = [
    "sha1_text",
    "sha1_file",
    "safe_json_load",
    "safe_json_dump",
    "load_yaml",
    "normalize_whitespace",
    "strip_control_chars",
    "extract_nested",
]

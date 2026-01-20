# review_pipeline/readers.py
from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Dict, List

from .utils import safe_json_load

logger = logging.getLogger(__name__)


def _as_records(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def _read_text(path: pathlib.Path) -> str:
    # utf-8-sig: 兼容 BOM；errors=replace：最大化容错
    return path.read_text(encoding="utf-8-sig", errors="replace")


def _read_json(path: pathlib.Path) -> List[Dict[str, Any]]:
    return _as_records(safe_json_load(_read_text(path)))


def _read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = safe_json_load(line)
            except Exception as exc:
                logger.warning("Bad JSONL line %s in %s: %s", idx, path, exc)
                continue

            # 兼容某些行是数组（少见但有）
            if isinstance(obj, dict):
                obj["__source_line"] = idx
                records.append(obj)
            elif isinstance(obj, list):
                for x in obj:
                    if isinstance(x, dict):
                        x["__source_line"] = idx
                        records.append(x)
            else:
                logger.warning("Line %s in %s is not object/array", idx, path)
    return records


def _looks_like_jsonl(text: str, probe_lines: int = 6) -> bool:
    """
    快速判定：如果有多行，并且前几行大多数能被解析为 dict，则认为是 JSONL。
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        return False

    ok = 0
    checked = 0
    for l in lines[:probe_lines]:
        checked += 1
        if not (l.startswith("{") and l.endswith("}")):
            continue
        try:
            obj = safe_json_load(l)
            if isinstance(obj, dict):
                ok += 1
        except Exception:
            pass

    # 至少命中 2 行，且命中率过半
    return ok >= 2 and ok >= (checked // 2)


def read_records(path: pathlib.Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()

    # 1) 明确 jsonl
    if suffix == ".jsonl":
        return _read_jsonl(path)

    # 2) 轻量探测：像 jsonl 就直接走 jsonl（避免先抛异常）
    text = _read_text(path)
    if _looks_like_jsonl(text):
        return _read_jsonl(path)

    # 3) 否则按标准 JSON 解析
    return _as_records(safe_json_load(text))


def detect_format(path: pathlib.Path) -> str:
    text = _read_text(path)
    if path.suffix.lower() == ".jsonl":
        return "jsonl"
    return "jsonl" if _looks_like_jsonl(text) else "json"

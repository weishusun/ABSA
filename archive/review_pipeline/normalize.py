"""Normalization into standard schema."""
from __future__ import annotations

import logging
import pathlib
from datetime import datetime
from typing import Any, Dict, Optional

from .utils import extract_nested, normalize_whitespace, sha1_text, strip_control_chars

logger = logging.getLogger(__name__)

STANDARD_FIELDS = [
    "domain",
    "brand",
    "model",
    "doc_id",
    "platform",
    "url",
    "ctime",
    "like_count",
    "reply_count",
    "content_raw",
    "content_clean",
    "sentence_idx",
    "sentence",
    "source_file",
    "source_line",
    "parse_error",
    "error_msg",
    "extra_json",
]


class Normalizer:
    def __init__(self, config: Dict[str, Any], domain: str, input_dir: pathlib.Path) -> None:
        self.config = config
        self.domain = domain
        self.input_dir = input_dir

    def _guess_brand_model(self, path: pathlib.Path) -> Dict[str, Optional[str]]:
        rel_parts = path.relative_to(self.input_dir).parts
        brand_override = self.config.get("brand_override")
        model_override = self.config.get("model_override")
        brand = brand_override
        model = model_override
        if brand is None and len(rel_parts) >= 2:
            brand = rel_parts[1] if rel_parts[0] == self.domain else rel_parts[0]
        if model is None and len(rel_parts) >= 3:
            model = rel_parts[2] if rel_parts[0] == self.domain else rel_parts[1]
        return {"brand": brand, "model": model}

    def _get_field(self, record: Dict[str, Any], key: Optional[str]) -> Optional[Any]:
        if not key:
            return None
        if "." in key:
            return extract_nested(record, key)
        return record.get(key)

    def normalize_record(self, record: Dict[str, Any], source_file: pathlib.Path) -> Dict[str, Any]:
        brand_model = self._guess_brand_model(source_file)
        content_field = self.config.get("content_field", "content")
        url_field = self.config.get("url_field", "url")
        platform_field = self.config.get("platform_field", "platform")
        ctime_field = self.config.get("ctime_field")
        like_field = self.config.get("like_field")
        reply_field = self.config.get("reply_field")

        content_raw = self._get_field(record, content_field) or ""
        url = self._get_field(record, url_field)
        platform = self._get_field(record, platform_field)
        ctime_raw = self._get_field(record, ctime_field)
        like_count = self._get_field(record, like_field)
        reply_count = self._get_field(record, reply_field)

        doc_id = record.get("uuid") or record.get("id")
        if not doc_id:
            doc_id = sha1_text(f"{content_raw}|{url or ''}")

        ctime = None
        if ctime_raw:
            try:
                ctime = str(self._parse_datetime(ctime_raw))
            except Exception:
                ctime = str(ctime_raw)

        extra_fields = self.config.get("extra_fields", [])
        extra_json = {k: self._get_field(record, k) for k in extra_fields}

        return {
            "domain": self.domain,
            "brand": brand_model.get("brand"),
            "model": brand_model.get("model"),
            "doc_id": doc_id,
            "platform": platform,
            "url": url,
            "ctime": ctime,
            "like_count": like_count,
            "reply_count": reply_count,
            "content_raw": content_raw,
            "content_clean": None,
            "sentence_idx": None,
            "sentence": None,
            "source_file": str(source_file),
            "source_line": record.get("__source_line"),
            "parse_error": False,
            "error_msg": None,
            "extra_json": extra_json if extra_json else None,
        }

    @staticmethod
    def _parse_datetime(value: Any) -> datetime:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(int(value))
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S"):
            try:
                return datetime.strptime(str(value), fmt)
            except Exception:
                continue
        return datetime.fromisoformat(str(value))


__all__ = ["Normalizer", "STANDARD_FIELDS"]

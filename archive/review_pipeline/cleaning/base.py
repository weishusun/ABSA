"""Base cleaner implementation."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from ..utils import normalize_whitespace, strip_control_chars

logger = logging.getLogger(__name__)

HTML_TAG_RE = re.compile(r"<[^>]+>")
EMOJI_RE = re.compile(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]")


class BaseCleaner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        patterns = config.get("noise_suffix_patterns", []) or []
        self.noise_suffix_patterns = [re.compile(p) for p in patterns]
        self.keep_emoji = bool(config.get("keep_emoji", False))
        self.keep_english = bool(config.get("keep_english", True))
        self.min_length = int(config.get("min_length", 1))

    def clean_text(self, text: str) -> str:
        original = text or ""
        text = strip_control_chars(original)
        text = HTML_TAG_RE.sub(" ", text)
        if not self.keep_emoji:
            text = EMOJI_RE.sub("", text)
        text = normalize_whitespace(text)
        for pattern in self.noise_suffix_patterns:
            text = pattern.sub("", text)
        if not self.keep_english:
            text = re.sub(r"[A-Za-z]", "", text)
        return text

    def is_valid(self, text: str) -> bool:
        return len(text) >= self.min_length

    def process(self, record: Dict[str, Any]) -> Dict[str, Any]:
        text = record.get("content_raw", "") or ""
        cleaned = self.clean_text(text)
        record["content_clean"] = cleaned
        record["parse_error"] = False if cleaned else record.get("parse_error", False)
        if not self.is_valid(cleaned):
            record["parse_error"] = True
            record["error_msg"] = "text too short"
        return record


__all__ = ["BaseCleaner"]

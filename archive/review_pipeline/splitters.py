"""Sentence splitters."""
from __future__ import annotations

import re
from typing import List

CHINESE_END_PUNCT = "。！？；…"
DEFAULT_MIN_LEN = 2
DEFAULT_MAX_LEN = 120


class SentenceSplitter:
    def __init__(self, min_len: int = DEFAULT_MIN_LEN, max_len: int = DEFAULT_MAX_LEN):
        self.min_len = min_len
        self.max_len = max_len
        self.pattern = re.compile(rf"[{CHINESE_END_PUNCT}]+")

    def split(self, text: str) -> List[str]:
        sentences: List[str] = []
        parts = self.pattern.split(text)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) > self.max_len:
                sentences.extend(self._split_long(part))
            elif len(part) >= self.min_len:
                sentences.append(part)
        return sentences

    def _split_long(self, text: str) -> List[str]:
        sub_parts = [p.strip() for p in re.split(r"[，,]", text) if p.strip()]
        merged: List[str] = []
        for sub in sub_parts:
            if len(sub) >= self.min_len:
                merged.append(sub)
        return merged


__all__ = ["SentenceSplitter"]

"""Car domain cleaner."""
from __future__ import annotations

import re
from typing import Any, Dict

from ..base import BaseCleaner


class CarCleaner(BaseCleaner):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.trim_patterns = [re.compile(p) for p in config.get("extra_noise_patterns", [])]

    def clean_text(self, text: str) -> str:
        text = super().clean_text(text)
        for pattern in self.trim_patterns:
            text = pattern.sub("", text)
        return text


__all__ = ["CarCleaner"]

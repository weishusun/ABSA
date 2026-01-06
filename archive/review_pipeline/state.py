"""State management for resume capability."""
from __future__ import annotations

import json
import logging
import pathlib
from typing import Dict, Optional

from .utils import safe_json_dump, sha1_file

logger = logging.getLogger(__name__)


class StateManager:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.state: Dict[str, Dict[str, str]] = {}
        if path.exists():
            try:
                self.state = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("Failed to load state file %s, starting fresh", path)
                self.state = {}

    def _key(self, file_path: pathlib.Path) -> str:
        return str(file_path)

    def _snapshot(self, file_path: pathlib.Path) -> Dict[str, str]:
        stat = file_path.stat()
        return {
            "size": str(stat.st_size),
            "mtime": str(int(stat.st_mtime)),
            "hash": sha1_file(file_path),
        }

    def is_unchanged(self, file_path: pathlib.Path) -> bool:
        key = self._key(file_path)
        if key not in self.state:
            return False
        return self.state[key] == self._snapshot(file_path)

    def update(self, file_path: pathlib.Path) -> None:
        key = self._key(file_path)
        self.state[key] = self._snapshot(file_path)

    def save(self) -> None:
        self.path.write_text(safe_json_dump(self.state), encoding="utf-8")


__all__ = ["StateManager"]

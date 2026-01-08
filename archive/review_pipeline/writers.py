"""Writers for cleaned data and manifests."""
from __future__ import annotations

import json
import logging
import pathlib
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from .normalize import STANDARD_FIELDS
from .utils import safe_json_dump

logger = logging.getLogger(__name__)


class ResultWriter:
    def __init__(self, output_dir: pathlib.Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_dataframe(self, rows: List[Dict[str, Any]], write_csv: bool = False) -> pathlib.Path:
        if not rows:
            raise ValueError("No rows to write")
        df = pd.DataFrame(rows, columns=STANDARD_FIELDS)
        parquet_path = self.output_dir / "clean_sentences.parquet"
        try:
            df.to_parquet(parquet_path, index=False)
            logger.info("Wrote parquet to %s", parquet_path)
        except Exception as exc:  # pragma: no cover
            logger.warning("Parquet writing failed (%s), falling back to CSV", exc)
            parquet_path = self.output_dir / "clean_sentences.csv"
            df.to_csv(parquet_path, index=False)
        if write_csv:
            csv_path = self.output_dir / "clean_sentences.csv"
            df.to_csv(csv_path, index=False)
            logger.info("Wrote CSV to %s", csv_path)
        return parquet_path

    def write_manifest(self, manifest: Dict[str, Any]) -> pathlib.Path:
        path = self.output_dir / "manifest.json"
        with path.open("w", encoding="utf-8") as f:
            f.write(safe_json_dump(manifest))
        return path


__all__ = ["ResultWriter"]

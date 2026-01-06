"""Main cleaning pipeline."""
from __future__ import annotations

import logging
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from .cleaning.base import BaseCleaner
from .cleaning.domains.beauty import BeautyCleaner
from .cleaning.domains.car import CarCleaner
from .cleaning.domains.laptop import LaptopCleaner
from .cleaning.domains.phone import PhoneCleaner
from .discovery import discover_files
from .normalize import Normalizer
from .readers import read_records
from .splitters import SentenceSplitter
from .state import StateManager
from .utils import load_yaml
from .writers import ResultWriter

logger = logging.getLogger(__name__)

CLEANER_MAP = {
    "phone": PhoneCleaner,
    "car": CarCleaner,
    "laptop": LaptopCleaner,
    "beauty": BeautyCleaner,
}


class CleanPipeline:
    def __init__(
        self,
        domain: str,
        input_dir: pathlib.Path,
        output_dir: pathlib.Path,
        config_path: pathlib.Path,
        workers: int,
        state: StateManager,
        force: bool,
    ) -> None:
        self.domain = domain
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.config = load_yaml(config_path)
        self.normalizer = Normalizer(self.config, domain, input_dir)
        cleaner_cls = CLEANER_MAP.get(domain, BaseCleaner)
        self.cleaner = cleaner_cls(self.config)
        splitter_conf = self.config.get("splitter", {}) if isinstance(self.config, dict) else {}
        self.splitter = SentenceSplitter(
            min_len=int(splitter_conf.get("min_len", 2)),
            max_len=int(splitter_conf.get("max_len", 120)),
        )
        self.state = state
        self.force = force
        self.workers = max(1, workers)
        self.writer = ResultWriter(output_dir)
        self.stats = {
            "files_total": 0,
            "files_processed": 0,
            "reviews_total": 0,
            "sentences_total": 0,
            "errors": 0,
            "platform_distribution": {},
            "empty_ratio": 0.0,
        }

    def run(self) -> None:
        start = time.time()
        files = discover_files(self.input_dir)
        self.stats["files_total"] = len(files)
        logger.info("Discovered %s input files", len(files))
        rows: List[Dict[str, Any]] = []

        def worker(path: pathlib.Path) -> Tuple[pathlib.Path, List[Dict[str, Any]]]:
            return path, self._process_file(path)

        if self.workers > 1:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = {executor.submit(worker, path): path for path in files}
                for future in as_completed(futures):
                    path, result_rows = future.result()
                    rows.extend(result_rows)
                    self.stats["files_processed"] += 1
        else:
            for path in files:
                rows.extend(self._process_file(path))
                self.stats["files_processed"] += 1

        if rows:
            self.writer.write_dataframe(rows, write_csv=bool(self.config.get("write_csv")))
        empty_sentences = len([r for r in rows if not r.get("sentence")])
        self.stats["empty_ratio"] = empty_sentences / max(len(rows), 1)
        self.stats["duration_sec"] = round(time.time() - start, 2)
        self.writer.write_manifest(self.stats)
        self.state.save()
        logger.info("Pipeline finished: %s sentences", self.stats["sentences_total"])

    def _process_file(self, path: pathlib.Path) -> List[Dict[str, Any]]:
        if not self.force and self.state.is_unchanged(path):
            logger.info("Skipping unchanged file %s", path)
            return []
        try:
            records = read_records(path)
            normalized = [self.normalizer.normalize_record(rec, path) for rec in records]
            cleaned = [self.cleaner.process(rec) for rec in normalized]
            sentences_rows = self._split_sentences(cleaned)
            self.state.update(path)
            return sentences_rows
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed processing %s: %s", path, exc)
            return []

    def _split_sentences(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        for rec in records:
            content = rec.get("content_clean") or ""
            sentences = self.splitter.split(content)
            if not sentences:
                sentences = [""]
            for idx, sentence in enumerate(sentences):
                row = dict(rec)
                row["sentence_idx"] = idx
                row["sentence"] = sentence
                row["parse_error"] = rec.get("parse_error", False) or sentence == ""
                if row["parse_error"]:
                    row["error_msg"] = row.get("error_msg") or "empty sentence"
                    self.stats["errors"] += 1
                output.append(row)
            self.stats["reviews_total"] += 1
            self.stats["sentences_total"] += len(sentences)
            platform = rec.get("platform") or "unknown"
            self.stats["platform_distribution"][platform] = (
                self.stats["platform_distribution"].get(platform, 0) + 1
            )
        return output


__all__ = ["CleanPipeline"]

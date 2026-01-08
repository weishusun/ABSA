#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step00: ingest raw JSON/JSONL under data/<domain>/<brand>/<model>/ to clean_sentences.parquet
- Supports JSON array, JSON object, JSONL
- Extracts content/ctime/id via common field candidates; falls back to hash for id
- Splits sentences by中文标点与换行
- Writes to parquet dataset (append-friendly), records ingest manifest & stats
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


CONTENT_KEYS = ["content", "text", "comment", "review", "body", "评价内容"]
CTIME_KEYS = ["ctime", "create_time", "comment_time", "time", "date", "createdAt", "created_at"]
ID_KEYS = ["id", "review_id", "doc_id", "comment_id", "content_id"]

SENT_SPLIT_RE = re.compile(r"[。！？!?；;]+|\n+")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--data-root", default="", help="root dir of raw JSON/JSONL; default data/<domain>")
    ap.add_argument("--output", default="", help="output parquet path or dataset dir; default outputs/<domain>/clean_sentences.parquet")
    ap.add_argument("--resume", action="store_true", help="skip files already in ingest_manifest.jsonl")
    ap.add_argument("--max-files", type=int, default=0, help="limit number of files for debug")
    ap.add_argument("--max-docs", type=int, default=0, help="limit total docs for debug")
    ap.add_argument("--chunk-size", type=int, default=5000, help="rows per parquet write chunk")
    return ap.parse_args()


def sentence_split(text: str) -> List[str]:
    parts = SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def find_first(obj: Dict, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in obj and obj[k] not in (None, ""):
            v = obj[k]
            if isinstance(v, (int, float)):
                return str(v)
            if isinstance(v, str):
                return v.strip()
    return None


def gen_doc_id(path: str, idx: int) -> str:
    return hashlib.md5(f"{path}#{idx}".encode("utf-8")).hexdigest()


def load_json_records(path: Path) -> Iterable[Tuple[Dict, int]]:
    def _iter_jsonl(p: Path) -> Iterable[Tuple[Dict, int]]:
        with p.open("r", encoding="utf-8-sig") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield obj, i

    suffix = path.suffix.lower()

    # .jsonl 明确走逐行
    if suffix == ".jsonl":
        yield from _iter_jsonl(path)
        return

    # .json：先尝试标准 JSON（对象/数组），失败且为 Extra data 时回退 JSONL
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        # 典型：文件实际是 JSONL（每行一个 JSON），但扩展名是 .json
        if getattr(e, "msg", "") == "Extra data" or "Extra data" in str(e):
            yield from _iter_jsonl(path)
            return
        raise

    if isinstance(data, list):
        for i, obj in enumerate(data):
            if isinstance(obj, dict):
                yield obj, i
    elif isinstance(data, dict):
        yield data, 0


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def dataset_dir_from_output(out: Path) -> Path:
    return out


def read_ingested(manifest: Path) -> set:
    done = set()
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                if "source_path" in rec:
                    done.add(rec["source_path"])
            except Exception:
                continue
    return done


def append_manifest(manifest: Path, rec: Dict) -> None:
    ensure_dir(manifest.parent)
    with manifest.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    domain = args.domain.strip().lower()
    if not domain:
        print("[FATAL] --domain is required", file=sys.stderr)
        return 2

    data_root = Path(args.data_root) if args.data_root else Path("data") / domain
    output_path = Path(args.output) if args.output else Path("outputs") / domain / "clean_sentences.parquet"
    dataset_dir = dataset_dir_from_output(output_path)
    log_dir = Path("outputs") / domain / "logs"
    ensure_dir(log_dir)
    error_log = log_dir / "ingest_errors.log"
    meta_dir = Path("outputs") / domain / "meta"
    ensure_dir(meta_dir)
    manifest_fp = meta_dir / "ingest_manifest.jsonl"
    stats_fp = meta_dir / "ingest_stats.json"

    done_files = read_ingested(manifest_fp) if args.resume else set()

    all_files = list(data_root.rglob("*.json")) + list(data_root.rglob("*.jsonl"))
    all_files = [p for p in all_files if p.is_file()]
    if args.max_files and args.max_files > 0:
        all_files = all_files[: args.max_files]

    total_docs = 0
    total_sent = 0
    ctime_vals: List[str] = []
    error_log_handle = error_log.open("a", encoding="utf-8")
    chunk_size = max(1, int(args.chunk_size))

    schema = pa.schema(
        [
            ("domain", pa.string()),
            ("brand", pa.string()),
            ("model", pa.string()),
            ("doc_id", pa.string()),
            ("sentence_idx", pa.int32()),
            ("sentence", pa.string()),
            ("ctime", pa.string()),
            ("source_path", pa.string()),
        ]
    )

    writer_opts = dict(compression="zstd")

    processed = 0

    for f in all_files:
        rel = f.relative_to(data_root)
        parts = rel.parts
        if len(parts) < 3:
            continue
        if str(f) in done_files:
            continue
        brand, model = parts[0], parts[1]

        try:
            records = list(load_json_records(f))
        except Exception as e:
            error_log_handle.write(f"{f}\tparse_error\t{e}\n")
            continue

        out_rows: List[Dict] = []
        sentences_written = 0
        file_ctimes: List[str] = []
        for idx, (obj, j) in enumerate(records):
            doc_id = find_first(obj, ID_KEYS) or gen_doc_id(str(f), j)
            content = find_first(obj, CONTENT_KEYS)
            if not content:
                continue
            ctime = find_first(obj, CTIME_KEYS) or ""
            sents = sentence_split(content)
            if not sents:
                continue
            for si, sent in enumerate(sents):
                out_rows.append(
                    {
                        "domain": domain,
                        "brand": brand,
                        "model": model,
                        "doc_id": doc_id,
                        "sentence_idx": si,
                        "sentence": sent,
                        "ctime": ctime,
                        "source_path": str(f),
                    }
                )
                sentences_written += 1
            total_docs += 1
            total_sent += len(sents)
            if ctime:
                ctime_vals.append(ctime)
                file_ctimes.append(ctime)

            if args.max_docs and total_docs >= args.max_docs:
                break

            if len(out_rows) >= chunk_size:
                table = pa.Table.from_pylist(out_rows, schema=schema)
                out_root = Path(args.output)
                out_root.mkdir(parents=True, exist_ok=True)
                pq.write_to_dataset(table, root_path=str(out_root), **writer_opts)
                out_rows.clear()

        if out_rows:
            table = pa.Table.from_pylist(out_rows, schema=schema)
            out_root = Path(args.output)
            out_root.mkdir(parents=True, exist_ok=True)
            pq.write_to_dataset(table, root_path=str(out_root), **writer_opts)
        append_manifest(
            manifest_fp,
            {
                "source_path": str(f),
                "docs": len(records),
                "sentences": sentences_written,
                "ctime_min": min(file_ctimes) if file_ctimes else None,
                "ctime_max": max(file_ctimes) if file_ctimes else None,
                "ts": datetime.now().isoformat(timespec="seconds"),
            },
        )
        processed += 1
        if args.max_docs and total_docs >= args.max_docs:
            break

    error_log_handle.close()

    stats = {
        "domain": domain,
        "data_root": str(data_root),
        "output_dataset": str(dataset_dir),
        "files_processed": processed,
        "docs": total_docs,
        "sentences": total_sent,
        "ctime_min": min(ctime_vals) if ctime_vals else None,
        "ctime_max": max(ctime_vals) if ctime_vals else None,
        "ts": datetime.now().isoformat(timespec="seconds"),
    }
    stats_fp.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] files={processed} docs={total_docs} sentences={total_sent} output={dataset_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

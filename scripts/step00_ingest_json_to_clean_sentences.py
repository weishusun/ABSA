#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step00: ingest raw JSON/JSONL under data/<domain>/<brand>/<model>/ to clean_sentences.parquet
- [安全增强] 强制输入目录必须包含 domain 关键词，防止误扫描其他领域
- [数据纯净] 每次运行强制覆盖输出文件，不再追加，防止跨领域数据污染
- [结构] 输出为单个 Parquet 文件，而非 Dataset 文件夹
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import shutil
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
    ap.add_argument("--output", default="", help="output parquet file path")
    # --resume 在强制覆盖模式下意义不大，但保留以防后续需要
    ap.add_argument("--resume", action="store_true",
                    help="[Deprecated in safe mode] skip files already in ingest_manifest")
    ap.add_argument("--max-files", type=int, default=0, help="limit number of files for debug")
    ap.add_argument("--max-docs", type=int, default=0, help="limit total docs for debug")
    ap.add_argument("--chunk-size", type=int, default=10000, help="rows per parquet write chunk")
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

    if suffix == ".jsonl":
        yield from _iter_jsonl(path)
        return

    try:
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
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


def append_manifest(manifest: Path, rec: Dict) -> None:
    ensure_dir(manifest.parent)
    with manifest.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def check_path_safety(domain: str, data_root: Path) -> None:
    """
    [安全熔断] 确保输入路径包含 domain 名称，防止扫描到父级目录导致数据污染。
    """
    abs_path = str(data_root.resolve()).lower()
    if domain not in abs_path:
        print(f"\n{'=' * 60}", file=sys.stderr)
        print(f"[FATAL 安全拦截] 输入路径异常！", file=sys.stderr)
        print(f"当前领域: {domain}", file=sys.stderr)
        print(f"输入路径: {data_root}", file=sys.stderr)
        print(f"错误原因: 输入路径未包含领域名称 '{domain}'。", file=sys.stderr)
        print(f"潜在风险: 可能误扫描了整个 inputs/ 目录，导致手机/汽车数据混杂。", file=sys.stderr)
        print(f"修正建议: 请将 data-root 设置为具体子目录，例如 inputs/{domain}", file=sys.stderr)
        print(f"{'=' * 60}\n", file=sys.stderr)
        sys.exit(1)


def prepare_output_file(output_path: Path) -> None:
    """
    [数据纯净] 强制清理旧的输出文件/文件夹。
    """
    if output_path.exists():
        print(f"[INFO] 检测到旧产物: {output_path}")
        if output_path.is_dir():
            print(f"[WARN] 旧产物是文件夹 (Dataset模式)，正在强制删除以切换为单文件模式...")
            shutil.rmtree(output_path)
        else:
            print(f"[INFO] 正在删除旧文件以确保数据纯净...")
            output_path.unlink()

    ensure_dir(output_path.parent)


def main() -> int:
    args = parse_args()
    domain = args.domain.strip().lower()
    if not domain:
        print("[FATAL] --domain is required", file=sys.stderr)
        return 2

    # 1. 路径解析与安全检查
    data_root = Path(args.data_root) if args.data_root else Path("data") / domain
    check_path_safety(domain, data_root)

    output_path = Path(args.output) if args.output else Path("outputs") / domain / "clean_sentences.parquet"

    # 2. 输出环境清理 (防止 Append 污染)
    prepare_output_file(output_path)

    # 3. 日志准备
    log_dir = Path("outputs") / domain / "logs"
    ensure_dir(log_dir)
    error_log = log_dir / "ingest_errors.log"
    meta_dir = Path("outputs") / domain / "meta"
    ensure_dir(meta_dir)
    manifest_fp = meta_dir / "ingest_manifest.jsonl"
    stats_fp = meta_dir / "ingest_stats.json"

    # 清空 manifest 以配合新的全量跑
    if manifest_fp.exists():
        manifest_fp.unlink()

    # 4. 扫描文件
    print(f"[INFO] Scanning files in: {data_root}")
    all_files = list(data_root.rglob("*.json")) + list(data_root.rglob("*.jsonl"))
    all_files = [p for p in all_files if p.is_file()]

    if args.max_files and args.max_files > 0:
        all_files = all_files[: args.max_files]

    if not all_files:
        print(f"[WARN] No .json/.jsonl files found in {data_root}")
        return 0

    print(f"[INFO] Found {len(all_files)} files. Starting ingest...")

    total_docs = 0
    total_sent = 0
    ctime_vals: List[str] = []

    # 清空错误日志
    with error_log.open("w", encoding="utf-8") as f:
        f.write(f"ts\tfile\terror\n")

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

    # 初始化 ParquetWriter (单文件模式)
    writer = None

    processed_files_count = 0
    out_rows: List[Dict] = []

    for f in all_files:
        # 路径解析: data_root/BRAND/MODEL/file.json
        try:
            rel = f.relative_to(data_root)
            parts = rel.parts
            # 兼容性处理：如果目录下直接是文件，brand/model 设为 unknown 或文件夹名
            if len(parts) >= 2:
                brand, model = parts[0], parts[1]
            elif len(parts) == 1:
                brand, model = parts[0], "unknown"
            else:
                brand, model = "unknown", "unknown"
        except ValueError:
            brand, model = "unknown", "unknown"

        try:
            records = list(load_json_records(f))
        except Exception as e:
            with error_log.open("a", encoding="utf-8") as elf:
                elf.write(f"{datetime.now()}\t{f}\t{e}\n")
            continue

        sentences_written_for_file = 0
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
                sentences_written_for_file += 1

            total_docs += 1
            total_sent += len(sents)
            if ctime:
                ctime_vals.append(ctime)
                file_ctimes.append(ctime)

            # 内存缓冲区写入磁盘
            if len(out_rows) >= chunk_size:
                table = pa.Table.from_pylist(out_rows, schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, schema, compression="zstd")
                writer.write_table(table)
                out_rows.clear()
                print(f"[PROGRESS] Processed {total_docs} docs...", end="\r")

            if args.max_docs and total_docs >= args.max_docs:
                break

        # 记录 Manifest
        append_manifest(
            manifest_fp,
            {
                "source_path": str(f),
                "docs": len(records),
                "sentences": sentences_written_for_file,
                "ctime_min": min(file_ctimes) if file_ctimes else None,
                "ctime_max": max(file_ctimes) if file_ctimes else None,
                "ts": datetime.now().isoformat(timespec="seconds"),
            },
        )
        processed_files_count += 1

        if args.max_docs and total_docs >= args.max_docs:
            break

    # 写入剩余数据
    if out_rows:
        table = pa.Table.from_pylist(out_rows, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(output_path, schema, compression="zstd")
        writer.write_table(table)
        out_rows.clear()

    if writer:
        writer.close()
    else:
        # 如果没有数据写入，生成一个空文件保持流程通畅
        print("[WARN] No data rows extracted. Creating empty parquet file.")
        empty_table = pa.Table.from_pylist([], schema=schema)
        writer = pq.ParquetWriter(output_path, schema, compression="zstd")
        writer.write_table(empty_table)
        writer.close()

    # 写入统计信息
    stats = {
        "domain": domain,
        "data_root": str(data_root),
        "output_file": str(output_path),
        "files_processed": processed_files_count,
        "docs": total_docs,
        "sentences": total_sent,
        "ctime_min": min(ctime_vals) if ctime_vals else None,
        "ctime_max": max(ctime_vals) if ctime_vals else None,
        "ts": datetime.now().isoformat(timespec="seconds"),
    }
    stats_fp.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[DONE] Success! Clean data saved to: {output_path}")
    print(f"       Docs: {total_docs} | Sentences: {total_sent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# -*- coding: utf-8 -*-
"""
sentiment_04_infer_asc.py
Route-B Sentiment: Full inference (ASC) on aspect_pairs_ds (streaming, resumable).

Key features
- Skip-if-exists now also updates checkpoint periodically to speed up resume scans
- Robust parquet reading: file -> row-group -> dictionary_decode (+ cast shard)
- Stream inference: never merges whole dataset; writes per row-group part parquet
- Resumable: checkpoint.json records done keys (relpath::rg=N); safe to Ctrl+C and --resume
- Aspect-aware text template: default "{aspect_l1}#{aspect_l2}：{sentence}"
- CUDA/FP16 controlled by flags

Typical usage (PowerShell)
python -u .\scripts\route_b_sentiment\sentiment_04_infer_asc.py `
  --input .\outputs\phone_v2\sentiment\aspect_pairs_ds `
  --output-dir .\outputs\phone_v2\sentiment\asc_pred_ds `
  --model-dir .\outputs\phone_v2\models\asc_lora_v1 `
  --base-model hfl/chinese-macbert-base `
  --batch-size 64 --max-length 256 --fp16 `
  --resume --heartbeat-sec 30
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from peft import PeftModel, PeftConfig  # type: ignore
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False


# -----------------------------
# basic utils
# -----------------------------
def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_info(msg: str) -> None:
    print(f"[INFO] {now_ts()} {msg}", flush=True)


def log_warn(msg: str) -> None:
    print(f"[WARN] {now_ts()} {msg}", flush=True)


def log_err(msg: str) -> None:
    print(f"[ERROR] {now_ts()} {msg}", flush=True)


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def rm_tree(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)


def read_json(p: Path) -> Dict:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json_atomic(p: Path, obj: Dict) -> None:
    safe_mkdir(p.parent)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def append_jsonl(p: Path, obj: Dict) -> None:
    safe_mkdir(p.parent)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# -----------------------------
# input discovery
# -----------------------------
def list_parquet_files(input_path: Path) -> Tuple[List[Path], Optional[Path]]:
    if input_path.is_file() and input_path.suffix.lower() == ".parquet":
        return [input_path], None
    files = sorted([p for p in input_path.rglob("*.parquet") if p.is_file()])
    return files, input_path


_SHARD_RE = re.compile(r"(?:^|[/\\])shard=(\d+)(?:[/\\]|$)")


def infer_shard_from_relpath(rel: str) -> Optional[int]:
    m = _SHARD_RE.search(rel)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


# -----------------------------
# device / model loading
# -----------------------------
def choose_device(no_cuda: bool) -> torch.device:
    if (not no_cuda) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_tokenizer(
    base_model: str,
    model_dir: Path,
    device: torch.device,
    fp16: bool,
    local_only: bool,
    hf_token: Optional[str],
    merge_lora: bool,
):
    token_kwargs = {"token": hf_token} if hf_token else {}
    common_kwargs = dict(local_files_only=bool(local_only), **token_kwargs)

    # tokenizer: prefer model_dir, fallback to base_model
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True, **common_kwargs)
        log_info(f"tokenizer loaded from model-dir: {model_dir}")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, **common_kwargs)
        log_info(f"tokenizer loaded from base-model: {base_model}")

    dtype = torch.float16 if (device.type == "cuda" and fp16) else torch.float32

    # 1) try full model from model_dir
    try:
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), torch_dtype=dtype, **common_kwargs)
        model.to(device)
        model.eval()
        log_info(f"model loaded as full checkpoint from model-dir: {model_dir}")
        return tokenizer, model
    except Exception as e:
        log_warn(f"full model load from model-dir failed, will try base+adapter. reason={e}")

    # 2) base + adapter
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=3,
        ignore_mismatched_sizes=True,
        torch_dtype=dtype,
        **common_kwargs,
    )
    if not _HAS_PEFT:
        raise RuntimeError("peft not installed, but model-dir is not a full model. Please: pip install peft")

    # validate adapter
    try:
        _ = PeftConfig.from_pretrained(str(model_dir), **common_kwargs)
    except Exception as e:
        raise RuntimeError(f"model-dir does not look like a PEFT adapter: {model_dir}. reason={e}")

    model = PeftModel.from_pretrained(model, str(model_dir), local_files_only=bool(local_only), **token_kwargs)

    if merge_lora:
        try:
            model = model.merge_and_unload()
            log_info("LoRA merged into base model for inference.")
        except Exception as e:
            log_warn(f"merge_and_unload failed; continue with PEFT wrapper. reason={e}")

    model.to(device)
    model.eval()
    log_info(f"model loaded as base+adapter: base={base_model}, adapter={model_dir}")
    return tokenizer, model


# -----------------------------
# parquet normalize
# -----------------------------
def _decode_dictionary_chunked(col: pa.ChunkedArray) -> pa.ChunkedArray:
    chunks: List[pa.Array] = []
    for ch in col.chunks:
        if pa.types.is_dictionary(ch.type):
            chunks.append(pc.dictionary_decode(ch))
        else:
            chunks.append(ch)
    return pa.chunked_array(chunks)


def normalize_table_schema(tbl: pa.Table) -> pa.Table:
    """
    - decode any dictionary columns (chunk-wise)
    - cast 'shard' to int32 if present
    """
    schema = tbl.schema
    new_cols: List[pa.ChunkedArray] = []
    new_fields: List[pa.Field] = []

    for i in range(tbl.num_columns):
        f = schema.field(i)
        col = tbl.column(i)

        if pa.types.is_dictionary(f.type):
            col2 = _decode_dictionary_chunked(col)
            f2 = pa.field(f.name, col2.type, nullable=True)
        else:
            col2 = col
            f2 = f

        if f2.name == "shard":
            try:
                col2 = pc.cast(col2, pa.int32(), safe=False)
                f2 = pa.field("shard", pa.int32(), nullable=True)
            except Exception:
                pass

        new_cols.append(col2)
        new_fields.append(f2)

    return pa.Table.from_arrays(new_cols, schema=pa.schema(new_fields))


# -----------------------------
# inference
# -----------------------------
def build_texts(
    tbl: pa.Table,
    sentence_col: str,
    aspect_l1_col: str,
    aspect_l2_col: str,
    text_template: str,
) -> List[str]:
    if sentence_col not in tbl.column_names:
        raise ValueError(f"Missing required column: {sentence_col}")

    sents = tbl[sentence_col].to_pylist()

    # aspect columns optional (fallback to empty)
    if aspect_l1_col in tbl.column_names:
        a1 = tbl[aspect_l1_col].to_pylist()
    else:
        a1 = [""] * len(sents)

    if aspect_l2_col in tbl.column_names:
        a2 = tbl[aspect_l2_col].to_pylist()
    else:
        a2 = [""] * len(sents)

    out: List[str] = []
    for s, x, y in zip(sents, a1, a2):
        s = "" if s is None else str(s)
        x = "" if x is None else str(x)
        y = "" if y is None else str(y)
        out.append(text_template.format(sentence=s, aspect_l1=x, aspect_l2=y))
    return out


@torch.no_grad()
def infer_probs(
    tokenizer,
    model,
    device: torch.device,
    fp16: bool,
    texts: List[str],
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    probs_all: List[np.ndarray] = []
    use_amp = (device.type == "cuda" and fp16)

    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tokenizer(
            chunk,
            truncation=True,
            max_length=int(max_length),
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = model(**enc)
        else:
            out = model(**enc)

        logits = out.logits
        probs = torch.softmax(logits, dim=-1).float().cpu().numpy()
        probs_all.append(probs)

    if not probs_all:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(probs_all, axis=0)


def build_output_table(input_table: pa.Table, probs: np.ndarray) -> pa.Table:
    if input_table.num_rows != probs.shape[0]:
        raise ValueError(f"row mismatch: table_rows={input_table.num_rows} probs_rows={probs.shape[0]}")

    pred_id = probs.argmax(axis=1).astype(np.int16)
    conf = probs.max(axis=1).astype(np.float32)
    pred_label = np.where(pred_id == 0, "NEG", np.where(pred_id == 1, "NEU", "POS")).tolist()

    out = input_table
    out = out.append_column("pred_id", pa.array(pred_id, type=pa.int16()))
    out = out.append_column("pred_label", pa.array(pred_label, type=pa.string()))
    out = out.append_column("confidence", pa.array(conf, type=pa.float32()))
    out = out.append_column("p_neg", pa.array(probs[:, 0].astype(np.float32), type=pa.float32()))
    out = out.append_column("p_neu", pa.array(probs[:, 1].astype(np.float32), type=pa.float32()))
    out = out.append_column("p_pos", pa.array(probs[:, 2].astype(np.float32), type=pa.float32()))
    return out


# -----------------------------
# checkpoint
# -----------------------------
def load_done_set(ckpt: Dict) -> Set[str]:
    done = ckpt.get("done", [])
    if isinstance(done, list):
        return set(done)
    return set()


def make_key(rel: str, rg: int) -> str:
    return f"{rel}::rg={rg}"


# -----------------------------
# main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="aspect_pairs_ds directory OR a single parquet file")
    ap.add_argument("--output-dir", required=True, help="output dataset directory")
    ap.add_argument("--model-dir", required=True, help="LoRA adapter dir (or full model dir)")
    ap.add_argument("--base-model", required=True, help="base model id or path, e.g. hfl/chinese-macbert-base")

    ap.add_argument("--sentence-col", default="sentence")
    ap.add_argument("--aspect-l1-col", default="aspect_l1")
    ap.add_argument("--aspect-l2-col", default="aspect_l2")
    ap.add_argument("--text-template", default="{aspect_l1}#{aspect_l2}：{sentence}")

    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)

    ap.add_argument("--fp16", action="store_true", help="enable fp16 autocast on CUDA")
    ap.add_argument("--no-cuda", action="store_true", help="force CPU even if CUDA is available")
    ap.add_argument("--merge-lora", action="store_true", help="merge LoRA into base for inference (optional)")

    ap.add_argument("--resume", action="store_true", help="resume from checkpoint.json")
    ap.add_argument("--overwrite", action="store_true", help="delete output-dir first")
    ap.add_argument("--max-files", type=int, default=0)

    ap.add_argument("--heartbeat-sec", type=int, default=30)
    ap.add_argument("--local-only", action="store_true", help="HF local_files_only")
    ap.add_argument("--hf-token-env", default="HF_TOKEN", help="env var name for HF token")
    ap.add_argument("--compression", default="snappy", choices=["snappy", "zstd", "gzip", "none"])

    args = ap.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)

    if not input_path.exists():
        log_err(f"input not found: {input_path}")
        return 2
    if not model_dir.exists():
        log_err(f"model-dir not found: {model_dir}")
        return 2

    if args.overwrite:
        log_warn(f"--overwrite enabled, deleting output-dir: {out_dir}")
        rm_tree(out_dir)

    safe_mkdir(out_dir)
    ckpt_path = out_dir / "checkpoint.json"
    manifest_path = out_dir / "manifest.jsonl"

    hf_token = os.environ.get(args.hf_token_env) if args.hf_token_env else None
    device = choose_device(bool(args.no_cuda))

    log_info(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} device={device} fp16={bool(args.fp16 and device.type=='cuda')}")
    if device.type == "cuda":
        try:
            log_info(f"gpu={torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    tokenizer, model = load_model_and_tokenizer(
        base_model=args.base_model,
        model_dir=model_dir,
        device=device,
        fp16=bool(args.fp16),
        local_only=bool(args.local_only),
        hf_token=hf_token,
        merge_lora=bool(args.merge_lora),
    )

    files, base_dir = list_parquet_files(input_path)
    if not files:
        log_err("no parquet files found under input.")
        return 2
    if args.max_files and args.max_files > 0:
        files = files[: int(args.max_files)]

    # resume
    ckpt: Dict = {}
    done: Set[str] = set()
    if args.resume and ckpt_path.exists():
        ckpt = read_json(ckpt_path)
        done = load_done_set(ckpt)
        log_info(f"resume enabled: loaded done={len(done)}")
    else:
        ckpt = {
            "version": 1,
            "created_at": now_ts(),
            "input": str(input_path),
            "output_dir": str(out_dir),
            "base_model": args.base_model,
            "model_dir": str(model_dir),
            "text_template": args.text_template,
            "done": [],
            "stats": {},
        }

    total_row_groups = 0
    for f in files:
        pf = pq.ParquetFile(str(f))
        total_row_groups += pf.num_row_groups

    compression = None if args.compression == "none" else args.compression

    t0 = time.time()
    last_hb = time.time()
    ckpt_write_interval = max(5, int(args.heartbeat_sec))
    last_ckpt_write = time.time()

    processed_rg = 0
    skipped_rg = 0
    written_parts = 0
    total_rows = 0

    log_info(f"input parquet files={len(files)} total_row_groups={total_row_groups}")

    try:
        for fi, f in enumerate(files):
            pf = pq.ParquetFile(str(f))

            # relative path key
            if base_dir is None:
                rel = f.name
            else:
                try:
                    rel = str(f.relative_to(base_dir))
                except Exception:
                    rel = f.name

            shard_val = infer_shard_from_relpath(rel)
            if shard_val is None:
                shard_val = -1

            for rg in range(pf.num_row_groups):
                key = make_key(rel, rg)
                processed_rg += 1

                if key in done:
                    skipped_rg += 1
                    continue

                # heartbeat
                if (time.time() - last_hb) >= max(1, int(args.heartbeat_sec)):
                    elapsed = (time.time() - t0) / 60.0
                    log_info(
                        f"[HEARTBEAT] elapsed={elapsed:.1f}m "
                        f"file={fi+1}/{len(files)} rg={processed_rg}/{total_row_groups} "
                        f"written_parts={written_parts} skipped_rg={skipped_rg} total_rows={total_rows}"
                    )
                    last_hb = time.time()

                # output part path (mirror parent + row-group part)
                rel_path = Path(rel)
                out_parent = out_dir / rel_path.parent
                safe_mkdir(out_parent)
                out_fp = out_parent / f"{rel_path.stem}-rg{rg:04d}.parquet"

                # extra safety: if file already exists, treat as done (unless overwrite mode already wiped output-dir)
                if out_fp.exists():
                    done.add(key)
                    skipped_rg += 1
                    now = time.time()
                    if (now - last_ckpt_write) >= ckpt_write_interval:
                        ckpt["done"] = sorted(done)
                        ckpt["stats"] = {
                            "processed_rg": processed_rg,
                            "total_row_groups": total_row_groups,
                            "written_parts": written_parts,
                            "skipped_rg": skipped_rg,
                            "total_rows": total_rows,
                            "last_update": now_ts(),
                        }
                        write_json_atomic(ckpt_path, ckpt)
                        last_ckpt_write = now
                    continue

                # read row group
                try:
                    tbl = pf.read_row_group(rg)
                except Exception as e:
                    log_err(f"failed reading row-group: file={f} rg={rg} err={e}")
                    return 2

                if tbl.num_rows == 0:
                    # write empty part for traceability
                    pq.write_table(tbl, str(out_fp), compression=compression)
                    append_jsonl(manifest_path, {
                        "ts": now_ts(),
                        "src_file": rel,
                        "row_group": rg,
                        "out_path": str(out_fp.relative_to(out_dir)),
                        "rows": 0,
                        "shard": int(shard_val),
                    })
                    done.add(key)
                    ckpt["done"] = sorted(done)
                    ckpt["stats"] = {
                        "processed_rg": processed_rg,
                        "total_row_groups": total_row_groups,
                        "written_parts": written_parts,
                        "skipped_rg": skipped_rg,
                        "total_rows": total_rows,
                        "last_update": now_ts(),
                    }
                    write_json_atomic(ckpt_path, ckpt)
                    continue

                # normalize schema
                try:
                    tbl = normalize_table_schema(tbl)
                except Exception as e:
                    log_err(f"failed normalize schema: file={f} rg={rg} err={e}")
                    return 2

                # build texts (aspect-aware)
                try:
                    texts = build_texts(
                        tbl,
                        sentence_col=args.sentence_col,
                        aspect_l1_col=args.aspect_l1_col,
                        aspect_l2_col=args.aspect_l2_col,
                        text_template=args.text_template,
                    )
                except Exception as e:
                    log_err(f"failed build texts: file={f} rg={rg} err={e}")
                    return 2

                # infer
                probs = infer_probs(
                    tokenizer, model, device,
                    fp16=bool(args.fp16),
                    texts=texts,
                    max_length=int(args.max_length),
                    batch_size=int(args.batch_size),
                )

                # build output table
                out_tbl = build_output_table(tbl, probs)

                # write
                try:
                    pq.write_table(out_tbl, str(out_fp), compression=compression)
                except Exception as e:
                    log_err(f"failed writing parquet: {out_fp} err={e}")
                    return 2

                rows = int(out_tbl.num_rows)
                total_rows += rows
                written_parts += 1

                append_jsonl(manifest_path, {
                    "ts": now_ts(),
                    "src_file": rel,
                    "row_group": rg,
                    "out_path": str(out_fp.relative_to(out_dir)),
                    "rows": rows,
                    "shard": int(shard_val),
                })

                done.add(key)

                # checkpoint each row-group (safe resume)
                ckpt["done"] = sorted(done)
        ckpt["stats"] = {
            "processed_rg": processed_rg,
            "total_row_groups": total_row_groups,
            "written_parts": written_parts,
            "skipped_rg": skipped_rg,
            "total_rows": total_rows,
            "last_update": now_ts(),
        }
        write_json_atomic(ckpt_path, ckpt)
        last_ckpt_write = time.time()

        ckpt["done"] = sorted(done)
        ckpt["stats"] = {
            "processed_rg": processed_rg,
            "total_row_groups": total_row_groups,
            "written_parts": written_parts,
            "skipped_rg": skipped_rg,
            "total_rows": total_rows,
            "last_update": now_ts(),
        }
        write_json_atomic(ckpt_path, ckpt)

        elapsed = (time.time() - t0) / 60.0
        log_info(f"DONE. elapsed={elapsed:.1f}m written_parts={written_parts} skipped_rg={skipped_rg} total_rows={total_rows} out_dir={out_dir}")
        return 0

    except KeyboardInterrupt:
        log_warn("Interrupted by user (Ctrl+C). Outputs kept. Re-run with --resume to continue.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

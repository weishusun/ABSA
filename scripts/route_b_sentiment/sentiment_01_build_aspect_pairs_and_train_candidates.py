# -*- coding: utf-8 -*-
"""
Route B - sentiment_01 (streaming, no DuckDB) - v3

Key goals:
- train_candidates: bounded pool, minute-level, optional single-aspect within pool only
- aspect_pairs_ds: STREAM by iter_batches (NOT row-group loop) to avoid 1.8M row groups
- visible incremental writes: shard=*/part-*.parquet
- resume: batch checkpoint (file_index + batch_index), idempotent shard writes per batch
- parquet metadata thrift limit: configurable + Windows C long clamp to avoid overflow
- Input schema must包含 domain/brand/model/doc_id/sentence_idx/sentence/aspect_l1/aspect_l2/(ctime); 若确实缺 ctime 可显式传 --allow-missing-ctime

Outputs:
  {output_dir}/train_candidates.parquet
  {output_dir}/aspect_pairs_ds/shard=*/part-*.parquet
  {output_dir}/aspect_pairs_ds/manifest.jsonl
  {output_dir}/aspect_pairs_ds/checkpoint.json

Recommended usage:

A) Minute-level train candidates (5k)
  python -u .\scripts\route_b_sentiment\sentiment_01_build_aspect_pairs_and_train_candidates.py `
    --input .\outputs\phone_v2\aspect_sentences.parquet `
    --output-dir .\outputs\phone_v2\sentiment `
    --max-train-rows 5000 `
    --train-pool-rows 200000 `
    --require-single-aspect `
    --overwrite `
    --threads 6 --heartbeat-sec 30 `
    --thrift-string-mb 2047 --thrift-container-mb 2047

B) Build ds (streaming, resumable)  ★强烈建议先删掉旧 ds（旧逻辑写了海量小文件）
  Remove-Item -Recurse -Force .\outputs\phone_v2\sentiment\aspect_pairs_ds

  python -u .\scripts\route_b_sentiment\sentiment_01_build_aspect_pairs_and_train_candidates.py `
    --input .\outputs\phone_v2\aspect_sentences.parquet `
    --output-dir .\outputs\phone_v2\sentiment `
    --write-ds --overwrite `
    --skip-train-candidates `
    --shard-n 64 `
    --ds-batch-rows 50000 `
    --threads 6 --heartbeat-sec 30 `
    --thrift-string-mb 2047 --thrift-container-mb 2047

Resume:
  python -u .\scripts\route_b_sentiment\sentiment_01_build_aspect_pairs_and_train_candidates.py `
    --input .\outputs\phone_v2\aspect_sentences.parquet `
    --output-dir .\outputs\phone_v2\sentiment `
    --write-ds --resume `
    --skip-train-candidates `
    --shard-n 64 `
    --ds-batch-rows 50000 `
    --threads 6 --heartbeat-sec 30 `
    --thrift-string-mb 2047 --thrift-container-mb 2047
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import duckdb

# Dependencies
try:
    import polars as pl
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.dataset as pds
except Exception as e:
    raise RuntimeError(
        "Missing dependencies. Please run: pip install -U polars pyarrow\n"
        f"Original import error: {e}"
    )

JOIN_KEYS = ["domain", "brand", "model", "doc_id", "sentence_idx"]
NEEDED_COLS = ["domain", "brand", "model", "doc_id", "ctime", "sentence_idx", "sentence", "aspect_l1", "aspect_l2"]

# Windows C long is often 32-bit; pyarrow parquet thrift limit setter uses C long
MAX_C_LONG = 2_147_483_647  # 2^31-1


def now_ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    units = ["KB", "MB", "GB", "TB", "PB"]
    x = float(n)
    for u in units:
        x /= 1024.0
        if x < 1024.0:
            return f"{x:.2f}{u}"
    return f"{x:.2f}EB"


def dir_parquet_stats(p: Path) -> Tuple[int, int]:
    """return (num_parquet_files, total_bytes) under directory"""
    if not p.exists():
        return 0, 0
    files, total = 0, 0
    for root, _, fns in os.walk(p):
        for fn in fns:
            if fn.lower().endswith(".parquet"):
                files += 1
                try:
                    total += (Path(root) / fn).stat().st_size
                except OSError:
                    pass
    return files, total


@dataclass
class Heartbeat:
    hb_sec: int
    stage: str = "init"
    start: float = time.time()
    last: float = 0.0
    lastwrite: str = "-"

    def tick(self, ds_dir: Optional[Path] = None, extra: str = "", force: bool = False) -> None:
        t = time.time()
        if (not force) and (t - self.last < self.hb_sec):
            return
        self.last = t
        elapsed_m = (t - self.start) / 60.0
        ds_files, ds_bytes = (0, 0) if ds_dir is None else dir_parquet_stats(ds_dir)
        print(
            f"[HEARTBEAT] {now_ts()} elapsed={elapsed_m:.1f}m stage={self.stage} "
            f"ds_files={ds_files} ds_size={human_bytes(ds_bytes)} lastwrite={self.lastwrite} {extra}",
            flush=True,
        )

    def mark_write(self, what: str, ds_dir: Optional[Path] = None, extra: str = "") -> None:
        self.lastwrite = f"{what}@{now_ts()}"
        self.tick(ds_dir=ds_dir, extra=extra, force=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)

    # Compatibility with old CLI
    ap.add_argument("--write-ds", action="store_true")
    ap.add_argument("--write-pairs-parquet", action="store_true")  # discouraged; kept for compatibility
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--resume", action="store_true")

    ap.add_argument("--shard-n", type=int, default=64)

    ap.add_argument("--skip-train-candidates", action="store_true")
    ap.add_argument("--max-train-rows", type=int, default=5000)
    ap.add_argument("--require-single-aspect", action="store_true")

    # Bounded pool for minute-level candidates (no full scan)
    ap.add_argument("--train-pool-rows", type=int, default=200000, help="Max rows loaded for train pool (bounded).")
    ap.add_argument("--train-pool-batch-rows", type=int, default=50000, help="Batch rows for loading train pool.")
    ap.add_argument("--train-hash-seed", type=int, default=0)

    # Old sampling flags (accepted, mapped approximately)
    ap.add_argument("--train-key-sample-method", choices=["reservoir", "bernoulli", "none"], default="none")
    ap.add_argument("--train-key-sample-rows", type=int, default=600000)
    ap.add_argument("--train-key-sample-frac", type=float, default=0.0)
    ap.add_argument("--train-key-inner-limit", type=int, default=0)

    # DS streaming / checkpointing
    ap.add_argument("--checkpoint-file", default="", help="Defaults to {output_dir}/aspect_pairs_ds/checkpoint.json")
    ap.add_argument("--manifest-file", default="", help="Defaults to {output_dir}/aspect_pairs_ds/manifest.jsonl")
    ap.add_argument("--ds-batch-rows", type=int, default=50000,
                    help="Rows per batch when streaming parquet for ds build (avoid row-group loop).")

    # Operational knobs
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--heartbeat-sec", type=int, default=30)

    # Deprecated DuckDB knobs (ignored but accepted)
    ap.add_argument("--duckdb-file", default="")
    ap.add_argument("--memory-limit", default="14GB")
    ap.add_argument("--temp-dir", default="")
    ap.add_argument("--enable-duckdb-progress", action="store_true")
    ap.add_argument("--progress-ms", type=int, default=2000)

    # Parquet thrift metadata limits (MB)
    ap.add_argument("--thrift-string-mb", type=int, default=512,
                    help="Override PyArrow thrift string size limit for parquet metadata (MB).")
    ap.add_argument("--thrift-container-mb", type=int, default=512,
                    help="Override PyArrow thrift container size limit for parquet metadata (MB).")

    ap.add_argument("--allow-missing-ctime", action="store_true",
                    help="显式允许输入缺少 ctime（默认不允许，缺失会报 FATAL）")

    return ap.parse_args()


def list_parquet_files(inp: Path) -> List[Path]:
    if inp.is_file():
        return [inp]
    if inp.is_dir():
        files = sorted([p for p in inp.rglob("*.parquet") if p.is_file()])
        if not files:
            raise FileNotFoundError(f"No parquet files found under directory: {inp}")
        return files
    raise FileNotFoundError(f"Input path not found: {inp}")


def atomic_write_json(fp: Path, obj: dict) -> None:
    ensure_dir(fp.parent)
    tmp = fp.with_suffix(fp.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, fp)


def load_checkpoint(fp: Path) -> dict:
    """
    New format: {"file_index": int, "batch_index": int}
    Backward-compat: if old checkpoint contains row_group_index, we reset batch_index=0.
    """
    if not fp.exists():
        return {"file_index": 0, "batch_index": 0}
    try:
        ckpt = json.loads(fp.read_text(encoding="utf-8"))
        if "batch_index" in ckpt:
            return {"file_index": int(ckpt.get("file_index", 0)), "batch_index": int(ckpt.get("batch_index", 0))}
        # old format
        if "row_group_index" in ckpt:
            return {"file_index": int(ckpt.get("file_index", 0)), "batch_index": 0}
        return {"file_index": int(ckpt.get("file_index", 0)), "batch_index": int(ckpt.get("batch_index", 0))}
    except Exception:
        return {"file_index": 0, "batch_index": 0}


def append_manifest(fp: Path, rec: dict) -> None:
    ensure_dir(fp.parent)
    with fp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


def quick_write_test(out_dir: Path) -> None:
    ensure_dir(out_dir)
    (out_dir / "write_test.txt").write_text(f"ok {datetime.now().isoformat()}\n", encoding="utf-8")


def _open_parquet_file(
    path: Path,
    thrift_string_limit: int,
    thrift_container_limit: int,
) -> pq.ParquetFile:
    return pq.ParquetFile(
        str(path),
        thrift_string_size_limit=thrift_string_limit,
        thrift_container_size_limit=thrift_container_limit,
    )


def _concat_tables_safe(tables: List[pa.Table]) -> pa.Table:
    """
    pyarrow deprecates promote=True in favor of promote_options.
    Keep compatibility across versions.
    """
    try:
        return pa.concat_tables(tables, promote_options="default")
    except TypeError:
        # older pyarrow
        return pa.concat_tables(tables, promote=True)


def build_train_candidates(
    parquet_files: List[Path],
    out_dir: Path,
    max_train_rows: int,
    pool_rows: int,
    pool_batch_rows: int,
    thrift_string_limit: int,
    thrift_container_limit: int,
    require_single_aspect: bool,
    seed: int,
    inner_limit: int,
    overwrite: bool,
    hb: Heartbeat,
) -> None:
    train_fp = out_dir / "train_candidates.parquet"
    if train_fp.exists() and train_fp.stat().st_size > 0 and not overwrite:
        print(f"[SKIP] train_candidates exists: {train_fp}", flush=True)
        return
    if overwrite and train_fp.exists():
        try:
            train_fp.unlink()
        except Exception:
            pass

    hb.stage = "train_pool_load"
    hb.tick(extra=f"(pool_rows={pool_rows} batch_rows={pool_batch_rows})", force=True)

    # Load bounded pool by iter_batches (avoid row-group loop; critical if row groups are tiny)
    batches: List[pa.RecordBatch] = []
    got = 0

    for f in parquet_files:
        pf = _open_parquet_file(f, thrift_string_limit, thrift_container_limit)
        cols = [c for c in NEEDED_COLS if c in pf.schema.names]
        if not cols:
            raise RuntimeError(f"No expected columns found in parquet schema: {f}")

        for rb in pf.iter_batches(batch_size=pool_batch_rows, columns=cols, use_threads=True):
            if rb.num_rows == 0:
                continue
            batches.append(rb)
            got += rb.num_rows
            hb.tick(extra=f"(pool_loaded={got})")
            if got >= pool_rows:
                break
        if got >= pool_rows:
            break

    if not batches:
        raise RuntimeError("train pool is empty. Check input parquet and columns.")

    pool_tbl = pa.Table.from_batches(batches)
    # If we overshot, slice down to exact pool_rows for stability
    if pool_tbl.num_rows > pool_rows:
        pool_tbl = pool_tbl.slice(0, pool_rows)

    df = pl.from_arrow(pool_tbl)

    # Optional per-product cap inside pool (compat: train-key-inner-limit)
    if inner_limit and inner_limit > 0:
        hb.stage = "train_pool_inner_limit"
        hb.tick()
        df = (
            df.with_columns(
                pl.int_range(0, pl.len()).over(["domain", "brand", "model"]).alias("_rn")
            )
            .filter(pl.col("_rn") < inner_limit)
            .drop("_rn")
        )

    # Require single aspect within POOL only (fast, no global groupby)
    if require_single_aspect:
        hb.stage = "train_pool_single_aspect"
        hb.tick()
        df = df.filter(pl.len().over(JOIN_KEYS) == 1)

    if df.height == 0:
        raise RuntimeError(
            "train pool after filtering is empty. "
            "Try increasing --train-pool-rows or disable --require-single-aspect."
        )

    hb.stage = "train_candidates_select"
    hb.tick()

    # Stable pseudo-random: hash struct and take top-N
    df = (
        df.with_columns(
            pl.struct(JOIN_KEYS + ["aspect_l1", "aspect_l2"]).hash(seed=seed).alias("_h")
        )
        .sort("_h")
        .head(max_train_rows)
        .drop("_h")
    )

    if df.height == 0:
        raise RuntimeError("train_candidates result is empty (unexpected).")

    hb.stage = "write_train_candidates"
    hb.tick()
    tmp = train_fp.with_suffix(".parquet.tmp")
    df.write_parquet(tmp, compression="zstd")
    os.replace(tmp, train_fp)
    hb.mark_write("train_candidates.parquet", extra=f"(rows={df.height}, size={human_bytes(train_fp.stat().st_size)})")
    print(f"[OK] wrote {train_fp} rows={df.height} size={human_bytes(train_fp.stat().st_size)}", flush=True)


def build_aspect_pairs_ds_stream(
    parquet_files: List[Path],
    ds_dir: Path,
    shard_n: int,
    ds_batch_rows: int,
    checkpoint_fp: Path,
    thrift_string_limit: int,
    thrift_container_limit: int,
    manifest_fp: Path,
    resume: bool,
    overwrite: bool,
    seed: int,
    hb: Heartbeat,
) -> None:
    if overwrite and ds_dir.exists():
        import shutil
        shutil.rmtree(ds_dir, ignore_errors=True)

    ensure_dir(ds_dir)

    ckpt = {"file_index": 0, "batch_index": 0}
    if resume:
        ckpt = load_checkpoint(checkpoint_fp)

    file_i0 = int(ckpt.get("file_index", 0))
    batch_i0 = int(ckpt.get("batch_index", 0))

    hb.stage = "ds_stream_init"
    hb.tick(
        ds_dir=ds_dir,
        extra=f"(resume={resume} start_file={file_i0} start_batch={batch_i0} ds_batch_rows={ds_batch_rows})",
        force=True,
    )

    # Stream by iter_batches (critical improvement)
    for fi in range(file_i0, len(parquet_files)):
        f = parquet_files[fi]
        pf = _open_parquet_file(f, thrift_string_limit, thrift_container_limit)

        cols = [c for c in NEEDED_COLS if c in pf.schema.names]
        if not cols:
            raise RuntimeError(f"No expected columns found in parquet schema: {f}")

        batch_iter = pf.iter_batches(batch_size=ds_batch_rows, columns=cols, use_threads=True)

        # resume: skip already processed batches for the first file
        start_b = batch_i0 if (resume and fi == file_i0) else 0
        if start_b > 0:
            batch_iter = itertools.islice(batch_iter, start_b, None)

        b_idx = start_b

        for rb in batch_iter:
            hb.stage = "ds_read_batch"
            hb.tick(ds_dir=ds_dir, extra=f"(file={fi+1}/{len(parquet_files)} batch={b_idx})")

            if rb.num_rows == 0:
                b_idx += 1
                atomic_write_json(checkpoint_fp, {"file_index": fi, "batch_index": b_idx})
                continue

            df = pl.from_arrow(rb)

            # Compute shard deterministically (vectorized)
            df = df.with_columns(
                (pl.struct(["domain", "brand", "model", "doc_id", "sentence_idx", "aspect_l1", "aspect_l2"])
                 .hash(seed=seed)
                 .abs() % shard_n
                 ).cast(pl.Int32).alias("shard")
            )

            hb.stage = "ds_write_partitions"
            hb.tick(ds_dir=ds_dir, extra=f"(batch_rows={df.height})")

            # One file per shard for this batch
            parts = df.partition_by("shard", as_dict=True)

            for shard_key, g in parts.items():
                # polars as_dict=True: key is tuple even for single column
                shard_id = int(shard_key[0]) if isinstance(shard_key, tuple) else int(shard_key)

                shard_path = ds_dir / f"shard={shard_id}"
                ensure_dir(shard_path)

                out_name = f"part-f{fi:04d}-b{b_idx:06d}.parquet"
                out_fp = shard_path / out_name
                tmp_fp = shard_path / (out_name + ".tmp")

                # Idempotent resume per-shard-per-batch:
                # if file already exists and non-empty, skip writing it.
                if resume and out_fp.exists():
                    try:
                        if out_fp.stat().st_size > 0:
                            continue
                    except OSError:
                        pass

                g.write_parquet(tmp_fp, compression="zstd")
                os.replace(tmp_fp, out_fp)

                append_manifest(manifest_fp, {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "file_index": fi,
                    "batch_index": b_idx,
                    "shard": shard_id,
                    "rows": int(g.height),
                    "path": str(out_fp).replace("\\", "/"),
                    "bytes": int(out_fp.stat().st_size),
                })
                hb.mark_write(f"shard={shard_id}/{out_name}", ds_dir=ds_dir, extra=f"(rows={g.height})")

            # Advance checkpoint ONLY after batch fully written
            b_idx += 1
            atomic_write_json(checkpoint_fp, {"file_index": fi, "batch_index": b_idx})

        # next file: reset batch offset
        batch_i0 = 0

    hb.stage = "ds_done"
    hb.tick(ds_dir=ds_dir, force=True)
    print(f"[OK] ds build done. ds_dir={ds_dir} checkpoint={checkpoint_fp} manifest={manifest_fp}", flush=True)


def validate_schema(input_path: Path, allow_missing_ctime: bool, thrift_string_limit: int, thrift_container_limit: int) -> None:
    try:
        dataset = pds.dataset(str(input_path), format="parquet")
        cols = set(dataset.schema.names)
    except Exception as e:
        print(f"[WARN] pyarrow schema read failed, fallback to duckdb: {type(e).__name__}", file=sys.stderr)
        try:
            con = duckdb.connect(database=":memory:")
            con.execute("PRAGMA threads=1;")

            # 注意：不要用 ? 参数绑定；用字面量路径并做转义
            p = str(input_path).replace("\\", "/").replace("'", "''")

            rel = con.sql(f"SELECT * FROM read_parquet('{p}') LIMIT 0")
            cols = set(rel.columns)

            con.close()
        except Exception as e2:
            print(f"[FATAL] duckdb schema fallback failed: {e2}", file=sys.stderr)
            raise SystemExit(2)

    required = {"domain", "brand", "model", "doc_id", "sentence_idx", "sentence", "aspect_l1", "aspect_l2"}
    if not allow_missing_ctime:
        required.add("ctime")
    missing = sorted(required - cols)
    if missing:
        print(
            f"[FATAL] 输入缺少字段: {missing}；请先跑 Step00/tag_aspects 生成带 ctime 的 aspect_sentences",
            file=sys.stderr,
        )
        raise SystemExit(2)


def compute_thrift_limits(args: argparse.Namespace) -> Tuple[int, int]:
    thrift_string_limit = int(args.thrift_string_mb) * 1024 * 1024
    thrift_container_limit = int(args.thrift_container_mb) * 1024 * 1024
    if thrift_string_limit > MAX_C_LONG:
        thrift_string_limit = MAX_C_LONG
    if thrift_container_limit > MAX_C_LONG:
        thrift_container_limit = MAX_C_LONG
    print(
        f"[INFO] thrift_string_limit={thrift_string_limit} bytes, thrift_container_limit={thrift_container_limit} bytes",
        flush=True,
    )
    return thrift_string_limit, thrift_container_limit


def apply_pyarrow_thrift_limits(thrift_string_limit: int, thrift_container_limit: int) -> bool:
    ok = False
    for mod_name in ("pyarrow._parquet", "pyarrow.parquet"):
        try:
            mod = __import__(mod_name, fromlist=["dummy"])
        except Exception:
            continue
        try:
            setter = getattr(mod, "set_thrift_string_size_limit", None) or getattr(mod, "_set_thrift_string_size_limit", None)
            if setter:
                setter(int(thrift_string_limit))
                ok = True
        except Exception:
            pass
        try:
            setter = getattr(mod, "set_thrift_container_size_limit", None) or getattr(mod, "_set_thrift_container_size_limit", None)
            if setter:
                setter(int(thrift_container_limit))
                ok = True
        except Exception:
            pass
    if not ok:
        print("[WARN] pyarrow has no thrift limit setter; metadata limits may still apply.", file=sys.stderr)
    return ok


def main() -> int:
    args = parse_args()

    # Threads: polars uses env var POLARS_MAX_THREADS
    if args.threads and int(args.threads) > 0:
        os.environ["POLARS_MAX_THREADS"] = str(int(args.threads))

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    ds_dir = out_dir / "aspect_pairs_ds"
    ensure_dir(out_dir)

    thrift_string_limit, thrift_container_limit = compute_thrift_limits(args)
    applied = apply_pyarrow_thrift_limits(thrift_string_limit, thrift_container_limit)
    print(f"[INFO] apply_thrift_limits ok={applied}", flush=True)

    validate_schema(
        in_path,
        allow_missing_ctime=bool(args.allow_missing_ctime),
        thrift_string_limit=thrift_string_limit,
        thrift_container_limit=thrift_container_limit,
    )

    hb = Heartbeat(hb_sec=int(args.heartbeat_sec))
    hb.tick(force=True)

    # Early write test (catch permission/lock issues)
    try:
        quick_write_test(out_dir)
    except Exception as e:
        print(f"[FATAL] cannot write to output-dir: {out_dir} err={e}", file=sys.stderr, flush=True)
        return 2

    parquet_files = list_parquet_files(in_path)
    print(f"[INFO] input parquet files: {len(parquet_files)}", flush=True)

    # Map old sampling flags to bounded pool rows (approximate, deterministic)
    pool_rows = int(args.train_pool_rows)
    method = str(args.train_key_sample_method)
    if method != "none":
        if method == "reservoir":
            pool_rows = max(pool_rows, int(args.train_key_sample_rows))
        elif method == "bernoulli" and float(args.train_key_sample_frac or 0.0) > 0:
            frac = float(args.train_key_sample_frac)
            pool_rows = max(
                pool_rows,
                int(args.max_train_rows) * max(10, int(1.0 / max(frac, 1e-6))),
            )
        print(f"[WARN] train-key-sample-method={method} mapped to bounded pool_rows={pool_rows} (no SQL sampling).",
              flush=True)

    # A) train_candidates (default ON unless skip)
    if not args.skip_train_candidates:
        build_train_candidates(
            parquet_files=parquet_files,
            out_dir=out_dir,
            max_train_rows=int(args.max_train_rows),
            pool_rows=int(pool_rows),
            pool_batch_rows=int(args.train_pool_batch_rows),
            thrift_string_limit=int(thrift_string_limit),
            thrift_container_limit=int(thrift_container_limit),
            require_single_aspect=bool(args.require_single_aspect),
            seed=int(args.train_hash_seed),
            inner_limit=int(args.train_key_inner_limit or 0),
            overwrite=bool(args.overwrite),
            hb=hb,
        )

    # B) ds build (streaming) if requested
    if args.write_ds:
        checkpoint_fp = Path(args.checkpoint_file) if args.checkpoint_file.strip() else (ds_dir / "checkpoint.json")
        manifest_fp = Path(args.manifest_file) if args.manifest_file.strip() else (ds_dir / "manifest.jsonl")

        build_aspect_pairs_ds_stream(
            parquet_files=parquet_files,
            ds_dir=ds_dir,
            shard_n=int(args.shard_n),
            ds_batch_rows=int(args.ds_batch_rows),
            checkpoint_fp=checkpoint_fp,
            thrift_string_limit=int(thrift_string_limit),
            thrift_container_limit=int(thrift_container_limit),
            manifest_fp=manifest_fp,
            resume=bool(args.resume),
            overwrite=bool(args.overwrite),
            seed=int(args.train_hash_seed),
            hb=hb,
        )

    if args.write_pairs_parquet:
        print("[WARN] --write-pairs-parquet is not implemented in v3 (would be huge). Use aspect_pairs_ds instead.",
              flush=True)

    hb.stage = "done"
    hb.tick(ds_dir=ds_dir, force=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

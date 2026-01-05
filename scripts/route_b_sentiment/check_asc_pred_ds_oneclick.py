# -*- coding: utf-8 -*-
"""
One-click checker for Route-B ASC inference output dataset (asc_pred_ds).

Checks:
- checkpoint.json stats
- 0-byte parquet files
- required columns
- prob normalization: p_neg+p_neu+p_pos ~= 1
- prob/conf ranges
- label distribution (ALL + confidence threshold)
- ctime scale detection (s vs ms) and date range
- top aspects share within POS/NEG (ALL + high confidence)

Run:
python -u .\scripts\route_b_sentiment\check_asc_pred_ds_oneclick.py --pred-ds .\outputs\phone_v2\sentiment\asc_pred_ds
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds


REQUIRED_COLS = [
    "pred_label", "confidence", "p_neg", "p_neu", "p_pos",
]

OPTIONAL_BIZ_COLS = [
    "brand", "model", "aspect_l1", "aspect_l2", "ctime"
]


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now_ts()}] {msg}", flush=True)


def load_json(p: Path) -> Dict:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def find_zero_byte_parquets(root: Path) -> int:
    cnt = 0
    for fp in root.rglob("*.parquet"):
        try:
            if fp.is_file() and fp.stat().st_size == 0:
                cnt += 1
        except Exception:
            pass
    return cnt


def to_numpy_float(arr: pa.Array) -> np.ndarray:
    # safe conversion for float arrays (may be ChunkedArray from dataset -> but in RecordBatch it's Array)
    return np.asarray(arr.to_numpy(zero_copy_only=False), dtype=np.float64)


def value_counts_to_dict(vc: pa.Array) -> Dict[str, int]:
    # vc is StructArray with fields ["values","counts"]
    out = {}
    if vc is None or len(vc) == 0:
        return out
    values = vc.field(0).to_pylist()
    counts = vc.field(1).to_pylist()
    for v, c in zip(values, counts):
        if v is None:
            continue
        out[str(v)] = out.get(str(v), 0) + int(c)
    return out


def merge_counts(dst: Dict[str, int], src: Dict[str, int]) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + int(v)


def topk_from_countmap(m: Dict[str, int], k: int) -> Tuple[int, list]:
    total = sum(m.values())
    items = sorted(m.items(), key=lambda x: x[1], reverse=True)[:k]
    return total, items


def convert_ctime_to_date(ts_val: int, unit: str) -> str:
    # unit: "s" or "ms"
    if unit == "ms":
        ts_val = int(ts_val) // 1000
    return datetime.fromtimestamp(int(ts_val)).date().isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-ds", required=True, help="asc_pred_ds directory")
    ap.add_argument("--conf-thr", type=float, default=0.65, help="high confidence threshold")
    ap.add_argument("--batch-size", type=int, default=65536, help="scan batch size")
    ap.add_argument("--topk", type=int, default=20, help="topk aspects to print for POS/NEG")
    ap.add_argument("--max-aspect-keys", type=int, default=500000, help="cap for unique aspect keys (safety)")
    ap.add_argument("--report-json", default="", help="write report json to this path (optional)")
    args = ap.parse_args()

    root = Path(args.pred_ds)
    if not root.exists():
        raise SystemExit(f"[FATAL] pred-ds not found: {root}")

    ckpt = load_json(root / "checkpoint.json")
    if ckpt:
        log(f"checkpoint.json found. stats={ckpt.get('stats', {})} done_count={len(ckpt.get('done', []))}")
    else:
        log("checkpoint.json NOT found (not fatal, but resume/progress trace may be missing).")

    manifest = root / "manifest.jsonl"
    log(f"manifest.jsonl exists={manifest.exists()}")

    zcnt = find_zero_byte_parquets(root)
    if zcnt > 0:
        log(f"[WARN] Found {zcnt} zero-byte parquet files. Recommend deleting them then re-run inference with --resume.")
    else:
        log("0-byte parquet files: none")

    # dataset open
    parquet_files = [str(p) for p in root.rglob("*.parquet") if p.is_file()]
    if not parquet_files:
        raise SystemExit(f"[FATAL] no parquet files found under: {root}")
    log(f"parquet files found: {len(parquet_files)}")
    dset = ds.dataset(parquet_files, format="parquet")

    cols = dset.schema.names
    log(f"dataset opened. total_columns={len(cols)}")

    missing = [c for c in REQUIRED_COLS if c not in cols]
    if missing:
        log(f"[FATAL] Missing required columns: {missing}")
        raise SystemExit(2)
    else:
        log(f"required columns OK: {REQUIRED_COLS}")

    # Decide which optional columns are present
    has = {c: (c in cols) for c in OPTIONAL_BIZ_COLS}
    log("optional columns present: " + ", ".join([f"{k}={v}" for k, v in has.items()]))

    # scanning columns (streaming)
    scan_cols = list(set(REQUIRED_COLS + [c for c in OPTIONAL_BIZ_COLS if has[c]]))
    scanner = dset.scanner(columns=scan_cols, batch_size=int(args.batch_size))

    # stats accumulators
    n_rows = 0

    # label distributions
    label_counts_all: Dict[str, int] = {}
    label_counts_hi: Dict[str, int] = {}

    # prob stats
    sum_min = float("inf")
    sum_max = float("-inf")
    sum_mean_acc = 0.0
    sum_cnt_acc = 0
    max_abs_sum_err = 0.0

    pmin = {"p_neg": float("inf"), "p_neu": float("inf"), "p_pos": float("inf")}
    pmax = {"p_neg": float("-inf"), "p_neu": float("-inf"), "p_pos": float("-inf")}
    conf_min = float("inf")
    conf_max = float("-inf")

    out_of_range_prob = 0
    out_of_range_conf = 0

    # time range (by ctime min/max)
    ctime_min = None
    ctime_max = None
    ctime_nonnull = 0

    # aspect top counts (use compute.value_counts over joined key)
    aspect_pos_all: Dict[str, int] = {}
    aspect_neg_all: Dict[str, int] = {}
    aspect_pos_hi: Dict[str, int] = {}
    aspect_neg_hi: Dict[str, int] = {}
    aspect_enabled = has.get("aspect_l1", False) and has.get("aspect_l2", False)

    conf_thr = float(args.conf_thr)
    max_keys = int(args.max_aspect_keys)
    aspect_disabled_reason = ""

    log(f"start scanning batches... batch_size={args.batch_size} conf_thr={conf_thr}")

    for bi, batch in enumerate(scanner.to_batches()):
        n = batch.num_rows
        n_rows += n

        # label counts
        vc_all = pc.value_counts(batch.column(batch.schema.get_field_index("pred_label")))
        merge_counts(label_counts_all, value_counts_to_dict(vc_all))

        # confidence
        conf_arr = batch.column(batch.schema.get_field_index("confidence"))
        conf_np = to_numpy_float(conf_arr)
        if conf_np.size:
            conf_min = min(conf_min, float(np.nanmin(conf_np)))
            conf_max = max(conf_max, float(np.nanmax(conf_np)))
            out_of_range_conf += int(np.sum((conf_np < -1e-6) | (conf_np > 1.0 + 1e-6)))

        # probs
        pneg = to_numpy_float(batch.column(batch.schema.get_field_index("p_neg")))
        pneu = to_numpy_float(batch.column(batch.schema.get_field_index("p_neu")))
        ppos = to_numpy_float(batch.column(batch.schema.get_field_index("p_pos")))

        if pneg.size:
            pmin["p_neg"] = min(pmin["p_neg"], float(np.nanmin(pneg)))
            pmax["p_neg"] = max(pmax["p_neg"], float(np.nanmax(pneg)))
            pmin["p_neu"] = min(pmin["p_neu"], float(np.nanmin(pneu)))
            pmax["p_neu"] = max(pmax["p_neu"], float(np.nanmax(pneu)))
            pmin["p_pos"] = min(pmin["p_pos"], float(np.nanmin(ppos)))
            pmax["p_pos"] = max(pmax["p_pos"], float(np.nanmax(ppos)))

            out_of_range_prob += int(np.sum((pneg < -1e-6) | (pneg > 1.0 + 1e-6)))
            out_of_range_prob += int(np.sum((pneu < -1e-6) | (pneu > 1.0 + 1e-6)))
            out_of_range_prob += int(np.sum((ppos < -1e-6) | (ppos > 1.0 + 1e-6)))

            s = pneg + pneu + ppos
            sum_min = min(sum_min, float(np.nanmin(s)))
            sum_max = max(sum_max, float(np.nanmax(s)))
            sum_mean_acc += float(np.nansum(s))
            sum_cnt_acc += int(np.sum(~np.isnan(s)))
            max_abs_sum_err = max(max_abs_sum_err, float(np.nanmax(np.abs(s - 1.0))))

        # high-confidence label counts
        mask_hi = pc.greater_equal(conf_arr, pa.scalar(conf_thr, type=pa.float32()))
        labels_hi = pc.filter(batch.column(batch.schema.get_field_index("pred_label")), mask_hi)
        vc_hi = pc.value_counts(labels_hi)
        merge_counts(label_counts_hi, value_counts_to_dict(vc_hi))

        # ctime range
        if has.get("ctime", False):
            ctime_arr = batch.column(batch.schema.get_field_index("ctime"))
            # try cast to int64 for min/max
            try:
                ctime_i64 = pc.cast(ctime_arr, pa.int64(), safe=False)
                cmin = pc.min(ctime_i64).as_py()
                cmax = pc.max(ctime_i64).as_py()
                if cmin is not None and cmax is not None:
                    ctime_nonnull += n  # approximate; if many nulls, this overcounts slightly (acceptable for range)
                    ctime_min = cmin if ctime_min is None else min(ctime_min, cmin)
                    ctime_max = cmax if ctime_max is None else max(ctime_max, cmax)
            except Exception:
                pass

        # aspect top counts (POS/NEG, all & hi)
        if aspect_enabled and not aspect_disabled_reason:
            try:
                a1 = batch.column(batch.schema.get_field_index("aspect_l1"))
                a2 = batch.column(batch.schema.get_field_index("aspect_l2"))
                # join key = "L1##L2"
                key = pc.binary_join_element_wise([pc.cast(a1, pa.string()), pc.cast(a2, pa.string())], "##")

                lab = batch.column(batch.schema.get_field_index("pred_label"))

                # POS/NEG all
                pos_mask = pc.equal(lab, pa.scalar("POS"))
                neg_mask = pc.equal(lab, pa.scalar("NEG"))

                pos_keys = pc.filter(key, pos_mask)
                neg_keys = pc.filter(key, neg_mask)

                merge_counts(aspect_pos_all, value_counts_to_dict(pc.value_counts(pos_keys)))
                merge_counts(aspect_neg_all, value_counts_to_dict(pc.value_counts(neg_keys)))

                # POS/NEG high confidence
                pos_hi_mask = pc.and_(pos_mask, mask_hi)
                neg_hi_mask = pc.and_(neg_mask, mask_hi)

                pos_hi_keys = pc.filter(key, pos_hi_mask)
                neg_hi_keys = pc.filter(key, neg_hi_mask)

                merge_counts(aspect_pos_hi, value_counts_to_dict(pc.value_counts(pos_hi_keys)))
                merge_counts(aspect_neg_hi, value_counts_to_dict(pc.value_counts(neg_hi_keys)))

                # safety cap
                if (len(aspect_pos_all) + len(aspect_neg_all)) > max_keys:
                    aspect_disabled_reason = f"unique aspect keys exceeded cap({max_keys}); stop collecting aspect stats to avoid memory risk."
                    log(f"[WARN] {aspect_disabled_reason}")

            except Exception as e:
                aspect_disabled_reason = f"aspect stats disabled due to error: {e}"
                log(f"[WARN] {aspect_disabled_reason}")

        if (bi + 1) % 20 == 0:
            log(f"scanned batches={bi+1} rows={n_rows}")

    # build report
    sum_mean = (sum_mean_acc / sum_cnt_acc) if sum_cnt_acc > 0 else float("nan")

    # infer ctime unit & date range
    time_info = {}
    if ctime_min is not None and ctime_max is not None:
        unit = "ms" if int(ctime_max) > 10**12 else "s"
        try:
            dt_min = convert_ctime_to_date(int(ctime_min), unit)
            dt_max = convert_ctime_to_date(int(ctime_max), unit)
        except Exception:
            dt_min, dt_max = None, None
        time_info = {"ctime_min": int(ctime_min), "ctime_max": int(ctime_max), "unit": unit, "dt_min": dt_min, "dt_max": dt_max}
    else:
        time_info = {"ctime_min": None, "ctime_max": None, "unit": None, "dt_min": None, "dt_max": None}

    # Print summary
    log("========================================")
    log(f"ROWS total = {n_rows}")
    log("---- label distribution (ALL) ----")
    for k, v in sorted(label_counts_all.items(), key=lambda x: x[1], reverse=True):
        log(f"{k}: {v}")

    log(f"---- label distribution (confidence>={conf_thr}) ----")
    for k, v in sorted(label_counts_hi.items(), key=lambda x: x[1], reverse=True):
        log(f"{k}: {v}")

    log("---- probability sanity ----")
    log(f"sum(p) min/max/mean = {sum_min:.6f} / {sum_max:.6f} / {sum_mean:.6f}")
    log(f"max |sum(p)-1| = {max_abs_sum_err:.6e}")
    log(f"p_neg min/max = {pmin['p_neg']:.6f} / {pmax['p_neg']:.6f}")
    log(f"p_neu min/max = {pmin['p_neu']:.6f} / {pmax['p_neu']:.6f}")
    log(f"p_pos min/max = {pmin['p_pos']:.6f} / {pmax['p_pos']:.6f}")
    log(f"confidence min/max = {conf_min:.6f} / {conf_max:.6f}")
    if out_of_range_prob:
        log(f"[WARN] prob out-of-range count ~= {out_of_range_prob}")
    if out_of_range_conf:
        log(f"[WARN] confidence out-of-range count ~= {out_of_range_conf}")

    log("---- time range (ctime) ----")
    log(json.dumps(time_info, ensure_ascii=False))

    # Top aspects
    if aspect_enabled and not aspect_disabled_reason:
        log("---- top aspects share within POS/NEG (ALL) ----")
        pos_total, pos_top = topk_from_countmap(aspect_pos_all, int(args.topk))
        neg_total, neg_top = topk_from_countmap(aspect_neg_all, int(args.topk))
        log(f"POS total={pos_total}")
        for k, c in pos_top:
            share = (c / pos_total) if pos_total > 0 else 0.0
            log(f"POS {k} cnt={c} share_in_pos={share:.6f}")
        log(f"NEG total={neg_total}")
        for k, c in neg_top:
            share = (c / neg_total) if neg_total > 0 else 0.0
            log(f"NEG {k} cnt={c} share_in_neg={share:.6f}")

        log(f"---- top aspects share within POS/NEG (confidence>={conf_thr}) ----")
        pos_total2, pos_top2 = topk_from_countmap(aspect_pos_hi, int(args.topk))
        neg_total2, neg_top2 = topk_from_countmap(aspect_neg_hi, int(args.topk))
        log(f"POS_hi total={pos_total2}")
        for k, c in pos_top2:
            share = (c / pos_total2) if pos_total2 > 0 else 0.0
            log(f"POS_hi {k} cnt={c} share_in_pos={share:.6f}")
        log(f"NEG_hi total={neg_total2}")
        for k, c in neg_top2:
            share = (c / neg_total2) if neg_total2 > 0 else 0.0
            log(f"NEG_hi {k} cnt={c} share_in_neg={share:.6f}")
    else:
        if not aspect_enabled:
            log("[WARN] aspect columns not present; skip aspect sanity.")
        if aspect_disabled_reason:
            log(f"[WARN] aspect sanity disabled: {aspect_disabled_reason}")

    # verdict (simple)
    verdict_ok = True
    reasons = []

    if n_rows <= 0:
        verdict_ok = False
        reasons.append("no rows in dataset")

    if max_abs_sum_err > 1e-2:
        verdict_ok = False
        reasons.append(f"prob sum error too large: {max_abs_sum_err}")

    if out_of_range_prob > 0:
        reasons.append(f"prob out-of-range detected: {out_of_range_prob}")

    if out_of_range_conf > 0:
        reasons.append(f"confidence out-of-range detected: {out_of_range_conf}")

    if "NEG" not in label_counts_all and "POS" not in label_counts_all:
        reasons.append("label distribution missing NEG/POS (check model output)")

    log("========================================")
    if verdict_ok:
        log("[PASS] basic checks passed.")
    else:
        log("[FAIL] checks failed: " + "; ".join(reasons))

    report = {
        "pred_ds": str(root),
        "rows": int(n_rows),
        "label_counts_all": label_counts_all,
        "label_counts_hi": label_counts_hi,
        "conf_thr": conf_thr,
        "prob_sum": {
            "min": sum_min, "max": sum_max, "mean": sum_mean, "max_abs_err": max_abs_sum_err,
        },
        "prob_range": {"pmin": pmin, "pmax": pmax},
        "confidence_range": {"min": conf_min, "max": conf_max},
        "out_of_range_prob": int(out_of_range_prob),
        "out_of_range_conf": int(out_of_range_conf),
        "time_info": time_info,
        "zero_byte_parquet_files": int(zcnt),
        "verdict_ok": bool(verdict_ok),
        "reasons": reasons,
        "generated_at": now_ts(),
    }

    if args.report_json:
        outp = Path(args.report_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"report json written: {outp}")

    return 0 if verdict_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

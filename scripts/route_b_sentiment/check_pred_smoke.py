# -*- coding: utf-8 -*-
"""
Smoke-check inference output dataset (pred_ds):
- verify 3-class outputs (p_pos/p_neu/p_neg or p0/p1/p2)
- quick label distribution
- probability min/max
- show top rows by p_neg

Usage (PowerShell):
  python -u .\scripts\route_b_sentiment\check_pred_smoke.py --pred-dir .\outputs\car\runs\qa_car_04\step04_pred_ds --max-files 3 --top-neg 10
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from collections import Counter

import pandas as pd


def find_parquets(pred_dir: Path) -> list[str]:
    pattern = str(pred_dir / "**" / "*.parquet")
    files = glob.glob(pattern, recursive=True)
    files = [f for f in files if Path(f).is_file()]
    files.sort()
    return files


def pick_label_col(cols: list[str]) -> str | None:
    if "pred_label" in cols:
        return "pred_label"
    if "label" in cols:
        return "label"
    return None


def detect_prob_cols(cols: list[str]) -> dict:
    # Preferred explicit names
    has_named = all(c in cols for c in ["p_pos", "p_neu", "p_neg"])
    if has_named:
        return {"mode": "named", "cols": ["p_pos", "p_neu", "p_neg"]}

    # Fallback index names
    has_idx3 = all(c in cols for c in ["p0", "p1", "p2"])
    if has_idx3:
        return {"mode": "idx3", "cols": ["p0", "p1", "p2"]}

    # 2-class suspicion
    has_idx2 = all(c in cols for c in ["p0", "p1"]) and "p2" not in cols
    if has_idx2:
        return {"mode": "idx2", "cols": ["p0", "p1"]}

    return {"mode": "none", "cols": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, help="pred_ds directory (contains checkpoint.json/manifest.jsonl and shard=*)")
    ap.add_argument("--max-files", type=int, default=3, help="how many parquet files to read (smoke)")
    ap.add_argument("--top-neg", type=int, default=10, help="print top rows by negative probability (if available)")
    ap.add_argument("--show-cols", action="store_true", help="print all columns of first file")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    if not pred_dir.exists():
        raise SystemExit(f"[FATAL] pred_dir not exists: {pred_dir}")

    files = find_parquets(pred_dir)
    print(f"[INFO] pred_dir={pred_dir}")
    print(f"[INFO] parquet_files={len(files)}")
    if not files:
        raise SystemExit("[FATAL] no parquet files found under pred_dir")

    files = files[: max(1, args.max_files)]
    print(f"[INFO] reading first {len(files)} file(s)")

    label_counter = Counter()
    prob_mins = {}
    prob_maxs = {}
    saw_cols = None
    prob_meta = None
    label_col = None

    # for top-neg sample
    top_neg_rows = []

    for i, fp in enumerate(files, 1):
        df = pd.read_parquet(fp)
        cols = list(df.columns)

        if saw_cols is None:
            saw_cols = cols
            label_col = pick_label_col(cols)
            prob_meta = detect_prob_cols(cols)

            print(f"[INFO] sample_file={fp}")
            print(f"[INFO] label_col={label_col}")
            print(f"[INFO] prob_mode={prob_meta['mode']} prob_cols={prob_meta['cols']}")
            if args.show_cols:
                print("[INFO] columns:", cols)

        # label distribution
        if label_col and label_col in df.columns:
            vc = df[label_col].value_counts(dropna=False).to_dict()
            for k, v in vc.items():
                label_counter[str(k)] += int(v)

        # probability ranges
        for c in prob_meta["cols"]:
            if c in df.columns:
                mn = float(df[c].min())
                mx = float(df[c].max())
                prob_mins[c] = mn if c not in prob_mins else min(prob_mins[c], mn)
                prob_maxs[c] = mx if c not in prob_maxs else max(prob_maxs[c], mx)

        # collect top-neg candidates
        if prob_meta["mode"] == "named" and "p_neg" in df.columns:
            sub = df[[label_col] + ["p_neg", "confidence", "sentence"] if label_col and "sentence" in df.columns and "confidence" in df.columns
                     else [c for c in [label_col, "p_neg", "confidence", "sentence"] if c and c in df.columns]].copy()
            sub = sub.sort_values("p_neg", ascending=False).head(args.top_neg)
            top_neg_rows.append(sub)
        elif prob_meta["mode"] == "idx3":
            # if you use p0/p1/p2, we treat p2 as "neg" candidate by convention (adjust if your mapping differs)
            if "p2" in df.columns:
                keep = [c for c in [label_col, "p2", "confidence", "sentence"] if c and c in df.columns]
                sub = df[keep].copy()
                sub = sub.sort_values("p2", ascending=False).head(args.top_neg)
                top_neg_rows.append(sub)

        print(f"[INFO] ({i}/{len(files)}) rows={len(df)}")

    print("\n===== SUMMARY =====")
    if label_counter:
        print("[LABEL] value_counts (aggregated):")
        for k, v in label_counter.most_common():
            print(f"  {k}: {v}")
    else:
        print("[LABEL] not found (no pred_label/label column)")

    if prob_meta["cols"]:
        print("[PROB] min/max:")
        for c in prob_meta["cols"]:
            if c in prob_mins:
                print(f"  {c}: min={prob_mins[c]:.6f} max={prob_maxs[c]:.6f}")
            else:
                print(f"  {c}: (missing)")
    else:
        print("[PROB] no prob columns found (expected p_pos/p_neu/p_neg or p0/p1/p2)")

    # detect head mismatch risk
    if prob_meta["mode"] == "idx2":
        print("\n[WARN] Only 2 prob columns found (p0,p1). This strongly suggests the model head is 2-class,")
        print("       while your LoRA training was 3-class (POS/NEU/NEG). Inference results may be invalid.")
    elif prob_meta["mode"] in ("none",):
        print("\n[WARN] No probability columns found. Inference output schema may be incomplete.")

    # show top-neg rows
    if top_neg_rows:
        print("\n===== TOP NEGATIVE SAMPLES =====")
        merged = pd.concat(top_neg_rows, ignore_index=True)
        if prob_meta["mode"] == "named" and "p_neg" in merged.columns:
            merged = merged.sort_values("p_neg", ascending=False).head(args.top_neg)
        elif prob_meta["mode"] == "idx3" and "p2" in merged.columns:
            merged = merged.sort_values("p2", ascending=False).head(args.top_neg)
        with pd.option_context("display.max_colwidth", 120, "display.width", 180):
            print(merged)
    else:
        print("\n[INFO] no negative-prob samples collected (no p_neg/p2 column or no sentence column).")

    print("\n[OK] smoke-check done.")


if __name__ == "__main__":
    main()

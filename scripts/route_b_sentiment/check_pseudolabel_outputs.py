# scripts/route_b_sentiment/check_pseudolabel_outputs.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

REQ_CAND = {"sentence", "aspect_l1", "aspect_l2"}
REQ_OUT  = {"pair_id", "label", "confidence", "sentence", "aspect_l1", "aspect_l2"}

def must_exist(p: Path):
    if not p.exists():
        raise SystemExit(f"[FATAL] missing file: {p}")
    if p.stat().st_size <= 0:
        raise SystemExit(f"[FATAL] empty file (0 bytes): {p}")

def load_parquet(p: Path, cols=None) -> pd.DataFrame:
    return pd.read_parquet(p, columns=cols)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--train", required=True)
    args = ap.parse_args()

    cand_p = Path(args.candidates)
    raw_p  = Path(args.raw)
    train_p= Path(args.train)

    for p in [cand_p, raw_p, train_p]:
        must_exist(p)

    print("[OK] files exist and non-empty")
    print(" candidates:", cand_p, cand_p.stat().st_size, "bytes")
    print(" raw       :", raw_p,  raw_p.stat().st_size,  "bytes")
    print(" train     :", train_p,train_p.stat().st_size,"bytes")
    print()

    # 1) schema check
    cand = load_parquet(cand_p)
    raw  = load_parquet(raw_p)
    trn  = load_parquet(train_p)

    print("[SCHEMA] candidates columns:", len(cand.columns))
    print(sorted(list(cand.columns))[:30], "..." if len(cand.columns)>30 else "")
    print("[SCHEMA] raw columns:", len(raw.columns))
    print(sorted(list(raw.columns))[:30], "..." if len(raw.columns)>30 else "")
    print("[SCHEMA] train columns:", len(trn.columns))
    print(sorted(list(trn.columns))[:30], "..." if len(trn.columns)>30 else "")
    print()

    miss = REQ_CAND - set(cand.columns)
    if miss:
        raise SystemExit(f"[FATAL] candidates missing required cols: {miss}")
    miss = REQ_OUT - set(raw.columns)
    if miss:
        raise SystemExit(f"[FATAL] pseudolabel_raw missing required cols: {miss}")
    miss = REQ_OUT - set(trn.columns)
    if miss:
        raise SystemExit(f"[FATAL] train_pseudolabel missing required cols: {miss}")

    # 2) basic stats
    def show_basic(name, df):
        print(f"[BASIC] {name}: rows={len(df):,} cols={len(df.columns)}")
        # pair_id uniqueness
        if "pair_id" in df.columns:
            u = df["pair_id"].astype(str).nunique(dropna=True)
            print(f"  pair_id unique: {u:,} ({u/len(df)*100:.1f}%)")
        # label dist
        if "label" in df.columns:
            vc = df["label"].astype(str).value_counts(dropna=False)
            print("  label dist:\n", vc.to_string())
        # confidence
        if "confidence" in df.columns:
            s = pd.to_numeric(df["confidence"], errors="coerce")
            print(f"  confidence: min={s.min():.3f} p50={s.median():.3f} p90={s.quantile(0.9):.3f} max={s.max():.3f}")
        print()

    show_basic("pseudolabel_raw", raw)
    show_basic("train_pseudolabel", trn)

    # 3) sanity rules
    # labels must be only POS/NEU/NEG
    bad_labels = set(trn["label"].astype(str).unique()) - {"POS", "NEU", "NEG"}
    if bad_labels:
        raise SystemExit(f"[FATAL] train labels contain unexpected values: {bad_labels}")

    # confidence should be within [0,1]
    conf = pd.to_numeric(trn["confidence"], errors="coerce")
    if conf.isna().any():
        print("[WARN] train confidence has NaN values")
    if (conf < 0).any() or (conf > 1).any():
        raise SystemExit("[FATAL] train confidence out of [0,1] range detected")

    # 4) alignment candidates vs train
    # if candidates have pair_id use it; otherwise align by (sentence, aspect_l1, aspect_l2)
    if "pair_id" in cand.columns:
        cand_ids = set(cand["pair_id"].astype(str))
        train_ids = set(trn["pair_id"].astype(str))
        hit = len(cand_ids & train_ids)
        print(f"[ALIGN] by pair_id: candidates={len(cand_ids):,} train={len(train_ids):,} hit={hit:,} hit_rate={hit/max(1,len(cand_ids))*100:.1f}%")
    else:
        cand_key = cand[["sentence","aspect_l1","aspect_l2"]].astype(str).agg("|".join, axis=1)
        trn_key  = trn[["sentence","aspect_l1","aspect_l2"]].astype(str).agg("|".join, axis=1)
        cand_set = set(cand_key.tolist())
        trn_set  = set(trn_key.tolist())
        hit = len(cand_set & trn_set)
        print(f"[ALIGN] by (sentence,aspect): candidates={len(cand_set):,} train={len(trn_set):,} hit={hit:,} hit_rate={hit/max(1,len(cand_set))*100:.1f}%")

    # 5) duplicates in train by pair_id (should be 1 per pair_id after top1 selection)
    dup = trn["pair_id"].astype(str).duplicated().sum()
    if dup > 0:
        raise SystemExit(f"[FATAL] train has duplicated pair_id rows: {dup}")

    print("\n[PASS] pseudolabel outputs look valid and ready for script03.")

if __name__ == "__main__":
    main()

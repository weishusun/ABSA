# -*- coding: utf-8 -*-
"""
coverage_apply_updates.py

Apply manual decisions from coverage_suggestions_<domain>.xlsx to:
- aspects/<domain>/lexicons/<L1>__<L2>.txt   (decision=ADD)
- aspects/<domain>/stoplist.txt             (decision=STOP)

Rules:
- decision column accepts: ADD / STOP / IGNORE (case-insensitive)
- For ADD:
    - prefer target_L1/target_L2 if both non-empty
    - else use best_L1/best_L2
- Deduplicate within file; keep UTF-8.
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Set
import re
import pandas as pd


def log(msg: str):
    print(msg, flush=True)


def norm_decision(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip().upper()
    s = re.sub(r"\s+", "", s)
    return s


def safe_name(s: str) -> str:
    # keep simple for Windows filenames
    s = str(s).strip()
    s = s.replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_")
    s = s.replace("?", "_").replace('"', "_").replace("<", "_").replace(">", "_").replace("|", "_")
    return s


def read_lines(p: Path) -> List[str]:
    if not p.exists():
        return []
    try:
        return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception:
        return []


def write_lines_unique(p: Path, lines: List[str]):
    p.parent.mkdir(parents=True, exist_ok=True)
    uniq = []
    seen = set()
    for ln in lines:
        t = str(ln).strip()
        if not t or t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    p.write_text("\n".join(uniq) + ("\n" if uniq else ""), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--in-xlsx", default=None)
    ap.add_argument("--sheet", default="suggestions")
    ap.add_argument("--decision-col", default="decision")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    in_xlsx = Path(args.in_xlsx) if args.in_xlsx else (repo / "outputs" / args.domain / f"coverage_suggestions_{args.domain}.xlsx")

    lexicon_dir = repo / "aspects" / args.domain / "lexicons"
    stoplist_path = repo / "aspects" / args.domain / "stoplist.txt"

    if not in_xlsx.exists():
        raise FileNotFoundError(f"Missing: {in_xlsx}")

    df = pd.read_excel(in_xlsx, sheet_name=args.sheet)
    if df.empty:
        log("empty sheet; nothing to apply.")
        return

    need_cols = {"term", args.decision_col, "best_L1", "best_L2"}
    for c in need_cols:
        if c not in df.columns:
            raise RuntimeError(f"Missing column: {c}. Found columns={list(df.columns)}")

    # optional target columns
    has_target = ("target_L1" in df.columns) and ("target_L2" in df.columns)

    add_rows = []
    stop_rows = []

    for _, r in df.iterrows():
        term = str(r.get("term", "")).strip()
        if not term:
            continue
        dec = norm_decision(r.get(args.decision_col, ""))

        if dec == "ADD":
            if has_target:
                tl1 = str(r.get("target_L1", "")).strip()
                tl2 = str(r.get("target_L2", "")).strip()
            else:
                tl1 = tl2 = ""
            bl1 = str(r.get("best_L1", "")).strip()
            bl2 = str(r.get("best_L2", "")).strip()

            l1 = tl1 if (tl1 and tl2) else bl1
            l2 = tl2 if (tl1 and tl2) else bl2
            if not l1 or not l2:
                # can't place automatically
                continue
            add_rows.append((term, l1, l2))

        elif dec == "STOP":
            stop_rows.append(term)

        else:
            continue

    log(f"[PLAN] ADD items: {len(add_rows)}")
    log(f"[PLAN] STOP items: {len(stop_rows)}")
    if args.dry_run:
        log("[DRY-RUN] no files written.")
        return

    # apply STOP
    stop_old = read_lines(stoplist_path)
    stop_new = stop_old + stop_rows
    write_lines_unique(stoplist_path, stop_new)
    log(f"[OK] stoplist updated: {stoplist_path} (added up to {len(stop_rows)})")

    # apply ADD
    # group by (l1,l2)
    grouped = {}
    for term, l1, l2 in add_rows:
        grouped.setdefault((l1, l2), []).append(term)

    for (l1, l2), terms in grouped.items():
        fn = safe_name(l1) + "__" + safe_name(l2) + ".txt"
        p = lexicon_dir / fn
        old = read_lines(p)
        new = old + terms
        write_lines_unique(p, new)
        log(f"[OK] lexicon updated: {p} (add up to {len(terms)})")

    log("[DONE] Now re-run scripts/tag_aspects.py (same baseline command) and check outputs/<domain>/aspect_coverage_<domain>.xlsx")


if __name__ == "__main__":
    main()

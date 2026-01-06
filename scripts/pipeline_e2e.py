#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
End-to-end pipeline:
00 ingest JSON/JSONL -> clean_sentences
tag -> aspect_sentences
01..05 + web via route_b_sentiment/pipeline.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--run-id", default="", help="run id; default {YYYYMMDD}_{domain}_e2e")
    ap.add_argument("--steps", default="00,tag,01,02,03,04,05,web", help="comma steps: 00,tag,01..05,web")
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def run_cmd(cmd: List[str]) -> None:
    print("[CMD]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def build_run_id(domain: str, run_id: str) -> str:
    if run_id:
        return run_id
    return f"{datetime.now().strftime('%Y%m%d')}_{domain}_e2e"


def write_meta(run_root: Path, meta: Dict) -> None:
    meta_dir = run_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    fp = meta_dir / "run.json"
    tmp = fp.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(fp)


def main() -> int:
    args = parse_args()
    domain = args.domain.lower()
    run_id = build_run_id(domain, args.run_id)
    run_root = Path("outputs") / domain / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    steps = [s.strip().lower() for s in args.steps.split(",") if s.strip()]
    meta = {
        "domain": domain,
        "run_id": run_id,
        "steps": [],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {},
        "outputs": {},
        "run_root": str(run_root),
    }

    clean_sentences = Path("outputs") / domain / "clean_sentences.parquet"
    aspect_sentences = Path("outputs") / domain / "aspect_sentences.parquet"

    # Step 00
    if "00" in steps:
        meta["steps"].append("00")
        cmd = [
            sys.executable,
            "-u",
            str(Path("scripts") / "step00_ingest_json_to_clean_sentences.py"),
            "--domain",
            domain,
            "--output",
            str(clean_sentences),
        ]
        if args.resume:
            cmd.append("--resume")
        run_cmd(cmd)
        meta["inputs"]["raw_data_root"] = str(Path("data") / domain)
        meta["outputs"]["clean_sentences"] = str(clean_sentences)
    else:
        if not clean_sentences.exists():
            print(f"[FATAL] clean_sentences missing: {clean_sentences}; include step 00 or provide file.", file=sys.stderr)
            return 2
        meta["outputs"]["clean_sentences"] = str(clean_sentences)

    # tag aspects
    if "tag" in steps:
        meta["steps"].append("tag")
        cmd = [
            sys.executable,
            "-u",
            str(Path("scripts") / "tag_aspects.py"),
            "--input",
            str(clean_sentences),
            "--config",
            str(Path("configs") / "domains" / domain / "aspects.yaml"),
            "--output-dir",
            str(Path("outputs") / domain),
        ]
        run_cmd(cmd)
        meta["outputs"]["aspect_sentences"] = str(aspect_sentences)
    else:
        if not aspect_sentences.exists():
            print(f"[FATAL] aspect_sentences missing: {aspect_sentences}; include 'tag' step.", file=sys.stderr)
            return 2
        meta["outputs"]["aspect_sentences"] = str(aspect_sentences)

    # downstream steps via existing pipeline
    rest_steps = [s for s in steps if s not in ("00", "tag")]
    if rest_steps:
        cmd = [
            sys.executable,
            "-u",
            str(Path("scripts") / "route_b_sentiment" / "pipeline.py"),
            "--domain",
            domain,
            "--run-id",
            run_id,
            "--input-aspect-sentences",
            str(aspect_sentences),
            "--steps",
            ",".join(rest_steps),
        ]
        if args.resume:
            cmd.append("--resume")
        run_cmd(cmd)

    write_meta(run_root, meta)
    print(f"[DONE] e2e steps={meta['steps']} run_root={run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
End-to-end pipeline:
00 ingest JSON/JSONL -> clean_sentences
tag -> aspect_sentences
01..05 + web via route_b_sentiment/pipeline.py

Workspace convention:
- If --workspace is provided, or env var ABSA_WORKSPACE is set:
  outputs go to <workspace>/outputs/...
- Otherwise outputs go to <repo_root>/outputs/...
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent  # repo root (scripts/..)


def resolve_workspace(workspace_arg: str) -> Path:
    """
    Returns the workspace root directory.
    - If workspace_arg is provided: use it
    - Else if ABSA_WORKSPACE env var is set: use it
    - Else: fallback to repo root
    """
    if workspace_arg:
        return Path(workspace_arg).expanduser().resolve()
    env_ws = os.environ.get("ABSA_WORKSPACE", "").strip()
    if env_ws:
        return Path(env_ws).expanduser().resolve()
    return ROOT


def outputs_root_of(workspace_root: Path) -> Path:
    """
    Returns the outputs root directory.
    - workspace == repo root => <repo_root>/outputs
    - otherwise => <workspace_root>/outputs
    """
    if workspace_root.resolve() == ROOT.resolve():
        return ROOT / "outputs"
    return workspace_root / "outputs"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> None:
    cwd = cwd or ROOT
    print("[CMD]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd))


def build_run_id(domain: str, run_id: str) -> str:
    if run_id:
        return run_id
    return f"{datetime.now().strftime('%Y%m%d')}_{domain}_e2e"


def write_meta(run_root: Path, meta: Dict) -> None:
    meta_dir = run_root / "meta"
    ensure_dir(meta_dir)
    fp = meta_dir / "run.json"
    tmp = fp.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(fp)


def normalize_step_token(tok: str) -> str:
    """
    Normalize step tokens:
    - "0" -> "00"
    - "00" -> "00"
    - "1" -> "01"
    - "01" -> "01"
    - ...
    - "tag" -> "tag"
    - "web" -> "web"
    """
    t = tok.strip().lower()
    if not t:
        return ""
    if t in ("tag", "web"):
        return t
    if t in ("ingest", "step00", "step0"):
        return "00"
    if t.isdigit():
        n = int(t)
        if n == 0:
            return "00"
        # keep 2-digit for 1..99
        return f"{n:02d}"
    # allow already-formatted "00","01"... or unknown tokens pass-through
    return t


def normalize_steps(raw_steps: str) -> List[str]:
    out: List[str] = []
    for part in (raw_steps or "").split(","):
        s = normalize_step_token(part)
        if not s:
            continue
        if s not in out:
            out.append(s)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--run-id", default="", help="run id; default {YYYYMMDD}_{domain}_e2e")
    ap.add_argument("--steps", default="00,tag,01,02,03,04,05,web", help="comma steps: 00,tag,01..05,web")
    ap.add_argument("--resume", action="store_true")

    # Workspace (recommended)
    ap.add_argument(
        "--workspace",
        default="",
        help="workspace root; default ABSA_WORKSPACE env var; fallback repo root",
    )

    # Step00 debug caps (passed through)
    ap.add_argument("--max-files", type=int, default=0, help="(Step00) limit number of files for debug")
    ap.add_argument("--max-docs", type=int, default=0, help="(Step00) limit total docs for debug")
    ap.add_argument("--chunk-size", type=int, default=0, help="(Step00) rows per parquet write chunk")

    # tag_aspects tuning (passed through)
    ap.add_argument("--tag-batch-size", type=int, default=0, help="(tag) batch size for tag_aspects.py")
    ap.add_argument("--uncovered-sample", type=int, default=0, help="(tag) uncovered sample size")
    ap.add_argument("--uncovered-topk", type=int, default=0, help="(tag) uncovered topk terms")
    ap.add_argument("--example-k", type=int, default=0, help="(tag) examples per uncovered term")

    return ap.parse_args()


def main() -> int:
    args = parse_args()
    domain = args.domain.strip().lower()
    run_id = build_run_id(domain, args.run_id)

    workspace_root = resolve_workspace(args.workspace)
    out_root = outputs_root_of(workspace_root)

    # Standard output layout
    domain_out = out_root / domain
    ensure_dir(domain_out)
    run_root = domain_out / "runs" / run_id
    ensure_dir(run_root)

    steps = normalize_steps(args.steps)

    meta: Dict = {
        "domain": domain,
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(ROOT),
        "workspace_root": str(workspace_root),
        "outputs_root": str(out_root),
        "steps_executed": [],
        "inputs": {},
        "outputs": {},
        "run_root": str(run_root),
    }

    clean_sentences = domain_out / "clean_sentences.parquet"
    aspect_sentences = domain_out / "aspect_sentences.parquet"

    # Step 00
    if "00" in steps:
        meta["steps_executed"].append("00")
        data_root = (ROOT / "data" / domain)
        cmd = [
            sys.executable,
            "-u",
            str(ROOT / "scripts" / "step00_ingest_json_to_clean_sentences.py"),
            "--domain",
            domain,
            "--data-root",
            str(data_root),
            "--output",
            str(clean_sentences),
        ]
        if args.resume:
            cmd.append("--resume")
        if args.max_files and args.max_files > 0:
            cmd += ["--max-files", str(args.max_files)]
        if args.max_docs and args.max_docs > 0:
            cmd += ["--max-docs", str(args.max_docs)]
        if args.chunk_size and args.chunk_size > 0:
            cmd += ["--chunk-size", str(args.chunk_size)]
        run_cmd(cmd, cwd=ROOT)
        meta["inputs"]["raw_data_root"] = str(data_root)
        meta["outputs"]["clean_sentences"] = str(clean_sentences)
    else:
        if not clean_sentences.exists():
            print(f"[FATAL] clean_sentences missing: {clean_sentences}; include step 00 or provide file.", file=sys.stderr)
            return 2
        meta["outputs"]["clean_sentences"] = str(clean_sentences)

    # tag aspects
    if "tag" in steps:
        meta["steps_executed"].append("tag")
        cfg = ROOT / "configs" / "domains" / domain / "aspects.yaml"
        cmd = [
            sys.executable,
            "-u",
            str(ROOT / "scripts" / "tag_aspects.py"),
            "--input",
            str(clean_sentences),
            "--config",
            str(cfg),
            "--output-dir",
            str(domain_out),
        ]
        if args.tag_batch_size and args.tag_batch_size > 0:
            cmd += ["--batch-size", str(args.tag_batch_size)]
        if args.uncovered_sample and args.uncovered_sample > 0:
            cmd += ["--uncovered-sample", str(args.uncovered_sample)]
        if args.uncovered_topk and args.uncovered_topk > 0:
            cmd += ["--uncovered-topk", str(args.uncovered_topk)]
        if args.example_k and args.example_k > 0:
            cmd += ["--example-k", str(args.example_k)]
        run_cmd(cmd, cwd=ROOT)
        meta["outputs"]["aspect_sentences"] = str(aspect_sentences)
    else:
        if not aspect_sentences.exists():
            print(f"[FATAL] aspect_sentences missing: {aspect_sentences}; include 'tag' step or ensure file exists.", file=sys.stderr)
            return 2
        meta["outputs"]["aspect_sentences"] = str(aspect_sentences)

    # downstream RouteB steps (01..05,web)
    rest_steps = [s for s in steps if s not in ("00", "tag")]
    if rest_steps:
        meta["steps_executed"].extend(rest_steps)
        cmd = [
            sys.executable,
            "-u",
            str(ROOT / "scripts" / "route_b_sentiment" / "pipeline.py"),
            "--domain",
            domain,
            "--run-id",
            run_id,
            "--input-aspect-sentences",
            str(aspect_sentences),
            "--steps",
            ",".join(rest_steps),
            "--workspace",
            str(workspace_root),
        ]
        if args.resume:
            cmd.append("--resume")
        run_cmd(cmd, cwd=ROOT)

    write_meta(run_root, meta)
    print(f"[DONE] steps_executed={meta['steps_executed']} run_root={run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

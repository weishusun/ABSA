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

# repo root (scripts/..)
ROOT = Path(__file__).resolve().parent.parent


def resolve_workspace(workspace_arg: str) -> Path:
    if workspace_arg:
        return Path(workspace_arg).expanduser().resolve()
    env_ws = os.environ.get("ABSA_WORKSPACE", "").strip()
    if env_ws:
        return Path(env_ws).expanduser().resolve()
    return ROOT


def outputs_root_of(workspace_root: Path) -> Path:
    if workspace_root.resolve() == ROOT.resolve():
        return ROOT / "outputs"
    return workspace_root / "outputs"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _has_json_or_jsonl(root: Path) -> bool:
    if not root.exists():
        return False
    for pat in ("*.json", "*.jsonl"):
        if any(root.rglob(pat)):
            return True
    return False


def resolve_step00_data_root(domain: str, workspace_root: Path, repo_root: Path) -> Path:
    ws_candidate = workspace_root / "data" / domain
    repo_candidate = repo_root / "data" / domain

    if _has_json_or_jsonl(ws_candidate):
        return ws_candidate
    if _has_json_or_jsonl(repo_candidate):
        return repo_candidate

    raise SystemExit(
        "[FATAL] Step00 找不到输入数据。请确认数据存放在：\n"
        f"  - {ws_candidate}\n"
        f"  - {repo_candidate}"
    )

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
        return f"{n:02d}"
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
    ap.add_argument("--run-id", default="", help="run id")
    ap.add_argument("--steps", default="00,tag,01,02,03,04,05,web")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workspace", default="")
    # 下传参数
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=0)
    ap.add_argument("--tag-batch-size", type=int, default=0)
    ap.add_argument("--uncovered-sample", type=int, default=0)
    ap.add_argument("--uncovered-topk", type=int, default=0)
    ap.add_argument("--example-k", type=int, default=0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    domain = args.domain.strip().lower()
    run_id = build_run_id(domain, args.run_id)

    workspace_root = resolve_workspace(args.workspace)
    out_root = outputs_root_of(workspace_root)

    domain_out = out_root / domain
    ensure_dir(domain_out)
    run_root = domain_out / "runs" / run_id
    ensure_dir(run_root)

    steps = normalize_steps(args.steps)
    # 预定义配置文件路径
    cfg = ROOT / "configs" / "domains" / domain / "aspects.yaml"

    meta: Dict = {
        "domain": domain,
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_root": str(run_root),
        "steps_executed": [],
        "inputs": {},
        "outputs": {},
    }

    clean_sentences = domain_out / "clean_sentences.parquet"
    aspect_sentences = domain_out / "aspect_sentences.parquet"

    # Step 00: Ingest
    if "00" in steps:
        meta["steps_executed"].append("00")
        data_root = resolve_step00_data_root(domain, workspace_root, ROOT)
        cmd = [
            sys.executable, "-u",
            str(ROOT / "scripts" / "step00_ingest_json_to_clean_sentences.py"),
            "--domain", domain,
            "--data-root", str(data_root),
            "--output", str(clean_sentences),
        ]
        if args.resume: cmd.append("--resume")
        if args.max_files > 0: cmd += ["--max-files", str(args.max_files)]
        run_cmd(cmd, cwd=ROOT)
        meta["outputs"]["clean_sentences"] = str(clean_sentences)

    # Tagging
    if "tag" in steps:
        meta["steps_executed"].append("tag")
        cmd = [
            sys.executable, "-u",
            str(ROOT / "scripts" / "tag_aspects.py"),
            "--input", str(clean_sentences),
            "--config", str(cfg),
            "--output-dir", str(domain_out),
        ]
        if args.tag_batch_size > 0: cmd += ["--batch-size", str(args.tag_batch_size)]
        run_cmd(cmd, cwd=ROOT)
        meta["outputs"]["aspect_sentences"] = str(aspect_sentences)

    # Downstream Steps (01..05, web)
    rest_steps = [s for s in steps if s not in ("00", "tag")]
    if rest_steps:
        meta["steps_executed"].extend(rest_steps)
        # 修复点：添加 --aspects-yaml 透传
        cmd = [
            sys.executable, "-u",
            str(ROOT / "scripts" / "route_b_sentiment" / "pipeline.py"),
            "--domain", domain,
            "--run-id", run_id,
            "--input-aspect-sentences", str(aspect_sentences),
            "--aspects-yaml", str(cfg),  # <--- 重要修复
            "--steps", ",".join(rest_steps),
            "--workspace", str(workspace_root),
        ]
        if args.resume: cmd.append("--resume")
        run_cmd(cmd, cwd=ROOT)

    write_meta(run_root, meta)
    print(f"[DONE] run_root={run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
End-to-end pipeline (workspace-aware):

00 ingest JSON/JSONL -> clean_sentences
tag -> aspect_sentences
01..05 (RouteB) -> runs/<run_id>/...

Workspace strategy:
- Prefer --workspace
- Else use env ABSA_WORKSPACE
- Else fallback to repo root

Data strategy:
- Prefer <workspace>/data/<domain> if it exists and has json/jsonl
- Else fallback to <repo>/data/<domain>

Outputs strategy:
- Always write Step00/tag outputs to <workspace>/outputs/<domain>/...
- RouteB runs currently still depend on route_b pipeline's own output logic
  (we will make route_b pipeline workspace-aware in the next module).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--run-id", default="", help="run id; default {YYYYMMDD}_{domain}_e2e")
    ap.add_argument("--steps", default="00,tag,01,02,03,04,05", help="comma steps: 00,tag,01..05,web")
    ap.add_argument("--resume", action="store_true")

    # Workspace-aware options
    ap.add_argument(
        "--workspace",
        default=None,
        help="workspace root; default ABSA_WORKSPACE env var; fallback repo root",
    )

    # Step00 passthrough (safe for smoke/debug)
    ap.add_argument("--max-files", type=int, default=None, help="(Step00) limit number of files for debug")
    ap.add_argument("--max-docs", type=int, default=None, help="(Step00) limit total docs for debug")
    ap.add_argument("--chunk-size", type=int, default=None, help="(Step00) rows per parquet write chunk")

    # tag_aspects passthrough (for coverage loop / future UI)
    ap.add_argument("--tag-batch-size", type=int, default=None, help="(tag) batch size for tag_aspects.py")
    ap.add_argument("--uncovered-sample", type=int, default=None, help="(tag) uncovered sample size")
    ap.add_argument("--uncovered-topk", type=int, default=None, help="(tag) uncovered topk terms")
    ap.add_argument("--example-k", type=int, default=None, help="(tag) examples per uncovered term")

    return ap.parse_args()


def repo_root() -> Path:
    return REPO_ROOT


def resolve_workspace(cli_workspace: Optional[str]) -> Path:
    if cli_workspace:
        return Path(cli_workspace).expanduser().resolve()
    env_ws = os.getenv("ABSA_WORKSPACE")
    if env_ws:
        return Path(env_ws).expanduser().resolve()
    return repo_root()


def _has_json_files(p: Path) -> bool:
    if not p.exists() or not p.is_dir():
        return False
    for pat in ("*.json", "*.jsonl"):
        try:
            for _ in p.rglob(pat):
                return True
        except Exception:
            # permissions or other issues: treat as present to avoid false negative
            return True
    return False


def choose_data_root(ws: Path, domain: str) -> Path:
    ws_data = ws / "data" / domain
    if _has_json_files(ws_data):
        return ws_data
    return repo_root() / "data" / domain


def outputs_root(ws: Path, domain: str) -> Path:
    out = ws / "outputs" / domain
    out.mkdir(parents=True, exist_ok=True)
    return out


def build_run_id(domain: str, run_id: str) -> str:
    if run_id:
        return run_id
    return f"{datetime.now().strftime('%Y%m%d')}_{domain}_e2e"


def normalize_step_token(tok: str) -> Optional[str]:
    t = (tok or "").strip().lower()
    if not t:
        return None
    if t in {"tag", "web"}:
        return t
    # accept 0/00 as step00
    if t.isdigit():
        try:
            n = int(t)
        except Exception:
            return t
        if n == 0:
            return "00"
        # normalize 1..99 to 2-digit, but our pipeline only uses 01..05
        return f"{n:02d}"
    # accept 00/01.. style
    if len(t) == 2 and t[0] == "0" and t[1].isdigit():
        return t
    return t


def normalize_steps(steps_str: str) -> List[str]:
    raw = [s for s in (steps_str or "").split(",")]
    out: List[str] = []
    seen = set()
    for s in raw:
        ns = normalize_step_token(s)
        if ns is None:
            continue
        if ns not in seen:
            out.append(ns)
            seen.add(ns)
    return out


def run_cmd(cmd: List[str], *, cwd: Path, env: Optional[Dict[str, str]] = None) -> None:
    print("[CMD]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd), env=env)


def write_meta(run_root: Path, meta: Dict) -> None:
    meta_dir = run_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    fp = meta_dir / "run.json"
    tmp = fp.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(fp)


def split_steps(steps: List[str]) -> Tuple[bool, bool, bool, List[str]]:
    want_00 = "00" in steps
    want_tag = "tag" in steps
    want_web = "web" in steps

    # RouteB steps are only 01..05 (normalized)
    route_steps: List[str] = [s for s in steps if s in {"01", "02", "03", "04", "05"}]
    # if user wants web exports, ensure 05 is included (web exports usually depend on aggregation)
    if want_web and "05" not in route_steps:
        route_steps.append("05")
    return want_00, want_tag, want_web, route_steps


def main() -> int:
    args = parse_args()
    domain = args.domain.lower().strip()
    run_id = build_run_id(domain, args.run_id)

    ws = resolve_workspace(args.workspace)
    out_root = outputs_root(ws, domain)
    run_root = out_root / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    steps = normalize_steps(args.steps)
    want_00, want_tag, want_web, route_steps = split_steps(steps)

    data_root = choose_data_root(ws, domain)
    clean_sentences = out_root / "clean_sentences.parquet"
    aspect_sentences = out_root / "aspect_sentences.parquet"
    aspects_yaml = repo_root() / "configs" / "domains" / domain / "aspects.yaml"

    meta: Dict = {
        "domain": domain,
        "run_id": run_id,
        "steps_requested": steps,
        "steps_executed": [],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "workspace": str(ws),
        "repo_root": str(repo_root()),
        "inputs": {},
        "outputs": {},
        "run_root": str(run_root),
    }

    # Step00: ingest
    if want_00:
        if not _has_json_files(data_root):
            print(
                f"[FATAL] No JSON/JSONL files found for domain='{domain}'.\n"
                f"Checked:\n  - workspace data: {ws / 'data' / domain}\n"
                f"  - repo data:      {repo_root() / 'data' / domain}\n"
                f"Put raw files under one of these folders, then rerun.",
                file=sys.stderr,
            )
            return 2

        cmd = [
            sys.executable,
            "-u",
            str(repo_root() / "scripts" / "step00_ingest_json_to_clean_sentences.py"),
            "--domain",
            domain,
            "--data-root",
            str(data_root),
            "--output",
            str(clean_sentences),
        ]
        if args.resume:
            cmd.append("--resume")
        if args.max_files is not None:
            cmd += ["--max-files", str(args.max_files)]
        if args.max_docs is not None:
            cmd += ["--max-docs", str(args.max_docs)]
        if args.chunk_size is not None:
            cmd += ["--chunk-size", str(args.chunk_size)]

        run_cmd(cmd, cwd=repo_root())
        meta["steps_executed"].append("00")
        meta["inputs"]["raw_data_root"] = str(data_root)
        meta["outputs"]["clean_sentences"] = str(clean_sentences)
    else:
        if not clean_sentences.exists():
            print(
                f"[FATAL] clean_sentences missing: {clean_sentences}\n"
                f"Either include step '00' or ensure the file exists at that location.",
                file=sys.stderr,
            )
            return 2
        meta["outputs"]["clean_sentences"] = str(clean_sentences)

    # tag aspects
    if want_tag:
        if not aspects_yaml.exists():
            print(f"[FATAL] aspects.yaml not found for domain='{domain}': {aspects_yaml}", file=sys.stderr)
            return 2

        cmd = [
            sys.executable,
            "-u",
            str(repo_root() / "scripts" / "tag_aspects.py"),
            "--input",
            str(clean_sentences),
            "--config",
            str(aspects_yaml),
            "--output-dir",
            str(out_root),
        ]
        if args.tag_batch_size is not None:
            cmd += ["--batch-size", str(args.tag_batch_size)]
        if args.uncovered_sample is not None:
            cmd += ["--uncovered-sample", str(args.uncovered_sample)]
        if args.uncovered_topk is not None:
            cmd += ["--uncovered-topk", str(args.uncovered_topk)]
        if args.example_k is not None:
            cmd += ["--example-k", str(args.example_k)]

        run_cmd(cmd, cwd=repo_root())
        meta["steps_executed"].append("tag")
        meta["inputs"]["aspects_yaml"] = str(aspects_yaml)
        meta["outputs"]["aspect_sentences"] = str(aspect_sentences)
    else:
        # only require aspect_sentences if later route steps exist
        if route_steps and (not aspect_sentences.exists()):
            print(
                f"[FATAL] aspect_sentences missing: {aspect_sentences}\n"
                f"Either include step 'tag' or ensure the file exists at that location.",
                file=sys.stderr,
            )
            return 2
        if aspect_sentences.exists():
            meta["outputs"]["aspect_sentences"] = str(aspect_sentences)

    # RouteB pipeline only if user requested 01..05
    if route_steps:
        if not aspect_sentences.exists():
            print(
                f"[FATAL] aspect_sentences missing: {aspect_sentences}\n"
                f"RouteB steps require aspect_sentences; include step 'tag' first.",
                file=sys.stderr,
            )
            return 2

        cmd = [
            sys.executable,
            "-u",
            str(repo_root() / "scripts" / "route_b_sentiment" / "pipeline.py"),
            "--domain",
            domain,
            "--run-id",
            run_id,
            "--input-aspect-sentences",
            str(aspect_sentences),
            "--steps",
            ",".join(route_steps),
            "--aspects-yaml",
            str(aspects_yaml),
        ]
        if args.resume:
            cmd.append("--resume")

        # keep cwd=repo_root for imports stability
        run_cmd(cmd, cwd=repo_root())
        meta["steps_executed"].extend(route_steps)

        if want_web:
            # Placeholder: route_b pipeline help doesn't expose "web" step.
            # For now, we treat web as "ensure 05 ran".
            meta["steps_executed"].append("web")

    write_meta(run_root, meta)
    print(f"[DONE] steps_executed={meta['steps_executed']} run_root={run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Route B domain pipeline wrapper.

Runs:
- sentiment_01_build_aspect_pairs_and_train_candidates.py
- sentiment_02_pseudolabel_openai.py
- sentiment_03_train_asc_lora.py
- sentiment_04_infer_asc.py
- sentiment_05_aggregate_and_build_excels.py
- export_web_tables_l1_11.py

Output layout (per docs):
outputs/<domain>/runs/<run_id>/
  step01_pairs/
  step02_pseudo/
  step03_model/
  step04_pred/
  step05_agg/
  web_exports/
  meta/run.json

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
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent

from scripts.route_b_sentiment._shared.config_resolver import resolve as resolve_configs  # noqa: E402


STEP_DIRS = {
    "01": "step01_pairs",
    "02": "step02_pseudo",
    "03": "step03_model",
    "04": "step04_pred",
    "05": "step05_agg",
    "web": "web_exports",
}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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


def run_root(domain: str, run_id: str, outputs_root: Path) -> Path:
    return outputs_root / domain / "runs" / run_id


def step_dir(rr: Path, step: str) -> Path:
    if step not in STEP_DIRS:
        raise ValueError(f"Unknown step for step_dir: {step}")
    return rr / STEP_DIRS[step]


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> None:
    cwd = cwd or ROOT
    print("[CMD]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd))


def write_meta(run_dir: Path, meta: Dict) -> None:
    meta_dir = run_dir / "meta"
    ensure_dir(meta_dir)
    out_fp = meta_dir / "run.json"
    tmp = out_fp.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out_fp)


def build_run_id(domain: str, run_id: str) -> str:
    if run_id:
        return run_id
    ts = datetime.now().strftime("%Y%m%d")
    return f"{ts}_{domain}_v0"


def normalize_step_token(tok: str) -> str:
    t = tok.strip().lower()
    if not t:
        return ""
    if t == "web":
        return "web"
    if t.isdigit():
        n = int(t)
        if n == 0:
            return "00"  # RouteB normally doesn't use 00, but keep safe
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
    ap.add_argument("--domain", required=True, help="phone|laptop|car|beauty")
    ap.add_argument("--run-id", default="", help="run identifier; default auto {YYYYMMDD}_{domain}_v0")
    ap.add_argument("--input-aspect-sentences", required=True, help="Path to aspect_sentences.parquet (tag output)")
    ap.add_argument("--aspects-yaml", default="", help="Override aspects config path; default resolved by domain")
    ap.add_argument("--steps", default="01,02,03,04,05,web", help="Comma-separated steps to run, e.g. 01,02 or 01,02,04,05")
    ap.add_argument("--resume", action="store_true", help="Pass --resume to step scripts when applicable (01/03/04)")

    ap.add_argument(
        "--workspace",
        default="",
        help="workspace root; default ABSA_WORKSPACE env var; fallback repo root",
    )

    ap.add_argument("--max-train-rows", type=int, default=5000, help="Step01 max train rows")
    ap.add_argument("--train-pool-rows", type=int, default=200000, help="Step01 pool rows")
    ap.add_argument("--shard-n", type=int, default=64, help="Step01 shard count for aspect_pairs_ds")
    ap.add_argument("--ds-batch-rows", type=int, default=50000, help="Step01 ds batch rows")
    ap.add_argument("--step02-max-rows", type=int, default=0, help="Step02 max rows (0 = no cap)")
    ap.add_argument("--base-model", default="hfl/chinese-macbert-base", help="Base model for step03/04")
    ap.add_argument("--fp16", action="store_true", help="Use --fp16 for step04")
    return ap.parse_args()


def step01(args: argparse.Namespace, rr: Path) -> Dict:
    out_dir = step_dir(rr, "01")
    ensure_dir(out_dir)
    cmd = [
        sys.executable,
        "-u",
        str(SCRIPT_DIR / "sentiment_01_build_aspect_pairs_and_train_candidates.py"),
        "--input",
        str(Path(args.input_aspect_sentences)),
        "--output-dir",
        str(out_dir),
        "--write-ds",
        "--shard-n",
        str(args.shard_n),
        "--ds-batch-rows",
        str(args.ds_batch_rows),
        "--max-train-rows",
        str(args.max_train_rows),
        "--train-pool-rows",
        str(args.train_pool_rows),
        "--overwrite",
    ]
    if args.resume:
        cmd.append("--resume")
    run_cmd(cmd)
    return {
        "train_candidates": str(out_dir / "train_candidates.parquet"),
        "aspect_pairs_ds": str(out_dir / "aspect_pairs_ds"),
        "step01_dir": str(out_dir),
    }


def step02(args: argparse.Namespace, rr: Path, step01_out: Dict) -> Dict:
    out_dir = step_dir(rr, "02")
    ensure_dir(out_dir)
    cmd = [
        sys.executable,
        "-u",
        str(SCRIPT_DIR / "sentiment_02_pseudolabel_openai.py"),
        "--input",
        step01_out["train_candidates"],
        "--output-dir",
        str(out_dir),
        "--batch-items",
        "20",
        "--confidence-thr",
        "0.85",
    ]
    if args.step02_max_rows and args.step02_max_rows > 0:
        cmd += ["--max-rows", str(args.step02_max_rows)]
    run_cmd(cmd)
    return {
        "train_pseudolabel": str(out_dir / "train_pseudolabel.parquet"),
        "step02_dir": str(out_dir),
    }


def step03(args: argparse.Namespace, rr: Path, step02_out: Dict) -> Dict:
    out_dir = step_dir(rr, "03")
    ensure_dir(out_dir)
    cmd = [
        sys.executable,
        "-u",
        str(SCRIPT_DIR / "sentiment_03_train_asc_lora.py"),
        "--train",
        step02_out["train_pseudolabel"],
        "--out-dir",
        str(out_dir),
        "--base-model",
        args.base_model,
    ]
    if args.resume:
        cmd.append("--resume")
    run_cmd(cmd)
    return {"model_dir": str(out_dir), "step03_dir": str(out_dir)}


def step04(args: argparse.Namespace, rr: Path, step01_out: Dict, step03_out: Dict) -> Dict:
    out_dir = step_dir(rr, "04")
    ensure_dir(out_dir)
    cmd = [
        sys.executable,
        "-u",
        str(SCRIPT_DIR / "sentiment_04_infer_asc.py"),
        "--input",
        step01_out["aspect_pairs_ds"],
        "--output-dir",
        str(out_dir),
        "--model-dir",
        step03_out["model_dir"],
        "--base-model",
        args.base_model,
        "--overwrite",
    ]
    if args.resume:
        cmd.append("--resume")
    if args.fp16:
        cmd.append("--fp16")
    run_cmd(cmd)
    return {"pred_ds": str(out_dir), "step04_dir": str(out_dir)}


def step05(args: argparse.Namespace, rr: Path, step04_out: Dict) -> Dict:
    out_dir = step_dir(rr, "05")
    ensure_dir(out_dir)
    cmd = [
        sys.executable,
        "-u",
        str(SCRIPT_DIR / "sentiment_05_aggregate_and_build_excels.py"),
        "--pred-ds",
        step04_out["pred_ds"],
        "--out-dir",
        str(out_dir),
    ]
    run_cmd(cmd)
    return {"agg_dir": str(out_dir), "step05_dir": str(out_dir)}


def step_web(args: argparse.Namespace, rr: Path, step04_out: Dict) -> Dict:
    out_dir = step_dir(rr, "web")
    ensure_dir(out_dir)
    cmd = [
        sys.executable,
        "-u",
        str(SCRIPT_DIR / "export_web_tables_l1_11.py"),
        "--pred-ds",
        step04_out["pred_ds"],
        "--out-dir",
        str(out_dir),
    ]
    run_cmd(cmd)
    return {"web_dir": str(out_dir)}


def main() -> int:
    args = parse_args()
    args.domain = args.domain.lower().strip()
    args.run_id = build_run_id(args.domain, args.run_id)

    workspace_root = resolve_workspace(args.workspace)
    out_root = outputs_root_of(workspace_root)

    rr = run_root(args.domain, args.run_id, out_root)
    ensure_dir(rr)

    # Resolve configs (kept for traceability in meta)
    aspects_yaml = Path(args.aspects_yaml) if args.aspects_yaml else None
    domain_yaml: Optional[Path] = None
    if aspects_yaml is None or not aspects_yaml.exists():
        aspects_yaml, domain_yaml = resolve_configs(args.domain)
    else:
        _, domain_yaml = resolve_configs(args.domain)

    # Basic input guard
    in_fp = Path(args.input_aspect_sentences)
    if not in_fp.exists():
        print(f"[FATAL] input-aspect-sentences not found: {in_fp}", file=sys.stderr)
        return 2

    meta: Dict = {
        "domain": args.domain,
        "run_id": args.run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(ROOT),
        "workspace_root": str(workspace_root),
        "outputs_root": str(out_root),
        "run_root": str(rr),
        "input_aspect_sentences": str(in_fp),
        "aspects_yaml": str(aspects_yaml) if aspects_yaml else "",
        "domain_yaml": str(domain_yaml) if domain_yaml else "",
        "steps_executed": [],
        "outputs": {},
    }

    steps = normalize_steps(args.steps)

    step_outputs: Dict[str, Dict] = {}

    for s in steps:
        if s in ("00",):  # RouteB should not run 00; ignore safely
            print("[WARN] step '00' is not a RouteB step; ignored.", flush=True)
            continue

        if s == "01":
            meta["steps_executed"].append("01")
            step_outputs["01"] = step01(args, rr)

        elif s == "02":
            meta["steps_executed"].append("02")
            deps1 = step_outputs.get("01") or step01(args, rr)
            step_outputs["01"] = deps1
            step_outputs["02"] = step02(args, rr, deps1)

        elif s == "03":
            meta["steps_executed"].append("03")
            deps1 = step_outputs.get("01") or step01(args, rr)
            step_outputs["01"] = deps1
            deps2 = step_outputs.get("02") or step02(args, rr, deps1)
            step_outputs["02"] = deps2
            step_outputs["03"] = step03(args, rr, deps2)

        elif s == "04":
            meta["steps_executed"].append("04")
            deps1 = step_outputs.get("01") or step01(args, rr)
            step_outputs["01"] = deps1
            deps3 = step_outputs.get("03")
            if deps3 is None:
                deps2 = step_outputs.get("02") or step02(args, rr, deps1)
                step_outputs["02"] = deps2
                deps3 = step03(args, rr, deps2)
                step_outputs["03"] = deps3
            step_outputs["04"] = step04(args, rr, deps1, deps3)

        elif s == "05":
            meta["steps_executed"].append("05")
            deps4 = step_outputs.get("04")
            if deps4 is None:
                deps1 = step_outputs.get("01") or step01(args, rr)
                step_outputs["01"] = deps1
                deps3 = step_outputs.get("03")
                if deps3 is None:
                    deps2 = step_outputs.get("02") or step02(args, rr, deps1)
                    step_outputs["02"] = deps2
                    deps3 = step03(args, rr, deps2)
                    step_outputs["03"] = deps3
                deps4 = step04(args, rr, deps1, deps3)
                step_outputs["04"] = deps4
            step_outputs["05"] = step05(args, rr, deps4)

        elif s == "web":
            meta["steps_executed"].append("web")
            deps4 = step_outputs.get("04")
            if deps4 is None:
                deps1 = step_outputs.get("01") or step01(args, rr)
                step_outputs["01"] = deps1
                deps3 = step_outputs.get("03")
                if deps3 is None:
                    deps2 = step_outputs.get("02") or step02(args, rr, deps1)
                    step_outputs["02"] = deps2
                    deps3 = step03(args, rr, deps2)
                    step_outputs["03"] = deps3
                deps4 = step04(args, rr, deps1, deps3)
                step_outputs["04"] = deps4
            step_outputs["web"] = step_web(args, rr, deps4)

        else:
            print(f"[WARN] unknown step ignored: {s}", flush=True)

    # collect outputs for meta
    meta["outputs"] = step_outputs
    write_meta(rr, meta)

    print(f"[DONE] pipeline steps={meta['steps_executed']} run_root={rr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Route B domain pipeline wrapper.
Runs sentiment_01~05 and export_web_tables_l1_11 with per-domain run_id outputs under outputs/<domain>/runs/<run_id>.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent

from scripts.route_b_sentiment._shared.config_resolver import resolve as resolve_configs  # noqa: E402
from scripts.route_b_sentiment._shared import paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, help="phone|laptop|car|beauty")
    ap.add_argument("--run-id", default="", help="run identifier; default auto {YYYYMMDD}_{domain}_v0")
    ap.add_argument("--input-aspect-sentences", required=True, help="Path to aspect_sentences.parquet (Step00 output)")
    ap.add_argument("--aspects-yaml", default="", help="Override aspects config path; default resolved by domain")
    ap.add_argument("--steps", default="01,02,03,04,05,web", help="Comma-separated steps to run, e.g., 01,02 or 01,02,04,05")
    ap.add_argument("--resume", action="store_true", help="Pass --resume to step scripts when applicable")
    ap.add_argument("--max-train-rows", type=int, default=5000, help="Step01 max train rows")
    ap.add_argument("--train-pool-rows", type=int, default=200000, help="Step01 pool rows")
    ap.add_argument("--shard-n", type=int, default=64, help="Step01 shard count for aspect_pairs_ds")
    ap.add_argument("--ds-batch-rows", type=int, default=50000, help="Step01 ds batch rows")
    ap.add_argument("--step02-max-rows", type=int, default=0, help="Step02 max rows (0 = no cap)")
    ap.add_argument("--base-model", default="hfl/chinese-macbert-base", help="Base model for step03/04")
    ap.add_argument("--fp16", action="store_true", help="Use --fp16 for step04")
    return ap.parse_args()


def run_cmd(cmd: List[str]) -> None:
    print("[CMD]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def write_meta(run_dir: Path, meta: Dict) -> None:
    meta_dir = run_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_fp = meta_dir / "run.json"
    tmp = out_fp.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out_fp)


def build_run_id(domain: str, run_id: str) -> str:
    if run_id:
        return run_id
    ts = datetime.now().strftime("%Y%m%d")
    return f"{ts}_{domain}_v0"


def step01(args: argparse.Namespace, run_root: Path) -> Dict:
    out_dir = paths.step_dir(args.domain, args.run_id, "01")
    paths.ensure_dir(out_dir)
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
    }


def step02(args: argparse.Namespace, step01_out: Dict) -> Dict:
    out_dir = paths.step_dir(args.domain, args.run_id, "02")
    paths.ensure_dir(out_dir)
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
    }


def step03(args: argparse.Namespace, step02_out: Dict) -> Dict:
    out_dir = paths.step_dir(args.domain, args.run_id, "03")
    paths.ensure_dir(out_dir)
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
    return {"model_dir": str(out_dir)}


def step04(args: argparse.Namespace, step01_out: Dict, step03_out: Dict) -> Dict:
    out_dir = paths.step_dir(args.domain, args.run_id, "04")
    paths.ensure_dir(out_dir)
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
        "--resume",
        "--overwrite",
    ]
    if args.fp16:
        cmd.append("--fp16")
    run_cmd(cmd)
    return {"pred_ds": str(out_dir)}


def step05(args: argparse.Namespace, step04_out: Dict) -> Dict:
    out_dir = paths.step_dir(args.domain, args.run_id, "05")
    paths.ensure_dir(out_dir)
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
    return {"agg_dir": str(out_dir)}


def step_web(args: argparse.Namespace, step04_out: Dict) -> Dict:
    out_dir = paths.step_dir(args.domain, args.run_id, "web")
    paths.ensure_dir(out_dir)
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
    args.domain = args.domain.lower()
    args.run_id = build_run_id(args.domain, args.run_id)

    run_root = paths.run_root(args.domain, args.run_id)
    paths.ensure_dir(run_root)

    aspects_yaml = Path(args.aspects_yaml) if args.aspects_yaml else None
    domain_yaml = None
    if aspects_yaml is None or not aspects_yaml.exists():
        aspects_yaml, domain_yaml = resolve_configs(args.domain)
    else:
        _, domain_yaml = resolve_configs(args.domain)

    meta = {
        "domain": args.domain,
        "run_id": args.run_id,
        "input_aspect_sentences": str(Path(args.input_aspect_sentences)),
        "aspects_yaml": str(aspects_yaml) if aspects_yaml else "",
        "domain_yaml": str(domain_yaml) if domain_yaml else "",
        "steps": [],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_root": str(run_root),
    }

    step_outputs: Dict[str, Dict] = {}
    steps = [s.strip().lower() for s in args.steps.split(",") if s.strip()]

    for s in steps:
        if s in ("01", "1"):
            meta["steps"].append("01")
            step_outputs["01"] = step01(args, run_root)
        elif s in ("02", "2"):
            meta["steps"].append("02")
            deps = step_outputs.get("01") or step01(args, run_root)
            step_outputs["02"] = step02(args, deps)
        elif s in ("03", "3"):
            meta["steps"].append("03")
            deps1 = step_outputs.get("01") or step01(args, run_root)
            deps2 = step_outputs.get("02") or step02(args, deps1)
            step_outputs["03"] = step03(args, deps2)
        elif s in ("04", "4"):
            meta["steps"].append("04")
            deps1 = step_outputs.get("01") or step01(args, run_root)
            deps3 = step_outputs.get("03")
            if deps3 is None:
                deps2 = step_outputs.get("02") or step02(args, deps1)
                deps3 = step03(args, deps2)
                step_outputs["03"] = deps3
            step_outputs["04"] = step04(args, deps1, deps3)
        elif s in ("05", "5"):
            meta["steps"].append("05")
            deps4 = step_outputs.get("04")
            if deps4 is None:
                deps1 = step_outputs.get("01") or step01(args, run_root)
                deps3 = step_outputs.get("03")
                if deps3 is None:
                    deps2 = step_outputs.get("02") or step02(args, deps1)
                    deps3 = step03(args, deps2)
                    step_outputs["03"] = deps3
                deps4 = step04(args, deps1, deps3)
                step_outputs["04"] = deps4
            step_outputs["05"] = step05(args, deps4)
        elif s == "web":
            meta["steps"].append("web")
            deps4 = step_outputs.get("04")
            if deps4 is None:
                deps1 = step_outputs.get("01") or step01(args, run_root)
                deps3 = step_outputs.get("03")
                if deps3 is None:
                    deps2 = step_outputs.get("02") or step02(args, deps1)
                    deps3 = step03(args, deps2)
                    step_outputs["03"] = deps3
                deps4 = step04(args, deps1, deps3)
                step_outputs["04"] = deps4
            step_outputs["web"] = step_web(args, deps4)
        else:
            print(f"[WARN] unknown step ignored: {s}", file=sys.stderr)

    meta["outputs"] = step_outputs
    write_meta(run_root, meta)
    print(f"[DONE] pipeline steps={meta['steps']} run_root={run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

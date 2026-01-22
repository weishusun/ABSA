# scripts/route_b_sentiment/pipeline.py
# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Route B domain pipeline wrapper.
(Modified: Step 02 Smart Skip & Step 04 Resume Fix & Cool-Down Support)
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
    ap.add_argument("--run-id", default="")
    ap.add_argument("--input-aspect-sentences", required=True)
    ap.add_argument("--aspects-yaml", default="")
    ap.add_argument("--steps", default="01,02,03,04,05,web")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workspace", default="")
    # Step 01 params
    ap.add_argument("--max-train-rows", type=int, default=5000)
    ap.add_argument("--train-pool-rows", type=int, default=200000)
    ap.add_argument("--shard-n", type=int, default=64)
    ap.add_argument("--ds-batch-rows", type=int, default=50000)
    # Step 02 params
    ap.add_argument("--step02-max-rows", type=int, default=0)
    # Step 03 params
    ap.add_argument("--base-model", default="hfl/chinese-macbert-base")
    ap.add_argument("--num-train-epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    # Step 04 params
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--reuse-model", default="")
    # [新增] Step 04 专用散热参数
    ap.add_argument("--step04-batch-size", type=int, default=16, help="推理时的 Batch Size")
    ap.add_argument("--step04-cool-down-time", type=float, default=0.0, help="推理散热时间(秒)")

    return ap.parse_args()


# --- Step Functions ---


def step01(args: argparse.Namespace, rr: Path) -> Dict:
    out_dir = step_dir(rr, "01")
    ensure_dir(out_dir)

    # ================= [新增] 智能跳过逻辑 =================
    # 如果是续传模式，且 Step 01 的核心产物 (aspect_pairs_ds) 已经存在，则直接跳过
    if args.resume and (out_dir / "aspect_pairs_ds").exists():
        print(f"[INFO] Step 01 output exists: {out_dir / 'aspect_pairs_ds'}")
        print("[INFO] Skipping Step 01 to save time (Resume mode is on).")
        return {
            "train_candidates": str(out_dir / "train_candidates.parquet"),
            "aspect_pairs_ds": str(out_dir / "aspect_pairs_ds"),
            "step01_dir": str(out_dir),
        }
    # ========================================================

    # Step 01 脚本内部有 "if exists and not overwrite: return" 逻辑，所以这里直接跑是安全的
    cmd = [
        sys.executable, "-u",
        str(SCRIPT_DIR / "sentiment_01_build_aspect_pairs_and_train_candidates.py"),
        "--input", str(Path(args.input_aspect_sentences)),
        "--output-dir", str(out_dir),
        "--write-ds",
        "--shard-n", str(args.shard_n),
        "--ds-batch-rows", str(args.ds_batch_rows),
        "--max-train-rows", str(args.max_train_rows),
        "--train-pool-rows", str(args.train_pool_rows),
        "--overwrite",  # Pipeline always passes overwrite, but script handles idempotency
    ]
    if args.resume: cmd.append("--resume")
    run_cmd(cmd)
    return {
        "train_candidates": str(out_dir / "train_candidates.parquet"),
        "aspect_pairs_ds": str(out_dir / "aspect_pairs_ds"),
        "step01_dir": str(out_dir),
    }

def step02(args: argparse.Namespace, rr: Path, step01_out: Dict) -> Dict:
    out_dir = step_dir(rr, "02")
    ensure_dir(out_dir)
    target_file = out_dir / "train_pseudolabel.parquet"

    # [关键修改] 智能跳过：如果结果文件存在且大于 0，直接复用，不调 API
    if target_file.exists() and target_file.stat().st_size > 0:
        print(f"[SKIP] Step 02 output exists: {target_file}")
        print(f"[INFO] Skipping API calls to save cost. If you want to regenerate, delete this file manually.")
        return {"train_pseudolabel": str(target_file), "step02_dir": str(out_dir)}

    cmd = [
        sys.executable, "-u",
        str(SCRIPT_DIR / "sentiment_02_pseudolabel_openai.py"),
        "--domain", args.domain,
        "--run-id", args.run_id,
        "--input-aspect-sentences", str(args.input_aspect_sentences),
        "--input-pairs-dir", step01_out["step01_dir"],
        "--output-dir", str(out_dir),
        "--batch-size", "20",
        "--confidence-thr", "0.0",
    ]
    if args.step02_max_rows and args.step02_max_rows > 0:
        cmd += ["--max-rows", str(args.step02_max_rows)]
    run_cmd(cmd)
    return {"train_pseudolabel": str(target_file), "step02_dir": str(out_dir)}


def step03(args: argparse.Namespace, rr: Path, step02_out: Dict) -> Dict:
    out_dir = step_dir(rr, "03")
    ensure_dir(out_dir)
    cmd = [
        sys.executable, "-u",
        str(SCRIPT_DIR / "sentiment_03_train_asc_lora.py"),
        "--train-file", step02_out["train_pseudolabel"],
        "--output-dir", str(out_dir),
        "--base-model", args.base_model,
        "--num-train-epochs", str(args.num_train_epochs),
        "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum),
        "--learning-rate", "1e-4",
    ]
    if args.resume: cmd.append("--resume")
    run_cmd(cmd)
    return {"model_dir": str(out_dir), "step03_dir": str(out_dir)}


def step04(args: argparse.Namespace, rr: Path, step01_out: Dict, step03_out: Dict) -> Dict:
    out_dir = step_dir(rr, "04")
    ensure_dir(out_dir)
    cmd = [
        sys.executable, "-u",
        str(SCRIPT_DIR / "sentiment_04_infer_asc.py"),
        "--input", step01_out["aspect_pairs_ds"],
        "--output-dir", str(out_dir),
        "--model-dir", step03_out["model_dir"],
        "--base-model", args.base_model,

        # [关键修改] 动态传入 UI 设定的参数
        "--batch-size", str(args.step04_batch_size),
        "--cool-down-time", str(args.step04_cool_down_time),
    ]

    # 逻辑修改：只有“不续传”的时候，才允许覆盖；续传时严禁覆盖
    if not args.resume:
        cmd.append("--overwrite")

    if args.resume:
        cmd.append("--resume")

        # [修改] 强制开启 FP16，不再判断 args.fp16
        # 原代码: if args.fp16: cmd.append("--fp16")
    cmd.append("--fp16")  # <--- 强制添加这一行

    run_cmd(cmd)
    return {"pred_ds": str(out_dir), "step04_dir": str(out_dir)}


def step05(args: argparse.Namespace, rr: Path, step04_out: Dict) -> Dict:
    out_dir = step_dir(rr, "05")
    ensure_dir(out_dir)
    cmd = [
        sys.executable, "-u",
        str(SCRIPT_DIR / "sentiment_05_aggregate_and_build_excels.py"),
        "--pred-ds", step04_out["pred_ds"],
        "--out-dir", str(out_dir),
        "--overwrite",  # <--- 【新增】强制覆盖旧 Excel，不再报错
    ]
    run_cmd(cmd)
    return {"agg_dir": str(out_dir), "step05_dir": str(out_dir)}

def step_web(args: argparse.Namespace, rr: Path, step04_out: Dict) -> Dict:
    out_dir = step_dir(rr, "web")
    ensure_dir(out_dir)
    cmd = [
        sys.executable, "-u",
        str(SCRIPT_DIR / "export_web_tables_l1_11.py"),
        "--pred-ds", step04_out["pred_ds"],
        "--out-dir", str(out_dir),
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

    # --- [核心修复] 配置解析逻辑 ---
    aspects_yaml = Path(args.aspects_yaml) if args.aspects_yaml else None
    domain_yaml: Optional[Path] = None

    if aspects_yaml and aspects_yaml.exists():
        print(f"[INFO] 使用手动指定的 aspects 配置文件: {aspects_yaml}")
        potential_domain_yaml = aspects_yaml.parent / "domain.yaml"
        if potential_domain_yaml.exists():
            domain_yaml = potential_domain_yaml
    else:
        print(f"[INFO] 未指定配置文件，正在从内置库解析: {args.domain}")
        aspects_yaml, domain_yaml = resolve_configs(args.domain)

    # 检查输入数据是否存在
    in_fp = Path(args.input_aspect_sentences)
    if not in_fp.exists():
        print(f"[FATAL] 找不到输入文件: {in_fp}", file=sys.stderr)
        return 2

    meta: Dict = {
        "domain": args.domain,
        "run_id": args.run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_root": str(rr),
        "steps_executed": [],
    }

    # 处理复用模型逻辑
    reuse_model_path = None
    if args.reuse_model:
        p = Path(args.reuse_model)
        if not p.is_absolute():
            p = workspace_root / args.reuse_model
        if not p.exists():
            print(f"[FATAL] 复用模型路径不存在: {p}", file=sys.stderr)
            return 2
        reuse_model_path = str(p)

    steps = normalize_steps(args.steps)
    step_outputs: Dict[str, Dict] = {}

    for s in steps:
        if s == "01":
            meta["steps_executed"].append("01")
            step_outputs["01"] = step01(args, rr)
        elif s == "02":
            if reuse_model_path: continue
            meta["steps_executed"].append("02")
            deps1 = step_outputs.get("01") or step01(args, rr)
            step_outputs["02"] = step02(args, rr, deps1)
        elif s == "03":
            if reuse_model_path:
                step_outputs["03"] = {"model_dir": reuse_model_path}
                continue
            meta["steps_executed"].append("03")
            deps1 = step_outputs.get("01") or step01(args, rr)
            deps2 = step_outputs.get("02") or step02(args, rr, deps1)
            step_outputs["03"] = step03(args, rr, deps2)
        elif s == "04":
            meta["steps_executed"].append("04")
            deps1 = step_outputs.get("01") or step01(args, rr)
            deps3 = {"model_dir": reuse_model_path} if reuse_model_path else (
                        step_outputs.get("03") or step03(args, rr, step_outputs.get("02") or step02(args, rr, deps1)))
            step_outputs["04"] = step04(args, rr, deps1, deps3)
        elif s == "05":
            meta["steps_executed"].append("05")
            deps4 = step_outputs.get("04")
            if deps4: step_outputs["05"] = step05(args, rr, deps4)
        elif s == "web":
            meta["steps_executed"].append("web")
            deps4 = step_outputs.get("04")
            if deps4: step_outputs["web"] = step_web(args, rr, deps4)

    write_meta(rr, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
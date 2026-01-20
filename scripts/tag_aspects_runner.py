#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tag_aspects_runner.py

Module-3 purpose:
- Provide a script-level auditable entrypoint for tag_aspects.py.
- Always writes:
    outputs/<domain>/meta/tag/logs/tag.log
    outputs/<domain>/meta/tag/manifest_tag_script.json
- Does NOT change tag_aspects.py internals.

Usage (PowerShell example; use forward slashes to avoid unicodeescape issues):
python -u scripts/tag_aspects_runner.py `
  --input E:/ABSA_WORKSPACE/outputs/car/clean_sentences.parquet `
  --config C:/Users/weish/ABSA/configs/domains/car/aspects.yaml `
  --output-dir E:/ABSA_WORKSPACE/outputs/car `
  --run-id 20260108_car_tag_smoke `
  -- --batch-size 2048
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from ops.manifest import run_logged, build_step_manifest, write_json_atomic

ROOT = Path(__file__).resolve().parent.parent  # repo root


def now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def infer_domain(config_path: Path, output_dir: Path) -> str:
    # Prefer configs/domains/<domain>/aspects.yaml
    try:
        parts = config_path.as_posix().split("/")
        if "configs" in parts and "domains" in parts:
            i = parts.index("domains")
            if i + 1 < len(parts):
                d = parts[i + 1].strip().lower()
                if d:
                    return d
    except Exception:
        pass

    # Fallback: output-dir tail name (outputs/<domain>)
    d2 = output_dir.name.strip().lower()
    return d2 or "unknown"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="clean_sentences.parquet (dataset/file)")
    ap.add_argument("--config", required=True, help="aspects.yaml")
    ap.add_argument("--output-dir", required=True, help="outputs/<domain> directory (e.g. E:/.../outputs/car)")
    ap.add_argument("--run-id", default="", help="audit run id for this tag invocation")
    ap.add_argument("--hash-first-mb", type=int, default=0)

    # forward extra args to tag_aspects.py
    ap.add_argument("passthrough", nargs=argparse.REMAINDER)
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    domain = infer_domain(config_path, output_dir)
    run_id = args.run_id.strip() or f"{datetime.now().strftime('%Y%m%d')}_{domain}_tag_{now_id()}"

    # meta location: outputs/<domain>/meta/tag/
    meta_dir = output_dir / "meta" / "tag"
    logs_dir = meta_dir / "logs"
    ensure_dir(logs_dir)

    log_fp = logs_dir / "tag.log"
    man_fp = meta_dir / "manifest_tag_script.json"

    # Build command
    cmd: List[str] = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "tag_aspects.py"),
        "--input",
        str(input_path),
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
    ]

    passthrough = args.passthrough or []
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    cmd.extend(passthrough)

    # Run tag_aspects.py as subprocess (do not raise on failure)
    run_info = run_logged(cmd, cwd=ROOT, log_path=log_fp, check=False)

    # Expected outputs (best-effort)
    aspect_sentences = output_dir / "aspect_sentences.parquet"
    overlaps_xlsx = output_dir / "aspect_lexicon_overlaps.xlsx"

    # Derive roots for manifest paths (best-effort)
    outputs_root = output_dir.parent  # outputs/<domain> -> outputs
    workspace_root = outputs_root.parent  # workspace

    man = build_step_manifest(
        domain=domain,
        run_id=run_id,
        step="tag",
        workspace_root=str(workspace_root),
        outputs_root=str(outputs_root),
        repo_root=str(ROOT),
        cmd=cmd,
        run_info=run_info,
        inputs=[input_path, config_path],
        outputs=[aspect_sentences, overlaps_xlsx, output_dir],
        params={"passthrough": passthrough},
        hash_first_mb=int(args.hash_first_mb or 0),
    )

    write_json_atomic(man_fp, man)

    if man.get("status") != "success":
        print(f"[DONE] status=failed manifest={man_fp} log={log_fp}", file=sys.stderr)
        return int(run_info.get("returncode", 2) or 2)

    print(f"[DONE] status=success manifest={man_fp} log={log_fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

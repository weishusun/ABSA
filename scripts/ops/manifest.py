# scripts/ops/manifest.py
# -*- coding: utf-8 -*-

import json
import subprocess
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    """返回当前 UTC 时间的 ISO 格式字符串"""
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, obj: Dict[str, Any], indent: int = 2) -> None:
    """原子写入 JSON 文件（先写临时文件再重命名），防止写入中断导致文件损坏"""
    # 确保父目录存在
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    tmp.replace(path)


def calculate_file_hash(path: Path, limit_mb: int = 0) -> Optional[str]:
    """计算文件 SHA256，支持仅计算前 N MB 以提高速度"""
    if not path.exists() or not path.is_file():
        return None
    try:
        h = hashlib.sha256()
        limit_bytes = limit_mb * 1024 * 1024
        with path.open("rb") as f:
            if limit_bytes > 0:
                chunk = f.read(limit_bytes)
                if chunk:
                    h.update(chunk)
            else:
                # 全量计算
                while True:
                    chunk = f.read(4096)
                    if not chunk:
                        break
                    h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def run_logged(
        cmd: List[str],
        cwd: Path,
        log_path: Path,
        check: bool = False
) -> Dict[str, Any]:
    """
    运行子进程，将标准输出和标准错误重定向到日志文件。
    返回执行信息字典（开始时间、结束时间、耗时、返回码）。
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    start_ts = utc_now_iso()
    t0 = time.time()

    # 使用 'w' 模式打开日志文件，实时写入
    with log_path.open("w", encoding="utf-8") as log_f:
        # Header
        log_f.write(f"=== CMD START: {start_ts} ===\n")
        log_f.write(f"CMD: {' '.join(cmd)}\n")
        log_f.write(f"CWD: {cwd}\n")
        log_f.write("-" * 60 + "\n")
        log_f.flush()

        # 启动子进程
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdout=log_f,
                stderr=subprocess.STDOUT,  # 将 stderr 合并到 stdout
                text=True,
                encoding="utf-8",
                errors="replace"
            )

            proc.wait()
            rc = proc.returncode
        except Exception as e:
            log_f.write(f"\n[INTERNAL ERROR] Failed to run process: {e}\n")
            rc = -1

        # Footer
        end_ts = utc_now_iso()
        elapsed = time.time() - t0
        log_f.write("\n" + "-" * 60 + "\n")
        log_f.write(f"=== CMD END: {end_ts} (RC={rc}, Elapsed={elapsed:.2f}s) ===\n")

    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)

    return {
        "start_at": start_ts,
        "end_at": end_ts,
        "elapsed_sec": elapsed,
        "returncode": rc,
        "log_path": str(log_path)
    }


def build_step_manifest(
        domain: str,
        run_id: str,
        step: str,
        workspace_root: str,
        outputs_root: str,
        repo_root: str,
        cmd: List[str],
        run_info: Dict[str, Any],
        inputs: List[Path],
        outputs: List[Path],
        params: Dict[str, Any],
        hash_first_mb: int = 0
) -> Dict[str, Any]:
    """
    构建标准化的 Step Manifest 字典。
    """

    # 处理输入文件元数据
    input_meta = {}
    for p in inputs:
        p_path = Path(p)
        if p_path.exists():
            input_meta[p_path.name] = {
                "path": str(p_path),
                "size": p_path.stat().st_size,
                "mtime": p_path.stat().st_mtime,
                "sha256_head": calculate_file_hash(p_path, hash_first_mb) if hash_first_mb else None
            }
        else:
            input_meta[p_path.name] = {"path": str(p_path), "exists": False}

    # 处理输出文件元数据
    output_meta = {}
    for p in outputs:
        p_path = Path(p)
        if p_path.exists():
            is_file = p_path.is_file()
            output_meta[p_path.name] = {
                "path": str(p_path),
                "exists": True,
                "type": "file" if is_file else "dir",
                "size": p_path.stat().st_size if is_file else 0,
            }
        else:
            output_meta[p_path.name] = {"path": str(p_path), "exists": False}

    return {
        "schema": "absa.manifest.step.v1",
        "domain": domain,
        "run_id": run_id,
        "step": step,
        "status": "success" if run_info["returncode"] == 0 else "failed",
        "created_at": utc_now_iso(),
        "context": {
            "workspace_root": workspace_root,
            "outputs_root": outputs_root,
            "repo_root": repo_root,
            "cwd": str(Path.cwd()),
            "cmd": cmd,
        },
        "execution": run_info,
        "inputs": input_meta,
        "outputs": output_meta,
        "parameters": params,
    }
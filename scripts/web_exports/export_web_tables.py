from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# 你项目里如果已有统一的时间/manifest工具函数，后续可以替换成共用实现。
SCHEMA_VERSION = "absa.web_exports.v1"


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_yaml_minimal(path: Path) -> dict:
    # 避免引入额外依赖；若你已在项目里使用 PyYAML，可直接换成 yaml.safe_load
    # 这里用极简兜底：仅支持 key: value 的扁平结构（足够 smoke 用）。
    data: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        data[k.strip()] = v.strip().strip("'\"")
    return data


def load_l1_list(aspects_yaml: Path) -> List[str]:
    """
    兼容策略：
    - 优先按你项目的 aspects.yaml 结构解析（若你已有成熟结构，建议替换本函数为正式解析器）
    - 兜底：从文件中粗略抓取 'l1:' 下的 '- xxx'
    """
    text = aspects_yaml.read_text(encoding="utf-8")
    l1: List[str] = []
    in_l1 = False
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.strip().startswith("l1:"):
            in_l1 = True
            continue
        if in_l1:
            if line.strip().startswith("- "):
                l1.append(line.strip()[2:].strip().strip("'\""))
            elif line and not line.startswith(" "):
                # 走到下一个顶层 key 了
                break
    if not l1:
        raise RuntimeError(f"Cannot parse L1 list from {aspects_yaml}")
    return l1


@dataclass
class Paths:
    repo_root: Path
    workspace_root: Path
    outputs_root: Path

    def domain_root(self, domain: str) -> Path:
        return self.outputs_root / domain

    def web_root(self, domain: str) -> Path:
        return self.domain_root(domain) / "web_exports"


def build_empty_tables(domain: str, l1_list: List[str]) -> Dict[str, pd.DataFrame]:
    # product_list：空表
    product_list = pd.DataFrame(
        columns=["domain", "product_id", "brand", "model", "first_day", "last_day", "total_cnt"]
    )

    # by_product tables：空表（但 schema 固定）
    l1_pie_by_product = pd.DataFrame(
        columns=["domain", "product_id", "l1", "pos_cnt", "neg_cnt", "neu_cnt", "total_cnt", "first_day", "last_day"]
    )
    l1_daily_by_product = pd.DataFrame(
        columns=["domain", "product_id", "l1", "day", "pos_cnt", "neg_cnt", "neu_cnt", "total_cnt"]
    )
    l1_weekly_by_product = pd.DataFrame(
        columns=["domain", "product_id", "l1", "week_start", "week_end", "pos_cnt", "neg_cnt", "neu_cnt", "total_cnt"]
    )

    # all_products tables：提供 L1 维度的空行也可以，但这里保持空表，避免网站误读为“有数据=0”
    l1_pie_all = pd.DataFrame(
        columns=["domain", "l1", "pos_cnt", "neg_cnt", "neu_cnt", "total_cnt", "first_day", "last_day"]
    )
    l1_daily_all = pd.DataFrame(
        columns=["domain", "l1", "day", "pos_cnt", "neg_cnt", "neu_cnt", "total_cnt"]
    )
    l1_weekly_all = pd.DataFrame(
        columns=["domain", "l1", "week_start", "week_end", "pos_cnt", "neg_cnt", "neu_cnt", "total_cnt"]
    )

    # domain 列预设类型一致性（可选）
    for df in [
        product_list,
        l1_pie_by_product,
        l1_daily_by_product,
        l1_weekly_by_product,
        l1_pie_all,
        l1_daily_all,
        l1_weekly_all,
    ]:
        if "domain" in df.columns:
            df["domain"] = df.get("domain", pd.Series(dtype="string"))

    return {
        "product_list.parquet": product_list,
        "l1_pie_alltime_by_product.parquet": l1_pie_by_product,
        "l1_daily_last7_by_product.parquet": l1_daily_by_product,
        "l1_weekly_last4_by_product.parquet": l1_weekly_by_product,
        "l1_pie_alltime_all_products.parquet": l1_pie_all,
        "l1_daily_last7_all_products.parquet": l1_daily_all,
        "l1_weekly_last4_all_products.parquet": l1_weekly_all,
    }


def write_tables(out_tables_dir: Path, tables: Dict[str, pd.DataFrame]) -> List[str]:
    ensure_dir(out_tables_dir)
    written: List[str] = []
    for name, df in tables.items():
        out_path = out_tables_dir / name
        df.to_parquet(out_path, index=False)
        written.append(str(out_path))
    return written


def copy_tree_overwrite(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--repo-root", default=r"C:\Users\weish\ABSA")
    ap.add_argument("--workspace-root", default=r"E:\ABSA_WORKSPACE")
    ap.add_argument("--outputs-root", default=r"E:\ABSA_WORKSPACE\outputs")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--smoke", action="store_true", help="Do not read routeb outputs; emit empty web tables with schema.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    domain = args.domain
    run_id = args.run_id or f"{datetime.now().strftime('%Y%m%d')}_{domain}_web"
    stamp = now_stamp()

    paths = Paths(
        repo_root=Path(args.repo_root),
        workspace_root=Path(args.workspace_root),
        outputs_root=Path(args.outputs_root),
    )

    # configs：从 repo 读取 Domain Pack
    aspects_yaml = paths.repo_root / "configs" / "domains" / domain / "aspects.yaml"
    domain_yaml = paths.repo_root / "configs" / "domains" / domain / "domain.yaml"
    if not aspects_yaml.exists():
        raise FileNotFoundError(f"missing aspects.yaml: {aspects_yaml}")
    if not domain_yaml.exists():
        raise FileNotFoundError(f"missing domain.yaml: {domain_yaml}")

    l1_list = load_l1_list(aspects_yaml)

    web_root = paths.web_root(domain)
    export_root = web_root / "exports" / stamp
    export_tables_dir = export_root / "tables"
    export_meta_dir = export_root / "meta"
    latest_root = web_root / "latest"
    latest_tables_dir = latest_root / "tables"
    latest_meta_dir = latest_root / "meta"

    ensure_dir(export_meta_dir)

    started_at = datetime.now().isoformat(timespec="seconds")

    if args.smoke:
        tables = build_empty_tables(domain=domain, l1_list=l1_list)
        outputs_written = write_tables(export_tables_dir, tables)
        mode = "smoke"
        inputs = {
            "configs": {
                "domain_yaml": str(domain_yaml),
                "aspects_yaml": str(aspects_yaml),
            },
            "source": None,
        }
    else:
        # 非 smoke：这里先定义“契约位”，避免你现在被迫绑定 routeb 的具体文件名。
        # 后续模块 3-2/3-3 再把 routeb 的最终聚合产物固定到这个路径。
        web_ready = paths.domain_root(domain) / "sentiment" / "web_ready.parquet"
        if not web_ready.exists():
            raise FileNotFoundError(
                f"missing web_ready input: {web_ready}\n"
                f"Tip: run with --smoke first, or later fix RouteB to write sentiment/web_ready.parquet"
            )
        df = pd.read_parquet(web_ready)

        # 这里按你最终契约来实现聚合；当前先占位抛错，防止静默生成错误表。
        raise NotImplementedError(
            "Non-smoke export is not implemented yet. "
            "Module 3 will next define the exact columns of sentiment/web_ready.parquet and implement aggregation."
        )

    ended_at = datetime.now().isoformat(timespec="seconds")

    manifest = {
        "schema": SCHEMA_VERSION,
        "domain": domain,
        "run_id": run_id,
        "step": "web_exports",
        "mode": mode,
        "created_at": started_at,
        "finished_at": ended_at,
        "status": "success",
        "inputs": inputs,
        "outputs": {
            "export_root": str(export_root),
            "tables": outputs_written,
        },
    }

    # 写 exports/<stamp>/meta/manifest_web.json
    atomic_write_json(export_meta_dir / "manifest_web.json", manifest)

    # 更新 latest：复制 tables + meta
    ensure_dir(latest_root)
    copy_tree_overwrite(export_tables_dir, latest_tables_dir)
    ensure_dir(latest_meta_dir)
    atomic_write_json(latest_meta_dir / "manifest_web.json", manifest)

    # 写 latest.json 指针
    atomic_write_json(web_root / "latest.json", {"stamp": stamp, "domain": domain, "run_id": run_id})

    print(f"[DONE] domain={domain} mode={mode} export={export_root}")
    print(f"[DONE] latest={latest_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

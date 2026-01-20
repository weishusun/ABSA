from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Sequence

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


SENTIMENT_MAP = {
    "POS": "POS",
    "POSITIVE": "POS",
    "NEG": "NEG",
    "NEGATIVE": "NEG",
    "NEU": "NEU",
    "NEUTRAL": "NEU",
}

COUNT_COLUMNS = ["pos_cnt", "neg_cnt", "neu_cnt", "total_cnt"]
PRODUCT_LIST_COLUMNS = ["domain", "product_id", "brand", "model", "first_day", "last_day", "total_cnt"]
L1_PIE_BY_PRODUCT_COLUMNS = [
    "domain",
    "product_id",
    "l1",
    "pos_cnt",
    "neg_cnt",
    "neu_cnt",
    "total_cnt",
    "first_day",
    "last_day",
]
L1_DAILY_BY_PRODUCT_COLUMNS = ["domain", "product_id", "l1", "day", "pos_cnt", "neg_cnt", "neu_cnt", "total_cnt"]
L1_WEEKLY_BY_PRODUCT_COLUMNS = [
    "domain",
    "product_id",
    "l1",
    "week_start",
    "week_end",
    "pos_cnt",
    "neg_cnt",
    "neu_cnt",
    "total_cnt",
]
L1_PIE_ALL_COLUMNS = ["domain", "l1", "pos_cnt", "neg_cnt", "neu_cnt", "total_cnt", "first_day", "last_day"]
L1_DAILY_ALL_COLUMNS = ["domain", "l1", "day", "pos_cnt", "neg_cnt", "neu_cnt", "total_cnt"]
L1_WEEKLY_ALL_COLUMNS = [
    "domain",
    "l1",
    "week_start",
    "week_end",
    "pos_cnt",
    "neg_cnt",
    "neu_cnt",
    "total_cnt",
]

EMPTY_COLUMN_DTYPES = {
    "domain": "string",
    "product_id": "string",
    "brand": "string",
    "model": "string",
    "l1": "string",
    "week_start": "datetime64[ns]",
    "week_end": "datetime64[ns]",
    "day": "datetime64[ns]",
    "first_day": "datetime64[ns]",
    "last_day": "datetime64[ns]",
    "pos_cnt": "int64",
    "neg_cnt": "int64",
    "neu_cnt": "int64",
    "total_cnt": "int64",
}


def first_non_null(series: pd.Series) -> object:
    idx = series.first_valid_index()
    return series.loc[idx] if idx is not None else pd.NA


def ensure_string_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        if column in df:
            df[column] = df[column].astype("string")


def ensure_int_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        if column in df:
            df[column] = df[column].fillna(0).astype("int64")


def ensure_date_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        if column in df:
            df[column] = pd.to_datetime(df[column], errors="coerce").dt.date


def empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    data = {}
    for column in columns:
        dtype = EMPTY_COLUMN_DTYPES.get(column, "object")
        data[column] = pd.Series(dtype=dtype)
    return pd.DataFrame(data)


def filter_last_n_days(df: pd.DataFrame, latest_day: pd.Timestamp, window_days: int) -> pd.DataFrame:
    if pd.isna(latest_day):
        return df.iloc[0:0]
    start = latest_day - pd.Timedelta(days=window_days - 1)
    mask = df["day"].notna() & (df["day"] >= start) & (df["day"] <= latest_day)
    return df.loc[mask]


def filter_last_n_weeks(df: pd.DataFrame, latest_week_start: pd.Timestamp, window_weeks: int) -> pd.DataFrame:
    if pd.isna(latest_week_start):
        return df.iloc[0:0]
    start = latest_week_start - pd.Timedelta(weeks=window_weeks - 1)
    mask = df["week_start"].notna() & (df["week_start"] >= start) & (df["week_start"] <= latest_week_start)
    return df.loc[mask]


def preprocess_web_ready(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    frame = df.copy()
    domain_series = frame["domain"] if "domain" in frame else pd.Series(pd.NA, index=frame.index)
    domain_series = domain_series.fillna(domain)
    frame["domain"] = domain_series.astype("string")
    product_id_series = frame["product_id"] if "product_id" in frame else pd.Series(pd.NA, index=frame.index)
    frame["product_id"] = product_id_series.fillna("unknown_product").astype("string")
    brand_series = frame["brand"] if "brand" in frame else pd.Series(pd.NA, index=frame.index)
    frame["brand"] = brand_series.astype("string")
    model_series = frame["model"] if "model" in frame else pd.Series(pd.NA, index=frame.index)
    frame["model"] = model_series.astype("string")
    l1_series = frame["l1"] if "l1" in frame else pd.Series(pd.NA, index=frame.index)
    frame["l1"] = l1_series.astype("string")
    sentiment_series = frame["sentiment"] if "sentiment" in frame else pd.Series(pd.NA, index=frame.index)
    sentiment_upper = sentiment_series.astype("string").str.upper()
    frame["sentiment"] = sentiment_upper.map(SENTIMENT_MAP).fillna("NEU")
    weight_series = frame["weight"] if "weight" in frame else pd.Series(1, index=frame.index)
    frame["weight"] = pd.to_numeric(weight_series, errors="coerce").fillna(1).astype("int64")
    day_series = frame["day"] if "day" in frame else pd.Series(pd.NaT, index=frame.index)
    frame["day"] = pd.to_datetime(day_series, errors="coerce").dt.normalize()
    frame["week_start"] = pd.NaT
    frame["week_end"] = pd.NaT
    valid_day = frame["day"].notna()
    if valid_day.any():
        week_start = frame.loc[valid_day, "day"] - pd.to_timedelta(
            frame.loc[valid_day, "day"].dt.weekday, unit="d"
        )
        frame.loc[valid_day, "week_start"] = week_start
        frame.loc[valid_day, "week_end"] = week_start + pd.Timedelta(days=6)
    frame["pos_cnt"] = (frame["sentiment"] == "POS").astype("int64") * frame["weight"]
    frame["neg_cnt"] = (frame["sentiment"] == "NEG").astype("int64") * frame["weight"]
    frame["neu_cnt"] = (frame["sentiment"] == "NEU").astype("int64") * frame["weight"]
    frame["total_cnt"] = frame["pos_cnt"] + frame["neg_cnt"] + frame["neu_cnt"]
    return frame


def build_product_list(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return empty_frame(PRODUCT_LIST_COLUMNS)
    grouped = df.groupby(["domain", "product_id"], as_index=False)
    result = grouped.agg(
        brand=("brand", first_non_null),
        model=("model", first_non_null),
        first_day=("day", "min"),
        last_day=("day", "max"),
        total_cnt=("total_cnt", "sum"),
    )
    ensure_string_columns(result, ["domain", "product_id", "brand", "model"])
    ensure_date_columns(result, ["first_day", "last_day"])
    ensure_int_columns(result, ["total_cnt"])
    return result[PRODUCT_LIST_COLUMNS]


def build_l1_pie_by_product(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return empty_frame(L1_PIE_BY_PRODUCT_COLUMNS)
    grouped = df.groupby(["domain", "product_id", "l1"], as_index=False)
    result = grouped.agg(
        pos_cnt=("pos_cnt", "sum"),
        neg_cnt=("neg_cnt", "sum"),
        neu_cnt=("neu_cnt", "sum"),
        total_cnt=("total_cnt", "sum"),
        first_day=("day", "min"),
        last_day=("day", "max"),
    )
    ensure_string_columns(result, ["domain", "product_id", "l1"])
    ensure_date_columns(result, ["first_day", "last_day"])
    ensure_int_columns(result, COUNT_COLUMNS)
    return result[L1_PIE_BY_PRODUCT_COLUMNS]


def build_l1_daily_last7_by_product(df: pd.DataFrame, latest_day: pd.Timestamp) -> pd.DataFrame:
    window = filter_last_n_days(df, latest_day, 7)
    if window.empty:
        return empty_frame(L1_DAILY_BY_PRODUCT_COLUMNS)
    grouped = window.groupby(["domain", "product_id", "l1", "day"], as_index=False)
    result = grouped.agg(
        pos_cnt=("pos_cnt", "sum"),
        neg_cnt=("neg_cnt", "sum"),
        neu_cnt=("neu_cnt", "sum"),
        total_cnt=("total_cnt", "sum"),
    )
    ensure_string_columns(result, ["domain", "product_id", "l1"])
    ensure_int_columns(result, COUNT_COLUMNS)
    ensure_date_columns(result, ["day"])
    return result[L1_DAILY_BY_PRODUCT_COLUMNS]


def build_l1_weekly_last4_by_product(df: pd.DataFrame, latest_week_start: pd.Timestamp) -> pd.DataFrame:
    window = filter_last_n_weeks(df, latest_week_start, 4)
    if window.empty:
        return empty_frame(L1_WEEKLY_BY_PRODUCT_COLUMNS)
    grouped = window.groupby(["domain", "product_id", "l1", "week_start"], as_index=False)
    result = grouped.agg(
        pos_cnt=("pos_cnt", "sum"),
        neg_cnt=("neg_cnt", "sum"),
        neu_cnt=("neu_cnt", "sum"),
        total_cnt=("total_cnt", "sum"),
    )
    result["week_end"] = result["week_start"] + pd.Timedelta(days=6)
    ensure_string_columns(result, ["domain", "product_id", "l1"])
    ensure_int_columns(result, COUNT_COLUMNS)
    ensure_date_columns(result, ["week_start", "week_end"])
    return result[L1_WEEKLY_BY_PRODUCT_COLUMNS]


def build_l1_pie_all_products(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return empty_frame(L1_PIE_ALL_COLUMNS)
    grouped = df.groupby(["domain", "l1"], as_index=False)
    result = grouped.agg(
        pos_cnt=("pos_cnt", "sum"),
        neg_cnt=("neg_cnt", "sum"),
        neu_cnt=("neu_cnt", "sum"),
        total_cnt=("total_cnt", "sum"),
        first_day=("day", "min"),
        last_day=("day", "max"),
    )
    ensure_string_columns(result, ["domain", "l1"])
    ensure_date_columns(result, ["first_day", "last_day"])
    ensure_int_columns(result, COUNT_COLUMNS)
    return result[L1_PIE_ALL_COLUMNS]


def build_l1_daily_last7_all_products(df: pd.DataFrame, latest_day: pd.Timestamp) -> pd.DataFrame:
    window = filter_last_n_days(df, latest_day, 7)
    if window.empty:
        return empty_frame(L1_DAILY_ALL_COLUMNS)
    grouped = window.groupby(["domain", "l1", "day"], as_index=False)
    result = grouped.agg(
        pos_cnt=("pos_cnt", "sum"),
        neg_cnt=("neg_cnt", "sum"),
        neu_cnt=("neu_cnt", "sum"),
        total_cnt=("total_cnt", "sum"),
    )
    ensure_string_columns(result, ["domain", "l1"])
    ensure_int_columns(result, COUNT_COLUMNS)
    ensure_date_columns(result, ["day"])
    return result[L1_DAILY_ALL_COLUMNS]


def build_l1_weekly_last4_all_products(df: pd.DataFrame, latest_week_start: pd.Timestamp) -> pd.DataFrame:
    window = filter_last_n_weeks(df, latest_week_start, 4)
    if window.empty:
        return empty_frame(L1_WEEKLY_ALL_COLUMNS)
    grouped = window.groupby(["domain", "l1", "week_start"], as_index=False)
    result = grouped.agg(
        pos_cnt=("pos_cnt", "sum"),
        neg_cnt=("neg_cnt", "sum"),
        neu_cnt=("neu_cnt", "sum"),
        total_cnt=("total_cnt", "sum"),
    )
    result["week_end"] = result["week_start"] + pd.Timedelta(days=6)
    ensure_string_columns(result, ["domain", "l1"])
    ensure_int_columns(result, COUNT_COLUMNS)
    ensure_date_columns(result, ["week_start", "week_end"])
    return result[L1_WEEKLY_ALL_COLUMNS]


def build_web_tables(df: pd.DataFrame, domain: str, l1_list: List[str]) -> Dict[str, pd.DataFrame]:
    normalized = preprocess_web_ready(df, domain)
    if normalized.empty:
        return build_empty_tables(domain=domain, l1_list=l1_list)
    df_l1 = normalized[normalized["l1"].notna()].copy()
    latest_day = df_l1["day"].max() if not df_l1.empty else pd.NaT
    latest_week_start = df_l1["week_start"].max() if not df_l1.empty else pd.NaT
    return {
        "product_list.parquet": build_product_list(normalized),
        "l1_pie_alltime_by_product.parquet": build_l1_pie_by_product(df_l1),
        "l1_daily_last7_by_product.parquet": build_l1_daily_last7_by_product(df_l1, latest_day),
        "l1_weekly_last4_by_product.parquet": build_l1_weekly_last4_by_product(df_l1, latest_week_start),
        "l1_pie_alltime_all_products.parquet": build_l1_pie_all_products(df_l1),
        "l1_daily_last7_all_products.parquet": build_l1_daily_last7_all_products(df_l1, latest_day),
        "l1_weekly_last4_all_products.parquet": build_l1_weekly_last4_all_products(df_l1, latest_week_start),
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
        web_ready = paths.domain_root(domain) / "sentiment" / "web_ready.parquet"
        if not web_ready.exists():
            raise FileNotFoundError(
                f"missing web_ready input: {web_ready}\n"
                f"Tip: run with --smoke first, or later fix RouteB to write sentiment/web_ready.parquet"
            )
        df = pd.read_parquet(web_ready)
        tables = build_web_tables(df, domain=domain, l1_list=l1_list)
        outputs_written = write_tables(export_tables_dir, tables)
        mode = "full"
        inputs = {
            "configs": {
                "domain_yaml": str(domain_yaml),
                "aspects_yaml": str(aspects_yaml),
            },
            "source": {"web_ready": str(web_ready)},
        }

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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None


# -----------------------------
# Helpers
# -----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_cmd(argv: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        shell=False,
    )


def print_check(label: str, ok: bool, note: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    if note:
        print(f"- [{tag}] {label}: {note}")
    else:
        print(f"- [{tag}] {label}")


def safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return None


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def repo_root_from_git() -> Path:
    res = run_cmd(["git", "rev-parse", "--show-toplevel"])
    if res.returncode == 0 and res.stdout.strip():
        return Path(res.stdout.strip())
    return Path(__file__).resolve().parents[2]


def list_parquet_files(dir_path: Path) -> list[Path]:
    if not dir_path.exists() or not dir_path.is_dir():
        return []
    return sorted(dir_path.glob("*.parquet"))


def parquet_num_rows(path: Path) -> int | None:
    """
    Best-effort row count.
    IMPORTANT: Do NOT treat 0 from parquet metadata as authoritative on Windows/pyarrow.
    """
    if pq is not None:
        try:
            pf = pq.ParquetFile(path)
            n = int(pf.metadata.num_rows)
            # Some environments may return 0 even when file has rows; treat 0 as "unknown".
            return n if n > 0 else None
        except Exception:
            return None
    return None


def parquet_read_columns_head(path: Path, columns: list[str], n: int = 2000) -> pd.DataFrame:
    """
    Robust small read with fallback:
      1) Try pyarrow.ParquetFile row-group read (fast & stable in many cases)
      2) If that fails, fallback to pandas.read_parquet (often works when ParquetFile fails on Windows)
    """
    # 1) pyarrow row-group path
    if pq is not None:
        try:
            pf = pq.ParquetFile(path)
            batches = []
            rows = 0
            for rg in range(pf.num_row_groups):
                tbl = pf.read_row_group(rg, columns=columns)
                df = tbl.to_pandas()
                batches.append(df)
                rows += len(df)
                if rows >= n:
                    break
            return pd.concat(batches, ignore_index=True) if batches else pd.DataFrame(columns=columns)
        except Exception:
            # fallback below
            pass

    # 2) pandas fallback
    df2 = pd.read_parquet(path, columns=columns)
    return df2.head(n)


def to_datetime_series(x: pd.Series) -> pd.Series:
    """
    Convert common formats into pandas datetime64[ns].
    Handles:
      - ISO date/datetime strings
      - epoch seconds (10 digits) / epoch milliseconds (13 digits)
    """
    if x is None or len(x) == 0:
        return pd.to_datetime(x, errors="coerce")

    s = x.copy()
    s_str = s.astype("string")

    is_digits = s_str.str.fullmatch(r"\d+").fillna(False)
    if bool(is_digits.any()):
        s_num = pd.to_numeric(s_str.where(is_digits), errors="coerce")
        med = float(s_num.dropna().median()) if not s_num.dropna().empty else 0.0
        unit = "ms" if med > 1e12 else "s"
        dt_epoch = pd.to_datetime(s_num, unit=unit, errors="coerce")

        dt_other = pd.to_datetime(s.where(~is_digits), errors="coerce")
        return dt_epoch.where(is_digits, dt_other)

    return pd.to_datetime(s, errors="coerce")


# -----------------------------
# Domain paths
# -----------------------------

@dataclass
class DomainPaths:
    domain: str
    base: Path
    clean_sentences: Path
    aspect_sentences: Path
    meta_step00_manifest: Path
    meta_tag_manifest: Path
    web_ready: Path
    web_exports_root: Path
    web_exports_latest_tables: Path
    web_exports_latest_meta: Path
    web_exports_latest_json: Path
    manifest_web: Path
    product_list: Path
    module5_coverage_xlsx: Path
    module5_counts_xlsx: Path
    module5_lexicon_overlaps_xlsx: Path


def build_domain_paths(workspace_root: Path, domain: str) -> DomainPaths:
    base = workspace_root / domain
    web_exports_root = base / "web_exports"
    latest = web_exports_root / "latest"
    latest_tables = latest / "tables"
    latest_meta = latest / "meta"
    return DomainPaths(
        domain=domain,
        base=base,
        clean_sentences=base / "clean_sentences.parquet",
        aspect_sentences=base / "aspect_sentences.parquet",
        meta_step00_manifest=base / "meta" / "step00" / "manifest_step00.json",
        meta_tag_manifest=base / "meta" / "tag" / "manifest_tag_script.json",
        web_ready=base / "sentiment" / "web_ready.parquet",
        web_exports_root=web_exports_root,
        web_exports_latest_tables=latest_tables,
        web_exports_latest_meta=latest_meta,
        web_exports_latest_json=web_exports_root / "latest.json",
        manifest_web=latest_meta / "manifest_web.json",
        product_list=latest_tables / "product_list.parquet",
        module5_coverage_xlsx=base / f"aspect_coverage_{domain}.xlsx",
        module5_counts_xlsx=base / f"aspect_counts_{domain}.xlsx",
        module5_lexicon_overlaps_xlsx=base / "aspect_lexicon_overlaps.xlsx",
    )


# -----------------------------
# Checks
# -----------------------------

def check_repo_basics(repo_root: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"ok": True}
    br = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    info["branch"] = br.stdout.strip() if br.returncode == 0 else "<unknown>"
    st = run_cmd(["git", "status", "--porcelain"], cwd=repo_root)
    info["dirty"] = (st.returncode == 0 and bool(st.stdout.strip()))
    return info


def check_config_sync(repo_root: Path) -> dict[str, Any]:
    script = repo_root / "scripts" / "tools" / "sync_legacy_configs.py"
    if not script.exists():
        return {"ok": False, "note": f"missing {script}"}
    py = repo_root / ".venv" / "Scripts" / "python.exe"
    res = run_cmd([str(py), "-u", str(script), "--check"], cwd=repo_root)
    return {"ok": (res.returncode == 0), "rc": res.returncode, "stdout": res.stdout, "stderr": res.stderr}


def check_workspace_io(dp: DomainPaths) -> dict[str, Any]:
    required = [
        dp.base,
        dp.clean_sentences,
        dp.aspect_sentences,
        dp.meta_step00_manifest,
        dp.meta_tag_manifest,
        dp.web_ready,
        dp.web_exports_latest_json,
        dp.web_exports_latest_tables,
        dp.manifest_web,
    ]
    ok = all(p.exists() for p in required)
    tables = [p.name for p in list_parquet_files(dp.web_exports_latest_tables)]
    ok = ok and (len(tables) >= 7)
    return {"ok": ok, "tables": tables}


def check_latest_json(dp: DomainPaths) -> dict[str, Any]:
    if not dp.web_exports_latest_json.exists():
        return {"ok": False, "note": "latest.json missing"}
    obj = safe_read_json(dp.web_exports_latest_json)
    if not obj:
        return {"ok": False, "note": "latest.json unreadable"}
    ok = isinstance(obj.get("stamp"), str) and obj.get("domain") == dp.domain and isinstance(obj.get("run_id"), str)
    return {"ok": ok, "data": obj}


def check_expected_tables(dp: DomainPaths) -> dict[str, Any]:
    expected = [
        "product_list.parquet",
        "l1_pie_alltime_by_product.parquet",
        "l1_daily_last7_by_product.parquet",
        "l1_weekly_last4_by_product.parquet",
        "l1_pie_alltime_all_products.parquet",
        "l1_daily_last7_all_products.parquet",
        "l1_weekly_last4_all_products.parquet",
    ]
    actual = [p.name for p in list_parquet_files(dp.web_exports_latest_tables)]
    missing = [x for x in expected if x not in actual]
    extra = [x for x in actual if x not in expected]
    return {"ok": (len(missing) == 0), "missing": missing, "extra": extra, "actual": actual}


def sanity_web_ready_day(dp: DomainPaths) -> dict[str, Any]:
    if not dp.web_ready.exists():
        return {"ok": False, "note": "web_ready.parquet missing"}

    try:
        df = parquet_read_columns_head(dp.web_ready, ["day"], n=5000)
    except Exception as exc:
        return {"ok": False, "note": f"read failed: {exc}"}

    if df.empty:
        return {"ok": False, "note": "web_ready empty (0 rows)"}

    day = to_datetime_series(df["day"])
    nulls = int(day.isna().sum())

    if day.dropna().empty:
        return {"ok": False, "note": "day all NaT", "null_day": nulls}

    dmin = day.dropna().min()
    dmax = day.dropna().max()

    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)

    ok = (nulls == 0) and (dmin.date() >= date(2000, 1, 1)) and (dmax.date() <= tomorrow)
    return {"ok": ok, "null_day": nulls, "min": dmin.isoformat(), "max": dmax.isoformat()}


def sanity_product_list_first_last(dp: DomainPaths) -> dict[str, Any]:
    if not dp.product_list.exists():
        return {"ok": False, "note": "product_list.parquet missing"}

    # Metadata row-count is not reliable enough to fail early on Windows.
    nrows_meta = parquet_num_rows(dp.product_list)

    try:
        df = parquet_read_columns_head(dp.product_list, ["first_day", "last_day"], n=50000)
    except Exception as exc:
        return {"ok": False, "note": f"read failed: {exc}", "nrows_meta": nrows_meta}

    if df.empty:
        # Now it is a real empty (based on actual read), not metadata guess.
        return {"ok": False, "note": "product_list empty (read returned 0 rows)", "nrows_meta": nrows_meta}

    first = to_datetime_series(df["first_day"])
    last = to_datetime_series(df["last_day"])
    null_first = int(first.isna().sum())
    null_last = int(last.isna().sum())

    mask = (~first.isna()) & (~last.isna())
    bad_order = int((first[mask] > last[mask]).sum())

    ok = (null_first == 0) and (null_last == 0) and (bad_order == 0)

    out = {
        "ok": ok,
        "nrows_meta": nrows_meta,
        "rows_sampled": int(len(df)),
        "null_first_day": null_first,
        "null_last_day": null_last,
        "bad_first_gt_last": bad_order,
    }
    if not first.dropna().empty:
        out["first_min"] = first.dropna().min().isoformat()
    if not last.dropna().empty:
        out["last_max"] = last.dropna().max().isoformat()
    if not ok:
        out["note"] = "first/last has nulls or order issues"
    return out


def module5_readiness(dp: DomainPaths) -> dict[str, Any]:
    coverage_ok = dp.module5_coverage_xlsx.exists()
    counts_ok = dp.module5_counts_xlsx.exists()
    lex_ok = dp.module5_lexicon_overlaps_xlsx.exists()
    return {"ok": (coverage_ok and counts_ok and lex_ok), "coverage": coverage_ok, "counts": counts_ok, "lexicon_overlaps": lex_ok}


def main() -> int:
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--domain", type=str)
    g.add_argument("--domains", type=str)
    parser.add_argument("--workspace-root", type=str, required=True)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    repo_root = repo_root_from_git()
    workspace_root = Path(args.workspace_root).resolve()

    domains = [args.domain.strip()] if args.domain else [x.strip() for x in args.domains.split(",") if x.strip()]

    out = Path(args.out) if args.out else (repo_root / "docs" / "audit" / f"AUDIT_{datetime.now().strftime('%Y%m%d')}.md")
    if not out.is_absolute():
        out = (repo_root / out).resolve()

    repo_info = check_repo_basics(repo_root)
    print_check("Repo 基础", True, f"{repo_info.get('branch')} (dirty={repo_info.get('dirty')})")

    cfg = check_config_sync(repo_root)
    print_check("配置一致性: sync_legacy_configs --check", bool(cfg.get("ok")), f"exit {cfg.get('rc')}")

    # Per-domain checks (print only; report可后续再扩展)
    for d in domains:
        dp = build_domain_paths(workspace_root, d)

        io = check_workspace_io(dp)
        print_check(f"[{d}] Workspace I/O", bool(io["ok"]))

        latest = check_latest_json(dp)
        print_check(f"[{d}] latest.json 可读", bool(latest["ok"]), latest.get("note", ""))

        tables = check_expected_tables(dp)
        print_check(f"[{d}] web_exports latest/tables 表集合", bool(tables["ok"]), f"missing={tables.get('missing')}")

        wr = sanity_web_ready_day(dp)
        print_check(f"[{d}] 内容级 sanity: web_ready.day", bool(wr["ok"]), wr.get("note", f"null_day={wr.get('null_day')}"))

        pl = sanity_product_list_first_last(dp)
        if pl.get("ok"):
            note = f"null_first={pl.get('null_first_day')} null_last={pl.get('null_last_day')} bad_order={pl.get('bad_first_gt_last')}"
        else:
            note = pl.get("note", "unknown")
        print_check(f"[{d}] 内容级 sanity: product_list first/last", bool(pl.get("ok")), note)

        mw_ok = dp.manifest_web.exists() and bool(safe_read_json(dp.manifest_web))
        mt_ok = dp.meta_tag_manifest.exists() and bool(safe_read_json(dp.meta_tag_manifest))
        print_check(f"[{d}] 审计文件: manifest_web.json", mw_ok)
        print_check(f"[{d}] 审计文件: manifest_tag_script.json", mt_ok)

        m5 = module5_readiness(dp)
        print_check(f"[{d}] 模块 5 readiness（coverage/counts/overlaps）", bool(m5["ok"]))

    ensure_parent(out)
    out.write_text(f"# ABSA Stage Audit\n\n- time_utc: `{utc_now_iso()}`\n", encoding="utf-8")
    print(f"[DONE] wrote report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from pathlib import Path
import re
import pandas as pd
import pyarrow.dataset as ds

INPUT = Path(r".\outputs\phone\clean_sentences.parquet")
OUT_DIR = Path(r".\outputs\phone\reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def safe_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r'[\\/:*?"<>|]+', "_", s)
    return s[:120] if len(s) > 120 else s

def export_one_product(table: pd.DataFrame, out_path: Path):
    # overview
    total_sent = len(table)
    total_docs = table["doc_id"].nunique()
    parse_err = int((table["parse_error"] == True).sum()) if "parse_error" in table.columns else 0

    # time range (ctime may be str)
    ctime = pd.to_datetime(table["ctime"], errors="coerce")
    tmin = ctime.min()
    tmax = ctime.max()

    overview = pd.DataFrame([
        ["total_sentences", total_sent],
        ["total_docs", total_docs],
        ["parse_error_sentences", parse_err],
        ["time_min", "" if pd.isna(tmin) else tmin.isoformat()],
        ["time_max", "" if pd.isna(tmax) else tmax.isoformat()],
    ], columns=["metric", "value"])

    # platform breakdown
    plat = (
        table["platform"].fillna("UNKNOWN")
        .value_counts()
        .rename_axis("platform")
        .reset_index(name="sentences")
    )

    # samples: 抽样一些句子（先做一个通用抽样，后面接入 aspect/sentiment 再细分抽样）
    samples = table.loc[:, ["sentence", "ctime", "platform", "url", "doc_id"]].dropna(subset=["sentence"])
    samples = samples.sample(n=min(200, len(samples)), random_state=42) if len(samples) > 0 else samples

    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        overview.to_excel(w, sheet_name="overview", index=False)
        plat.to_excel(w, sheet_name="platform_breakdown", index=False)
        samples.to_excel(w, sheet_name="samples", index=False)

def main():
    # 用 dataset 方式避免一次性加载全表
    dataset = ds.dataset(str(INPUT), format="parquet")

    # 先取所有 (brand, model) 组合（这是小表）
    pairs = dataset.to_table(columns=["brand", "model"]).to_pandas().drop_duplicates()

    for _, row in pairs.iterrows():
        brand = row["brand"] if pd.notna(row["brand"]) else "UNKNOWN"
        model = row["model"] if pd.notna(row["model"]) else "UNKNOWN"

        # 过滤读取：只读该产品的必要列
        filt = (ds.field("brand") == brand) & (ds.field("model") == model)
        cols = ["brand","model","doc_id","platform","url","ctime","sentence","parse_error"]
        table = dataset.to_table(filter=filt, columns=cols).to_pandas()

        out_path = OUT_DIR / safe_name(brand) / f"{safe_name(model)}.xlsx"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        export_one_product(table, out_path)

    # 生成一个总索引（方便你快速查）
    index_path = OUT_DIR / "INDEX.xlsx"
    idx = pairs.copy()
    idx["report_path"] = idx.apply(lambda r: str((OUT_DIR / safe_name(r["brand"]) / f"{safe_name(r['model'])}.xlsx").resolve()), axis=1)
    idx.to_excel(index_path, index=False)
    print("[OK] Reports written to:", OUT_DIR)

if __name__ == "__main__":
    main()

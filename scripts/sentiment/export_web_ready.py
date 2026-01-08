from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

SCHEMA = pa.schema(
    [
        pa.field("domain", pa.string()),
        pa.field("product_id", pa.string()),
        pa.field("brand", pa.string()),
        pa.field("model", pa.string()),
        pa.field("l1", pa.string()),
        pa.field("sentiment", pa.string()),
        pa.field("day", pa.date32()),
        pa.field("weight", pa.int64()),
    ],
    metadata={"schema": "absa.sentiment.web_ready.v1"},
)

READ_COLUMNS = ["domain", "brand", "model", "aspect_l1", "pred_label", "ctime"]
OUTPUT_COLUMNS = [field.name for field in SCHEMA]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_sentiment(value: object | None) -> str:
    if value is None:
        return "NEU"
    candidate = str(value).strip().upper()
    if not candidate:
        return "NEU"
    if candidate in {"POS", "POSITIVE", "+", "2"}:
        return "POS"
    if candidate in {"NEG", "NEGATIVE", "-", "0"}:
        return "NEG"
    if candidate in {"NEU", "NEUTRAL", "1"}:
        return "NEU"
    # fallback: try numeric mapping
    if candidate.isdigit():
        mapping = {"0": "NEG", "1": "NEU", "2": "POS"}
        return mapping.get(candidate, "NEU")
    return "NEU"


def clean_text(series: pd.Series) -> pd.Series:
    cleaned = series.where(series.notna(), "")
    stripped = cleaned.astype("string").str.strip()
    return stripped.replace("", pd.NA)


def build_product_id(brand: pd.Series, model: pd.Series) -> pd.Series:
    product_id = pd.Series(index=brand.index, dtype="string")
    both = brand.notna() & model.notna()
    product_id[both] = brand[both] + "__" + model[both]
    brand_only = brand.notna() & model.isna()
    product_id[brand_only] = brand[brand_only]
    model_only = brand.isna() & model.notna()
    product_id[model_only] = model[model_only]
    product_id = product_id.fillna("unknown_product")
    return product_id


def determine_step04_root(
    workspace_root: Path, domain: str, run_path: Path | None
) -> Path:
    if run_path:
        return run_path
    runs_dir = workspace_root / domain / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs directory not found: {runs_dir}")
    runs = sorted([entry for entry in runs_dir.iterdir() if entry.is_dir()], key=lambda p: p.name)
    if not runs:
        raise FileNotFoundError(f"no run directories under {runs_dir}")
    return runs[-1] / "step04_pred"


def transform_batch(batch: pd.DataFrame) -> pd.DataFrame:
    if batch.empty:
        return batch
    frame = batch.rename(
        columns={
            "aspect_l1": "l1",
            "pred_label": "sentiment",
        }
    )
    for expected in ("l1", "sentiment"):
        if expected not in frame.columns:
            frame[expected] = pd.NA
    frame["brand"] = clean_text(frame.get("brand", pd.Series(dtype="object")))
    frame["model"] = clean_text(frame.get("model", pd.Series(dtype="object")))
    frame["product_id"] = build_product_id(frame["brand"], frame["model"])
    frame["domain"] = frame.get("domain", pd.NA).where(frame["domain"].notna(), "unknown_domain")
    frame["domain"] = frame["domain"].astype("string")
    frame["l1"] = frame["l1"].astype("string")
    frame["sentiment"] = frame["sentiment"].apply(normalize_sentiment).astype("string")
    # ctime is epoch seconds stored as string (e.g. "1766475000")
    ctime_raw = frame.get("ctime")

    # 1) coerce to numeric; non-numeric become NaN
    ctime_num = pd.to_numeric(ctime_raw, errors="coerce")

    # 2) parse as unix epoch seconds (UTC)
    ctime = pd.to_datetime(ctime_num, errors="coerce", unit="s", utc=True)

    # 3) day derived from UTC date
    day = ctime.dt.date

    frame["day"] = ctime.dt.tz_localize(None).dt.normalize().dt.date
    if "weight" in frame.columns:
        frame["weight"] = (
            pd.to_numeric(frame["weight"], errors="coerce")
            .fillna(1)
            .astype("int64")
        )
    else:
        frame["weight"] = 1
    ordered = frame[OUTPUT_COLUMNS].copy()
    ordered["brand"] = frame["brand"]
    ordered["model"] = frame["model"]
    ordered["day"] = frame["day"]
    ordered["weight"] = frame["weight"]
    return ordered


def create_empty_table() -> pa.Table:
    arrays = [pa.array([], type=field.type) for field in SCHEMA]
    return pa.Table.from_arrays(arrays, names=[field.name for field in SCHEMA])


def main() -> int:
    parser = argparse.ArgumentParser(description="Build web_ready.parquet from RouteB Step04 output")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--workspace-root", default=r"E:\ABSA_WORKSPACE\outputs")
    parser.add_argument("--step04-path", type=Path, help="Explicit path to step04_pred dataset")
    parser.add_argument("--output", type=Path, help="Target path for web_ready.parquet")
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root)
    domain = args.domain
    step04_root = determine_step04_root(
        workspace_root=workspace_root,
        domain=domain,
        run_path=args.step04_path,
    )
    if not step04_root.exists():
        raise FileNotFoundError(f"step04_pred path does not exist: {step04_root}")
    parquet_files = sorted(step04_root.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no parquet files found under: {step04_root}")

    dataset = ds.dataset([str(p) for p in parquet_files], format="parquet")

    scanner = dataset.scanner(columns=READ_COLUMNS, use_threads=True)

    output_path = args.output or workspace_root / domain / "sentiment" / "web_ready.parquet"
    ensure_dir(output_path.parent)

    rows_written = 0
    writer = pq.ParquetWriter(output_path, schema=SCHEMA)
    try:
        for batch in scanner.to_batches():
            df = batch.to_pandas()
            transformed = transform_batch(df)
            if transformed.empty:
                continue
            table = pa.Table.from_pandas(transformed, schema=SCHEMA, safe=False)
            writer.write_table(table)
            rows_written += len(transformed)
    finally:
        if rows_written == 0:
            writer.write_table(create_empty_table())
        writer.close()

    print(f"[DONE] domain={domain} rows={rows_written} out={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

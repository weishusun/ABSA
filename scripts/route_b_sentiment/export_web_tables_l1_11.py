# scripts/route_b_sentiment/export_web_tables_l1_11.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import duckdb
import pandas as pd


def p2duck(p: Path) -> str:
    return p.resolve().as_posix()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_df(df: pd.DataFrame, out_parquet: Path, out_csv: Path) -> None:
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")


def build_preds_raw(con: duckdb.DuckDBPyConnection, pred_ds: Path) -> str:
    base = p2duck(pred_ds)
    patterns = [
        f"{base}/shard=*/part-*.parquet",  # your pred_full
        f"{base}/shard=*/pred.parquet",    # fallback
        f"{base}/*.parquet",
    ]
    last_err = None
    for pat in patterns:
        try:
            con.execute(f"CREATE OR REPLACE TEMP VIEW preds_raw AS SELECT * FROM read_parquet('{pat}');")
            con.execute("SELECT * FROM preds_raw LIMIT 0").df()
            return pat
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Cannot read parquet under pred_ds={pred_ds}. Last error={last_err!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-ds", required=True, help="pred_full dir, contains shard=*/part-*.parquet")
    ap.add_argument("--out-dir", required=True, help="output folder for web tables")
    ap.add_argument("--conf-threshold", type=float, default=0.80, help="hard stats threshold (confidence>=t)")
    ap.add_argument("--l1-n", type=int, default=11, help="number of L1 segments for pie (default 11)")
    ap.add_argument("--time-col", default="ctime", help="time column name in pred_ds (default ctime)")
    ap.add_argument("--use-soft", action="store_true", help="also export soft-weighted time series (optional)")
    args = ap.parse_args()

    pred_ds = Path(args.pred_ds)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    out_7d = out_dir / "last7d_day"
    out_1m = out_dir / "last1m_week"
    ensure_dir(out_7d)
    ensure_dir(out_1m)

    t = float(args.conf_threshold)
    l1_n = int(args.l1_n)

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=8;")

    used_pattern = build_preds_raw(con, pred_ds)
    cols = con.execute("SELECT * FROM preds_raw LIMIT 0").df().columns.tolist()
    colset = set(cols)

    # --- label/conf fallback ---
    label_expr = """
      coalesce(
        pred_label,
        CASE
          WHEN pred_id=0 THEN 'NEG'
          WHEN pred_id=1 THEN 'NEU'
          WHEN pred_id=2 THEN 'POS'
          ELSE NULL
        END
      )
    """
    conf_expr = "coalesce(confidence, greatest(p_neg, p_neu, p_pos))"

    # --- time parse: robust + binder-safe ---
    time_col = args.time_col
    if time_col not in colset:
        raise SystemExit(f"[FATAL] time_col={time_col} not found in preds_raw columns.")

    ts_parse_expr = f"""
    CASE
      WHEN {time_col} IS NULL THEN NULL
      ELSE
        coalesce(
          -- epoch numeric or numeric string
          CASE
            WHEN try_cast({time_col} AS BIGINT) IS NOT NULL THEN
              CASE
                WHEN try_cast({time_col} AS BIGINT) > 100000000000 THEN
                  to_timestamp(try_cast({time_col} AS DOUBLE) / 1000.0)  -- ms
                ELSE
                  to_timestamp(try_cast({time_col} AS DOUBLE))           -- sec
              END
            ELSE NULL
          END,
          -- timestamp/date strings
          try_cast(CAST({time_col} AS VARCHAR) AS TIMESTAMP),
          try_strptime(CAST({time_col} AS VARCHAR), '%Y-%m-%d %H:%M:%S'),
          try_strptime(CAST({time_col} AS VARCHAR), '%Y/%m/%d %H:%M:%S'),
          try_strptime(CAST({time_col} AS VARCHAR), '%Y-%m-%d'),
          try_strptime(CAST({time_col} AS VARCHAR), '%Y/%m/%d')
        )
    END
    """

    # Only columns we need
    must_cols = ["brand", "model", "aspect_l1", "pred_label", "pred_id", "p_pos", "p_neu", "p_neg", "confidence"]
    missing = [c for c in ["brand", "model", "aspect_l1", "aspect_l2", "p_pos", "p_neu", "p_neg"] if c not in colset]
    if missing:
        raise SystemExit(f"[FATAL] preds_raw missing required columns: {missing}")

    # Base view with dt (DATE), keep tz-unaware
    con.execute(f"""
      CREATE OR REPLACE TEMP VIEW base AS
      SELECT
        brand,
        model,
        aspect_l1,
        aspect_l2,
        {label_expr} AS label,
        p_pos, p_neu, p_neg,
        {conf_expr} AS confidence,
        CAST(({ts_parse_expr}) AS DATE) AS dt
      FROM preds_raw
      WHERE ({ts_parse_expr}) IS NOT NULL;
    """)

    # Hard view (confidence>=t)
    con.execute(f"""
      CREATE OR REPLACE TEMP VIEW hard AS
      SELECT * FROM base WHERE confidence >= {t};
    """)

    # Determine max date (use hard or base? use base to avoid losing days)
    max_dt = con.execute("SELECT max(dt) FROM base;").fetchone()[0]
    if max_dt is None:
        raise SystemExit("[FATAL] dt is NULL for all rows after parsing ctime.")

    # Window starts (relative to max_dt in your 30d dataset)
    # 7 days: max_dt-6..max_dt
    # 30 days: max_dt-29..max_dt
    start_7d = con.execute("SELECT (CAST(? AS DATE) - INTERVAL 6 DAY)::DATE;", [max_dt]).fetchone()[0]
    start_30d = con.execute("SELECT (CAST(? AS DATE) - INTERVAL 29 DAY)::DATE;", [max_dt]).fetchone()[0]

    # --- Build L1 mapping into exactly 11 segments (Top10 + 其他 if needed) ---
    l1_cnt = con.execute(f"""
      SELECT aspect_l1, COUNT(*) AS cnt
      FROM hard
      WHERE dt BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
      GROUP BY 1
      ORDER BY cnt DESC;
    """, [start_30d, max_dt]).df()

    distinct_l1 = l1_cnt["aspect_l1"].tolist()
    if len(distinct_l1) <= l1_n:
        keep = distinct_l1
        use_other = False
    else:
        # Top (l1_n-1) + 其他 = l1_n segments
        keep = distinct_l1[: max(1, l1_n - 1)]
        use_other = True

    map_rows: List[Tuple[str, str]] = []
    keep_set = set(keep)
    for l1 in distinct_l1:
        if l1 in keep_set:
            map_rows.append((l1, l1))
        else:
            map_rows.append((l1, "其他"))

    l1_map_df = pd.DataFrame(map_rows, columns=["aspect_l1_raw", "aspect_l1_11"]).drop_duplicates()
    con.register("l1_map_df", l1_map_df)
    con.execute("CREATE OR REPLACE TEMP VIEW l1_map AS SELECT * FROM l1_map_df;")

    # Apply mapping
    con.execute("""
      CREATE OR REPLACE TEMP VIEW hard_m AS
      SELECT
        h.brand, h.model,
        coalesce(m.aspect_l1_11, h.aspect_l1) AS aspect_l1_11,
        h.label,
        h.dt,
        h.p_pos, h.p_neu, h.p_neg
      FROM hard h
      LEFT JOIN l1_map m ON h.aspect_l1 = m.aspect_l1_raw;
    """)

    if args.use_soft:
        con.execute("""
          CREATE OR REPLACE TEMP VIEW base_m AS
          SELECT
            b.brand, b.model,
            coalesce(m.aspect_l1_11, b.aspect_l1) AS aspect_l1_11,
            b.label,
            b.dt,
            b.p_pos, b.p_neu, b.p_neg
          FROM base b
          LEFT JOIN l1_map m ON b.aspect_l1 = m.aspect_l1_raw;
        """)

    # Export metadata + mapping
    meta = {
        "pred_pattern": used_pattern,
        "max_dt": str(max_dt),
        "start_7d": str(start_7d),
        "start_30d": str(start_30d),
        "conf_threshold": t,
        "l1_n": l1_n,
        "l1_distinct_in_30d_hard": int(len(distinct_l1)),
        "l1_use_other": bool(use_other),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    write_df(l1_map_df, out_dir / "l1_mapping.parquet", out_dir / "l1_mapping.csv")

    # Helper SQL snippets
    def export_pie(folder: Path, start_dt: str, end_dt: str, tag: str) -> None:
        # POS pie
        df_pos = con.execute(f"""
          WITH agg AS (
            SELECT
              brand, model, aspect_l1_11,
              COUNT(*) AS cnt
            FROM hard_m
            WHERE dt BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
              AND label='POS'
            GROUP BY 1,2,3
          )
          SELECT
            brand, model, aspect_l1_11, cnt,
            cnt * 1.0 / NULLIF(SUM(cnt) OVER (PARTITION BY brand, model), 0) AS share
          FROM agg
          ORDER BY brand, model, cnt DESC;
        """, [start_dt, end_dt]).df()

        df_neg = con.execute(f"""
          WITH agg AS (
            SELECT
              brand, model, aspect_l1_11,
              COUNT(*) AS cnt
            FROM hard_m
            WHERE dt BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
              AND label='NEG'
            GROUP BY 1,2,3
          )
          SELECT
            brand, model, aspect_l1_11, cnt,
            cnt * 1.0 / NULLIF(SUM(cnt) OVER (PARTITION BY brand, model), 0) AS share
          FROM agg
          ORDER BY brand, model, cnt DESC;
        """, [start_dt, end_dt]).df()

        # all-products pie (optional but useful)
        df_all_pos = con.execute(f"""
          WITH agg AS (
            SELECT aspect_l1_11, COUNT(*) AS cnt
            FROM hard_m
            WHERE dt BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
              AND label='POS'
            GROUP BY 1
          )
          SELECT
            aspect_l1_11, cnt,
            cnt * 1.0 / NULLIF(SUM(cnt) OVER (), 0) AS share
          FROM agg
          ORDER BY cnt DESC;
        """, [start_dt, end_dt]).df()

        df_all_neg = con.execute(f"""
          WITH agg AS (
            SELECT aspect_l1_11, COUNT(*) AS cnt
            FROM hard_m
            WHERE dt BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
              AND label='NEG'
            GROUP BY 1
          )
          SELECT
            aspect_l1_11, cnt,
            cnt * 1.0 / NULLIF(SUM(cnt) OVER (), 0) AS share
          FROM agg
          ORDER BY cnt DESC;
        """, [start_dt, end_dt]).df()

        write_df(df_pos, folder / f"pie_product_pos_{tag}.parquet", folder / f"pie_product_pos_{tag}.csv")
        write_df(df_neg, folder / f"pie_product_neg_{tag}.parquet", folder / f"pie_product_neg_{tag}.csv")
        write_df(df_all_pos, folder / f"pie_all_pos_{tag}.parquet", folder / f"pie_all_pos_{tag}.csv")
        write_df(df_all_neg, folder / f"pie_all_neg_{tag}.parquet", folder / f"pie_all_neg_{tag}.csv")

    def export_counts(folder: Path, start_dt: str, end_dt: str, tag: str) -> None:
        # All products, by L1_11 and sentiment (POS/NEU/NEG) counts
        df_overall = con.execute(f"""
          SELECT
            aspect_l1_11,
            SUM(CASE WHEN label='POS' THEN 1 ELSE 0 END) AS pos_cnt,
            SUM(CASE WHEN label='NEG' THEN 1 ELSE 0 END) AS neg_cnt,
            SUM(CASE WHEN label='NEU' THEN 1 ELSE 0 END) AS neu_cnt,
            COUNT(*) AS total_cnt
          FROM hard_m
          WHERE dt BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
          GROUP BY 1
          ORDER BY total_cnt DESC;
        """, [start_dt, end_dt]).df()

        # Products dimension (for dropdown)
        df_products = con.execute(f"""
          SELECT
            brand, model,
            COUNT(*) AS total_cnt,
            SUM(CASE WHEN label='POS' THEN 1 ELSE 0 END) AS pos_cnt,
            SUM(CASE WHEN label='NEG' THEN 1 ELSE 0 END) AS neg_cnt,
            SUM(CASE WHEN label='NEU' THEN 1 ELSE 0 END) AS neu_cnt
          FROM hard_m
          WHERE dt BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
          GROUP BY 1,2
          ORDER BY total_cnt DESC;
        """, [start_dt, end_dt]).df()

        # L1 dimension (for dropdown order)
        df_l1 = con.execute(f"""
          SELECT aspect_l1_11, COUNT(*) AS total_cnt
          FROM hard_m
          WHERE dt BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
          GROUP BY 1
          ORDER BY total_cnt DESC;
        """, [start_dt, end_dt]).df()

        write_df(df_overall, folder / f"overall_l1_sentiment_{tag}.parquet", folder / f"overall_l1_sentiment_{tag}.csv")
        write_df(df_products, folder / f"products_{tag}.parquet", folder / f"products_{tag}.csv")
        write_df(df_l1, folder / f"l1_list_{tag}.parquet", folder / f"l1_list_{tag}.csv")

    def export_timeseries_day_7d(folder: Path, start_dt: str, end_dt: str, tag: str) -> None:
        # Product x L1 x day => pos/neg/neu counts (hard)
        df_ts_prod = con.execute(f"""
          SELECT
            dt AS time_bucket,
            brand, model,
            aspect_l1_11,
            SUM(CASE WHEN label='POS' THEN 1 ELSE 0 END) AS pos_cnt,
            SUM(CASE WHEN label='NEG' THEN 1 ELSE 0 END) AS neg_cnt,
            SUM(CASE WHEN label='NEU' THEN 1 ELSE 0 END) AS neu_cnt,
            COUNT(*) AS total_cnt
          FROM hard_m
          WHERE dt BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
          GROUP BY 1,2,3,4
          ORDER BY time_bucket, brand, model, aspect_l1_11;
        """, [start_dt, end_dt]).df()

        # All products (no brand/model) for global chart
        df_ts_all = con.execute(f"""
          SELECT
            dt AS time_bucket,
            aspect_l1_11,
            SUM(CASE WHEN label='POS' THEN 1 ELSE 0 END) AS pos_cnt,
            SUM(CASE WHEN label='NEG' THEN 1 ELSE 0 END) AS neg_cnt,
            SUM(CASE WHEN label='NEU' THEN 1 ELSE 0 END) AS neu_cnt,
            COUNT(*) AS total_cnt
          FROM hard_m
          WHERE dt BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
          GROUP BY 1,2
          ORDER BY time_bucket, aspect_l1_11;
        """, [start_dt, end_dt]).df()

        write_df(df_ts_prod, folder / f"ts_product_l1_day_{tag}.parquet", folder / f"ts_product_l1_day_{tag}.csv")
        write_df(df_ts_all, folder / f"ts_all_l1_day_{tag}.parquet", folder / f"ts_all_l1_day_{tag}.csv")

    def export_timeseries_week_1m(folder: Path, start_dt: str, end_dt: str, tag: str) -> None:
        # week bucket (DATE) - start of week
        df_ts_prod = con.execute(f"""
          SELECT
            CAST(date_trunc('week', CAST(dt AS TIMESTAMP)) AS DATE) AS time_bucket,
            brand, model,
            aspect_l1_11,
            SUM(CASE WHEN label='POS' THEN 1 ELSE 0 END) AS pos_cnt,
            SUM(CASE WHEN label='NEG' THEN 1 ELSE 0 END) AS neg_cnt,
            SUM(CASE WHEN label='NEU' THEN 1 ELSE 0 END) AS neu_cnt,
            COUNT(*) AS total_cnt
          FROM hard_m
          WHERE dt BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
          GROUP BY 1,2,3,4
          ORDER BY time_bucket, brand, model, aspect_l1_11;
        """, [start_dt, end_dt]).df()

        df_ts_all = con.execute(f"""
          SELECT
            CAST(date_trunc('week', CAST(dt AS TIMESTAMP)) AS DATE) AS time_bucket,
            aspect_l1_11,
            SUM(CASE WHEN label='POS' THEN 1 ELSE 0 END) AS pos_cnt,
            SUM(CASE WHEN label='NEG' THEN 1 ELSE 0 END) AS neg_cnt,
            SUM(CASE WHEN label='NEU' THEN 1 ELSE 0 END) AS neu_cnt,
            COUNT(*) AS total_cnt
          FROM hard_m
          WHERE dt BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
          GROUP BY 1,2
          ORDER BY time_bucket, aspect_l1_11;
        """, [start_dt, end_dt]).df()

        write_df(df_ts_prod, folder / f"ts_product_l1_week_{tag}.parquet", folder / f"ts_product_l1_week_{tag}.csv")
        write_df(df_ts_all, folder / f"ts_all_l1_week_{tag}.parquet", folder / f"ts_all_l1_week_{tag}.csv")

    # --- Export 7d/day ---
    tag7 = f"{str(start_7d)}_to_{str(max_dt)}_t{t:.2f}".replace(".", "p")
    export_pie(out_7d, str(start_7d), str(max_dt), "7d")
    export_counts(out_7d, str(start_7d), str(max_dt), "7d")
    export_timeseries_day_7d(out_7d, str(start_7d), str(max_dt), "7d")

    # --- Export 1m/week (30 days, weekly buckets) ---
    tag30 = f"{str(start_30d)}_to_{str(max_dt)}_t{t:.2f}".replace(".", "p")
    export_pie(out_1m, str(start_30d), str(max_dt), "1m")
    export_counts(out_1m, str(start_30d), str(max_dt), "1m")
    export_timeseries_week_1m(out_1m, str(start_30d), str(max_dt), "1m")

    # Optional: soft weighted time series (global + product) — only if you need smoother curves
    if args.use_soft:
        # 7d day soft (weights)
        df_soft_7d = con.execute("""
          SELECT
            dt AS time_bucket,
            brand, model,
            aspect_l1_11,
            SUM(p_pos) AS w_pos,
            SUM(p_neg) AS w_neg,
            SUM(p_neu) AS w_neu
          FROM base_m
          WHERE dt BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
          GROUP BY 1,2,3,4
          ORDER BY time_bucket, brand, model, aspect_l1_11;
        """, [start_7d, max_dt]).df()
        write_df(df_soft_7d, out_7d / "ts_product_l1_day_soft_7d.parquet", out_7d / "ts_product_l1_day_soft_7d.csv")

        # 1m week soft
        df_soft_1m = con.execute("""
          SELECT
            CAST(date_trunc('week', CAST(dt AS TIMESTAMP)) AS DATE) AS time_bucket,
            brand, model,
            aspect_l1_11,
            SUM(p_pos) AS w_pos,
            SUM(p_neg) AS w_neg,
            SUM(p_neu) AS w_neu
          FROM base_m
          WHERE dt BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
          GROUP BY 1,2,3,4
          ORDER BY time_bucket, brand, model, aspect_l1_11;
        """, [start_30d, max_dt]).df()
        write_df(df_soft_1m, out_1m / "ts_product_l1_week_soft_1m.parquet", out_1m / "ts_product_l1_week_soft_1m.csv")

    print("[DONE] Exported web tables.")
    print(f"[INFO] pred_pattern={used_pattern}")
    print(f"[INFO] max_dt={max_dt} start_7d={start_7d} start_30d={start_30d} conf_t={t} l1_n={l1_n}")
    print(f"[OUT] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

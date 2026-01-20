# scripts/route_b_sentiment/sentiment_05_aggregate_and_build_excels.py
# -*- coding: utf-8 -*-
# 说明：汇总 Excel 支持 --excel-name/--domain，默认 aspect_sentiment_counts_{domain}.xlsx；存在同名文件且未 --overwrite 会直接 FATAL 退出。

import argparse
from pathlib import Path
import sys
import re

import duckdb
import pandas as pd


def p2duck(p: Path) -> str:
    """WindowsPath -> POSIX path for duckdb, avoids Python 3.11 f-string backslash issues."""
    return p.resolve().as_posix()


def _excel_safe_write(df: pd.DataFrame, writer: pd.ExcelWriter, sheet: str, max_rows: int = 1_000_000):
    if len(df) > max_rows:
        # Excel 行数限制 1,048,576；这里留出余量
        print(f"[WARN] Sheet {sheet} rows={len(df)} exceeds Excel limit; skip writing this sheet.")
        return
    df.to_excel(writer, index=False, sheet_name=sheet)


def _try_create_preds_raw(con: duckdb.DuckDBPyConnection, pred_ds: Path) -> str:
    """
    Try multiple common patterns to build preds_raw view.
    Returns the pattern used (for logging).
    """
    base = p2duck(pred_ds)

    patterns = [
        f"{base}/shard=*/part-*.parquet",  # your pred_full
        f"{base}/shard=*/pred.parquet",    # old style
        f"{base}/*.parquet",
    ]

    last_err = None
    for pat in patterns:
        try:
            con.execute(f"""
                CREATE OR REPLACE TEMP VIEW preds_raw AS
                SELECT * FROM read_parquet('{pat}');
            """)
            # force schema resolution
            _ = con.execute("SELECT * FROM preds_raw LIMIT 0").df()
            print(f"[OK] pred pattern matched: {pat}")
            return pat
        except Exception as e:
            last_err = e

    # fallback: recursive scan and pass file list into read_parquet([...])
    files = sorted(pred_ds.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {pred_ds}")

    # Build SQL list with proper escaping
    file_list_sql = "[" + ",".join(duckdb.escape_literal(p2duck(f)) for f in files) + "]"
    try:
        con.execute(f"""
            CREATE OR REPLACE TEMP VIEW preds_raw AS
            SELECT * FROM read_parquet({file_list_sql});
        """)
        _ = con.execute("SELECT * FROM preds_raw LIMIT 0").df()
        print(f"[OK] pred files matched via recursive list: n_files={len(files)}")
        return "recursive_file_list"
    except Exception as e:
        if last_err is not None:
            print("[ERROR] glob patterns failed with last error:", repr(last_err), file=sys.stderr)
        raise e


def infer_domain(args_domain: str, pred_ds: Path, out_dir: Path, con: duckdb.DuckDBPyConnection) -> str:
    if args_domain:
        return args_domain
    for p in (pred_ds, out_dir):
        m = re.search(r"outputs[/\\\\]([^/\\\\]+)", str(p))
        if m:
            return m.group(1)
    try:
        sample = con.execute("SELECT domain FROM preds WHERE domain IS NOT NULL LIMIT 1").fetchone()
        if sample and sample[0]:
            return str(sample[0])
    except Exception:
        pass
    return "domain"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-ds", required=True, help="pred_full dir (partitioned shard=* with part-*.parquet)")
    ap.add_argument("--pairs-parquet", required=False, default="",
                    help="(optional) aspect_pairs.parquet for url/sentence/ctime补齐；如 pred_ds 已带这些列，可不提供")
    ap.add_argument("--out-dir", required=True, help="output base dir (e.g., outputs/phone_v2/sentiment/runs/... )")
    ap.add_argument("--by-product-dir", default="", help="outputs/.../by_product (optional)")
    ap.add_argument("--build-by-product", action="store_true", help="是否重建按产品 Excel（可选，可能较慢）")
    ap.add_argument("--excel-name", default="", help="可选，指定汇总 Excel 文件名；默认按 domain 命名")
    ap.add_argument("--domain", default="", help="可选，显式指定 domain，用于默认 Excel 命名")
    ap.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的输出文件")

    # 口径参数
    ap.add_argument("--conf-threshold", type=float, default=0.80, help="hard 口径阈值：confidence>=t")
    ap.add_argument("--time-grain", default="day", choices=["none", "day", "week", "month"],
                    help="时间分桶粒度。none=不输出时间分桶；day/week/month=输出时间维度聚合表。")
    ap.add_argument("--time-col", default="ctime", help="时间列名（预测表通常为 ctime）")

    # 例句
    ap.add_argument("--make-examples", action="store_true")
    ap.add_argument("--examples-min-confidence", type=float, default=-1.0,
                    help="例句最小 confidence（默认=-1 表示不筛；建议设为与 --conf-threshold 相同）")
    ap.add_argument("--pos-topk", type=int, default=3)
    ap.add_argument("--neg-topk", type=int, default=3)

    # 调试
    ap.add_argument("--max-products", type=int, default=0, help="0=全部；调试用")
    args = ap.parse_args()

    pred_ds = Path(args.pred_ds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_dir = Path(args.by_product_dir) if args.by_product_dir else None
    if args.build_by_product:
        if by_dir is None:
            raise SystemExit("[FATAL] 你开启了 --build-by-product，但未提供 --by-product-dir")
        by_dir.mkdir(parents=True, exist_ok=True)

    # 输出目录
    sentiment_dir = out_dir / "sentiment"
    sentiment_dir.mkdir(parents=True, exist_ok=True)

    # 统一输出命名：all / hard / soft
    agg_all_parquet = sentiment_dir / "aspect_sentiment_agg.parquet"  # 保持旧名：all
    agg_hard_parquet = sentiment_dir / f"aspect_sentiment_agg_hard_t{args.conf_threshold:.2f}.parquet".replace(".", "p")
    agg_soft_parquet = sentiment_dir / "aspect_sentiment_agg_soft.parquet"

    # 时间序列
    ts_all_parquet = sentiment_dir / "aspect_sentiment_timeseries.parquet"       # all 宽表
    ts_all_long_parquet = sentiment_dir / "aspect_sentiment_timeseries_long.parquet"  # all 长表

    ts_hard_parquet = sentiment_dir / f"aspect_sentiment_timeseries_hard_t{args.conf_threshold:.2f}.parquet".replace(".", "p")
    ts_hard_long_parquet = sentiment_dir / f"aspect_sentiment_timeseries_long_hard_t{args.conf_threshold:.2f}.parquet".replace(".", "p")

    ts_soft_parquet = sentiment_dir / "aspect_sentiment_timeseries_soft.parquet"
    ts_soft_long_parquet = sentiment_dir / "aspect_sentiment_timeseries_long_soft.parquet"

    ts_xlsx = out_dir / "aspect_sentiment_timeseries_phone.xlsx"

    examples_parquet = sentiment_dir / "examples.parquet"

    # duckdb
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=8;")

    # ========== 1) 读取 preds_raw ==========
    used_pattern = _try_create_preds_raw(con, pred_ds)

    pred_cols = con.execute("SELECT * FROM preds_raw LIMIT 0").df().columns.tolist()
    pred_cols_set = set(pred_cols)

    # optional pairs join
    use_pairs = False
    pairs_path = None
    if args.pairs_parquet:
        pairs_parquet = Path(args.pairs_parquet)
        if pairs_parquet.exists():
            use_pairs = True
            pairs_path = p2duck(pairs_parquet)
        else:
            print(f"[WARN] pairs-parquet not found: {pairs_parquet} ; ignore.")

    # Detect pair_id availability
    has_pair_id = "pair_id" in pred_cols_set

    # ========== 2) 标准化 preds 视图（统一字段）==========
    # label 兜底：优先 pred_label，否则 pred_id 映射
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

    # url 字段可能不存在
    url_expr = "url" if "url" in pred_cols_set else "NULL::VARCHAR AS url"

    # time col
    time_col = args.time_col
    if time_col not in pred_cols_set:
        # allow missing time col -> NULL
        print(f"[WARN] time-col '{time_col}' not found in preds_raw; ctime will be NULL.")
        time_expr = "NULL::VARCHAR AS ctime"
    else:
        time_expr = f"{time_col} AS ctime"

    # sentence 可能存在（你已验证有）
    sentence_expr = "sentence" if "sentence" in pred_cols_set else "NULL::VARCHAR AS sentence"

    # 基础字段应存在：domain/brand/model/aspect_l1/aspect_l2
    # 若缺失，则设 NULL，避免 SQL 失败（但业务上应当有）
    def col_or_null(name: str, typ: str = "VARCHAR") -> str:
        return name if name in pred_cols_set else f"NULL::{typ} AS {name}"

    domain_expr = col_or_null("domain", "VARCHAR")
    brand_expr = col_or_null("brand", "VARCHAR")
    model_expr = col_or_null("model", "VARCHAR")
    a1_expr = col_or_null("aspect_l1", "VARCHAR")
    a2_expr = col_or_null("aspect_l2", "VARCHAR")

    # 概率字段
    # 你的列是 p_neg/p_neu/p_pos，都存在
    if not {"p_neg", "p_neu", "p_pos"}.issubset(pred_cols_set):
        raise SystemExit("[FATAL] preds_raw missing one of p_neg/p_neu/p_pos; cannot aggregate soft/hard reliably.")

    # 如果提供 pairs 且 pred 有 pair_id，则 join 补齐 sentence/url/ctime（必要时）
    if use_pairs and has_pair_id:
        con.execute(f"""
            CREATE OR REPLACE TEMP VIEW pairs AS
            SELECT
                pair_id,
                domain,
                brand,
                model,
                aspect_l1,
                aspect_l2,
                sentence,
                url,
                {args.time_col} AS ctime
            FROM read_parquet('{pairs_path}');
        """)
        con.execute(f"""
            CREATE OR REPLACE TEMP VIEW preds AS
            SELECT
                coalesce(pr.domain, pa.domain) AS domain,
                coalesce(pr.brand,  pa.brand)  AS brand,
                coalesce(pr.model,  pa.model)  AS model,
                coalesce(pr.aspect_l1, pa.aspect_l1) AS aspect_l1,
                coalesce(pr.aspect_l2, pa.aspect_l2) AS aspect_l2,
                {label_expr} AS pred_label,
                pr.p_pos, pr.p_neu, pr.p_neg,
                {conf_expr} AS confidence,
                coalesce(pr.{args.time_col}, pa.ctime) AS ctime,
                coalesce(pr.sentence, pa.sentence) AS sentence,
                coalesce(pr.url, pa.url) AS url
            FROM preds_raw pr
            LEFT JOIN pairs pa USING(pair_id);
        """)
        print("[OK] preds view built via pair_id join (pairs_parquet applied).")
    else:
        if use_pairs and not has_pair_id:
            print("[WARN] pairs-parquet provided but preds_raw has no pair_id; skip join, use preds_raw directly.")
        con.execute(f"""
            CREATE OR REPLACE TEMP VIEW preds AS
            SELECT
                {domain_expr} AS domain,
                {brand_expr}  AS brand,
                {model_expr}  AS model,
                {a1_expr}     AS aspect_l1,
                {a2_expr}     AS aspect_l2,
                {label_expr}  AS pred_label,
                p_pos, p_neu, p_neg,
                {conf_expr}   AS confidence,
                {time_expr},
                {sentence_expr} AS sentence,
                {url_expr}
            FROM preds_raw;
        """)
        print("[OK] preds view built directly from preds_raw (no join).")

    domain_val = infer_domain(args.domain, pred_ds, out_dir, con)
    excel_name = args.excel_name.strip() or f"aspect_sentiment_counts_{domain_val}.xlsx"
    agg_xlsx = out_dir / excel_name
    if agg_xlsx.exists() and not args.overwrite:
        print(f"[FATAL] Excel 输出已存在且未指定 --overwrite: {agg_xlsx}", file=sys.stderr)
        raise SystemExit(2)

    # ========== 3) all 聚合（不带时间）==========
    con.execute(f"""
      COPY (
        SELECT
          domain, brand, model, aspect_l1, aspect_l2,
          SUM(CASE WHEN pred_label='POS' THEN 1 ELSE 0 END) AS pos_cnt,
          SUM(CASE WHEN pred_label='NEU' THEN 1 ELSE 0 END) AS neu_cnt,
          SUM(CASE WHEN pred_label='NEG' THEN 1 ELSE 0 END) AS neg_cnt,
          COUNT(*) AS total_cnt,
          (SUM(CASE WHEN pred_label='POS' THEN 1 ELSE 0 END) * 1.0) / NULLIF(COUNT(*),0) AS pos_rate,
          (SUM(CASE WHEN pred_label='NEU' THEN 1 ELSE 0 END) * 1.0) / NULLIF(COUNT(*),0) AS neu_rate,
          (SUM(CASE WHEN pred_label='NEG' THEN 1 ELSE 0 END) * 1.0) / NULLIF(COUNT(*),0) AS neg_rate,
          (SUM(CASE WHEN pred_label='POS' THEN 1 ELSE 0 END) - SUM(CASE WHEN pred_label='NEG' THEN 1 ELSE 0 END)) * 1.0
            / NULLIF(COUNT(*),0) AS sent_score
        FROM preds
        GROUP BY 1,2,3,4,5
      ) TO '{p2duck(agg_all_parquet)}'
      (FORMAT PARQUET, COMPRESSION ZSTD);
    """)
    print(f"[OK] wrote: {agg_all_parquet}")

    # ========== 4) hard 聚合（confidence>=t + coverage）==========
    t = float(args.conf_threshold)
    con.execute(f"""
      COPY (
        WITH all_cnt AS (
          SELECT
            domain, brand, model, aspect_l1, aspect_l2,
            COUNT(*) AS total_cnt_all
          FROM preds
          GROUP BY 1,2,3,4,5
        ),
        hard_cnt AS (
          SELECT
            domain, brand, model, aspect_l1, aspect_l2,
            SUM(CASE WHEN pred_label='POS' THEN 1 ELSE 0 END) AS pos_cnt_hard,
            SUM(CASE WHEN pred_label='NEU' THEN 1 ELSE 0 END) AS neu_cnt_hard,
            SUM(CASE WHEN pred_label='NEG' THEN 1 ELSE 0 END) AS neg_cnt_hard,
            COUNT(*) AS total_cnt_hard
          FROM preds
          WHERE confidence >= {t}
          GROUP BY 1,2,3,4,5
        )
        SELECT
          a.domain, a.brand, a.model, a.aspect_l1, a.aspect_l2,
          a.total_cnt_all,
          coalesce(h.total_cnt_hard, 0) AS total_cnt_hard,
          coalesce(h.pos_cnt_hard, 0) AS pos_cnt_hard,
          coalesce(h.neu_cnt_hard, 0) AS neu_cnt_hard,
          coalesce(h.neg_cnt_hard, 0) AS neg_cnt_hard,
          CASE WHEN coalesce(h.total_cnt_hard,0)=0 THEN NULL ELSE (h.pos_cnt_hard*1.0/h.total_cnt_hard) END AS pos_rate_hard,
          CASE WHEN coalesce(h.total_cnt_hard,0)=0 THEN NULL ELSE (h.neu_cnt_hard*1.0/h.total_cnt_hard) END AS neu_rate_hard,
          CASE WHEN coalesce(h.total_cnt_hard,0)=0 THEN NULL ELSE (h.neg_cnt_hard*1.0/h.total_cnt_hard) END AS neg_rate_hard,
          CASE WHEN coalesce(h.total_cnt_hard,0)=0 THEN NULL ELSE ((h.pos_cnt_hard - h.neg_cnt_hard)*1.0/h.total_cnt_hard) END AS sent_score_hard,
          CASE WHEN a.total_cnt_all=0 THEN NULL ELSE (coalesce(h.total_cnt_hard,0)*1.0/a.total_cnt_all) END AS coverage
        FROM all_cnt a
        LEFT JOIN hard_cnt h
        USING(domain, brand, model, aspect_l1, aspect_l2)
      ) TO '{p2duck(agg_hard_parquet)}'
      (FORMAT PARQUET, COMPRESSION ZSTD);
    """)
    print(f"[OK] wrote: {agg_hard_parquet}")

    # ========== 5) soft 聚合（概率加权）==========
    con.execute(f"""
      COPY (
        SELECT
          domain, brand, model, aspect_l1, aspect_l2,
          COUNT(*) AS total_cnt,
          SUM(p_pos) AS w_pos,
          SUM(p_neu) AS w_neu,
          SUM(p_neg) AS w_neg,
          (SUM(p_pos)+SUM(p_neu)+SUM(p_neg)) AS w_total,
          CASE WHEN (SUM(p_pos)+SUM(p_neu)+SUM(p_neg))=0 THEN NULL ELSE (SUM(p_pos)/(SUM(p_pos)+SUM(p_neu)+SUM(p_neg))) END AS pos_share_soft,
          CASE WHEN (SUM(p_pos)+SUM(p_neu)+SUM(p_neg))=0 THEN NULL ELSE (SUM(p_neu)/(SUM(p_pos)+SUM(p_neu)+SUM(p_neg))) END AS neu_share_soft,
          CASE WHEN (SUM(p_pos)+SUM(p_neu)+SUM(p_neg))=0 THEN NULL ELSE (SUM(p_neg)/(SUM(p_pos)+SUM(p_neu)+SUM(p_neg))) END AS neg_share_soft,
          CASE WHEN (SUM(p_pos)+SUM(p_neu)+SUM(p_neg))=0 THEN NULL ELSE ((SUM(p_pos)-SUM(p_neg))/(SUM(p_pos)+SUM(p_neu)+SUM(p_neg))) END AS sent_score_soft
        FROM preds
        GROUP BY 1,2,3,4,5
      ) TO '{p2duck(agg_soft_parquet)}'
      (FORMAT PARQUET, COMPRESSION ZSTD);
    """)
    print(f"[OK] wrote: {agg_soft_parquet}")

    # ========== 6) 写汇总 Excel（一个文件，多 sheet）==========
    df_all = con.sql(f"SELECT * FROM read_parquet('{p2duck(agg_all_parquet)}')").df()
    df_hard = con.sql(f"SELECT * FROM read_parquet('{p2duck(agg_hard_parquet)}')").df()
    df_soft = con.sql(f"SELECT * FROM read_parquet('{p2duck(agg_soft_parquet)}')").df()

    with pd.ExcelWriter(agg_xlsx, engine="openpyxl") as w:
        _excel_safe_write(df_all, w, "all_summary")
        _excel_safe_write(df_hard, w, f"hard_t{t:.2f}".replace(".", "p"))
        _excel_safe_write(df_soft, w, "soft_summary")
    print(f"[OK] wrote: {agg_xlsx}")

    # ========== 7) 时间分桶（all/hard/soft）==========
    if args.time_grain != "none":
        # 统一把 ctime 转 timestamp：
        # DuckDB Binder 会检查 CASE 的所有分支，所以任何算术都必须在 try_cast 后进行
        ts_parse_expr = f"""
        CASE
          WHEN ctime IS NULL THEN NULL
          ELSE
            coalesce(
              -- 1) 数字 epoch（ctime 可能是数字，也可能是数字字符串）
              CASE
                WHEN try_cast(ctime AS BIGINT) IS NOT NULL THEN
                  CASE
                    WHEN try_cast(ctime AS BIGINT) > 100000000000 THEN
                      to_timestamp(try_cast(ctime AS DOUBLE) / 1000.0)  -- ms
                    ELSE
                      to_timestamp(try_cast(ctime AS DOUBLE))           -- sec
                  END
                ELSE NULL
              END,

              -- 2) 字符串时间（ctime 是日期/时间字符串）
              try_cast(CAST(ctime AS VARCHAR) AS TIMESTAMP),
              try_strptime(CAST(ctime AS VARCHAR), '%Y-%m-%d %H:%M:%S'),
              try_strptime(CAST(ctime AS VARCHAR), '%Y/%m/%d %H:%M:%S'),
              try_strptime(CAST(ctime AS VARCHAR), '%Y-%m-%d'),
              try_strptime(CAST(ctime AS VARCHAR), '%Y/%m/%d')
            )
        END
        """


        if args.time_grain == "day":
            bucket_expr = "CAST(date_trunc('day', ts) AS DATE)"
        elif args.time_grain == "week":
            bucket_expr = "CAST(date_trunc('week', ts) AS DATE)"
        else:
            bucket_expr = "CAST(date_trunc('month', ts) AS DATE)"


        con.execute(f"""
            CREATE OR REPLACE TEMP VIEW preds_ts AS
            WITH t AS (
              SELECT
                *,
                ({ts_parse_expr}) AS ts
              FROM preds
            )
            SELECT
              domain, brand, model, aspect_l1, aspect_l2,
              pred_label, p_pos, p_neu, p_neg, confidence,
              ts,
              {bucket_expr} AS time_bucket
            FROM t
            WHERE ts IS NOT NULL;
        """)

        # ---- all 宽表 ----
        con.execute(f"""
          COPY (
            SELECT
              domain, brand, model, aspect_l1, aspect_l2,
              time_bucket,
              SUM(CASE WHEN pred_label='POS' THEN 1 ELSE 0 END) AS pos_cnt,
              SUM(CASE WHEN pred_label='NEU' THEN 1 ELSE 0 END) AS neu_cnt,
              SUM(CASE WHEN pred_label='NEG' THEN 1 ELSE 0 END) AS neg_cnt,
              COUNT(*) AS total_cnt,
              (SUM(CASE WHEN pred_label='POS' THEN 1 ELSE 0 END) * 1.0) / NULLIF(COUNT(*),0) AS pos_rate,
              (SUM(CASE WHEN pred_label='NEU' THEN 1 ELSE 0 END) * 1.0) / NULLIF(COUNT(*),0) AS neu_rate,
              (SUM(CASE WHEN pred_label='NEG' THEN 1 ELSE 0 END) * 1.0) / NULLIF(COUNT(*),0) AS neg_rate,
              (SUM(CASE WHEN pred_label='POS' THEN 1 ELSE 0 END) - SUM(CASE WHEN pred_label='NEG' THEN 1 ELSE 0 END)) * 1.0
                / NULLIF(COUNT(*),0) AS sent_score
            FROM preds_ts
            GROUP BY 1,2,3,4,5,6
            ORDER BY 2,3,4,5,6
          ) TO '{p2duck(ts_all_parquet)}'
          (FORMAT PARQUET, COMPRESSION ZSTD);
        """)
        print(f"[OK] wrote: {ts_all_parquet}")

        # ---- all 长表 ----
        con.execute(f"""
          COPY (
            SELECT
              domain, brand, model, aspect_l1, aspect_l2,
              time_bucket,
              pred_label AS label,
              COUNT(*) AS cnt
            FROM preds_ts
            GROUP BY 1,2,3,4,5,6,7
            ORDER BY 2,3,4,5,6,7
          ) TO '{p2duck(ts_all_long_parquet)}'
          (FORMAT PARQUET, COMPRESSION ZSTD);
        """)
        print(f"[OK] wrote: {ts_all_long_parquet}")

        # ---- hard 宽表（含 coverage）----
        con.execute(f"""
          COPY (
            WITH all_cnt AS (
              SELECT
                domain, brand, model, aspect_l1, aspect_l2, time_bucket,
                COUNT(*) AS total_cnt_all
              FROM preds_ts
              GROUP BY 1,2,3,4,5,6
            ),
            hard_cnt AS (
              SELECT
                domain, brand, model, aspect_l1, aspect_l2, time_bucket,
                SUM(CASE WHEN pred_label='POS' THEN 1 ELSE 0 END) AS pos_cnt_hard,
                SUM(CASE WHEN pred_label='NEU' THEN 1 ELSE 0 END) AS neu_cnt_hard,
                SUM(CASE WHEN pred_label='NEG' THEN 1 ELSE 0 END) AS neg_cnt_hard,
                COUNT(*) AS total_cnt_hard
              FROM preds_ts
              WHERE confidence >= {t}
              GROUP BY 1,2,3,4,5,6
            )
            SELECT
              a.domain, a.brand, a.model, a.aspect_l1, a.aspect_l2, a.time_bucket,
              a.total_cnt_all,
              coalesce(h.total_cnt_hard, 0) AS total_cnt_hard,
              coalesce(h.pos_cnt_hard, 0) AS pos_cnt_hard,
              coalesce(h.neu_cnt_hard, 0) AS neu_cnt_hard,
              coalesce(h.neg_cnt_hard, 0) AS neg_cnt_hard,
              CASE WHEN coalesce(h.total_cnt_hard,0)=0 THEN NULL ELSE (h.pos_cnt_hard*1.0/h.total_cnt_hard) END AS pos_rate_hard,
              CASE WHEN coalesce(h.total_cnt_hard,0)=0 THEN NULL ELSE (h.neu_cnt_hard*1.0/h.total_cnt_hard) END AS neu_rate_hard,
              CASE WHEN coalesce(h.total_cnt_hard,0)=0 THEN NULL ELSE (h.neg_cnt_hard*1.0/h.total_cnt_hard) END AS neg_rate_hard,
              CASE WHEN coalesce(h.total_cnt_hard,0)=0 THEN NULL ELSE ((h.pos_cnt_hard-h.neg_cnt_hard)*1.0/h.total_cnt_hard) END AS sent_score_hard,
              CASE WHEN a.total_cnt_all=0 THEN NULL ELSE (coalesce(h.total_cnt_hard,0)*1.0/a.total_cnt_all) END AS coverage
            FROM all_cnt a
            LEFT JOIN hard_cnt h
            USING(domain, brand, model, aspect_l1, aspect_l2, time_bucket)
            ORDER BY brand, model, aspect_l1, aspect_l2, time_bucket
          ) TO '{p2duck(ts_hard_parquet)}'
          (FORMAT PARQUET, COMPRESSION ZSTD);
        """)
        print(f"[OK] wrote: {ts_hard_parquet}")

        # ---- hard 长表 ----
        con.execute(f"""
          COPY (
            WITH hard_rows AS (
              SELECT
                domain, brand, model, aspect_l1, aspect_l2, time_bucket,
                pred_label AS label,
                COUNT(*) AS cnt_hard
              FROM preds_ts
              WHERE confidence >= {t}
              GROUP BY 1,2,3,4,5,6,7
            ),
            all_rows AS (
              SELECT
                domain, brand, model, aspect_l1, aspect_l2, time_bucket,
                COUNT(*) AS total_cnt_all
              FROM preds_ts
              GROUP BY 1,2,3,4,5,6
            )
            SELECT
              a.domain, a.brand, a.model, a.aspect_l1, a.aspect_l2, a.time_bucket,
              h.label,
              coalesce(h.cnt_hard, 0) AS cnt_hard,
              a.total_cnt_all,
              CASE WHEN a.total_cnt_all=0 THEN NULL ELSE (coalesce(h.cnt_hard,0)*1.0/a.total_cnt_all) END AS coverage_by_label
            FROM all_rows a
            LEFT JOIN hard_rows h
            USING(domain, brand, model, aspect_l1, aspect_l2, time_bucket)
            ORDER BY brand, model, aspect_l1, aspect_l2, time_bucket, label
          ) TO '{p2duck(ts_hard_long_parquet)}'
          (FORMAT PARQUET, COMPRESSION ZSTD);
        """)
        print(f"[OK] wrote: {ts_hard_long_parquet}")

        # ---- soft 宽表 ----
        con.execute(f"""
          COPY (
            SELECT
              domain, brand, model, aspect_l1, aspect_l2,
              time_bucket,
              COUNT(*) AS total_cnt,
              SUM(p_pos) AS w_pos,
              SUM(p_neu) AS w_neu,
              SUM(p_neg) AS w_neg,
              (SUM(p_pos)+SUM(p_neu)+SUM(p_neg)) AS w_total,
              CASE WHEN (SUM(p_pos)+SUM(p_neu)+SUM(p_neg))=0 THEN NULL ELSE (SUM(p_pos)/(SUM(p_pos)+SUM(p_neu)+SUM(p_neg))) END AS pos_share_soft,
              CASE WHEN (SUM(p_pos)+SUM(p_neu)+SUM(p_neg))=0 THEN NULL ELSE (SUM(p_neu)/(SUM(p_pos)+SUM(p_neu)+SUM(p_neg))) END AS neu_share_soft,
              CASE WHEN (SUM(p_pos)+SUM(p_neu)+SUM(p_neg))=0 THEN NULL ELSE (SUM(p_neg)/(SUM(p_pos)+SUM(p_neu)+SUM(p_neg))) END AS neg_share_soft,
              CASE WHEN (SUM(p_pos)+SUM(p_neu)+SUM(p_neg))=0 THEN NULL ELSE ((SUM(p_pos)-SUM(p_neg))/(SUM(p_pos)+SUM(p_neu)+SUM(p_neg))) END AS sent_score_soft
            FROM preds_ts
            GROUP BY 1,2,3,4,5,6
            ORDER BY brand, model, aspect_l1, aspect_l2, time_bucket
          ) TO '{p2duck(ts_soft_parquet)}'
          (FORMAT PARQUET, COMPRESSION ZSTD);
        """)
        print(f"[OK] wrote: {ts_soft_parquet}")

        # ---- soft 长表 ----
        con.execute(f"""
          COPY (
            SELECT
              domain, brand, model, aspect_l1, aspect_l2,
              time_bucket,
              'POS' AS label, SUM(p_pos) AS w
            FROM preds_ts
            GROUP BY 1,2,3,4,5,6
            UNION ALL
            SELECT
              domain, brand, model, aspect_l1, aspect_l2,
              time_bucket,
              'NEU' AS label, SUM(p_neu) AS w
            FROM preds_ts
            GROUP BY 1,2,3,4,5,6
            UNION ALL
            SELECT
              domain, brand, model, aspect_l1, aspect_l2,
              time_bucket,
              'NEG' AS label, SUM(p_neg) AS w
            FROM preds_ts
            GROUP BY 1,2,3,4,5,6
          ) TO '{p2duck(ts_soft_long_parquet)}'
          (FORMAT PARQUET, COMPRESSION ZSTD);
        """)
        print(f"[OK] wrote: {ts_soft_long_parquet}")

        # ---- 时间序列 Excel（多 sheet）----
        df_ts_all = con.sql(f"SELECT * FROM read_parquet('{p2duck(ts_all_parquet)}')").df()
        df_ts_all_long = con.sql(f"SELECT * FROM read_parquet('{p2duck(ts_all_long_parquet)}')").df()
        df_ts_hard = con.sql(f"SELECT * FROM read_parquet('{p2duck(ts_hard_parquet)}')").df()
        df_ts_hard_long = con.sql(f"SELECT * FROM read_parquet('{p2duck(ts_hard_long_parquet)}')").df()
        df_ts_soft = con.sql(f"SELECT * FROM read_parquet('{p2duck(ts_soft_parquet)}')").df()
        df_ts_soft_long = con.sql(f"SELECT * FROM read_parquet('{p2duck(ts_soft_long_parquet)}')").df()

        with pd.ExcelWriter(ts_xlsx, engine="openpyxl") as w:
            _excel_safe_write(df_ts_all, w, f"all_{args.time_grain}")
            _excel_safe_write(df_ts_all_long, w, f"all_{args.time_grain}_long")
            _excel_safe_write(df_ts_hard, w, f"hard_t{t:.2f}_{args.time_grain}".replace(".", "p"))
            _excel_safe_write(df_ts_hard_long, w, f"hard_t{t:.2f}_{args.time_grain}_long".replace(".", "p"))
            _excel_safe_write(df_ts_soft, w, f"soft_{args.time_grain}")
            _excel_safe_write(df_ts_soft_long, w, f"soft_{args.time_grain}_long")
        print(f"[OK] wrote: {ts_xlsx}")

    # ========== 8) 代表例句（可选）==========
    if args.make_examples:
        ex_min_conf = args.examples_min_confidence
        conf_filter = ""
        if ex_min_conf is not None and ex_min_conf >= 0:
            conf_filter = f"AND confidence >= {float(ex_min_conf)}"

        pos_sql = f"""
          SELECT * EXCLUDE(rn) FROM (
            SELECT
              domain, brand, model, aspect_l1, aspect_l2,
              'POS' AS label,
              p_pos AS prob,
              confidence,
              sentence, url, ctime,
              row_number() OVER (
                PARTITION BY domain,brand,model,aspect_l1,aspect_l2
                ORDER BY p_pos DESC
              ) AS rn
            FROM preds
            WHERE sentence IS NOT NULL {conf_filter}
          )
          WHERE rn <= {int(args.pos_topk)}
        """

        neg_sql = f"""
          SELECT * EXCLUDE(rn) FROM (
            SELECT
              domain, brand, model, aspect_l1, aspect_l2,
              'NEG' AS label,
              p_neg AS prob,
              confidence,
              sentence, url, ctime,
              row_number() OVER (
                PARTITION BY domain,brand,model,aspect_l1,aspect_l2
                ORDER BY p_neg DESC
              ) AS rn
            FROM preds
            WHERE sentence IS NOT NULL {conf_filter}
          )
          WHERE rn <= {int(args.neg_topk)}
        """

        con.execute(f"""
          COPY (
            {pos_sql}
            UNION ALL
            {neg_sql}
          ) TO '{p2duck(examples_parquet)}'
          (FORMAT PARQUET, COMPRESSION ZSTD);
        """)
        print(f"[OK] wrote: {examples_parquet}")

    # ========== 9) by_product Excel（可选）==========
    if args.build_by_product:
        # 产品列表来自 all 汇总（体量小）
        products = con.sql(f"""
          SELECT DISTINCT brand, model
          FROM read_parquet('{p2duck(agg_all_parquet)}')
          ORDER BY brand, model
        """).df()

        if args.max_products and args.max_products > 0:
            products = products.head(args.max_products)

        index_rows = []

        for i, r in products.iterrows():
            brand = str(r["brand"])
            model = str(r["model"])
            fn = f"{brand}__{model}.xlsx".replace("/", "_").replace("\\", "_").replace(":", "_")
            fp = by_dir / fn

            df_p_all = con.sql(f"""
              SELECT *
              FROM read_parquet('{p2duck(agg_all_parquet)}')
              WHERE brand = {duckdb.escape_literal(brand)}
                AND model = {duckdb.escape_literal(model)}
              ORDER BY aspect_l1, aspect_l2
            """).df()

            df_p_hard = con.sql(f"""
              SELECT *
              FROM read_parquet('{p2duck(agg_hard_parquet)}')
              WHERE brand = {duckdb.escape_literal(brand)}
                AND model = {duckdb.escape_literal(model)}
              ORDER BY aspect_l1, aspect_l2
            """).df()

            df_p_soft = con.sql(f"""
              SELECT *
              FROM read_parquet('{p2duck(agg_soft_parquet)}')
              WHERE brand = {duckdb.escape_literal(brand)}
                AND model = {duckdb.escape_literal(model)}
              ORDER BY aspect_l1, aspect_l2
            """).df()

            # pivot：用 hard 的 sent_score 更适合作为“稳健展示”
            if len(df_p_hard) > 0 and "sent_score_hard" in df_p_hard.columns:
                pivot = df_p_hard.pivot_table(index="aspect_l1", columns="aspect_l2", values="sent_score_hard", aggfunc="mean")
            else:
                pivot = df_p_all.pivot_table(index="aspect_l1", columns="aspect_l2", values="sent_score", aggfunc="mean")

            with pd.ExcelWriter(fp, engine="openpyxl") as w:
                _excel_safe_write(df_p_hard, w, "hard_summary")
                _excel_safe_write(df_p_all, w, "all_summary")
                _excel_safe_write(df_p_soft, w, "soft_summary")
                pivot.to_excel(w, sheet_name="l2_sentiment")

                if args.time_grain != "none":
                    # 写入 hard 时间序列（更贴近交付口径）
                    if ts_hard_parquet.exists():
                        df_p_ts_hard = con.sql(f"""
                          SELECT *
                          FROM read_parquet('{p2duck(ts_hard_parquet)}')
                          WHERE brand = {duckdb.escape_literal(brand)}
                            AND model = {duckdb.escape_literal(model)}
                          ORDER BY aspect_l1, aspect_l2, time_bucket
                        """).df()
                        _excel_safe_write(df_p_ts_hard, w, f"ts_hard_{args.time_grain}")

                    # 写入 soft 时间序列（趋势口径）
                    if ts_soft_parquet.exists():
                        df_p_ts_soft = con.sql(f"""
                          SELECT *
                          FROM read_parquet('{p2duck(ts_soft_parquet)}')
                          WHERE brand = {duckdb.escape_literal(brand)}
                            AND model = {duckdb.escape_literal(model)}
                          ORDER BY aspect_l1, aspect_l2, time_bucket
                        """).df()
                        _excel_safe_write(df_p_ts_soft, w, f"ts_soft_{args.time_grain}")

                if args.make_examples and examples_parquet.exists():
                    df_ex = con.sql(f"""
                      SELECT *
                      FROM read_parquet('{p2duck(examples_parquet)}')
                      WHERE brand = {duckdb.escape_literal(brand)}
                        AND model = {duckdb.escape_literal(model)}
                      ORDER BY aspect_l1, aspect_l2, label, prob DESC
                    """).df()
                    _excel_safe_write(df_ex, w, "examples")

            index_rows.append({"brand": brand, "model": model, "file": fn})

            if (i + 1) % 50 == 0:
                print(f"[PROGRESS] wrote {i+1}/{len(products)} product excels")

        index_xlsx = by_dir / "INDEX.xlsx"
        pd.DataFrame(index_rows).to_excel(index_xlsx, index=False)
        print(f"[OK] wrote: {index_xlsx}")

    print("[DONE] Step05 聚合与报表生成完成。")
    print(f"[INFO] pred_source={used_pattern}")
    print(f"[INFO] hard_threshold={t}")
    if args.time_grain != "none":
        print(f"[INFO] time_grain={args.time_grain}")
        print(f"[INFO] all_ts_long={ts_all_long_parquet}")
        print(f"[INFO] hard_ts_long={ts_hard_long_parquet}")
        print(f"[INFO] soft_ts_long={ts_soft_long_parquet}")


if __name__ == "__main__":
    main()

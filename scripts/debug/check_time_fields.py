# scripts/check_time_fields.py
# -*- coding: utf-8 -*-

import os
import re
import argparse
import duckdb


TIME_LIKE_PAT = re.compile(r"(time|date|ctime|created|updated|timestamp|ts)", re.I)


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def check_one_parquet(con: duckdb.DuckDBPyConnection, parquet_path: str, limit: int = 10):
    if not os.path.exists(parquet_path):
        print(f"[MISS] {parquet_path}")
        return

    print_header(f"FILE: {parquet_path}")

    # 1) 列信息
    cols = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{parquet_path}')").fetchdf()
    print("[COLUMNS]")
    print(cols.to_string(index=False))

    col_names = cols["column_name"].tolist()
    time_like_cols = [c for c in col_names if TIME_LIKE_PAT.search(c or "")]

    print("\n[TIME-LIKE COLUMNS]")
    if time_like_cols:
        print(", ".join(time_like_cols))
    else:
        print("(none)")

    # 2) 重点检查 ctime
    if "ctime" not in col_names:
        print("\n[CHECK ctime] ctime column NOT FOUND.")
        return

    # 3) ctime 类型、抽样值、空值统计（修复 DuckDB 聚合 + 非聚合冲突）
    stats = con.execute(
        f"""
        SELECT
            ANY_VALUE(typeof(ctime)) AS ctype,
            count(*) AS n_rows,
            count(ctime) AS n_notnull,
            sum(CASE WHEN ctime IS NULL THEN 1 ELSE 0 END) AS n_null
        FROM read_parquet('{parquet_path}')
        """
    ).fetchdf()
    print("\n[CTIME BASIC STATS]")
    print(stats.to_string(index=False))

    sample = con.execute(
        f"""
        SELECT ctime
        FROM read_parquet('{parquet_path}')
        WHERE ctime IS NOT NULL
        LIMIT {int(limit)}
        """
    ).fetchall()
    print(f"\n[CTIME SAMPLE TOP {limit}]")
    for i, (v,) in enumerate(sample, 1):
        print(f"{i:02d}: {v!r}")

    # 4) 解析能力测试：尽量在 DuckDB 内部做 “可解释的” 解析
    #    - 如果 ctime 是整数：判断秒/毫秒时间戳
    #    - 如果 ctime 是字符串：try_cast 到 TIMESTAMP（能解析很多常见格式）
    parse_probe = con.execute(
        f"""
        WITH t AS (
            SELECT ctime
            FROM read_parquet('{parquet_path}')
            WHERE ctime IS NOT NULL
        ),
        typed AS (
            SELECT
                typeof(ctime) AS ctype,
                ctime
            FROM t
            LIMIT 10000
        )
        SELECT
            ctype,
            -- 尝试按字符串解析
            sum(CASE WHEN try_cast(ctime AS TIMESTAMP) IS NOT NULL THEN 1 ELSE 0 END) AS ok_as_timestamp,
            -- 尝试按数字时间戳解析（秒/毫秒）
            sum(
                CASE
                    WHEN try_cast(ctime AS BIGINT) IS NOT NULL
                     AND (
                        try_cast(ctime AS BIGINT) BETWEEN 1000000000 AND 4000000000
                        OR try_cast(ctime AS BIGINT) BETWEEN 1000000000000 AND 4000000000000
                     )
                    THEN 1 ELSE 0
                END
            ) AS ok_as_epoch
        FROM typed
        GROUP BY ctype
        """
    ).fetchdf()

    print("\n[PARSE PROBE on sample<=10000 non-null rows]")
    print(parse_probe.to_string(index=False))

    # 5) 给出 min/max（分别用原值、字符串timestamp、epoch->timestamp 三种方式尽量推断）
    # 注意：这里只做“尽量推断”，不保证所有格式都能解析
    minmax = con.execute(
        f"""
        WITH t AS (
            SELECT ctime
            FROM read_parquet('{parquet_path}')
            WHERE ctime IS NOT NULL
        ),
        as_ts AS (
            SELECT
                try_cast(ctime AS TIMESTAMP) AS ts1,
                -- epoch seconds / milliseconds
                CASE
                    WHEN try_cast(ctime AS BIGINT) BETWEEN 1000000000 AND 4000000000
                        THEN to_timestamp(try_cast(ctime AS BIGINT))
                    WHEN try_cast(ctime AS BIGINT) BETWEEN 1000000000000 AND 4000000000000
                        THEN to_timestamp(try_cast(ctime AS BIGINT) / 1000)
                    ELSE NULL
                END AS ts2,
                ctime
            FROM t
        )
        SELECT
            min(ctime) AS min_raw,
            max(ctime) AS max_raw,
            min(ts1) AS min_ts_trycast,
            max(ts1) AS max_ts_trycast,
            min(ts2) AS min_ts_epoch,
            max(ts2) AS max_ts_epoch
        FROM as_ts
        """
    ).fetchdf()

    print("\n[MIN/MAX INFERENCE]")
    print(minmax.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, help="e.g. phone")
    ap.add_argument("--limit", type=int, default=10, help="sample size for ctime values")
    args = ap.parse_args()

    # 你项目的两个核心 parquet（按你当前流程）
    clean_path = os.path.join("outputs", args.domain, "clean_sentences.parquet")
    aspect_path = os.path.join("outputs", args.domain, "aspect_sentences.parquet")

    con = duckdb.connect(database=":memory:")
    # 让读取更稳一点
    con.execute("PRAGMA threads=8;")

    check_one_parquet(con, clean_path, limit=args.limit)
    check_one_parquet(con, aspect_path, limit=args.limit)

    con.close()


if __name__ == "__main__":
    main()

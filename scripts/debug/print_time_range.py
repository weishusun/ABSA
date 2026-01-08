# scripts/print_time_range.py
# -*- coding: utf-8 -*-

import argparse
import os
import duckdb


def to_sql_path(p: str) -> str:
    return os.path.abspath(p).replace("\\", "/")


def show_range(con, parquet_path: str, name: str):
    q = f"""
    WITH t AS (
      SELECT try_cast(ctime AS TIMESTAMP) AS ts
      FROM read_parquet('{to_sql_path(parquet_path)}')
      WHERE ctime IS NOT NULL
    )
    SELECT
      '{name}' AS table_name,
      count(*) AS n_rows,
      sum(CASE WHEN ts IS NULL THEN 1 ELSE 0 END) AS n_parse_fail,
      min(ts) AS min_ts,
      max(ts) AS max_ts
    FROM t;
    """
    df = con.execute(q).fetchdf()
    print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, help="e.g. phone")
    args = ap.parse_args()

    clean_path = os.path.join("outputs", args.domain, "clean_sentences.parquet")
    aspect_path = os.path.join("outputs", args.domain, "aspect_sentences.parquet")

    con = duckdb.connect(":memory:")
    con.execute("PRAGMA threads=8;")

    if os.path.exists(clean_path):
        show_range(con, clean_path, "clean_sentences")
    else:
        print(f"[MISS] {clean_path}")

    if os.path.exists(aspect_path):
        show_range(con, aspect_path, "aspect_sentences")
    else:
        print(f"[MISS] {aspect_path}")

    con.close()


if __name__ == "__main__":
    main()

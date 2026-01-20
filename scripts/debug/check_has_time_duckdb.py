# scripts/check_has_time_duckdb.py
# -*- coding: utf-8 -*-

import argparse
import os
import duckdb


def to_sql_path(p: str) -> str:
    # DuckDB 更偏好 / 分隔符
    return os.path.abspath(p).replace("\\", "/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="parquet file OR directory")
    ap.add_argument("--col", default="ctime")
    ap.add_argument("--show", type=int, default=200, help="show first N column names")
    args = ap.parse_args()

    p = args.path
    if not os.path.exists(p):
        raise SystemExit(f"[MISS] {p}")

    # 如果是目录，就用递归 glob
    if os.path.isdir(p):
        patt = to_sql_path(os.path.join(p, "**", "*.parquet"))
        src = f"read_parquet('{patt}')"
    else:
        src = f"read_parquet('{to_sql_path(p)}')"

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=8;")

    print("=" * 80)
    print(f"PATH: {os.path.abspath(p)}")
    print("=" * 80)

    # 仅取 schema：LIMIT 0
    try:
        df = con.execute(f"DESCRIBE SELECT * FROM {src} LIMIT 0").fetchdf()
    except Exception as e:
        print("[FAIL] DuckDB cannot DESCRIBE this parquet.")
        print("Error:", repr(e))
        raise
    finally:
        con.close()

    cols = df["column_name"].tolist()
    print(f"NUM_COLS: {len(cols)}")

    if len(cols) <= args.show:
        print("COLUMNS:", ", ".join(cols))
    else:
        print("COLUMNS(head):", ", ".join(cols[: args.show]))
        print("...")

    has = args.col in cols
    print(f"\nHAS_COLUMN[{args.col}]: {has}")
    if has:
        ctype = df.loc[df["column_name"] == args.col, "column_type"].iloc[0]
        print(f"TYPE[{args.col}]: {ctype}")


if __name__ == "__main__":
    main()

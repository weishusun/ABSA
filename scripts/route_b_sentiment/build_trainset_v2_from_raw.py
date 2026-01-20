import argparse
import duckdb
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--pos-min", type=float, default=0.75)
    ap.add_argument("--neu-min", type=float, default=0.65)
    ap.add_argument("--neg-min", type=float, default=0.00)  # NEG 默认全保留

    ap.add_argument("--cap-neu", type=int, default=2500)
    ap.add_argument("--cap-pos", type=int, default=3000)

    ap.add_argument("--oversample-neg", type=int, default=8)  # 复制倍数；设 1 表示不复制
    args = ap.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA enable_progress_bar=false;")

    # 1) 过滤：label + confidence
    con.execute(f"""
    CREATE TEMP TABLE base AS
    SELECT *
    FROM read_parquet('{raw.as_posix()}')
    WHERE
      (label='POS' AND confidence >= {args.pos_min})
      OR (label='NEU' AND confidence >= {args.neu_min})
      OR (label='NEG' AND confidence >= {args.neg_min});
    """)

    # 2) 截断 POS/NEU，避免淹没 NEG
    con.execute(f"""
    CREATE TEMP TABLE pos AS
    SELECT * FROM base WHERE label='POS'
    USING SAMPLE {args.cap_pos} ROWS;
    """)
    con.execute(f"""
    CREATE TEMP TABLE neu AS
    SELECT * FROM base WHERE label='NEU'
    USING SAMPLE {args.cap_neu} ROWS;
    """)
    con.execute("""
    CREATE TEMP TABLE neg AS
    SELECT * FROM base WHERE label='NEG';
    """)

    # 3) oversample NEG（训练用，不影响最终统计）
    k = max(1, int(args.oversample_neg))
    con.execute(f"""
    CREATE TEMP TABLE neg_os AS
    SELECT n.*
    FROM neg n, range(0, {k}) t(i);
    """)

    # 4) 合并训练集
    con.execute("""
    CREATE TEMP TABLE train AS
    SELECT * FROM pos
    UNION ALL SELECT * FROM neu
    UNION ALL SELECT * FROM neg_os;
    """)

    total = con.execute("SELECT COUNT(*) FROM train").fetchone()[0]
    dist = con.execute("SELECT label, COUNT(*) FROM train GROUP BY 1 ORDER BY 2 DESC").fetchall()
    print(f"[TRAIN_V2] total_rows={total} dist={dist}")

    con.execute(f"COPY train TO '{out.as_posix()}' (FORMAT PARQUET);")
    print("[OK] wrote:", out)

if __name__ == "__main__":
    main()

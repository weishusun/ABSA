import argparse
import duckdb
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="train_pseudolabel.parquet")
    ap.add_argument("--raw", required=False, help="pseudolabel_raw.parquet (optional)")
    args = ap.parse_args()

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA enable_progress_bar=false;")

    def describe(path: str):
        rows = con.execute("DESCRIBE SELECT * FROM read_parquet(?) LIMIT 1", [path]).fetchall()
        return [(str(r[0]), str(r[1])) for r in rows]

    def find_label_col(cols):
        # 常见列名
        for cand in ["label", "labels", "sentiment", "pseudo_label", "pred_label", "gold_label"]:
            if cand in cols:
                return cand
        for c in cols:
            if "label" in c.lower() or "sent" in c.lower():
                return c
        return None

    def find_conf_col(cols):
        for cand in ["confidence", "conf", "score"]:
            if cand in cols:
                return cand
        return None

    def summarize(name: str, path: str):
        print("=" * 90)
        print(f"[{name}] {path}")
        if not Path(path).exists():
            print(f"[{name}][ERROR] file not found")
            return

        sch = describe(path)
        cols = [c for c, _ in sch]
        print(f"[{name}] schema:")
        for c, t in sch:
            print(f"  - {c}: {t}")

        label_col = find_label_col(cols)
        conf_col = find_conf_col(cols)

        print(f"[{name}] detected label_col = {label_col}")
        print(f"[{name}] detected conf_col  = {conf_col}")

        if label_col is None:
            print(f"[{name}][ERROR] cannot detect label column; please tell me the label column name.")
            return

        total = con.execute("SELECT COUNT(*) FROM read_parquet(?)", [path]).fetchone()[0]
        print(f"[{name}] total_rows = {int(total):,}")

        # label 分布
        dist = con.execute(f"""
            SELECT CAST({label_col} AS VARCHAR) AS label, COUNT(*) AS n
            FROM read_parquet(?)
            GROUP BY 1
            ORDER BY n DESC
        """, [path]).fetchall()
        print(f"[{name}] label distribution:")
        for label, n in dist:
            n = int(n)
            print(f"  - {label}: {n:,} ({n / total:.2%})")

        # 如果有 confidence，给出分位数（对判断“NEG 被阈值砍掉”非常关键）
        if conf_col is not None:
            qs = con.execute(f"""
                SELECT
                  quantile_cont({conf_col}, 0.00) AS q00,
                  quantile_cont({conf_col}, 0.10) AS q10,
                  quantile_cont({conf_col}, 0.25) AS q25,
                  quantile_cont({conf_col}, 0.50) AS q50,
                  quantile_cont({conf_col}, 0.75) AS q75,
                  quantile_cont({conf_col}, 0.90) AS q90,
                  quantile_cont({conf_col}, 1.00) AS q100
                FROM read_parquet(?)
            """, [path]).fetchone()
            print(f"[{name}] confidence quantiles (all rows):")
            print("  q00/q10/q25/q50/q75/q90/q100 =",
                  [float(x) if x is not None else None for x in qs])

            # 按 label 看 confidence 分布（NEG 是否整体更低）
            by = con.execute(f"""
                SELECT
                  CAST({label_col} AS VARCHAR) AS label,
                  COUNT(*) AS n,
                  AVG({conf_col}) AS avg_conf,
                  MAX({conf_col}) AS max_conf,
                  quantile_cont({conf_col}, 0.50) AS med_conf,
                  quantile_cont({conf_col}, 0.90) AS q90_conf
                FROM read_parquet(?)
                GROUP BY 1
                ORDER BY n DESC
            """, [path]).fetchall()
            print(f"[{name}] confidence by label:")
            for label, n, avgc, maxc, medc, q90c in by:
                print(f"  - {label}: n={int(n):,} avg={avgc} med={medc} q90={q90c} max={maxc}")

    summarize("TRAIN", args.train)
    if args.raw:
        summarize("RAW", args.raw)

if __name__ == "__main__":
    main()

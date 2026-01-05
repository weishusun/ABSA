import argparse
import glob
import duckdb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-ds", required=True, help="asc_pred_ds directory")
    ap.add_argument("--sample-n", type=int, default=200000)
    args = ap.parse_args()

    files = glob.glob(rf"{args.pred_ds}\shard=*\*.parquet")
    print("=" * 90)
    print("[INFO] pred_ds =", args.pred_ds)
    print("[INFO] parquet_files =", len(files))
    if not files:
        raise SystemExit("No parquet files found under shard=*/*.parquet")

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA enable_progress_bar=false;")

    # A) 全量行数（对账用）
    n_total = con.execute("SELECT COUNT(*) FROM read_parquet(?)", [files]).fetchone()[0]
    print("[A] total_rows =", int(n_total))

    # B) pred_label 分布（全量）
    rows = con.execute("""
        SELECT pred_label, COUNT(*) AS n
        FROM read_parquet(?)
        GROUP BY 1
        ORDER BY n DESC
    """, [files]).fetchall()
    print("\n[B] pred_label distribution:")
    for label, n in rows:
        print(f"  - {label}: {int(n)} ({int(n)/n_total:.2%})")

    # C) 仅按概率 argmax 的分布（全量）
    rows = con.execute("""
        SELECT
          CASE
            WHEN p_neg >= p_neu AND p_neg >= p_pos THEN 'NEG'
            WHEN p_neu >= p_neg AND p_neu >= p_pos THEN 'NEU'
            ELSE 'POS'
          END AS argmax_label,
          COUNT(*) AS n
        FROM read_parquet(?)
        GROUP BY 1
        ORDER BY n DESC
    """, [files]).fetchall()
    print("\n[C] argmax(p_*) distribution:")
    for label, n in rows:
        print(f"  - {label}: {int(n)} ({int(n)/n_total:.2%})")

    # D) pred_label 与 argmax_label 是否一致（全量）
    r = con.execute("""
        WITH s AS (
          SELECT
            CAST(pred_label AS VARCHAR) AS pred_label,
            CASE
              WHEN p_neg >= p_neu AND p_neg >= p_pos THEN 'NEG'
              WHEN p_neu >= p_neg AND p_neu >= p_pos THEN 'NEU'
              ELSE 'POS'
            END AS argmax_label
          FROM read_parquet(?)
        )
        SELECT
          COUNT(*) AS n,
          SUM(CASE WHEN pred_label = argmax_label THEN 1 ELSE 0 END) AS match,
          SUM(CASE WHEN pred_label <> argmax_label THEN 1 ELSE 0 END) AS mismatch
        FROM s
    """, [files]).fetchone()
    n, match, mismatch = map(int, r)
    print("\n[D] pred_label vs argmax(p_*) consistency (FULL):")
    print(f"  n={n} match={match} mismatch={mismatch} mismatch_rate={mismatch/max(1,n):.2%}")

    # E) pred_id 与 pred_label 的对应关系（全量，应该接近一一映射）
    rows = con.execute("""
        SELECT pred_id, pred_label, COUNT(*) AS n
        FROM read_parquet(?)
        GROUP BY 1,2
        ORDER BY pred_id, n DESC
    """, [files]).fetchall()
    print("\n[E] pred_id -> pred_label mapping (top counts):")
    # 只打印每个 pred_id 的前 3 个 label，避免刷屏
    cur = None
    shown = 0
    for pid, plabel, cnt in rows:
        if pid != cur:
            cur = pid
            shown = 0
            print(f"  pred_id={pid}:")
        if shown < 3:
            print(f"    - {plabel}: {int(cnt)}")
            shown += 1

    # F) 概率/置信度形态（全量摘要）
    r = con.execute("""
        SELECT
          MIN(confidence), MAX(confidence), AVG(confidence),
          MAX(p_neg), MAX(p_neu), MAX(p_pos),
          AVG(GREATEST(p_neg,p_neu,p_pos)) AS avg_maxprob,
          MAX(GREATEST(p_neg,p_neu,p_pos)) AS max_maxprob
        FROM read_parquet(?)
    """, [files]).fetchone()
    print("\n[F] probability/confidence summary (FULL):")
    print("  confidence min/max/avg =", r[0], r[1], r[2])
    print("  max p_neg/p_neu/p_pos  =", r[3], r[4], r[5])
    print("  maxprob avg/max        =", r[6], r[7])

    # G) ctime 是否真为空（全量）
    r = con.execute("""
        SELECT
          COUNT(*) AS n,
          SUM(CASE WHEN ctime IS NULL OR TRIM(ctime)='' THEN 1 ELSE 0 END) AS n_empty,
          SUM(CASE WHEN ctime IS NOT NULL AND TRIM(ctime)<>'' THEN 1 ELSE 0 END) AS n_nonempty,
          MIN(ctime) AS min_ctime_str,
          MAX(ctime) AS max_ctime_str
        FROM read_parquet(?)
    """, [files]).fetchone()
    print("\n[G] ctime availability (FULL):")
    print("  n =", int(r[0]))
    print("  empty_or_null =", int(r[1]))
    print("  nonempty      =", int(r[2]))
    print("  min_str / max_str =", r[3], "/", r[4])

    print("\n[DONE] Paste this output back for next-step decision.")

if __name__ == "__main__":
    main()

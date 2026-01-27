# scripts/tools/aggregate_to_db.py
import argparse
import sqlite3
import duckdb
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="聚合 Step 04 结果为统计数据并入库 (支持去重覆盖)")
    parser.add_argument("--pred-ds", required=True, help="Step 04 输出目录 (step04_pred 或 asc_pred_ds)")
    parser.add_argument("--db-path", default="stats.db", help="数据库路径")
    return parser.parse_args()


def main():
    args = parse_args()
    pred_path = Path(args.pred_ds)

    if not pred_path.exists():
        print(f"[ERROR] 输入路径不存在: {pred_path}")
        sys.exit(1)

    print(f"🚀 [DuckDB] 正在读取并聚合数据: {pred_path}")

    # 1. 构造 DuckDB SQL
    # 使用 COALESCE 确保最终返回的一定是 TIMESTAMP 类型，解决类型冲突问题
    date_expr = """
    strftime(
        COALESCE(
            -- 1. 优先尝试：如果是 Unix 时间戳 (数字或数字字符串)
            CASE 
                WHEN try_cast(ctime AS BIGINT) IS NOT NULL THEN
                    CASE 
                        -- 情况A: 微秒级 (16位, > 10^14) -> 转秒
                        WHEN try_cast(ctime AS BIGINT) > 100000000000000 THEN to_timestamp(try_cast(ctime AS BIGINT) / 1000000)

                        -- 情况B: 毫秒级 (13位, > 10^11) -> 转秒
                        WHEN try_cast(ctime AS BIGINT) > 100000000000 THEN to_timestamp(try_cast(ctime AS BIGINT) / 1000)

                        -- 情况C: 秒级 (10位左右)
                        ELSE to_timestamp(try_cast(ctime AS BIGINT))
                    END
                ELSE NULL
            END,

            -- 2. 其次尝试：标准转换 (处理 '2026-01-01' 或 原生 TIMESTAMP 类型)
            try_cast(ctime AS TIMESTAMP),

            -- 3. 最后尝试：特殊格式 (如 '2026/01/01')
            try_cast(strptime(ctime, '%Y/%m/%d') as TIMESTAMP)
        ),
    '%Y-%m-%d')
    """

    query = f"""
    SELECT 
        {date_expr} as date,
        brand, 
        model,
        aspect_l1 as aspect,
        pred_label as sentiment,
        COUNT(*) as count
    FROM read_parquet('{str(pred_path)}/**/*.parquet', hive_partitioning=true)
    WHERE date IS NOT NULL
    GROUP BY 1, 2, 3, 4, 5
    ORDER BY 1, 2, 3
    """

    conn = None
    try:
        # 2. 执行聚合 (DuckDB)
        df_stats = duckdb.query(query).to_df()

        if df_stats.empty:
            print(
                "⚠️ [WARN] 聚合结果为空！请检查 Parquet 文件中是否包含 ctime, brand, model, aspect_l1, pred_label 字段。")
            sys.exit(0)

        # 获取新数据的时间范围
        min_date = df_stats['date'].min()
        max_date = df_stats['date'].max()
        count_rows = len(df_stats)

        print(f"📊 聚合完成！共 {count_rows} 条记录。")
        print(f"📅 数据时间范围: {min_date} 至 {max_date}")

        # 3. 初始化 SQLite 数据库连接
        conn = sqlite3.connect(args.db_path)
        cursor = conn.cursor()

        # 开启事务
        cursor.execute("BEGIN TRANSACTION")

        # 4. 建表 (如果不存在)
        # 注意：这里去掉了 UNIQUE 约束的 ON CONFLICT REPLACE，因为我们将在逻辑层处理去重
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_sentiment_stats (
            date TEXT,
            brand TEXT,
            model TEXT,
            aspect TEXT,
            sentiment TEXT,
            count INTEGER
        )
        """)
        # 创建索引以加速查询和删除
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON daily_sentiment_stats (date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_query ON daily_sentiment_stats (brand, model, date)")

        # 5. 【核心逻辑】先清理重叠时间段的旧数据 (幂等性处理)
        print(f"🧹 正在清理旧数据 (时间段: {min_date} ~ {max_date})...")
        cursor.execute(
            "DELETE FROM daily_sentiment_stats WHERE date >= ? AND date <= ?",
            (min_date, max_date)
        )
        deleted_count = cursor.rowcount
        print(f"   -> 已删除 {deleted_count} 条旧记录。")

        # 6. 批量插入新数据
        print(f"📥 正在写入 {count_rows} 条新数据...")
        data_to_insert = df_stats.values.tolist()

        cursor.executemany("""
        INSERT INTO daily_sentiment_stats (date, brand, model, aspect, sentiment, count)
        VALUES (?, ?, ?, ?, ?, ?)
        """, data_to_insert)

        # 提交事务
        conn.commit()
        print(f"✅ 入库成功！数据库: {args.db_path}")

    except Exception as e:
        print(f"❌ 入库失败: {e}")
        if conn:
            print("↺ 正在回滚事务...")
            conn.rollback()
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    main()
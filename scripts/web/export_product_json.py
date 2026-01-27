# scripts/web/export_product_json.py
import sqlite3
import pandas as pd
import json
import argparse
import sys
import os
from pathlib import Path
from datetime import timedelta

# 尝试自动定位项目根目录 (假设脚本在 scripts/web/ 下)
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[1]  # 往上跳两级到项目根目录


def get_db_path(domain, workspace_root=None):
    """根据领域自动构建数据库路径"""
    if workspace_root:
        root = Path(workspace_root)
    else:
        # 尝试从环境变量读取，或者默认使用项目下的 outputs
        root = Path(os.environ.get("ABSA_WORKSPACE", PROJECT_ROOT))

    db_path = root / "outputs" / domain / "stats.db"
    return db_path


def get_product_dashboard_data(db_path):
    """读取 stats.db，生成按【产品 (Brand + Model)】维度的结构化 JSON 数据"""

    # 1. 数据库路径检查
    if not isinstance(db_path, Path):
        db_path = Path(db_path)

    if not db_path.exists():
        return {"error": f"Database not found: {db_path}"}

    # 2. 读取数据
    try:
        conn = sqlite3.connect(str(db_path))
        query = "SELECT date, brand, model, aspect, sentiment, count FROM daily_sentiment_stats"
        df = pd.read_sql(query, conn)
        conn.close()
    except Exception as e:
        return {"error": f"Database error: {str(e)}"}

    if df.empty:
        return {"error": "Database is empty"}

    # 3. 数据预处理
    df['date'] = pd.to_datetime(df['date'])
    max_date = df['date'].max()  # 动态锚点时间（基于数据中的最后一天）

    df['product_key'] = df['brand'] + " " + df['model']
    unique_products = df['product_key'].unique()

    final_output = {
        "meta": {
            "last_updated": max_date.strftime('%Y-%m-%d'),
            "data_source": str(db_path.name),
            "total_products": len(unique_products)
        },
        "products": {}
    }

    # 4. 遍历产品生成数据
    for product_key in unique_products:
        df_prod = df[df['product_key'] == product_key].copy()

        product_data = {
            "brand_aspect_dist": {},
            "period_stats": {}
        }

        # --- 板块 1: L1 方面分布 (保持不变) ---
        aspect_grp = df_prod.groupby(['aspect', 'sentiment'])['count'].sum().reset_index()

        def get_dist_list(sent_label):
            d = aspect_grp[aspect_grp['sentiment'] == sent_label][['aspect', 'count']].to_dict('records')
            d.sort(key=lambda x: x['count'], reverse=True)
            return d

        product_data["brand_aspect_dist"] = {
            "POS": get_dist_list("POS"),
            "NEG": get_dist_list("NEG")
        }

        # --- 板块 2 & 3: 多时间窗口统计 (修复核心) ---
        periods_config = {
            "last_7_days": {
                "days": 6,  # 修正：6天前 + 今天 = 7天
                "rule": "D",
                "label": "day"
            },
            "last_1_month": {
                "days": 29,  # 修正：29天前 + 今天 = 30天
                "rule": "W",
                "label": "week"
            },
            "last_3_months": {"days": 90, "rule": "ME", "label": "month"},
            "last_6_months": {"days": 180, "rule": "ME", "label": "month"},
            "last_12_months": {"days": 365, "rule": "ME", "label": "month"}
        }

        for p_name, cfg in periods_config.items():
            # 计算起始日期
            start_date = max_date - timedelta(days=cfg['days'])
            df_period = df_prod[df_prod['date'] >= start_date].copy()

            if df_period.empty:
                product_data["period_stats"][p_name] = None
                continue

            # (1) 总量统计
            summary_s = df_period.groupby('sentiment')['count'].sum()
            summary = {
                "POS": int(summary_s.get("POS", 0)),
                "NEG": int(summary_s.get("NEG", 0)),
                "NEU": int(summary_s.get("NEU", 0)),
                "Total": int(df_period['count'].sum())
            }

            # (2) 趋势图 (Resample 修复)
            # 先按天聚合，解决同一天多条记录的问题
            daily_agg = df_period.groupby(['date', 'sentiment'])['count'].sum().unstack(fill_value=0)

            # 【核心修复逻辑】
            if cfg['rule'] == 'D':
                # 按天聚合，默认即可
                resampled = daily_agg.resample(cfg['rule']).sum().fillna(0)
            else:
                # 按周(W)或月(ME)聚合时，强制使用 label='left'
                # 效果：周聚合时，标签为“本周一”的日期，而不是“下周日”，避免日期超出 max_date
                resampled = daily_agg.resample(cfg['rule'], label='left', closed='left').sum().fillna(0)

            trend_list = []
            for ts, row in resampled.iterrows():
                trend_list.append({
                    "date": ts.strftime('%Y-%m-%d'),
                    "POS": int(row.get("POS", 0)),
                    "NEG": int(row.get("NEG", 0)),
                    "NEU": int(row.get("NEU", 0))
                })

            product_data["period_stats"][p_name] = {
                "granularity": cfg['label'],
                "summary": summary,
                "trend": trend_list
            }

        final_output["products"][product_key] = product_data

    return final_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export product dashboard JSON from stats.db")
    parser.add_argument("--domain", required=True, help="Domain name (e.g., phone, car, laptop)")
    parser.add_argument("--workspace", default=None, help="Root workspace path (optional)")
    parser.add_argument("--output", default=None, help="Output JSON file path (optional)")

    args = parser.parse_args()

    # 1. 自动定位 DB
    db_path = get_db_path(args.domain, args.workspace)
    print(f"🚀 [Domain: {args.domain}] Connecting to: {db_path}")

    # 2. 生成数据
    data = get_product_dashboard_data(db_path)

    if "error" in data:
        print(f"❌ Error: {data['error']}")
        sys.exit(1)

    # 3. 保存结果
    # 默认保存在 outputs/{domain}/dashboard_data.json，方便前端读取
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = db_path.parent / "dashboard_data.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Success! JSON saved to: {out_path}")
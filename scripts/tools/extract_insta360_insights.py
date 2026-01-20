import pandas as pd
from pathlib import Path
import glob
from tqdm import tqdm
from openai import OpenAI
import os
import json

# 1. 配置
RUN_ID = "20260113_insta360_e2e"
DOMAIN = "insta360"
RUN_ROOT = Path(f"outputs/{DOMAIN}/runs/{RUN_ID}")
xlsx_path = RUN_ROOT / "step05_agg" / f"aspect_sentiment_counts_{DOMAIN}.xlsx"
pred_dir = RUN_ROOT / "step04_pred"
output_report = RUN_ROOT / "影石全系舆情报告_LLM订正版.xlsx"

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL")
)


def llm_refine_quotes(model_name, aspect, sentiment, candidates):
    """请求 LLM 从候选池中筛选最符合要求的 4 条原文"""
    sentiment_str = "积极/优点" if sentiment == "POS" else "消极/槽点/缺点"

    prompt = f"""
你是一位专业的舆情分析专家。我们要为产品【{model_name}】的【{aspect}】维度挑选【{sentiment_str}】的原文引用。

【筛选标准】：
1. 必须是直接针对【{model_name}】的描述。
2. 如果是对比句（例如“影石比大疆好”），虽然包含负面词，但对影石是褒义的，这种【绝对不要】。
3. 剔除广告、复读机文字，保留真实的真实用户体感。
4. 确保情感倾向与【{sentiment_str}】严格一致。

【候选池】：
{chr(10).join([f"[{i}] {c}" for i, c in enumerate(candidates)])}

请从上方候选池中选出最合适的 4 条（如果不足 4 条则全选），直接返回编号和原文，格式如下：
1. 原文内容
2. 原文内容
...
"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",  # 或 deepseek-chat
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        content = response.choices[0].message.content
        lines = [line.split('. ', 1)[-1].strip(' "') for line in content.split('\n') if
                 line.strip() and line[0].isdigit()]
        return (lines + ["-"] * 4)[:4]
    except Exception as e:
        print(f"LLM 报错: {e}")
        return (candidates + ["-"] * 4)[:4]


def main():
    print("正在加载统计排名并扫描数据分片...")
    df_summary = pd.read_excel(xlsx_path, sheet_name="all_summary")
    df_l2 = df_summary[df_summary['aspect_l2'] != '_L1'].copy()

    # 1. 建立更大规模的采样池 (每项采样 20 条交由 LLM 挑选)
    tasks = {}
    models = df_l2['model'].unique()
    for m in models:
        m_data = df_l2[df_l2['model'] == m]
        for asp in m_data.sort_values('pos_cnt', ascending=False).head(10)['aspect_l2'].tolist():
            tasks[(m, asp, 'POS')] = []
        for asp in m_data.sort_values('neg_cnt', ascending=False).head(10)['aspect_l2'].tolist():
            tasks[(m, asp, 'NEG')] = []

    shard_files = glob.glob(str(pred_dir / "shard=*" / "*.parquet"))
    for f in tqdm(shard_files, desc="全量采样中"):
        df_part = pd.read_parquet(f)
        for row in df_part.itertuples():
            key = (row.model, row.aspect_l2, row.pred_label)
            if key in tasks and len(tasks[key]) < 20:  # 采样 20 条
                if row.sentence not in tasks[key]:
                    tasks[key].append(row.sentence)

    # 2. LLM 二次订正
    print("开始 LLM 二次订正与筛选...")
    final_rows = []
    # 限制处理前几个型号进行测试，或者全量跑
    for (model, aspect, sentiment), candidates in tqdm(tasks.items(), desc="LLM 订正进度"):
        if not candidates:
            refined = ["-"] * 4
        else:
            refined = llm_refine_quotes(model, aspect, sentiment, candidates)

        # 找到对应的声量计数
        count_row = df_l2[(df_l2['model'] == model) & (df_l2['aspect_l2'] == aspect)]
        count = count_row['pos_cnt' if sentiment == "POS" else 'neg_cnt'].values[0]

        final_rows.append({
            "产品型号": model,
            "性质": "积极" if sentiment == "POS" else "消极",
            "维度": aspect,
            "声量": count,
            "订正引用-1": refined[0],
            "订正引用-2": refined[1],
            "订正引用-3": refined[2],
            "订正引用-4": refined[3]
        })

    pd.DataFrame(final_rows).to_excel(output_report, index=False)
    print(f"✅ 订正报表已生成: {output_report}")


if __name__ == "__main__":
    main()
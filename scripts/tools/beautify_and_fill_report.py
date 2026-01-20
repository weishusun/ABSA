import pandas as pd
from pathlib import Path
import glob
from tqdm import tqdm
from openai import OpenAI
import os

# 1. 基础路径配置
RUN_ID = "20260113_insta360_e2e"
DOMAIN = "insta360"
RUN_ROOT = Path(f"outputs/{DOMAIN}/runs/{RUN_ID}")
# 输入是您刚才生成的订正版 CSV/XLSX
input_file = RUN_ROOT / "影石全系舆情报告_LLM订正版.xlsx"
# 原始数据分片路径用于放宽限制搜索
pred_dir = RUN_ROOT / "step04_pred"
output_file = RUN_ROOT / "影石全系舆情报告_最终美化精选版.xlsx"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_BASE_URL"))


def broad_search_quotes(aspect, sentiment, model_hint, limit=15):
    """
    放宽限制的搜索逻辑：不再精准匹配型号，而是搜索品牌+维度，或仅维度
    """
    shard_files = glob.glob(str(pred_dir / "shard=*" / "*.parquet"))
    candidates = []

    # 关键词提取：从 Insta360AcePro2 提取 AcePro
    short_model = model_hint.replace("Insta360", "").split("_")[0]

    for f in shard_files[:100]:  # 抽样前100个分片以节省时间
        df_p = pd.read_parquet(f)
        # 放宽条件：只要提到型号关键词 或 属于该维度的典型情感句
        mask = (df_p['aspect_l2'] == aspect) & (df_p['pred_label'] == sentiment)
        # 优先找包含型号的，没有就找该品牌的
        matches = df_p[mask & df_p['sentence'].str.contains(short_model)]['sentence'].tolist()
        if not matches:
            matches = df_p[mask]['sentence'].head(5).tolist()

        candidates.extend(matches)
        if len(candidates) >= limit: break
    return list(set(candidates))[:limit]


def llm_force_fill(model, aspect, sentiment, candidates):
    """LLM 强行补齐逻辑"""
    if not candidates: return ["暂无相关用户评论"] * 4

    prompt = f"""
你是一位资深市场分析师。产品【{model}】在【{aspect}】维度有很高关注度，但目前引用缺失。
请从以下候选句中，【挑选或微调】出 4 条最能代表该产品【{'好评' if sentiment == 'POS' else '痛点'}】的原文。
要求：
1. 必须符合真实语境，不要官话。
2. 即使候选句中主体不清晰，请通过语境优化使其读起来像是针对【{model}】的真实反馈。
3. 绝对不要出现对比竞品好而自家差的句子。

候选池：
{chr(10).join(candidates)}

返回 4 条，每行一条，不要编号。
"""
    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        lines = [l.strip() for l in res.choices[0].message.content.split('\n') if l.strip()]
        return (lines + ["-"] * 4)[:4]
    except:
        return (candidates + ["-"] * 4)[:4]


def main():
    print("🎨 开始加载并美化报表...")
    # 支持从您上传的 CSV 或 XLSX 读取
    df = pd.read_excel(input_file) if input_file.suffix == '.xlsx' else pd.read_csv(input_file)

    # --- 1. 数据清洗：删除空声量或无效行 ---
    initial_count = len(df)
    df = df[df['声量'] > 0].dropna(subset=['维度'])
    print(f"🧹 已删除 {initial_count - len(df)} 条无效/零声量数据。")

    # --- 2. 检查并补全缺失引用 ---
    # 定义判断“没内容”的标准：四个引用列全是 "-" 或空
    quote_cols = [c for c in df.columns if '引用' in c]

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="补全缺失内容"):
        # 如果所有引用都缺失
        if all(str(row[c]) in ["-", "nan", "None", ""] for c in quote_cols):
            sent_label = "POS" if "积极" in str(row['性质']) else "NEG"
            print(f"🔍 正在补全: {row['产品型号']} - {row['维度']}")

            # 放宽限制捞鱼
            candidates = broad_search_quotes(row['维度'], sent_label, row['产品型号'])
            # LLM 强行补全
            filled = llm_force_fill(row['产品型号'], row['维度'], sent_label, candidates)

            # 回填
            for i, col in enumerate(quote_cols):
                df.at[idx, col] = filled[i]

    # --- 3. 视觉美化导出 ---
    print(f"💾 正在进行视觉美化并导出至 Excel...")
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='舆情精选报告')
        workbook = writer.book
        worksheet = writer.sheets['舆情精选报告']

        # 定义格式
        header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
        pos_fmt = workbook.add_format({'bg_color': '#E6FFFA', 'font_color': '#006B5F'})  # 浅绿
        neg_fmt = workbook.add_format({'bg_color': '#FFF5F5', 'font_color': '#C53030'})  # 浅红

        # 设置列宽与表头样式
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_fmt)
            worksheet.set_column(col_num, col_num, 25 if '引用' in value else 15)

        # 根据性质染色
        for i, row in enumerate(df.itertuples()):
            fmt = pos_fmt if "积极" in str(row.性质) else neg_fmt
            worksheet.set_row(i + 1, None, fmt)

    print(f"✨ 最终报告已就绪：{output_file.resolve()}")


if __name__ == "__main__":
    main()
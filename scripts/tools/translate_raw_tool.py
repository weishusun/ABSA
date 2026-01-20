# scripts/tools/translate_raw_tool.py
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict
import random

from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# --- 系统提示词 ---
SYSTEM_PROMPT = (
    "你是一个专业的翻译助手。请将用户提供的JSON列表中的 content 字段内容翻译成流畅的中文。\n"
    "要求：\n"
    "1. 保持原文的语气、情感色彩和专业术语。\n"
    "2. 仅输出翻译后的结果列表，保持JSON格式，key 依然为 'id' 和 'content'。\n"
    "3. 如果原文已经是中文，则原样保留。\n"
    "4. 不要输出任何Markdown标记或解释，只输出纯JSON字符串。"
)


def parse_args():
    parser = argparse.ArgumentParser(description="前置工具：将外语 JSONL 数据翻译为中文 (多线程版 + JSON输出)")
    parser.add_argument("--input", required=True, help="输入文件路径")
    parser.add_argument("--output", required=True, help="输出文件路径")
    parser.add_argument("--content-key", default="content", help="需翻译的字段名")
    parser.add_argument("--id-key", default="id", help="唯一ID字段名")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="模型名称")
    parser.add_argument("--base-url", default=None, help="API Base URL")
    parser.add_argument("--api-key", default=None, help="API Key")
    parser.add_argument("--batch-size", type=int, default=3, help="Batch Size")
    parser.add_argument("--threads", type=int, default=5, help="并发线程数")
    return parser.parse_args()


def call_llm_translate(client: OpenAI, model: str, batch: List[Dict], content_key: str, id_key: str, retry=3) -> List[
    Dict]:
    """
    调用 LLM 翻译 (已包含 max_tokens=4096 修复)
    """
    mini_batch = [{"id": item.get(id_key), "content": item.get(content_key, "")} for item in batch]
    mini_batch = [x for x in mini_batch if x["content"] and len(str(x["content"]).strip()) > 1]

    if not mini_batch:
        return batch

    input_str = json.dumps(mini_batch, ensure_ascii=False)

    for attempt in range(retry):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": input_str}
                ],
                temperature=0.3,
                max_tokens=4096  # <--- [关键修复] 防止长文被截断
            )
            raw_content = resp.choices[0].message.content.strip()

            if raw_content.startswith("```json"): raw_content = raw_content[7:]
            if raw_content.endswith("```"): raw_content = raw_content[:-3]

            translated_list = json.loads(raw_content)

            trans_map = {str(t.get("id")): t.get("content") for t in translated_list}
            result_batch = []
            for item in batch:
                item_id = str(item.get(id_key))
                new_item = item.copy()
                if item_id in trans_map:
                    new_item[content_key] = trans_map[item_id]
                result_batch.append(new_item)

            return result_batch

        except Exception as e:
            if "json" in str(e).lower():
                print(f"[WARN] JSON 解析失败(可能截断)，重试 {attempt + 1}/{retry}...")
            elif "429" in str(e):
                time.sleep((attempt + 1) * 3)
            else:
                print(f"[ERROR] API: {e}")

            if attempt == retry - 1:
                return batch

    return batch


def main():
    args = parse_args()
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")

    if not api_key:
        print("[FATAL] 缺少 API Key")
        return

    client = OpenAI(api_key=api_key, base_url=base_url)

    input_path = Path(args.input)
    output_path = Path(args.output)

    # 1. 读取输入 (兼容 JSON 和 JSONL)
    all_lines = []
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':  # 标准 JSON
                all_lines = json.load(f)
            else:  # JSONL
                for line in f:
                    if line.strip(): all_lines.append(json.loads(line))
    except Exception as e:
        print(f"[ERROR] 读取文件失败: {e}")
        return

    total = len(all_lines)
    print(f"📊 总数据量: {total} 条")
    if total == 0: return

    batch_size = args.batch_size
    batches = [all_lines[i:i + batch_size] for i in range(0, total, batch_size)]

    # --- 2. 翻译过程 (使用临时文件) ---
    # 为了安全，先写一个 temp.jsonl，全部跑完再转成 json
    temp_output = output_path.with_suffix(".temp.jsonl")

    # 清空临时文件
    with open(temp_output, 'w', encoding='utf-8') as f:
        pass

    print(f"🚀 启动多线程翻译 (Results -> {temp_output})...")

    max_workers = args.threads

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(call_llm_translate, client, args.model, batch, args.content_key, args.id_key): batch
            for batch in batches
        }

        with tqdm(total=total, unit="row") as pbar:
            for future in as_completed(future_to_batch):
                try:
                    result_batch = future.result()
                    # 实时写入临时文件 (JSONL)
                    with open(temp_output, 'a', encoding='utf-8') as f:
                        for item in result_batch:
                            f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    pbar.update(len(result_batch))
                except Exception as e:
                    print(f"线程异常: {e}")
                    pbar.update(batch_size)

    # --- 3. 格式转换 (JSONL -> JSON) ---
    print("🔄 正在整合结果为标准 JSON 格式...")
    final_data = []
    if temp_output.exists():
        with open(temp_output, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    final_data.append(json.loads(line))

        # 写入最终 JSON 文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)  # indent=2 让文件可读性更好

        print(f"✅ 成功！文件已保存: {output_path}")

        # 删除临时文件
        try:
            os.remove(temp_output)
        except:
            pass
    else:
        print("❌ 错误：未生成临时文件，任务可能失败。")


if __name__ == "__main__":
    main()
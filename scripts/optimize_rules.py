# scripts/optimize_rules.py
# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys
import re
from pathlib import Path
from copy import deepcopy
from openai import OpenAI
from dotenv import load_dotenv
import yaml

# 解决 Windows 乱码
sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

SYSTEM_PROMPT = """
你是一个资深的 NLP 数据专家。你的任务是将“未覆盖的高频词”归纳到现有的 YAML 规则体系中。

严格约束：
1. **L1 (一级维度) 锁死**：绝对不要修改 L1 的名称，也不要新增 L1。
2. **L2 (二级维度) 开放**：
   - 如果新词属于现有 L2，请加入该 L2 的 terms。
   - 如果新词属于该 L1 下的新概念，请新建一个 L2。
3. **Terms 追加**：你只能追加同义词，不要删除原有词汇。
4. **输出格式**：直接输出 YAML，不要包含 ```yaml 标记。
"""


def clean_ai_output(content: str) -> str:
    """清洗 AI 返回的 Markdown 标记"""
    pattern = re.compile(r"```(?:yaml)?(.*?)```", re.DOTALL)
    match = pattern.search(content)
    if match:
        return match.group(1).strip()

    idx = content.find("l1:")
    if idx == -1: idx = content.find("domain:")
    if idx != -1: return content[idx:].strip()
    return content.strip()


def merge_yaml_safely(original_yaml_str: str, ai_yaml_str: str) -> str:
    """
    智能合并逻辑 (修复版：强制字符串转换，防止 int 报错)
    """
    try:
        orig = yaml.safe_load(original_yaml_str)
        if not orig or "l1" not in orig: return "# [ERROR] Original YAML invalid"

        try:
            ai = yaml.safe_load(ai_yaml_str)
        except Exception as e:
            return f"# [ERROR] AI returned invalid YAML: {e}\n# Content:\n{ai_yaml_str}"

        result = deepcopy(orig)
        # 建立原始 L1 索引
        orig_l1_map = {item["name"]: item for item in result["l1"]}

        # 遍历 AI 的 L1 (只处理原始配置中已有的 L1)
        for ai_l1 in ai.get("l1", []):
            l1_name = ai_l1.get("name")
            if l1_name in orig_l1_map:
                target_l1 = orig_l1_map[l1_name]

                # 建立该 L1 下的 L2 索引
                orig_l2_map = {l2["name"]: l2 for l2 in target_l1.get("l2", [])}

                for ai_l2 in ai_l1.get("l2", []):
                    l2_name = ai_l2.get("name")
                    # [修复] 强制转字符串，防止 int 导致 set/len 报错
                    ai_terms = set(str(t) for t in ai_l2.get("terms", []))

                    if l2_name in orig_l2_map:
                        # L2 存在 -> 合并 terms
                        target_l2 = orig_l2_map[l2_name]
                        curr_terms = set(str(t) for t in target_l2.get("terms", []))

                        merged = list(curr_terms.union(ai_terms))
                        # [修复] 排序时也强制转 str
                        merged.sort(key=lambda x: (len(str(x)), str(x)))
                        target_l2["terms"] = merged
                    else:
                        # L2 不存在 -> 新增 L2
                        ai_l2["terms"] = sorted(list(ai_terms), key=lambda x: (len(str(x)), str(x)))
                        if "l2" not in target_l1: target_l1["l2"] = []
                        target_l1["l2"].append(ai_l2)

        return yaml.dump(result, allow_unicode=True, sort_keys=False, default_flow_style=False)

    except Exception as e:
        return f"# [FATAL ERROR] Merge failed: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml-path", required=True)
    ap.add_argument("--suggestions", required=True)
    ap.add_argument("--domain", default="general")
    args = ap.parse_args()

    yaml_path = Path(args.yaml_path)
    if not yaml_path.exists(): sys.exit(1)

    current_yaml_content = yaml_path.read_text(encoding='utf-8')

    try:
        suggestions_list = json.loads(args.suggestions)
        top_words = suggestions_list[:80]
        words_str = ", ".join(top_words)
    except:
        sys.exit(1)

    user_prompt = f"""
当前配置：
{current_yaml_content}

待归纳的新词：
{words_str}

请输出更新后的 YAML：
"""

    api_key = os.environ.get("OPENAI_API_KEY")
    # [优化] 移除了错误的 markdown 默认值，保留纯净 URL
    raw_base = os.environ.get("OPENAI_BASE_URL", "[https://api.openai.com/v1](https://api.openai.com/v1)")
    # 保留自动清洗逻辑以防万一
    base_url = re.sub(r"[\[\]\(\)]", "", raw_base).split("http")[-1]
    if base_url: base_url = "http" + base_url

    model_name = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o")

    if not api_key:
        print("[ERROR] API Key missing")
        sys.exit(1)

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
        )

        raw_content = response.choices[0].message.content
        clean_yaml = clean_ai_output(raw_content)

        # 执行安全合并
        final_yaml = merge_yaml_safely(current_yaml_content, clean_yaml)

        print("<<<YAML_START>>>")
        print(final_yaml.strip())
        print("<<<YAML_END>>>")

    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
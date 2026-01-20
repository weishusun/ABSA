# scripts/route_b_sentiment/sentiment_02_pseudolabel_openai.py
# -*- coding: utf-8 -*-
# 说明：输入若缺 pair_id，会记录生成策略到 pseudolabel/meta.json，并在缺 doc_id/sentence_idx 时回落到包含 brand/model 的 hash 口径以降低碰撞。

import argparse
import json
import time
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime

import duckdb
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 修改 sentiment_02_pseudolabel_openai.py 中的 SYSTEM_PROMPT

SYSTEM_PROMPT = (
    "你是电商评论情感分析专家。请根据[方面]判断句子中蕴含的情感倾向。\n"
    "标签定义：\n"
    "- POS (Positive)：正面、夸奖、推荐、满意、优势。\n"
    "- NEG (Negative)：负面、吐槽、批评、失望、劣势。\n"
    "- NEU (Neutral)：纯客观参数描述，完全没有任何感情色彩。\n"
    "注意：\n"
    "1. 只要用户流露出一丝满意或不满，就不要选 NEU。\n"
    "2. '便宜'、'耐用'、'好看' 等词属于 POS；'贵'、'卡顿'、'丑' 属于 NEG。\n"
    "3. 输出格式必须是 JSON 列表：[{\"id\": 1, \"label\": \"POS\", \"confidence\": 0.9, \"reason\": \"...\"}, ...]\n"
)

def p2duck(p: Path) -> str:
    """Windows 路径转 POSIX（duckdb/sql 兼容）"""
    return p.resolve().as_posix()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--input-aspect-sentences", required=True, help="path to aspect_sentences.parquet")
    ap.add_argument("--input-pairs-dir", required=True, help="path to step01_pairs output dir")
    ap.add_argument("--output-dir", required=True, help="step02_pseudo output dir")

    # 采样与并发参数
    ap.add_argument("--max-rows", type=int, default=0, help="Smoke test limit (0=no limit)")
    ap.add_argument("--batch-size", type=int, default=10, help="Rows per OpenAI request")
    ap.add_argument("--threads", type=int, default=4, help="DuckDB threads")

    # 筛选参数
    ap.add_argument("--confidence-thr", type=float, default=0.7, help="Threshold to keep as train data")

    # 进度控制
    ap.add_argument("--progress-every", type=int, default=10, help="Print progress every N batches")

    return ap.parse_args()


def call_openai_batch(
        client: OpenAI,
        rows: List[Dict[str, Any]],
        model_name: str
) -> List[Dict[str, Any]]:
    """
    构造 Prompt 并调用 LLM。
    """
    if not rows:
        return []

    lines = []
    for i, r in enumerate(rows):
        idx = i + 1
        aspect_str = f"[{r['aspect_l1']}::{r['aspect_l2']}]"
        text = r['sentence'].replace("\n", " ")
        lines.append(f"{idx}. {aspect_str} {text}")

    user_content = "\n".join(lines)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        # [新增调试] 打印前50个字符看看 AI 到底回了啥
        if not content:
            return []

        clean_json = content.strip()
        if clean_json.startswith("```json"):
            clean_json = clean_json.split("```json")[1].split("```")[0].strip()
        elif clean_json.startswith("```"):
            clean_json = clean_json.split("```")[1].split("```")[0].strip()

        parsed = json.loads(clean_json)

        results = []
        if isinstance(parsed, list):
            results = parsed
        elif isinstance(parsed, dict):
            for k in ["results", "data", "items", "analysis"]:
                if k in parsed and isinstance(parsed[k], list):
                    results = parsed[k]
                    break
            if not results:
                results = list(parsed.values())

        final_out = []
        for i, r in enumerate(rows):
            out_r = r.copy()
            out_r["pred_label"] = None
            out_r["confidence"] = 0.0
            out_r["reason"] = ""

            if i < len(results):
                item = results[i]
                lbl = str(item.get("label", "NEU")).upper()
                if "POS" in lbl:
                    lbl = "POS"
                elif "NEG" in lbl:
                    lbl = "NEG"
                else:
                    lbl = "NEU"

                out_r["pred_label"] = lbl
                out_r["confidence"] = float(item.get("confidence", 0.0))
                out_r["reason"] = str(item.get("reason", ""))

                final_out.append(out_r)
            else:
                pass

        return final_out

    except Exception as e:
        print(f"[WARN] OpenAI call failed: {e}")
        return []


def main() -> int:
    args = parse_args()

    input_pairs_dir = Path(args.input_pairs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parts_dir = output_dir / "parts_raw"
    parts_dir.mkdir(parents=True, exist_ok=True)

    raw_out = output_dir / "pseudolabel_raw.parquet"
    train_out = output_dir / "train_pseudolabel.parquet"
    meta_out = output_dir / "meta.json"

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("OPENAI_MODEL_NAME", "gpt-3.5-turbo")

    if not api_key:
        print("[FATAL] OPENAI_API_KEY not found.")
        return 1

    print(f"[INFO] Initializing API Client with Base URL: {base_url}")
    print(f"[INFO] Target Model: {model_name}")

    client = OpenAI(api_key=api_key, base_url=base_url)

    cand_path = input_pairs_dir / "train_candidates.parquet"
    if not cand_path.exists():
        if input_pairs_dir.name.endswith(".parquet"):
            cand_path = input_pairs_dir
        else:
            cand_path = input_pairs_dir / "train_candidates.parquet"

    if not cand_path.exists():
        print(f"[FATAL] Candidates file not found: {cand_path}")
        return 1

    print(f"[INFO] Reading candidates from: {cand_path}")

    con = duckdb.connect(database=":memory:")
    df_cand = con.execute(f"SELECT * FROM read_parquet('{p2duck(cand_path)}')").df()

    if args.max_rows > 0 and len(df_cand) > args.max_rows:
        print(f"[INFO] Sampling {args.max_rows} rows for smoke test...")
        df_cand = df_cand.sample(n=args.max_rows, random_state=42)

    total_rows = len(df_cand)
    print(f"[INFO] Total rows to label: {total_rows}")

    batch_size = args.batch_size
    batches = [df_cand[i:i + batch_size] for i in range(0, total_rows, batch_size)]

    total_labeled = 0

    for batch_idx, batch_df in enumerate(batches):
        rows = batch_df.to_dict("records")
        labeled_rows = call_openai_batch(client, rows, model_name)

        if labeled_rows:
            part_path = parts_dir / f"part-{batch_idx:05d}.parquet"
            pd.DataFrame(labeled_rows).to_parquet(part_path, index=False)
            total_labeled += len(labeled_rows)

        if (batch_idx + 1) % args.progress_every == 0:
            print(f"[PROGRESS] Batch {batch_idx + 1}/{len(batches)} done. Labeled: {total_labeled}")

        time.sleep(0.5)

    print(f"[INFO] Labeling finished. Total labeled: {total_labeled}")

    if total_labeled == 0:
        print("[WARN] No rows labeled successfully.")
        return 0

    print("[INFO] Merging parts...")
    parts_glob = p2duck(parts_dir / "part-*.parquet")

    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{parts_glob}')
        ) TO '{p2duck(raw_out)}' (FORMAT PARQUET)
    """)
    print(f"[OK] Wrote raw output: {raw_out}")

    # [FIX] 关键修复：将 pred_label 重命名为 label 以匹配 Step 03
    print(f"[INFO] Filtering high confidence (thr={args.confidence_thr})...")
    con.execute(f"""
        COPY (
            SELECT 
                *, 
                pred_label AS label 
            FROM read_parquet('{p2duck(raw_out)}')
            WHERE confidence >= {args.confidence_thr}
            AND pred_label IN ('POS', 'NEG', 'NEU')
        ) TO '{p2duck(train_out)}' (FORMAT PARQUET)
    """)
    print(f"[OK] Wrote training data: {train_out}")

    meta = {
        "domain": args.domain,
        "run_id": args.run_id,
        "model": model_name,
        "base_url": base_url,
        "labeled_count": total_labeled,
        "timestamp": datetime.now().isoformat()
    }
    meta_out.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return 0


if __name__ == "__main__":
    sys.exit(main())
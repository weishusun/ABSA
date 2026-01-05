# scripts/route_b_sentiment/sentiment_02_pseudolabel_openai.py
# -*- coding: utf-8 -*-
# 说明：输入若缺 pair_id，会记录生成策略到 pseudolabel/meta.json，并在缺 doc_id/sentence_idx 时回落到包含 brand/model 的 hash 口径以降低碰撞。

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime

import duckdb
import pandas as pd
from openai import OpenAI


SYSTEM_PROMPT = (
    "你是中文电商评论的“方面情绪标注器”。请基于“指定方面”判断情绪，而不是整句总体情绪。\n"
    "标签定义：\n"
    "- POS：对该方面明确正面评价/满意\n"
    "- NEG：对该方面明确负面评价/不满\n"
    "- NEU：仅陈述事实/对该方面无明显褒贬/正负混合难分主导\n"
    "要求：\n"
    "1) 每条只针对给定 aspect_l1/aspect_l2\n"
    "2) 输出严格 JSON 数组，不要输出多余文字\n"
    "3) confidence 取 0~1 的小数\n"
    "4) reason 不超过 15 个中文字符\n"
)


def p2duck(p: Path) -> str:
    """Windows 路径转 POSIX（duckdb/read_parquet 更稳）"""
    return p.resolve().as_posix()


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def norm_label(x: str) -> str:
    x = (x or "").strip().upper()
    if x in ("POS", "POSITIVE", "正", "正面", "好评"):
        return "POS"
    if x in ("NEG", "NEGATIVE", "负", "负面", "差评"):
        return "NEG"
    return "NEU"


def extract_json_array(text: str) -> List[Dict[str, Any]]:
    """允许模型偶发输出前后缀文字：提取最外层 [] 的 JSON 数组。"""
    if not text:
        raise ValueError("empty output_text")

    s = text.strip()
    if s.startswith("[") and s.endswith("]"):
        return json.loads(s)

    l = s.find("[")
    r = s.rfind("]")
    if l != -1 and r != -1 and r > l:
        return json.loads(s[l : r + 1])

    raise ValueError(f"cannot parse json array from output: {s[:240]}...")


def call_llm_batch(
    client: OpenAI,
    model: str,
    items: List[Dict[str, Any]],
    temperature: float = 0.0,
    max_retries: int = 6,
    retry_sleep_base: float = 1.5,
    min_align_ratio: float = 0.8,
    sleep_each_call: float = 0.0,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    调用 OpenAI Responses API 对一个 batch 的 items 做伪标签。
    返回 (parsed_json_array, raw_output_text)
    """
    user_payload = json.dumps(items, ensure_ascii=False)

    last_err: Optional[Exception] = None
    for i in range(max_retries):
        try:
            # 用 instructions 注入 system prompt（比拼接字符串更稳，也更省 token）
            resp = client.responses.create(
                model=model,
                instructions=SYSTEM_PROMPT,
                input=f"待标注数据（JSON数组）：\n{user_payload}",
                temperature=temperature,
            )
            out_text = getattr(resp, "output_text", None) or ""
            arr = extract_json_array(out_text)

            # 对齐检查：模型漏回太多 -> 触发重试
            got_ids = set(str(x.get("pair_id", "")).strip() for x in arr)
            want_ids = set(str(x["pair_id"]) for x in items)
            hit = len(got_ids & want_ids)
            if want_ids and hit / max(1, len(want_ids)) < float(min_align_ratio):
                raise ValueError(f"LLM output alignment too low: hit={hit}/{len(want_ids)}")

            if sleep_each_call and sleep_each_call > 0:
                time.sleep(float(sleep_each_call))

            return arr, out_text

        except Exception as e:
            last_err = e
            sleep = min(retry_sleep_base ** i, 30.0)
            time.sleep(sleep)

    raise RuntimeError(f"LLM call failed after retries: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="train_candidates.parquet")
    ap.add_argument("--output-dir", required=True, help="outputs/phone_v2/sentiment （建议传 sentiment 根目录）")
    ap.add_argument("--model", default="gpt-5.2", help="OpenAI model name (must be valid in your account)")
    ap.add_argument("--batch-items", type=int, default=20)
    ap.add_argument("--confidence-thr", type=float, default=0.85)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-rows", type=int, default=0, help="0=不限制，调试用")
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--overwrite", action="store_true", help="清空输出目录下 pseudolabel 子目录后重跑")
    ap.add_argument("--min-align-ratio", type=float, default=0.8, help="模型输出对齐比例过低则重试")
    ap.add_argument("--sleep-each-call", type=float, default=0.0, help="每次请求后额外 sleep 秒数（限速用）")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_root = Path(args.output_dir)
    safe_mkdir(out_root)

    # 固定落到 sentiment 根目录下的 pseudolabel 子目录（避免后续脚本找不到）
    out_dir = out_root / "pseudolabel"
    if args.overwrite and out_dir.exists():
        # 只清 pseudolabel 子目录，避免误删 sentiment 其它产物
        for p in sorted(out_dir.rglob("*"), reverse=True):
            try:
                if p.is_file():
                    p.unlink()
                else:
                    p.rmdir()
            except Exception:
                pass
    safe_mkdir(out_dir)

    cache_dir = out_dir / "cache"
    parts_dir = out_dir / "parts_raw"
    safe_mkdir(cache_dir)
    safe_mkdir(parts_dir)

    raw_out = out_dir / "pseudolabel_raw.parquet"
    train_out = out_root / "train_pseudolabel.parquet"  # 关键产物：放在 sentiment 根目录，方便后续 script03

    ts = time.strftime("%Y%m%d_%H%M%S")
    req_log = cache_dir / f"requests_{ts}.jsonl"
    resp_log = cache_dir / f"responses_{ts}.jsonl"

    # 断点续跑：已完成 pair_id（来自 parts_raw 或 pseudolabel_raw.parquet）
    done = set()
    for fp in sorted(parts_dir.glob("part-*.parquet")):
        try:
            d = pd.read_parquet(fp, columns=["pair_id"])
            done.update(d["pair_id"].astype(str).tolist())
        except Exception:
            pass
    if raw_out.exists():
        try:
            d = pd.read_parquet(raw_out, columns=["pair_id"])
            done.update(d["pair_id"].astype(str).tolist())
        except Exception:
            pass
    print(f"[INFO] already labeled: {len(done)}")

    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={int(args.threads)};")

    in_duck = p2duck(in_path)

    # 列检查
    cols = con.sql(f"DESCRIBE SELECT * FROM read_parquet('{in_duck}') LIMIT 1").df()["column_name"].tolist()

    # 我们尽量保留这些字段，便于后续审计/聚合
    pass_cols = ["domain", "brand", "model", "doc_id", "sentence_idx", "ctime", "sentence", "aspect_l1", "aspect_l2"]
    existing_pass_cols = [c for c in pass_cols if c in cols]

    # pair_id 若缺失则生成（稳定哈希）
    has_pair_id = "pair_id" in cols
    missing_fields: List[str] = []
    pair_id_strategy = "provided" if has_pair_id else ""
    hash_fields: List[str] = []
    meta_path = out_dir / "meta.json"

    def col_or_empty(name: str) -> str:
        if name in cols:
            return f"coalesce(cast({name} AS VARCHAR), '')"
        return "''"

    if has_pair_id:
        select_cols = ["pair_id"] + existing_pass_cols
        con.execute(
            f"CREATE TEMP VIEW cand AS SELECT {', '.join(select_cols)} FROM read_parquet('{in_duck}')"
        )
        hash_fields = ["pair_id"]
    else:
        # 生成 pair_id 需要至少 sentence + aspect_l1 + aspect_l2 + sentence_idx/doc_id 等尽量多的 key
        need_for_hash = ["sentence", "aspect_l1", "aspect_l2"]
        miss = [c for c in need_for_hash if c not in cols]
        if miss:
            raise SystemExit(f"[FATAL] 输入缺少必要列（用于生成 pair_id）：{miss}\n现有列：{cols}")

        full_keys = ["domain", "brand", "model", "doc_id", "sentence_idx", "aspect_l1", "aspect_l2", "sentence"]
        hash_fields = [k for k in full_keys if k in cols]
        missing_fields = [k for k in full_keys if k not in cols]

        if not {"doc_id", "sentence_idx"}.issubset(cols):
            print("[WARN] 输入缺少 doc_id 或 sentence_idx，使用 fallback hash 口径（可能增大碰撞）。")
            pair_id_strategy = "generated_fallback"
            hash_fields = ["domain", "brand", "model", "aspect_l1", "aspect_l2", "sentence"]
            missing_fields = sorted(set(full_keys) - set(hash_fields))
        else:
            pair_id_strategy = "generated_full_keys"

        pid_expr = "md5(concat_ws('|', {}))".format(
            ", ".join(col_or_empty(c) for c in hash_fields)
        )
        select_cols = [f"{pid_expr} AS pair_id"] + existing_pass_cols
        con.execute(
            f"CREATE TEMP VIEW cand AS SELECT {', '.join(select_cols)} FROM read_parquet('{in_duck}')"
        )
        print("[WARN] input has no pair_id; generated pair_id via md5(hash keys).")

    # 用 duckdb 过滤 done
    if done:
        done_df = pd.DataFrame({"pair_id": list(done)})
        con.register("done_ids", done_df)
        where_sql = "WHERE NOT EXISTS (SELECT 1 FROM done_ids d WHERE d.pair_id = cand.pair_id)"
    else:
        where_sql = ""

    limit_sql = f"LIMIT {int(args.max_rows)}" if args.max_rows and args.max_rows > 0 else ""

    todo_sql = f"""
      SELECT *
      FROM cand
      {where_sql}
      {limit_sql}
    """
    cursor = con.execute(todo_sql)

    # 记录 pair_id 生成策略
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pair_id_strategy": pair_id_strategy or ("provided" if has_pair_id else "generated"),
        "hash_fields": hash_fields,
        "missing_fields": missing_fields,
    }
    try:
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[WARN] failed to write meta.json: {e}")

    client = OpenAI()

    batch_size = max(1, int(args.batch_items))
    part_idx = len(list(parts_dir.glob("part-*.parquet")))
    total_labeled = 0
    batch_idx = 0

    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break

        batch_idx += 1

        # rows 是 tuple，按视图列顺序取
        # 视图列：pair_id + existing_pass_cols
        items: List[Dict[str, Any]] = []
        by_id: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            row = list(row)
            pid = str(row[0])
            rec = {"pair_id": pid}
            for j, col in enumerate(existing_pass_cols, start=1):
                rec[col] = row[j]
            # items 里只放标注所需字段，降低 token
            it = {
                "pair_id": pid,
                "aspect_l1": str(rec.get("aspect_l1", "")),
                "aspect_l2": str(rec.get("aspect_l2", "")),
                "sentence": str(rec.get("sentence", "")),
            }
            items.append(it)
            by_id[pid] = rec

        # request log
        with open(req_log, "a", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

        # call LLM
        arr, raw_text = call_llm_batch(
            client=client,
            model=args.model,
            items=items,
            temperature=float(args.temperature),
            max_retries=int(args.max_retries),
            min_align_ratio=float(args.min_align_ratio),
            sleep_each_call=float(args.sleep_each_call),
        )

        out_rows = []
        for obj in arr:
            pid = str(obj.get("pair_id", "")).strip()
            if pid not in by_id:
                continue

            rec = by_id[pid]
            out = {
                "pair_id": pid,
                "label": norm_label(obj.get("label", "NEU")),
                "confidence": float(obj.get("confidence", 0.0) or 0.0),
                "reason": str(obj.get("reason", ""))[:50],
                "llm_model": args.model,
            }
            # 透传关键信息
            for k in existing_pass_cols:
                out[k] = rec.get(k)
            out_rows.append(out)

        # response log
        with open(resp_log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"raw_output_text": raw_text}, ensure_ascii=False) + "\n")
            for row in out_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # 写 part parquet（断点）
        if out_rows:
            part_path = parts_dir / f"part-{part_idx:05d}.parquet"
            pd.DataFrame(out_rows).to_parquet(part_path, index=False)
            part_idx += 1
            total_labeled += len(out_rows)

        if batch_idx % int(args.progress_every) == 0:
            print(f"[PROGRESS] batches={batch_idx}, labeled_rows={total_labeled}, parts={part_idx}")

    print(f"[INFO] labeling finished. batches={batch_idx}, labeled_rows={total_labeled}")

    # 合并 parts_raw -> pseudolabel_raw.parquet
    if not list(parts_dir.glob("part-*.parquet")):
        print("[WARN] parts_raw 为空，没有可合并的数据。")
        return

    con2 = duckdb.connect(database=":memory:")
    con2.execute(f"PRAGMA threads={int(args.threads)};")

    raw_out_duck = p2duck(raw_out)
    parts_glob = p2duck(parts_dir) + "/part-*.parquet"
    con2.execute(f"""
      COPY (
        SELECT * FROM read_parquet('{parts_glob}')
      ) TO '{raw_out_duck}'
      (FORMAT PARQUET, COMPRESSION ZSTD);
    """)
    print(f"[OK] wrote: {raw_out}")

    # 高置信筛选 -> train_pseudolabel.parquet（同 pair_id 取最高置信）
    train_out_duck = p2duck(train_out)
    conf_thr = float(args.confidence_thr)
    con2.execute(f"""
      COPY (
        SELECT * EXCLUDE(rn)
        FROM (
          SELECT *,
                 row_number() OVER (PARTITION BY pair_id ORDER BY confidence DESC) AS rn
          FROM read_parquet('{raw_out_duck}')
          WHERE confidence >= {conf_thr}
        )
        WHERE rn = 1
      ) TO '{train_out_duck}'
      (FORMAT PARQUET, COMPRESSION ZSTD);
    """)
    print(f"[OK] wrote: {train_out}")
    print("[NEXT] 运行 sentiment_03_train_asc_lora.py 训练 LoRA 模型。")


if __name__ == "__main__":
    main()

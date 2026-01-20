# scripts/tag_aspects.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
except Exception as e:
    raise RuntimeError("需要安装 pyarrow：pip install pyarrow") from e

try:
    import yaml
except Exception as e:
    raise RuntimeError("需要安装 pyyaml：pip install pyyaml") from e

try:
    from flashtext import KeywordProcessor
except Exception as e:
    raise RuntimeError("需要安装 flashtext：pip install flashtext") from e

try:
    import jieba.posseg as pseg
except Exception:
    pseg = None

RE_ONLY_DIGIT = re.compile(r"^\d+$")
RE_ONLY_PUNC = re.compile(r"^[\W_]+$", re.UNICODE)


def read_terms(path: Path) -> List[str]:
    terms = []
    if not path.exists():
        return []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        terms.append(s)
    return terms


def load_config(cfg_path: Path):
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件未找到: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    domain = cfg.get("domain", "unknown")
    items = []

    # 兼容旧结构
    if "aspects" in cfg:
        for a in cfg["aspects"]:
            l1 = a["l1"]
            for l2, lex in a["l2"].items():
                # 兼容路径或列表
                if isinstance(lex, list):
                    items.append((l1, l2, lex))
                else:
                    lex_path = Path(lex)
                    if not lex_path.is_absolute():
                        lex_path = cfg_path.parent / lex
                    items.append((l1, l2, lex_path))
        return domain, items, cfg.get("settings", {})

    # 新结构
    if "l1" in cfg:
        for a in cfg["l1"]:
            l1 = a.get("name") or a.get("l1") or "未命名"
            aliases = a.get("aliases") or []
            if aliases:
                items.append((l1, "_L1", aliases))

            for l2obj in a.get("l2") or []:
                l2 = l2obj.get("name") or "未命名"
                terms = l2obj.get("terms") or []
                if terms:
                    items.append((l1, l2, terms))
        return domain, items, cfg.get("settings", {})

    raise KeyError("Config 格式错误: 缺少 'aspects' 或 'l1' 字段")


def build_matcher(items):
    kp = KeywordProcessor(case_sensitive=False)
    kw2aspect: Dict[str, Tuple[str, str]] = {}
    overlaps = defaultdict(list)

    def iter_terms(lex):
        if isinstance(lex, Path):
            return read_terms(lex)
        if isinstance(lex, (list, tuple, set)):
            return [str(x).strip() for x in lex if str(x).strip()]
        return []

    count = 0
    for l1, l2, lex in items:
        for kw in iter_terms(lex):
            k = kw.strip()
            if not k or RE_ONLY_DIGIT.match(k) or RE_ONLY_PUNC.match(k):
                continue

            # 冲突检测
            if k in kw2aspect and kw2aspect[k] != (l1, l2):
                overlaps[k].append((l1, l2))
            else:
                kw2aspect[k] = (l1, l2)
                kp.add_keyword(k)
                count += 1

    print(f"[INIT] 已加载 {count} 个关键词规则")
    return kp, kw2aspect, overlaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=50000, help="降低默认批次大小以获得更快反馈")
    ap.add_argument("--uncovered-sample", type=int, default=50000)
    ap.add_argument("--uncovered-topk", type=int, default=300)
    ap.add_argument("--example-k", type=int, default=3)
    args = ap.parse_args()

    in_path = Path(args.input)
    cfg_path = Path(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[START] 开始处理，输入文件: {in_path.name}")
    print(f"[INFO] 正在加载规则配置...")
    domain, items, settings = load_config(cfg_path)
    kp, kw2aspect, overlaps = build_matcher(items)

    # 输出冲突词
    if overlaps:
        overlap_path = out_dir / "aspect_lexicon_overlaps.xlsx"
        rows = [{"keyword": kw, "conflicts": str(pairs)} for kw, pairs in overlaps.items()]
        pd.DataFrame(rows).to_excel(overlap_path, index=False)
        print(f"[WARN] 发现关键词冲突，已导出至: {overlap_path}")

    # 准备数据集
    dataset = ds.dataset(str(in_path), format="parquet")
    cols = ["domain", "brand", "model", "doc_id", "platform", "url", "ctime", "sentence_idx", "sentence"]
    schema_cols = set(dataset.schema.names)
    cols = [c for c in cols if c in schema_cols]

    if "sentence" not in cols:
        raise RuntimeError("输入数据缺少 'sentence' 列")

    # 扫描器
    scanner = dataset.scanner(columns=cols, batch_size=args.batch_size)

    # 统计变量
    total_sent = 0
    covered_sent = 0
    l1_hits = Counter()
    l2_hits = Counter()
    prod_counts = Counter()

    uncovered_samples = []
    uncovered_terms = Counter()
    term_examples = defaultdict(list)

    # 输出文件
    out_parquet = out_dir / "aspect_sentences.parquet"
    writer = None

    def ensure_writer(table: pa.Table):
        nonlocal writer
        if writer is None:
            writer = pq.ParquetWriter(out_parquet, table.schema, compression="zstd")

    start_time = time.time()
    last_log_time = start_time

    print(f"[INFO] 开始流式处理 (Batch Size: {args.batch_size})...")

    # --- 主循环优化版 ---
    for bi, rb in enumerate(scanner.to_batches()):
        df = rb.to_pandas()
        batch_rows = []

        # 优化点 1: 使用 itertuples 替代 iterrows (速度提升 10x)
        # 优化点 2: 预先获取列索引，避免 getattr 开销
        has_brand = 'brand' in df.columns
        has_model = 'model' in df.columns
        meta_cols = [c for c in cols if c not in ['sentence', 'aspect_l1', 'aspect_l2', 'hit_term']]

        for r in df.itertuples(index=False):
            # 动态获取 sentence，兼容不同列顺序
            sent = getattr(r, "sentence", None)
            total_sent += 1

            if not isinstance(sent, str) or not sent.strip():
                continue

            # 核心匹配
            hits = kp.extract_keywords(sent)

            if not hits:
                # 采样未覆盖
                if len(uncovered_samples) < args.uncovered_sample:
                    if len(sent) >= 6: uncovered_samples.append(sent)
                continue

            covered_sent += 1
            hits = list(dict.fromkeys(hits))  # 去重

            for kw in hits:
                l1, l2 = kw2aspect.get(kw, ("未归类", "未归类"))
                l1_hits[l1] += 1
                l2_hits[(l1, l2)] += 1

                # 统计
                b_val = getattr(r, "brand", "UNKNOWN") if has_brand else "UNKNOWN"
                m_val = getattr(r, "model", "UNKNOWN") if has_model else "UNKNOWN"
                prod_counts[(b_val, m_val, l1, l2)] += 1

                # 构造行数据
                row_data = {
                    "sentence": sent,
                    "aspect_l1": l1,
                    "aspect_l2": l2,
                    "hit_term": kw
                }
                # 填充元数据
                for mc in meta_cols:
                    row_data[mc] = getattr(r, mc, None)

                batch_rows.append(row_data)

        # 写入磁盘
        if batch_rows:
            t = pa.Table.from_pandas(pd.DataFrame(batch_rows), preserve_index=False)
            ensure_writer(t)
            writer.write_table(t)

        # 优化点 3: 高频日志反馈 (每批次都打印)
        curr_time = time.time()
        elapsed = curr_time - start_time
        speed = total_sent / elapsed if elapsed > 0 else 0

        print(
            f"[PROGRESS] Batch {bi + 1}: 已处理 {total_sent:,} 条 | 命中率 {covered_sent / total_sent:.1%} | 速度 {int(speed)} 条/秒")

    if writer:
        writer.close()
    else:
        # 防止空文件报错
        print("[WARN] 没有匹配到任何数据，生成空文件...")
        empty_df = pd.DataFrame(columns=cols + ["aspect_l1", "aspect_l2", "hit_term"])
        t = pa.Table.from_pandas(empty_df)
        with pq.ParquetWriter(out_parquet, t.schema) as writer:
            writer.write_table(t)

    # -----------------------------
    # 后处理：未覆盖词挖掘
    # -----------------------------
    print("[INFO] 正在挖掘未覆盖的高频词...")
    if pseg is not None and uncovered_samples:
        for s in uncovered_samples:
            try:
                for w, f in pseg.cut(s):
                    if len(w) < 2 or RE_ONLY_DIGIT.match(w) or RE_ONLY_PUNC.match(w):
                        continue
                    if f.startswith("n") or f == "eng":  # 仅名词和英文
                        uncovered_terms[w] += 1
                        if len(term_examples[w]) < args.example_k:
                            term_examples[w].append(s[:100])
            except:
                pass

    # 导出统计报表
    print("[INFO] 正在生成统计报表...")

    # ... (报表生成逻辑保持不变) ...
    l1_df = pd.DataFrame([{"aspect_l1": k, "hit_count": v} for k, v in l1_hits.most_common()])
    l2_df = pd.DataFrame([{"aspect_l1": k[0], "aspect_l2": k[1], "hit_count": v} for k, v in l2_hits.most_common()])

    uncovered_rows = []
    for term, cnt in uncovered_terms.most_common(args.uncovered_topk):
        ex = term_examples.get(term, [])
        uncovered_rows.append({
            "term": term, "count": cnt,
            "ex1": ex[0] if len(ex) > 0 else "",
            "ex2": ex[1] if len(ex) > 1 else ""
        })

    cov_path = out_dir / f"aspect_coverage_{domain}.xlsx"
    with pd.ExcelWriter(cov_path) as writer:
        l1_df.to_excel(writer, sheet_name="L1_Stats", index=False)
        l2_df.to_excel(writer, sheet_name="L2_Stats", index=False)
        pd.DataFrame(uncovered_rows).to_excel(writer, sheet_name="Uncovered_Terms", index=False)

    print(f"[DONE] 完成！总耗时: {int(time.time() - start_time)}秒")
    print(f"       输出文件: {out_parquet}")


if __name__ == "__main__":
    main()
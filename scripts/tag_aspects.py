# scripts/tag_aspects.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
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
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        terms.append(s)
    return terms


def load_config(cfg_path: Path):
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    domain = cfg.get("domain", "unknown")
    items = []

    # 旧结构：顶层 aspects: [{l1:..., l2:{<l2>: <lex_path>}}]
    if "aspects" in cfg:
        for a in cfg["aspects"]:
            l1 = a["l1"]
            for l2, lex in a["l2"].items():
                lex_path = Path(lex)
                if not lex_path.is_absolute():
                    lex_path = Path(lex)  # 相对路径：相对于 repo root
                items.append((l1, l2, lex_path))
        return domain, items, cfg.get("settings", {})

    # 新结构：顶层 l1: [{name, aliases, l2:[{name, terms:[]}, ...]}]
    if "l1" in cfg:
        for a in cfg["l1"]:
            l1 = a.get("name") or a.get("l1") or "未命名"
            aliases = a.get("aliases") or []
            # 可选：用 L1 aliases 兜底命中（映射到一个虚拟 L2）
            if aliases:
                items.append((l1, "_L1", aliases))

            for l2obj in a.get("l2") or []:
                l2 = l2obj.get("name") or "未命名"
                terms = l2obj.get("terms") or []
                if terms:
                    items.append((l1, l2, terms))
        return domain, items, cfg.get("settings", {})

    raise KeyError("config missing top-level key: 'aspects' (old) or 'l1' (new)")


def build_matcher(items):
    kp = KeywordProcessor(case_sensitive=False)
    kw2aspect: Dict[str, Tuple[str, str]] = {}
    overlaps = defaultdict(list)

    def iter_terms(lex):
        # 旧结构：lex 是 Path（词表文件）
        if isinstance(lex, Path):
            if not lex.exists():
                raise FileNotFoundError(f"lexicon 不存在：{lex}")
            return read_terms(lex)

        # 新结构：lex 是 list/tuple/set（terms 内联）
        if isinstance(lex, (list, tuple, set)):
            return [str(x).strip() for x in lex if str(x).strip()]

        raise TypeError(f"unsupported lexicon spec: {type(lex)}")

    for l1, l2, lex in items:
        for kw in iter_terms(lex):
            k = kw.strip()
            if not k or RE_ONLY_DIGIT.match(k) or RE_ONLY_PUNC.match(k):
                continue
            if k in kw2aspect and kw2aspect[k] != (l1, l2):
                overlaps[k].append((l1, l2))
            else:
                kw2aspect[k] = (l1, l2)
                kp.add_keyword(k)

    return kp, kw2aspect, overlaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="clean_sentences.parquet")
    ap.add_argument("--config", required=True, help="configs/aspects_phone.yaml")
    ap.add_argument("--output-dir", required=True, help="输出目录，例如 outputs/phone")
    ap.add_argument("--batch-size", type=int, default=100000)
    ap.add_argument("--uncovered-sample", type=int, default=80000, help="未覆盖句子抽样量（用于挖漏网词）")
    ap.add_argument("--uncovered-topk", type=int, default=300, help="未覆盖 top terms 输出数量")
    ap.add_argument("--example-k", type=int, default=3, help="每个未覆盖 term 保存例句数量")
    args = ap.parse_args()

    in_path = Path(args.input)
    cfg_path = Path(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    domain, items, settings = load_config(cfg_path)
    kp, kw2aspect, overlaps = build_matcher(items)

    # 输出冲突词
    overlap_path = out_dir / "aspect_lexicon_overlaps.xlsx"
    if overlaps:
        rows = []
        for kw, pairs in overlaps.items():
            rows.append({"keyword": kw, "conflicts": str(pairs)})
        pd.DataFrame(rows).to_excel(overlap_path, index=False)
        print(f"[WARN] lexicon overlaps found, wrote: {overlap_path}")

    # 读取 parquet（流式）
    dataset = ds.dataset(str(in_path), format="parquet")

    cols = ["domain", "brand", "model", "doc_id", "platform", "url", "ctime", "sentence_idx", "sentence"]
    # 容错：有些列可能不存在，做一个交集
    schema_cols = set(dataset.schema.names)
    cols = [c for c in cols if c in schema_cols]
    if "sentence" not in cols:
        raise RuntimeError("输入 parquet 缺少 sentence 列。")

    scanner = dataset.scanner(columns=cols, batch_size=args.batch_size)

    total_sent = 0
    covered_sent = 0

    # 方面命中计数（注意：这里计的是 hit 次数，不是句子数）
    l1_hits = Counter()
    l2_hits = Counter()

    # 产品级计数：brand/model/l1/l2 -> hit_count
    prod_counts = Counter()

    # 未覆盖抽样（用于挖漏项）
    uncovered_samples: List[str] = []

    # 未覆盖 term + 例句
    uncovered_terms = Counter()
    term_examples = defaultdict(list)

    def add_term_example(term: str, sent: str, k: int):
        if len(term_examples[term]) >= k:
            return
        s = sent.strip()
        if len(s) > 160:
            s = s[:160] + "…"
        term_examples[term].append(s)

    def maybe_sample_uncovered(s: str):
        if len(uncovered_samples) >= args.uncovered_sample:
            return
        if s and len(s) >= 6:
            uncovered_samples.append(s)

    # 输出：只写命中方面的长表
    out_parquet = out_dir / "aspect_sentences.parquet"
    writer = None

    def ensure_writer(table: pa.Table):
        nonlocal writer
        if writer is None:
            writer = pq.ParquetWriter(out_parquet, table.schema, compression="zstd")

    # 主循环
    for bi, rb in enumerate(scanner.to_batches()):
        df = rb.to_pandas()

        for _, r in df.iterrows():
            sent = r.get("sentence", None)
            total_sent += 1
            if not isinstance(sent, str) or not sent.strip():
                continue

            hits = kp.extract_keywords(sent)
            if not hits:
                maybe_sample_uncovered(sent)
                continue

            # 句子层覆盖
            covered_sent += 1

            # 命中去重（避免同一词多次）
            hits = list(dict.fromkeys(hits))

            rows = []
            for kw in hits:
                l1, l2 = kw2aspect.get(kw, ("未归类", "未归类"))
                l1_hits[l1] += 1
                l2_hits[(l1, l2)] += 1

                brand = r.get("brand", "UNKNOWN")
                model = r.get("model", "UNKNOWN")
                prod_counts[(brand, model, l1, l2)] += 1

                row = {
                    "sentence": sent,
                    "aspect_l1": l1,
                    "aspect_l2": l2,
                    "hit_term": kw,
                }
                # 附加元信息列（存在才写）
                for c in ["domain", "brand", "model", "doc_id", "platform", "url", "ctime", "sentence_idx"]:
                    if c in df.columns:
                        row[c] = r.get(c, None)
                rows.append(row)

            if rows:
                t = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
                ensure_writer(t)
                writer.write_table(t)

        if (bi + 1) % 5 == 0:
            print(f"[INFO] batches={bi+1} total_sent={total_sent:,} covered_sent={covered_sent:,}")

    if writer is not None:
        writer.close()

    # -----------------------------
    # 未覆盖残差挖掘（名词/英文）
    # -----------------------------
    if pseg is not None and uncovered_samples:
        for s in uncovered_samples:
            for w, f in pseg.cut(s):
                w = w.strip()
                if not w or len(w) < 2:
                    continue
                if RE_ONLY_DIGIT.match(w) or RE_ONLY_PUNC.match(w):
                    continue
                if f.startswith("n") or f == "eng":
                    uncovered_terms[w] += 1
                    add_term_example(w, s, args.example_k)

    # -----------------------------
    # 覆盖率 Gate + 报告
    # -----------------------------
    cover_rate = covered_sent / total_sent if total_sent else 0.0

    l1_gate = settings.get("coverage_gate", {}).get("l1_min_rate", 0.85)
    l2_gate = settings.get("coverage_gate", {}).get("l2_min_rate", 0.70)
    unclassified_max = settings.get("coverage_gate", {}).get("unclassified_max_rate", 0.10)

    # 用 “命中方面句子数” 来衡量 L2 覆盖：这里采用 covered_rate 作为 L2 覆盖近似
    # 更严格的 L2 覆盖可后续用“命中 L2 的句子占比”单独计算（需要去重句子 id）。
    cov_summary = pd.DataFrame([
        {"metric": "domain", "value": domain},
        {"metric": "total_sentences", "value": total_sent},
        {"metric": "covered_sentences", "value": covered_sent},
        {"metric": "covered_rate", "value": cover_rate},
        {"metric": "l1_min_rate_gate", "value": l1_gate},
        {"metric": "l2_min_rate_gate", "value": l2_gate},
        {"metric": "unclassified_max_rate", "value": unclassified_max},
        {"metric": "output_aspect_sentences_parquet", "value": str(out_parquet)},
    ])

    l1_df = pd.DataFrame([{"aspect_l1": k, "hit_count": v} for k, v in l1_hits.most_common()])
    l2_df = pd.DataFrame([{"aspect_l1": k[0], "aspect_l2": k[1], "hit_count": v} for k, v in l2_hits.most_common()])

    # 未覆盖 top terms（带例句）
    rows = []
    for term, cnt in uncovered_terms.most_common(args.uncovered_topk):
        ex = term_examples.get(term, [])
        rows.append({
            "term": term,
            "count": cnt,
            "ex1": ex[0] if len(ex) > 0 else "",
            "ex2": ex[1] if len(ex) > 1 else "",
            "ex3": ex[2] if len(ex) > 2 else "",
        })
    uncovered_df = pd.DataFrame(rows)

    coverage_xlsx = out_dir / f"aspect_coverage_{domain}.xlsx"
    with pd.ExcelWriter(coverage_xlsx, engine="openpyxl") as w:
        cov_summary.to_excel(w, sheet_name="summary", index=False)
        l1_df.to_excel(w, sheet_name="l1_hits", index=False)
        l2_df.to_excel(w, sheet_name="l2_hits", index=False)
        uncovered_df.to_excel(w, sheet_name="uncovered_top_terms", index=False)

    # 产品级统计
    prod_rows = []
    for (brand, model, l1, l2), cnt in prod_counts.items():
        prod_rows.append({"brand": brand, "model": model, "aspect_l1": l1, "aspect_l2": l2, "hit_count": cnt})
    prod_df = pd.DataFrame(prod_rows)
    if not prod_df.empty:
        prod_df = prod_df.sort_values(["brand", "model", "hit_count"], ascending=[True, True, False])
    prod_xlsx = out_dir / f"aspect_counts_{domain}.xlsx"
    prod_df.to_excel(prod_xlsx, index=False)

    print(f"[OK] wrote: {out_parquet}")
    print(f"[OK] wrote: {coverage_xlsx}")
    print(f"[OK] wrote: {prod_xlsx}")
    print(f"[INFO] covered_rate={cover_rate:.4f} (covered_sent={covered_sent:,} / total_sent={total_sent:,})")


if __name__ == "__main__":
    main()

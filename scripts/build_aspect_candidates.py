# scripts/build_aspect_candidates.py
# -*- coding: utf-8 -*-
"""
从 clean_sentences.parquet 自动抽取“方面候选词/短语”（偏 L2 颗粒度）
输出：aspect_candidates.xlsx（候选词/短语 + tf/df + 覆盖率 + 示例句 + POS 分布）

设计目标：
- 工程化：可复用到 phone/car/laptop/beauty，只改 --input/--domain
- 可扩展：可加 stopwords、可采样、可限制最大处理句子数
- 大数据可跑：pyarrow.dataset 批处理流式读取，不一次性加载 270w 句子
"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

try:
    import pyarrow.dataset as ds
except Exception as e:
    raise RuntimeError("需要安装 pyarrow 才能读取 parquet。请先 pip install pyarrow") from e

# jieba POS 分词（速度较慢，建议配合采样/最大句子数）
try:
    import jieba
    import jieba.posseg as pseg
except Exception as e:
    raise RuntimeError("需要安装 jieba。请先 pip install jieba") from e


# -----------------------------
# 配置：通用停用词 & 噪声过滤
# -----------------------------
DEFAULT_STOPWORDS: Set[str] = {
    # 常见虚词/口头语
    "这个", "那个", "就是", "还是", "但是", "不过", "然后", "因为", "所以", "而且", "如果", "虽然",
    "感觉", "觉得", "有点", "一点", "比较", "非常", "特别", "真的", "太", "挺", "很", "超级",
    "可能", "应该", "反正", "直接", "基本", "一般", "确实", "目前", "现在", "之前", "之后",
    "吧", "呀", "啊", "呢", "了", "的", "得", "地", "着", "么", "嘛",
    # 泛化词（通常不是方面本体）
    "东西", "产品", "手机", "电脑", "车", "车辆", "车子", "品牌", "型号", "功能",
    "问题", "情况", "地方", "方面", "体验", "使用", "时候", "时间", "原因",
    # 评价词（情绪词）— 这些不当作方面候选
    "好", "不好", "差", "垃圾", "拉胯", "满意", "失望", "后悔", "推荐", "不推荐", "值", "不值", "翻车",
}

# 过滤规则：纯数字、纯符号、过短、过长
RE_ONLY_DIGIT = re.compile(r"^\d+$")
RE_ONLY_PUNC = re.compile(r"^[\W_]+$", re.UNICODE)
RE_HAS_CHN_OR_ENG = re.compile(r"[A-Za-z\u4e00-\u9fff]+")


def load_stopwords(path: Optional[str]) -> Set[str]:
    if not path:
        return set(DEFAULT_STOPWORDS)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"stopwords 文件不存在：{p}")
    words = set(DEFAULT_STOPWORDS)
    for line in p.read_text(encoding="utf-8").splitlines():
        w = line.strip()
        if w and not w.startswith("#"):
            words.add(w)
    return words


def deterministic_keep(text: str, rate: float) -> bool:
    """
    稳定采样：同一句子在不同运行中保持是否被采样一致（便于复现）。
    用 blake2b 取 64-bit，映射到 [0,1)。
    """
    if rate >= 1.0:
        return True
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    v = int.from_bytes(h, "little") / (2**64)
    return v < rate


def is_valid_term(term: str, stopwords: Set[str], min_len: int, max_len: int) -> bool:
    term = term.strip()
    if not term:
        return False
    if len(term) < min_len or len(term) > max_len:
        return False
    if term in stopwords:
        return False
    if RE_ONLY_DIGIT.match(term):
        return False
    if RE_ONLY_PUNC.match(term):
        return False
    if not RE_HAS_CHN_OR_ENG.search(term):
        return False
    return True


def keep_pos(flag: str) -> bool:
    """
    POS 过滤：方面多为名词类。
    jieba 常见：n/nr/ns/nt/nz/eng 等
    """
    if not flag:
        return False
    if flag.startswith("n"):
        return True
    if flag == "eng":  # 例如 iPhone, A17, OLED 等
        return True
    return False


@dataclass
class TermStat:
    tf: int = 0
    df: int = 0
    examples: List[str] = field(default_factory=list)
    pos_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


def add_example(examples: List[str], sent: str, k: int) -> None:
    if len(examples) >= k:
        return
    sent = sent.strip()
    if not sent:
        return
    # 去掉极长示例，防止 Excel 过长
    if len(sent) > 180:
        sent = sent[:180] + "…"
    examples.append(sent)


def iter_sentence_batches(
    parquet_path: Path,
    sentence_col: str,
    brand_col: str,
    model_col: str,
    batch_size: int,
) -> Iterable[Tuple[List[str], List[str], List[str]]]:
    dataset = ds.dataset(str(parquet_path), format="parquet")
    scanner = dataset.scanner(columns=[sentence_col, brand_col, model_col], batch_size=batch_size)
    for rb in scanner.to_batches():
        # 用 pylist 避免 pandas 大开销
        sents = rb.column(0).to_pylist()
        brands = rb.column(1).to_pylist()
        models = rb.column(2).to_pylist()
        yield sents, brands, models


def build_candidates(
    parquet_path: Path,
    domain: str,
    out_xlsx: Path,
    sentence_col: str = "sentence",
    brand_col: str = "brand",
    model_col: str = "model",
    sample_rate: float = 0.15,
    max_sentences: int = 500_000,
    batch_size: int = 50_000,
    min_term_len: int = 2,
    max_term_len: int = 8,
    max_examples: int = 3,
    topk_output: int = 3000,
    enable_phrases: bool = True,
    phrase_max_tokens: int = 3,
    stopwords_path: Optional[str] = None,
) -> None:
    stopwords = load_stopwords(stopwords_path)

    # 统计容器
    stats: Dict[str, TermStat] = defaultdict(TermStat)

    total_seen = 0
    total_kept = 0
    total_empty = 0

    # jieba 初始化（提升后续性能）
    jieba.initialize()

    def update_term(term: str, pos: str, sent: str, seen_in_sent: Set[str]):
        nonlocal stats
        st = stats[term]
        st.tf += 1
        st.pos_counts[pos] += 1
        if term not in seen_in_sent:
            st.df += 1
            seen_in_sent.add(term)
        add_example(st.examples, sent, max_examples)

    # 主循环：流式处理
    for sents, _, _ in iter_sentence_batches(parquet_path, sentence_col, brand_col, model_col, batch_size):
        for sent in sents:
            total_seen += 1
            if sent is None:
                total_empty += 1
                continue
            sent = str(sent).strip()
            if not sent:
                total_empty += 1
                continue

            if not deterministic_keep(sent, sample_rate):
                continue

            total_kept += 1
            if total_kept > max_sentences:
                break

            seen_in_sent: Set[str] = set()

            # POS 分词
            tokens = []
            for w, f in pseg.cut(sent):
                w = w.strip()
                if not w:
                    continue
                if not keep_pos(f):
                    continue
                if not is_valid_term(w, stopwords, min_term_len, max_term_len):
                    continue
                tokens.append((w, f))
                update_term(w, f, sent, seen_in_sent)

            # 短语：连续名词拼接（例如 “系统 广告” -> “系统广告”）
            if enable_phrases and tokens:
                # 仅用 token 的词，不含 pos
                words = [t[0] for t in tokens]
                # 生成 2-gram / 3-gram
                nmax = max(2, min(phrase_max_tokens, 5))
                for n in range(2, nmax + 1):
                    for i in range(0, len(words) - n + 1):
                        phrase = "".join(words[i:i+n])
                        if not is_valid_term(phrase, stopwords, min_term_len, max_term_len):
                            continue
                        update_term(phrase, f"phrase{n}", sent, seen_in_sent)

        if total_kept > max_sentences:
            break

        if total_seen % (batch_size * 2) == 0:
            print(f"[INFO] seen={total_seen:,} sampled={total_kept:,} unique_terms={len(stats):,}")

    print(f"[DONE] seen={total_seen:,} sampled={total_kept:,} empty={total_empty:,} unique_terms={len(stats):,}")

    # 组织输出：按 df 优先（覆盖面），再按 tf
    rows = []
    for term, st in stats.items():
        pos_top = sorted(st.pos_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        rows.append({
            "domain": domain,
            "term": term,
            "tf": st.tf,
            "df": st.df,
            "df_rate_in_sample": (st.df / total_kept) if total_kept else 0.0,
            "pos_top3": " | ".join([f"{p}:{c}" for p, c in pos_top]),
            "example_1": st.examples[0] if len(st.examples) > 0 else "",
            "example_2": st.examples[1] if len(st.examples) > 1 else "",
            "example_3": st.examples[2] if len(st.examples) > 2 else "",
        })

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        raise RuntimeError("候选表为空：可能 sample_rate 太低或过滤过严。请提高 sample_rate 或放宽 min_term_len。")

    df_out.sort_values(["df", "tf"], ascending=[False, False], inplace=True)
    df_out = df_out.head(topk_output).reset_index(drop=True)

    # 写 Excel：candidates + run_stats
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    run_stats = pd.DataFrame([
        ["input", str(parquet_path)],
        ["domain", domain],
        ["sentence_col", sentence_col],
        ["sample_rate", sample_rate],
        ["max_sentences", max_sentences],
        ["batch_size", batch_size],
        ["min_term_len", min_term_len],
        ["max_term_len", max_term_len],
        ["enable_phrases", enable_phrases],
        ["phrase_max_tokens", phrase_max_tokens],
        ["topk_output", topk_output],
        ["total_seen", total_seen],
        ["total_sampled", total_kept],
        ["total_empty", total_empty],
        ["unique_terms", len(stats)],
    ], columns=["key", "value"])

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        df_out.to_excel(w, sheet_name="candidates", index=False)
        run_stats.to_excel(w, sheet_name="run_stats", index=False)

    print(f"[OK] wrote: {out_xlsx}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="clean_sentences.parquet 路径")
    ap.add_argument("--domain", default="phone", help="领域标识：phone/car/laptop/beauty")
    ap.add_argument("--output", default="", help="输出 xlsx 路径（默认在输入同目录）")

    ap.add_argument("--sentence-col", default="sentence")
    ap.add_argument("--brand-col", default="brand")
    ap.add_argument("--model-col", default="model")

    ap.add_argument("--sample-rate", type=float, default=0.15, help="稳定采样比例，默认 0.15（建议 0.1~0.3）")
    ap.add_argument("--max-sentences", type=int, default=500_000, help="最多处理的采样句子数")
    ap.add_argument("--batch-size", type=int, default=50_000, help="pyarrow batch 大小")

    ap.add_argument("--min-term-len", type=int, default=2)
    ap.add_argument("--max-term-len", type=int, default=8)
    ap.add_argument("--max-examples", type=int, default=3)
    ap.add_argument("--topk-output", type=int, default=3000)

    ap.add_argument("--no-phrases", action="store_true", help="关闭短语拼接（默认开启）")
    ap.add_argument("--phrase-max-tokens", type=int, default=3, help="短语最大 token 数（2~3 常用）")

    ap.add_argument("--stopwords", default="", help="可选：自定义 stopwords.txt（每行一个词，#开头为注释）")

    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"input 不存在：{in_path}")

    if args.output:
        out_xlsx = Path(args.output)
    else:
        out_xlsx = in_path.parent / f"aspect_candidates_{args.domain}.xlsx"

    build_candidates(
        parquet_path=in_path,
        domain=args.domain,
        out_xlsx=out_xlsx,
        sentence_col=args.sentence_col,
        brand_col=args.brand_col,
        model_col=args.model_col,
        sample_rate=args.sample_rate,
        max_sentences=args.max_sentences,
        batch_size=args.batch_size,
        min_term_len=args.min_term_len,
        max_term_len=args.max_term_len,
        max_examples=args.max_examples,
        topk_output=args.topk_output,
        enable_phrases=(not args.no_phrases),
        phrase_max_tokens=args.phrase_max_tokens,
        stopwords_path=(args.stopwords.strip() or None),
    )


if __name__ == "__main__":
    main()

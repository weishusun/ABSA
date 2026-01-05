# -*- coding: utf-8 -*-
"""
llm_autofill_decisions.py (fixed)

Fixes:
- Force dtype of decision/target columns to string to avoid pandas FutureWarning.
- Preflight request to fail fast on API key / model / SDK issues.
- Use Responses API structured outputs:
  - Prefer client.responses.parse(text_format=PydanticModel)
  - Fallback to client.responses.create(text.format=json_schema) if parse is unavailable.

Refs (OpenAI docs):
- Structured Outputs guide (Responses API + Pydantic) and supported models
- Responses API reference
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional, Literal, Dict, Tuple, Set

import pandas as pd
from pydantic import BaseModel, Field

from openai import OpenAI


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
    print(f"[{ts()}] {msg}", flush=True)


def parse_lexicon_pairs(lexicon_dir: Path) -> Set[Tuple[str, str]]:
    pairs: Set[Tuple[str, str]] = set()
    if not lexicon_dir.exists():
        return pairs
    for p in lexicon_dir.glob("*.txt"):
        stem = p.stem
        if "__" not in stem:
            continue
        l1, l2 = stem.split("__", 1)
        l1, l2 = l1.strip(), l2.strip()
        if l1 and l2:
            pairs.add((l1, l2))
    return pairs


class LLMDecision(BaseModel):
    decision: Literal["ADD", "STOP", "IGNORE"]
    target_L1: Optional[str] = None
    target_L2: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str


def load_cache(cache_path: Path) -> Dict[str, dict]:
    if not cache_path.exists():
        return {}
    out = {}
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                term = str(obj.get("term", "")).strip()
                if term:
                    out[term] = obj
            except Exception:
                continue
    return out


def append_cache(cache_path: Path, obj: dict):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def is_empty_cell(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    s = str(v).strip()
    return (s == "" or s.lower() == "nan")


def normalize_string_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = ""
        # force to pandas string dtype, and replace NaN with ""
        df[c] = df[c].astype("string")
        df[c] = df[c].fillna("")
    return df


def build_prompt(row: pd.Series, allowed_pairs: Set[Tuple[str, str]], domain: str) -> str:
    term = str(row.get("term", "")).strip()
    dfv = row.get("df", "")
    tfv = row.get("tf", "")
    sample_n = int(row.get("sample_n", 0) or 0)
    covered = int(row.get("covered_in_sample", 0) or 0)
    uncovered = int(row.get("uncovered_in_sample", 0) or 0)

    best_l1 = "" if pd.isna(row.get("best_L1", "")) else str(row.get("best_L1", "")).strip()
    best_l2 = "" if pd.isna(row.get("best_L2", "")) else str(row.get("best_L2", "")).strip()
    rec = "" if pd.isna(row.get("recommendation", "")) else str(row.get("recommendation", "")).strip()
    reason = "" if pd.isna(row.get("reason", "")) else str(row.get("reason", "")).strip()
    examples = "" if pd.isna(row.get("examples", "")) else str(row.get("examples", "")).strip()

    l1_to_l2 = {}
    for (l1, l2) in allowed_pairs:
        l1_to_l2.setdefault(l1, set()).add(l2)

    allowed_lines = []
    for l1 in sorted(l1_to_l2.keys()):
        l2s = "、".join(sorted(l1_to_l2[l1]))
        allowed_lines.append(f"- {l1}: {l2s}")
    allowed_text = "\n".join(allowed_lines)

    instructions = f"""
你在做中文评论 ABSA-like pipeline 的“覆盖率闭环扩词”自动决策。
当前领域：{domain}

任务：对候选词 term 给出决策 decision ∈ {{ADD, STOP, IGNORE}}，并在 ADD 时映射到一个现有方面类别 (target_L1,target_L2)。

判定准则：
- ADD：term 是“方面实体/部件/性能维度/体验维度”的稳定关键词；加入后能提升方面覆盖率；必须映射到【允许的 L1/L2 列表】中的一个组合。
- STOP：term 多为品牌/平台/人名/泛化抽象词/口水情绪词/语气词/与方面无关的常见噪声。
- IGNORE：term 过于模糊、歧义大、样本不足、或不值得纳入 stoplist；暂不处理。

重要约束：
- 只能从【允许的 L1/L2 列表】里选择 target_L1/target_L2（禁止编造新类）。
- decision=ADD 时 target_L1/target_L2 必填；否则必须为空。
- rationale 中文最多 2 句；confidence 0~1。
""".strip()

    user = f"""
候选词：{term}
统计：df={dfv}, tf={tfv}
抽样：sample_n={sample_n}, covered_in_sample={covered}, uncovered_in_sample={uncovered}
脚本提示：recommendation={rec}, best=({best_l1},{best_l2}), reason={reason}

示例句（可能包含换行）：
{examples}

允许的 L1/L2 列表：
{allowed_text}

请输出结构化结果。
""".strip()

    return instructions + "\n\n" + user


def extract_output_text(resp) -> str:
    # Prefer convenience property if present
    if hasattr(resp, "output_text") and isinstance(resp.output_text, str):
        return resp.output_text
    # Fallback: try to find first message text content
    try:
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", None) == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) in ("output_text", "text") and hasattr(c, "text"):
                        return c.text
    except Exception:
        pass
    raise RuntimeError("Cannot extract output text from Responses API result.")


def call_openai_structured(client: OpenAI, model: str, prompt: str) -> LLMDecision:
    # Prefer responses.parse if available (per Structured Outputs guide)
    if hasattr(client.responses, "parse"):
        resp = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": "你是严谨的中文NLP工程助手，只输出结构化结果。"},
                {"role": "user", "content": prompt},
            ],
            text_format=LLMDecision,
            temperature=0,
        )
        return resp.output_parsed

    # Fallback: responses.create with json_schema format
    schema = LLMDecision.model_json_schema()
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "你是严谨的中文NLP工程助手，只输出结构化结果。"},
            {"role": "user", "content": prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "LLMDecision",
                    "schema": schema,
                    "strict": True,
                },
            }
        },
        temperature=0,
    )
    txt = extract_output_text(resp)
    return LLMDecision.model_validate_json(txt)


def preflight(client: OpenAI, model: str):
    # Fail fast: key/model/permission/sdk
    p = "候选词：屏幕\n请输出：{decision:'IGNORE', target_L1:null, target_L2:null, confidence:1, rationale:'preflight'}"
    try:
        _ = call_openai_structured(client, model, p)
        log("[OK] preflight success (API key/model/SDK seems OK).")
    except Exception as e:
        raise RuntimeError(f"Preflight failed: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--in-xlsx", default=None)
    ap.add_argument("--sheet", default="suggestions")
    ap.add_argument("--out-xlsx", default=None)
    ap.add_argument("--model", default="gpt-4o-mini-2024-07-18")

    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--only-empty", action="store_true")
    ap.add_argument("--review-only", action="store_true")
    ap.add_argument("--min-sample", type=int, default=1)
    ap.add_argument("--auto-accept-high-consistency", action="store_true")
    ap.add_argument("--cache-jsonl", default=None)
    ap.add_argument("--no-preflight", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    out_dir = repo / "outputs" / args.domain

    in_xlsx = Path(args.in_xlsx) if args.in_xlsx else (out_dir / f"coverage_suggestions_{args.domain}.xlsx")
    out_xlsx = Path(args.out_xlsx) if args.out_xlsx else (out_dir / f"coverage_suggestions_{args.domain}.llm.xlsx")
    cache_path = Path(args.cache_jsonl) if args.cache_jsonl else (out_dir / "llm_decisions_cache.jsonl")

    lexicon_dir = repo / "aspects" / args.domain / "lexicons"
    allowed_pairs = parse_lexicon_pairs(lexicon_dir)
    if not allowed_pairs:
        raise RuntimeError(f"No lexicon pairs found under: {lexicon_dir}")

    if not in_xlsx.exists():
        raise FileNotFoundError(f"Missing: {in_xlsx}")

    df = pd.read_excel(in_xlsx, sheet_name=args.sheet)
    if df.empty:
        log("Empty input sheet; nothing to do.")
        return

    # Ensure/normalize string columns to avoid dtype warnings
    df = normalize_string_cols(df, [
        "decision", "target_L1", "target_L2",
        "llm_rationale", "llm_status", "llm_model",
    ])
    if "llm_confidence" not in df.columns:
        df["llm_confidence"] = pd.NA

    cache = load_cache(cache_path)
    log(f"Loaded cache items: {len(cache)} from {cache_path}")

    client = OpenAI()

    if not args.no_preflight:
        preflight(client, args.model)

    filled = 0
    sent = 0
    skipped = 0

    for idx, row in df.iterrows():
        term = str(row.get("term", "")).strip()
        if not term:
            continue

        if args.max_rows and filled >= args.max_rows:
            break

        if args.only_empty and not is_empty_cell(row.get("decision", "")):
            skipped += 1
            continue

        sample_n = int(row.get("sample_n", 0) or 0)
        if sample_n < args.min_sample:
            if is_empty_cell(row.get("decision", "")):
                df.at[idx, "decision"] = "IGNORE"
                df.at[idx, "llm_confidence"] = 0.0
                df.at[idx, "llm_rationale"] = "样本不足（sample_n过小），暂不处理。"
                df.at[idx, "llm_model"] = "rule"
                df.at[idx, "llm_status"] = "rule_min_sample"
                filled += 1
            continue

        rec = str(row.get("recommendation", "")).strip().upper()
        reason = str(row.get("reason", "")).strip()

        if args.review_only and rec == "ADD":
            skipped += 1
            continue

        # auto-accept high consistency without calling LLM
        if args.auto_accept_high_consistency and rec == "ADD" and "high_consistency" in reason:
            best_l1 = "" if pd.isna(row.get("best_L1", "")) else str(row.get("best_L1", "")).strip()
            best_l2 = "" if pd.isna(row.get("best_L2", "")) else str(row.get("best_L2", "")).strip()
            if (best_l1, best_l2) in allowed_pairs:
                df.at[idx, "decision"] = "ADD"
                df.at[idx, "target_L1"] = best_l1
                df.at[idx, "target_L2"] = best_l2
                df.at[idx, "llm_confidence"] = 0.9
                df.at[idx, "llm_rationale"] = "规则自动接受：high_consistency 且 best_L1/L2 合法。"
                df.at[idx, "llm_model"] = "rule"
                df.at[idx, "llm_status"] = "rule_high_consistency"
                filled += 1
                continue

        # cache
        if term in cache:
            obj = cache[term]
            df.at[idx, "decision"] = obj.get("decision", "") or ""
            df.at[idx, "target_L1"] = obj.get("target_L1", "") or ""
            df.at[idx, "target_L2"] = obj.get("target_L2", "") or ""
            df.at[idx, "llm_confidence"] = obj.get("confidence", pd.NA)
            df.at[idx, "llm_rationale"] = obj.get("rationale", "") or ""
            df.at[idx, "llm_model"] = obj.get("model", "") or ""
            df.at[idx, "llm_status"] = "cache"
            filled += 1
            continue

        prompt = build_prompt(row, allowed_pairs, args.domain)
        sent += 1

        try:
            dec = call_openai_structured(client, args.model, prompt)
            decision = dec.decision
            t1 = (dec.target_L1 or "").strip()
            t2 = (dec.target_L2 or "").strip()
            status = "ok"

            if decision == "ADD":
                if not t1 or not t2 or (t1, t2) not in allowed_pairs:
                    best_l1 = "" if pd.isna(row.get("best_L1", "")) else str(row.get("best_L1", "")).strip()
                    best_l2 = "" if pd.isna(row.get("best_L2", "")) else str(row.get("best_L2", "")).strip()
                    if (best_l1, best_l2) in allowed_pairs:
                        t1, t2 = best_l1, best_l2
                        status = "fallback_to_best"
                    else:
                        decision = "IGNORE"
                        t1, t2 = "", ""
                        status = "invalid_target_forced_ignore"
            else:
                t1, t2 = "", ""

            df.at[idx, "decision"] = decision
            df.at[idx, "target_L1"] = t1
            df.at[idx, "target_L2"] = t2
            df.at[idx, "llm_confidence"] = float(dec.confidence)
            df.at[idx, "llm_rationale"] = dec.rationale
            df.at[idx, "llm_model"] = args.model
            df.at[idx, "llm_status"] = status

            append_cache(cache_path, {
                "term": term,
                "decision": decision,
                "target_L1": t1 or None,
                "target_L2": t2 or None,
                "confidence": float(dec.confidence),
                "rationale": dec.rationale,
                "model": args.model,
                "status": status,
                "ts": ts(),
            })

            filled += 1
            if filled % 20 == 0:
                log(f"progress: filled={filled}, sent_to_llm={sent}, skipped={skipped}")

        except Exception as e:
            df.at[idx, "llm_status"] = f"error:{type(e).__name__}"
            df.at[idx, "llm_rationale"] = f"{type(e).__name__}: {e}"
            # continue processing remaining rows
            continue

    out_dir.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        df.to_excel(w, sheet_name=args.sheet, index=False)

    log(f"[OK] wrote: {out_xlsx}")
    log(f"[OK] cache: {cache_path}")
    log(f"summary: filled={filled}, sent_to_llm={sent}, skipped={skipped}")


if __name__ == "__main__":
    main()

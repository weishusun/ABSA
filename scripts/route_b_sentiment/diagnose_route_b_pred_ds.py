import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import duckdb


POSSIBLE_MANIFEST_KEYS = ["path", "file", "filepath", "relpath", "uri", "parquet", "part"]


def _load_manifest_files(manifest_path: Path, pred_ds_dir: Path) -> List[str]:
    files = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # 如果 manifest 不是 jsonl，而是直接每行一个文件名
                p = line
            else:
                p = None
                if isinstance(obj, dict):
                    for k in POSSIBLE_MANIFEST_KEYS:
                        if k in obj and obj[k]:
                            p = obj[k]
                            break
                if p is None:
                    # 兜底：尝试把整行当路径
                    p = line

            # 路径归一化
            p = str(p)
            p_path = Path(p)
            if not p_path.is_absolute():
                # 相对路径一般相对 pred_ds_dir
                p_path = (pred_ds_dir / p_path).resolve()

            if p_path.exists() and p_path.suffix.lower() == ".parquet":
                files.append(str(p_path))
            else:
                # 有些 manifest 可能写了目录/非 parquet，忽略但提示
                pass

    # 去重但保持顺序
    seen = set()
    uniq = []
    for x in files:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def _glob_parquets(pred_ds_dir: Path) -> List[str]:
    files = sorted([str(p.resolve()) for p in pred_ds_dir.glob("*.parquet")])
    return files


def _duckdb_count(con: duckdb.DuckDBPyConnection, files: List[str]) -> int:
    if not files:
        return 0
    # DuckDB 支持 read_parquet(list_of_files)
    q = "SELECT COUNT(*) AS n FROM read_parquet(?);"
    return int(con.execute(q, [files]).fetchone()[0])


def _duckdb_schema_preview(con: duckdb.DuckDBPyConnection, one_file: str) -> List[Tuple[str, str]]:
    q = "DESCRIBE SELECT * FROM read_parquet(?) LIMIT 1;"
    rows = con.execute(q, [one_file]).fetchall()
    # rows: [(column_name, column_type, null, key, default, extra), ...] 在不同版本略有差异
    out = []
    for r in rows:
        out.append((str(r[0]), str(r[1])))
    return out


def _find_label_col(schema: List[Tuple[str, str]]) -> Optional[str]:
    cols = [c for c, _ in schema]
    candidates = ["label", "labels", "y", "sentiment", "gold_label", "pseudo_label", "pred_label"]
    for c in candidates:
        if c in cols:
            return c
    # 兜底：包含 label 字样
    for c in cols:
        if "label" in c.lower():
            return c
    return None


def _load_model_mapping(model_dir: Path) -> Tuple[Optional[dict], Optional[dict]]:
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return None, None
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    id2label = cfg.get("id2label")
    label2id = cfg.get("label2id")
    return id2label, label2id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-ds", required=True, help="asc_pred_ds 目录，例如 outputs/phone_v2/sentiment/asc_pred_ds")
    ap.add_argument("--manifest", default=None, help="manifest.jsonl 路径；默认用 pred-ds/manifest.jsonl")
    ap.add_argument("--model-dir", required=True, help="模型目录，例如 outputs/phone_v2/models/asc_lora_v1")
    ap.add_argument("--train-pseudo", required=True, help="train_pseudolabel.parquet 路径")
    ap.add_argument("--sample-n", type=int, default=200000, help="用于映射一致性检查的抽样行数")
    args = ap.parse_args()

    pred_ds_dir = Path(args.pred_ds).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else (pred_ds_dir / "manifest.jsonl").resolve()
    model_dir = Path(args.model_dir).resolve()
    train_pseudo_path = Path(args.train_pseudo).resolve()

    print("=" * 90)
    print(f"[INFO] pred_ds_dir     = {pred_ds_dir}")
    print(f"[INFO] manifest_path  = {manifest_path}")
    print(f"[INFO] model_dir      = {model_dir}")
    print(f"[INFO] train_pseudo   = {train_pseudo_path}")
    print("=" * 90)

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA enable_progress_bar=false;")

    # ---------------------------
    # STEP 1: rowcount: manifest vs glob
    # ---------------------------
    print("\n[STEP 1] RowCount 口径核对：manifest vs 目录glob")
    manifest_files = []
    if manifest_path.exists():
        manifest_files = _load_manifest_files(manifest_path, pred_ds_dir)
        print(f"[STEP 1] manifest 引用 parquet 文件数: {len(manifest_files)}")
    else:
        print("[STEP 1][WARN] manifest 不存在，将跳过 manifest 口径统计。")

    glob_files = _glob_parquets(pred_ds_dir)
    print(f"[STEP 1] 目录 glob parquet 文件数: {len(glob_files)}")

    n_manifest = _duckdb_count(con, manifest_files) if manifest_files else None
    n_glob = _duckdb_count(con, glob_files) if glob_files else 0

    if n_manifest is not None:
        print(f"[STEP 1] ROWS(manifest) = {n_manifest:,}")
    print(f"[STEP 1] ROWS(glob)     = {n_glob:,}")

    if n_manifest is not None and n_glob != n_manifest:
        print("[STEP 1][ALERT] manifest 口径与 glob 口径不一致：目录可能混入旧 part 或重复写出。")
    else:
        print("[STEP 1] rowcount 口径一致或无法对比。")

    # ---------------------------
    # STEP 2: mapping consistency
    # ---------------------------
    print("\n[STEP 2] 映射一致性核对：pred_label vs argmax(p_*)，以及模型 id2label")
    id2label, label2id = _load_model_mapping(model_dir)
    print(f"[STEP 2] model config id2label = {id2label}")
    print(f"[STEP 2] model config label2id = {label2id}")

    if not glob_files:
        print("[STEP 2][ERROR] pred_ds 目录下没有 parquet，无法继续。")
        return

    schema = _duckdb_schema_preview(con, glob_files[0])
    cols = [c for c, _ in schema]
    print("[STEP 2] schema preview (first file):")
    for c, t in schema:
        print(f"  - {c}: {t}")

    required_prob_cols = ["p_neg", "p_neu", "p_pos"]
    has_prob_cols = all(c in cols for c in required_prob_cols)
    has_pred_label = "pred_label" in cols

    if not (has_prob_cols and has_pred_label):
        print("[STEP 2][WARN] 需要列 pred_label + p_neg/p_neu/p_pos 才能做快速一致性检查。")
        print("[STEP 2][WARN] 当前列缺失："
              f"{'pred_label ' if not has_pred_label else ''}"
              f"{' '.join([c for c in required_prob_cols if c not in cols])}")
        print("[STEP 2] 你需要告诉我 04 输出到底有哪些列名（或我再给你做 array probs 的版本）。")
    else:
        # 抽样一致性检查
        sample_n = args.sample_n
        q = f"""
        WITH s AS (
            SELECT
                pred_label,
                p_neg, p_neu, p_pos,
                CASE
                    WHEN p_neg >= p_neu AND p_neg >= p_pos THEN 'NEG'
                    WHEN p_neu >= p_neg AND p_neu >= p_pos THEN 'NEU'
                    ELSE 'POS'
                END AS argmax_label
            FROM read_parquet(?)
            USING SAMPLE {sample_n} ROWS
        )
        SELECT
            COUNT(*) AS n,
            SUM(CASE WHEN CAST(pred_label AS VARCHAR) = argmax_label THEN 1 ELSE 0 END) AS n_match,
            SUM(CASE WHEN CAST(pred_label AS VARCHAR) <> argmax_label THEN 1 ELSE 0 END) AS n_mismatch,
            AVG(GREATEST(p_neg, p_neu, p_pos)) AS avg_maxprob,
            MAX(GREATEST(p_neg, p_neu, p_pos)) AS max_maxprob
        FROM s;
        """
        n, n_match, n_mismatch, avg_maxprob, max_maxprob = con.execute(q, [glob_files]).fetchone()
        print(f"[STEP 2] sample_n={n:,} match={n_match:,} mismatch={n_mismatch:,}")
        print(f"[STEP 2] maxprob avg={avg_maxprob:.4f} max={max_maxprob:.4f}")

        # 如果 pred_label 不是字符串（比如 int id），再提供一个提示
        # 这里无法直接完美判断，但 mismatch 很高基本就说明问题
        if n_mismatch > 0:
            mismatch_rate = n_mismatch / max(1, n)
            print(f"[STEP 2][ALERT] mismatch_rate={mismatch_rate:.2%}：要么 pred_label 不是 NEG/NEU/POS 字符串，"
                  "要么 p_* 列与 label 映射错位（这是 NEG 消失的高概率根因）。")
        else:
            print("[STEP 2] pred_label 与 argmax(p_*) 一致（至少在抽样上）。映射问题概率降低。")

    # ---------------------------
    # STEP 3: train_pseudolabel label distribution
    # ---------------------------
    print("\n[STEP 3] train_pseudolabel 标签分布（判断 NEG 是否训练缺失）")
    if not train_pseudo_path.exists():
        print("[STEP 3][ERROR] train_pseudolabel 文件不存在。")
        return

    # 读 schema 找 label 列
    q_schema = "DESCRIBE SELECT * FROM read_parquet(?) LIMIT 1;"
    schema_tp = [(r[0], r[1]) for r in con.execute(q_schema, [str(train_pseudo_path)]).fetchall()]
    label_col = _find_label_col(schema_tp)
    print("[STEP 3] train_pseudolabel schema preview:")
    for c, t in schema_tp:
        print(f"  - {c}: {t}")
    print(f"[STEP 3] detected label column = {label_col}")

    if label_col is None:
        print("[STEP 3][ERROR] 未能自动识别标签列名。请告诉我 train_pseudolabel 里标签列叫什么。")
        return

    q_dist = f"""
    SELECT
        CAST({label_col} AS VARCHAR) AS label,
        COUNT(*) AS n
    FROM read_parquet(?)
    GROUP BY 1
    ORDER BY n DESC;
    """
    rows = con.execute(q_dist, [str(train_pseudo_path)]).fetchall()
    total = sum(int(r[1]) for r in rows)
    print(f"[STEP 3] total_rows={total:,}")
    for label, n in rows:
        print(f"  - {label}: {int(n):,} ({int(n)/max(1,total):.2%})")

    print("\n[DONE] 三步诊断已完成。把以上输出完整复制给我，我就能给你最小改动修复路径。")


if __name__ == "__main__":
    main()

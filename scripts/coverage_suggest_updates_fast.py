# -*- coding: utf-8 -*-
"""
coverage_suggest_updates_fast.py

Goal:
- Suggest lexicon/stoplist updates to improve aspect coverage.
- Avoid full-scan-per-term, avoid heavy DISTINCT on huge aspect parquet.
- Show continuous progress (stage logs + heartbeat + tqdm + checkpoint/resume).
"""

import argparse
import re
import time
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional

import duckdb
import pandas as pd
from tqdm import tqdm


# -----------------------------
# Basic config
# -----------------------------
SENTIMENT_LIKE = {
    "好", "很好", "不错", "一般", "垃圾", "差", "太差", "满意", "不满意",
    "喜欢", "不喜欢", "推荐", "不推荐", "后悔", "值", "不值",
    "香", "拉胯", "翻车", "真香", "绝了", "无语",
}

KEY_CANDIDATES = ["domain", "brand", "model", "doc_id", "sentence_idx"]


# -----------------------------
# Logging & heartbeat
# -----------------------------
def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_info(msg: str):
    print(f"[{ts()}][INFO] {msg}", flush=True)


def log_warn(msg: str):
    print(f"[{ts()}][WARN] {msg}", flush=True)


class Heartbeat:
    """
    Print a heartbeat log every `interval` seconds while a long step is running.
    Ensures you always see that the program is alive.
    """
    def __init__(self, label: str, interval: int = 10):
        self.label = label
        self.interval = max(1, int(interval))
        self._stop = threading.Event()
        self._t0 = None
        self._th = None

    def __enter__(self):
        self._t0 = time.time()

        def _run():
            while not self._stop.is_set():
                elapsed = time.time() - self._t0
                print(f"[HB] {self.label} | elapsed={elapsed/60:.1f} min", flush=True)
                self._stop.wait(self.interval)

        self._th = threading.Thread(target=_run, daemon=True)
        self._th.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._th:
            self._th.join(timeout=1)
        elapsed = time.time() - (self._t0 or time.time())
        print(f"[HB] {self.label} | done | elapsed={elapsed/60:.1f} min", flush=True)


# -----------------------------
# Excel / term sources
# -----------------------------
def safe_read_excel_any_sheets(xlsx_path: Path) -> Dict[str, pd.DataFrame]:
    try:
        xls = pd.ExcelFile(xlsx_path)
        out = {}
        for s in xls.sheet_names:
            try:
                out[s] = pd.read_excel(xlsx_path, sheet_name=s)
            except Exception:
                continue
        return out
    except Exception:
        return {}


def extract_terms_from_unmapped(unmapped_path: Path) -> pd.DataFrame:
    if not unmapped_path.exists():
        return pd.DataFrame(columns=["term", "df", "tf", "source"])
    df = pd.read_excel(unmapped_path)
    if df is None or df.empty:
        return pd.DataFrame(columns=["term", "df", "tf", "source"])

    cols = {str(c).lower(): c for c in df.columns}
    term_col = None
    for k in ["term", "关键词", "词", "token"]:
        if k in cols:
            term_col = cols[k]
            break
    if term_col is None:
        for c in df.columns:
            if "term" in str(c).lower():
                term_col = c
                break
    if term_col is None:
        return pd.DataFrame(columns=["term", "df", "tf", "source"])

    out = pd.DataFrame()
    out["term"] = df[term_col].astype(str).str.strip()
    out["df"] = pd.to_numeric(df[cols["df"]], errors="coerce") if "df" in cols else pd.NA
    out["tf"] = pd.to_numeric(df[cols["tf"]], errors="coerce") if "tf" in cols else pd.NA
    out["source"] = "unmapped_terms"
    out = out.dropna(subset=["term"]).drop_duplicates(subset=["term"])
    return out


def extract_terms_from_coverage(coverage_xlsx: Path) -> pd.DataFrame:
    if not coverage_xlsx.exists():
        return pd.DataFrame(columns=["term", "df", "tf", "source"])
    sheets = safe_read_excel_any_sheets(coverage_xlsx)
    if not sheets:
        return pd.DataFrame(columns=["term", "df", "tf", "source"])

    picked = []
    for sname, df in sheets.items():
        if df is None or df.empty:
            continue
        lname = sname.lower()
        has_term_col = any((str(c).lower() == "term" or "term" in str(c).lower()) for c in df.columns)
        if ("uncovered" in lname or "top" in lname or "term" in lname) and has_term_col:
            picked.append((sname, df))

    if not picked:
        for sname, df in sheets.items():
            if df is None or df.empty:
                continue
            has_term_col = any((str(c).lower() == "term" or "term" in str(c).lower()) for c in df.columns)
            if has_term_col:
                picked.append((sname, df))

    rows = []
    for sname, df in picked:
        term_col = None
        for c in df.columns:
            if str(c).lower() == "term" or "term" in str(c).lower():
                term_col = c
                break
        if term_col is None:
            continue

        cols = {str(c).lower(): c for c in df.columns}
        tmp = pd.DataFrame()
        tmp["term"] = df[term_col].astype(str).str.strip()
        tmp["df"] = pd.to_numeric(df[cols["df"]], errors="coerce") if "df" in cols else pd.NA
        tmp["tf"] = pd.to_numeric(df[cols["tf"]], errors="coerce") if "tf" in cols else pd.NA
        tmp["source"] = f"coverage:{sname}"
        tmp = tmp.dropna(subset=["term"]).drop_duplicates(subset=["term"])
        rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["term", "df", "tf", "source"])
    return pd.concat(rows, ignore_index=True)


# -----------------------------
# Lexicon / stoplist helpers
# -----------------------------
def load_existing_lexicon_terms(lexicon_dir: Path) -> set:
    terms = set()
    if not lexicon_dir.exists():
        return terms
    for p in lexicon_dir.glob("*.txt"):
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                t = line.strip()
                if t:
                    terms.add(t)
        except Exception:
            continue
    return terms


def load_stoplist(stoplist_path: Path) -> set:
    if not stoplist_path.exists():
        return set()
    try:
        return {line.strip() for line in stoplist_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    except Exception:
        return set()


def is_valid_term(t: str) -> bool:
    if not t:
        return False
    t = str(t).strip()
    if len(t) < 2:
        return False
    if t in SENTIMENT_LIKE:
        return False
    if re.fullmatch(r"[\d\W_]+", t):
        return False
    return True


# -----------------------------
# DuckDB utilities (avoid pyarrow schema thrift limit)
# -----------------------------
def _get_parquet_columns_via_duckdb(parquet_path: Path) -> List[str]:
    con = duckdb.connect(database=":memory:")
    try:
        con.execute("PRAGMA threads=8;")
        cols = con.execute(
            f"SELECT * FROM read_parquet('{parquet_path.as_posix()}') LIMIT 0"
        ).df().columns.tolist()
        return cols
    finally:
        con.close()


def duckdb_setup(
    con: duckdb.DuckDBPyConnection,
    threads: int,
    temp_dir: Path,
    memory_gb: Optional[int],
    enable_progress: bool,
    progress_ms: int,
):
    con.execute(f"PRAGMA threads={int(threads)};")
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"PRAGMA temp_directory='{temp_dir.as_posix()}';")

    # 防止内存打满导致系统整体变慢/假死
    if memory_gb and memory_gb > 0:
        con.execute(f"PRAGMA memory_limit='{int(memory_gb)}GB';")

    if enable_progress:
        try:
            con.execute("PRAGMA enable_progress_bar;")
            try:
                con.execute(f"PRAGMA progress_bar_time={int(progress_ms)};")
            except Exception:
                pass
        except Exception:
            log_warn("DuckDB progress bar not supported in this version.")

    # ---- debug: show settings actually applied ----
    try:
        log_info(f"DuckDB PRAGMA threads -> {con.execute('PRAGMA threads;').fetchall()}")
    except Exception as e:
        log_warn(f"PRAGMA threads check failed: {e}")

    def _safe_setting(name: str):
        try:
            sql = f"SELECT current_setting('{name}');"
            return con.execute(sql).fetchall()
        except Exception as e:
            return f"<not available: {e}>"

    log_info(f"DuckDB current_setting('threads') -> {_safe_setting('threads')}")
    log_info(f"DuckDB current_setting('temp_directory') -> {_safe_setting('temp_directory')}")
    log_info(f"DuckDB current_setting('memory_limit') -> {_safe_setting('memory_limit')}")



def sql_escape(s: str) -> str:
    return s.replace("'", "''")


# -----------------------------
# Pool building (fast & stable)
# -----------------------------
def build_pool_tagged(
    con: duckdb.DuckDBPyConnection,
    clean_path: Path,
    aspect_path: Path,
    sent_col: str,
    l1_col: str,
    l2_col: str,
    join_keys: List[str],
    pool_random: int,
    pool_uncovered: int,
    uncovered_sample_factor: int,
    heartbeat_sec: int,
):
    """
    Build:
      - pool_random: random sample from clean
      - pool_uncovered: "uncovered-ish" sample from clean using sample anti-join strategy
      - pool_raw_dedup: merged & dedup
      - aspect_in_pool: only aspects for pool keys (reduce join)
      - pool_tagged: pool sentences left-joined with aspect_in_pool

    Key improvement:
      - Never do `SELECT DISTINCT keys FROM aspect_sentences` on the full dataset.
      - Never do `clean_samp LEFT JOIN full aspect` which can explode intermediate rows.
    """
    keys_sql = ", ".join(join_keys)
    null_check_key = join_keys[0]  # for LEFT JOIN null check

    # 1) pool_random
    log_info(f"building pool_random={pool_random} ...")
    t0 = time.time()
    with Heartbeat("pool_random", interval=heartbeat_sec):
        try:
            con.execute(f"""
                CREATE TEMP TABLE pool_random AS
                SELECT {keys_sql}, {sent_col} AS sentence
                FROM read_parquet('{clean_path.as_posix()}')
                USING SAMPLE {int(pool_random)};
            """)
        except Exception:
            con.execute(f"""
                CREATE TEMP TABLE pool_random AS
                SELECT {keys_sql}, {sent_col} AS sentence
                FROM read_parquet('{clean_path.as_posix()}')
                LIMIT {int(pool_random)};
            """)
    n_rand = int(con.execute("SELECT COUNT(*) AS n FROM pool_random;").df().iloc[0]["n"])
    log_info(f"pool_random rows={n_rand} done in {time.time()-t0:.1f}s")

    # 2) pool_uncovered via sample anti-join (stable)
    if pool_uncovered <= 0:
        log_info("pool_uncovered skipped (pool_uncovered=0)")
        con.execute("CREATE TEMP TABLE pool_uncovered AS SELECT * FROM pool_random LIMIT 0;")
    else:
        clean_samp_n = int(pool_uncovered * max(1, uncovered_sample_factor))
        log_info(f"building pool_uncovered={pool_uncovered} via sample anti-join (clean_samp_n={clean_samp_n}) ...")
        t0 = time.time()

        # 2.1 clean_samp
        log_info("building clean_samp ...")
        with Heartbeat("clean_samp_for_uncovered", interval=heartbeat_sec):
            try:
                con.execute(f"""
                    CREATE TEMP TABLE clean_samp AS
                    SELECT {keys_sql}, {sent_col} AS sentence
                    FROM read_parquet('{clean_path.as_posix()}')
                    USING SAMPLE {int(clean_samp_n)};
                """)
            except Exception:
                con.execute(f"""
                    CREATE TEMP TABLE clean_samp AS
                    SELECT {keys_sql}, {sent_col} AS sentence
                    FROM read_parquet('{clean_path.as_posix()}')
                    LIMIT {int(clean_samp_n)};
                """)
        n_samp = int(con.execute("SELECT COUNT(*) AS n FROM clean_samp;").df().iloc[0]["n"])
        log_info(f"clean_samp rows={n_samp}")

        # 2.2 clean_keys distinct (small)
        log_info("building clean_keys (DISTINCT keys from clean_samp) ...")
        with Heartbeat("clean_keys(distinct)", interval=heartbeat_sec):
            con.execute(f"""
                CREATE TEMP TABLE clean_keys AS
                SELECT DISTINCT {keys_sql}
                FROM clean_samp;
            """)
        n_ck = int(con.execute("SELECT COUNT(*) AS n FROM clean_keys;").df().iloc[0]["n"])
        log_info(f"clean_keys rows={n_ck}")

        # 2.3 covered_in_sample: scan aspect once, only output DISTINCT keys
        log_info("building covered_in_sample (scan aspect once, join to clean_keys) ...")
        with Heartbeat("covered_in_sample", interval=heartbeat_sec):
            con.execute(f"""
                CREATE TEMP TABLE covered_in_sample AS
                SELECT DISTINCT a.{keys_sql}
                FROM read_parquet('{aspect_path.as_posix()}') a
                JOIN clean_keys c
                USING({keys_sql});
            """)
        n_cov = int(con.execute("SELECT COUNT(*) AS n FROM covered_in_sample;").df().iloc[0]["n"])
        log_info(f"covered_in_sample rows={n_cov}")

        # 2.4 pool_uncovered: anti-join on small covered_in_sample
        log_info("building pool_uncovered (anti-join within sample) ...")
        with Heartbeat("pool_uncovered(sample_anti_join)", interval=heartbeat_sec):
            con.execute(f"""
                CREATE TEMP TABLE pool_uncovered AS
                SELECT s.{keys_sql}, s.sentence
                FROM clean_samp s
                LEFT JOIN covered_in_sample k
                USING({keys_sql})
                WHERE k.{null_check_key} IS NULL
                LIMIT {int(pool_uncovered)};
            """)
        n_unc = int(con.execute("SELECT COUNT(*) AS n FROM pool_uncovered;").df().iloc[0]["n"])
        log_info(f"pool_uncovered rows={n_unc} done in {time.time()-t0:.1f}s")

    # 3) merge + dedup
    log_info("merging pools (random + uncovered) ...")
    t0 = time.time()
    with Heartbeat("merge_pools", interval=heartbeat_sec):
        con.execute("""
            CREATE TEMP TABLE pool_raw AS
            SELECT * FROM pool_random
            UNION ALL
            SELECT * FROM pool_uncovered;
        """)
    with Heartbeat("dedup_pool_raw", interval=heartbeat_sec):
        con.execute(f"""
            CREATE TEMP TABLE pool_raw_dedup AS
            SELECT DISTINCT {keys_sql}, sentence
            FROM pool_raw;
        """)
    n_dedup = int(con.execute("SELECT COUNT(*) AS n FROM pool_raw_dedup;").df().iloc[0]["n"])
    log_info(f"pool_raw_dedup rows={n_dedup} done in {time.time()-t0:.1f}s")

    # 4) restrict aspect to pool keys first (reduces join)
    log_info("building pool_keys (distinct keys from pool) ...")
    t0 = time.time()
    with Heartbeat("pool_keys(distinct)", interval=heartbeat_sec):
        con.execute(f"""
            CREATE TEMP TABLE pool_keys AS
            SELECT DISTINCT {keys_sql}
            FROM pool_raw_dedup;
        """)
    n_pk = int(con.execute("SELECT COUNT(*) AS n FROM pool_keys;").df().iloc[0]["n"])
    log_info(f"pool_keys rows={n_pk} done in {time.time()-t0:.1f}s")

    log_info("building aspect_in_pool (scan aspect once, keep only pool keys) ...")
    t0 = time.time()
    with Heartbeat("aspect_in_pool", interval=heartbeat_sec):
        con.execute(f"""
            CREATE TEMP TABLE aspect_in_pool AS
            SELECT a.{keys_sql}, a.{l1_col} AS L1, a.{l2_col} AS L2
            FROM read_parquet('{aspect_path.as_posix()}') a
            JOIN pool_keys k
            USING({keys_sql});
        """)
    n_ap = int(con.execute("SELECT COUNT(*) AS n FROM aspect_in_pool;").df().iloc[0]["n"])
    log_info(f"aspect_in_pool rows={n_ap} done in {time.time()-t0:.1f}s")

    # 5) pool_tagged
    log_info("building pool_tagged (pool left join aspect_in_pool) ...")
    t0 = time.time()
    with Heartbeat("pool_tagged", interval=heartbeat_sec):
        con.execute(f"""
            CREATE TEMP TABLE pool_tagged AS
            SELECT p.{keys_sql}, p.sentence, a.L1, a.L2
            FROM pool_raw_dedup p
            LEFT JOIN aspect_in_pool a
            USING({keys_sql});
        """)
    n_pt = int(con.execute("SELECT COUNT(*) AS n FROM pool_tagged;").df().iloc[0]["n"])
    log_info(f"pool_tagged rows={n_pt} done in {time.time()-t0:.1f}s")


# -----------------------------
# Term suggestion logic
# -----------------------------
def suggest_for_term_from_pool(con: duckdb.DuckDBPyConnection, term: str, sample_n: int, example_n: int) -> Dict[str, Any]:
    t = sql_escape(term)

    samp = con.execute(f"""
        WITH samp AS (
            SELECT sentence, L1, L2
            FROM pool_tagged
            WHERE strpos(sentence, '{t}') > 0
            LIMIT {int(sample_n)}
        )
        SELECT * FROM samp;
    """).df()

    if samp.empty:
        return {
            "sample_n": 0,
            "covered_in_sample": 0,
            "uncovered_in_sample": 0,
            "best_L1": "",
            "best_L2": "",
            "best_cnt": 0,
            "best_share": 0.0,
            "second_cnt": 0,
            "covered_ratio": 0.0,
            "examples": ""
        }

    sample_got = len(samp)
    covered_in_sample = int(samp["L1"].notna().sum())
    uncovered_in_sample = sample_got - covered_in_sample
    covered_ratio = covered_in_sample / max(sample_got, 1)

    dist = (
        samp.dropna(subset=["L1", "L2"])
            .groupby(["L1", "L2"], as_index=False)
            .size()
            .sort_values("size", ascending=False)
    )

    best_L1 = best_L2 = ""
    best_cnt = second_cnt = 0
    best_share = 0.0
    if not dist.empty:
        best_L1 = str(dist.iloc[0]["L1"])
        best_L2 = str(dist.iloc[0]["L2"])
        best_cnt = int(dist.iloc[0]["size"])
        second_cnt = int(dist.iloc[1]["size"]) if len(dist) > 1 else 0
        if covered_in_sample > 0:
            best_share = best_cnt / covered_in_sample

    examples = "\n".join(samp["sentence"].astype(str).head(int(example_n)).tolist())

    return {
        "sample_n": sample_got,
        "covered_in_sample": covered_in_sample,
        "uncovered_in_sample": uncovered_in_sample,
        "best_L1": best_L1,
        "best_L2": best_L2,
        "best_cnt": best_cnt,
        "best_share": float(best_share),
        "second_cnt": second_cnt,
        "covered_ratio": float(covered_ratio),
        "examples": examples
    }


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--topk", type=int, default=250)
    ap.add_argument("--sample-n", type=int, default=50)
    ap.add_argument("--example-n", type=int, default=3)

    ap.add_argument("--pool-random", type=int, default=300000, help="随机池大小")
    ap.add_argument("--pool-uncovered", type=int, default=20000, help="未覆盖池大小（推荐 0~20000）")
    ap.add_argument("--uncovered-sample-factor", type=int, default=3,
                    help="clean_samp = pool_uncovered * factor（越大越慢，推荐 2~5）")

    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--duckdb-memory-gb", type=int, default=22, help="DuckDB memory_limit(GB)，建议 20~24")
    ap.add_argument("--enable-duckdb-progress", action="store_true", help="开启 DuckDB 进度条（可能刷屏）")
    ap.add_argument("--progress-ms", type=int, default=2000)
    ap.add_argument("--heartbeat-sec", type=int, default=10)

    ap.add_argument("--checkpoint-every", type=int, default=10, help="每处理 N 个 term 写一次 partial parquet")
    ap.add_argument("--resume", action="store_true", help="若存在 partial parquet，则跳过已完成 term")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    out_dir = repo / "outputs" / args.domain

    clean_path = out_dir / "clean_sentences.parquet"
    aspect_path = out_dir / "aspect_sentences.parquet"
    coverage_xlsx = out_dir / f"aspect_coverage_{args.domain}.xlsx"
    unmapped_xlsx = repo / "aspects" / args.domain / "unmapped_terms.xlsx"

    lexicon_dir = repo / "aspects" / args.domain / "lexicons"
    stoplist_path = repo / "aspects" / args.domain / "stoplist.txt"

    if not clean_path.exists():
        raise FileNotFoundError(f"Missing: {clean_path}")
    if not aspect_path.exists():
        raise FileNotFoundError(f"Missing: {aspect_path}")

    clean_cols = _get_parquet_columns_via_duckdb(clean_path)
    aspect_cols = _get_parquet_columns_via_duckdb(aspect_path)

    sent_col = "sentence" if "sentence" in clean_cols else ("sent" if "sent" in clean_cols else None)
    if not sent_col:
        raise RuntimeError(f"Cannot find sentence column. cols={clean_cols[:80]}")

    l1_col = "aspect_l1" if "aspect_l1" in aspect_cols else ("L1" if "L1" in aspect_cols else None)
    l2_col = "aspect_l2" if "aspect_l2" in aspect_cols else ("L2" if "L2" in aspect_cols else None)
    if not l1_col or not l2_col:
        raise RuntimeError(f"Cannot detect aspect columns. cols={aspect_cols[:80]}")

    join_keys = [k for k in KEY_CANDIDATES if (k in clean_cols and k in aspect_cols)]
    if "doc_id" not in join_keys or "sentence_idx" not in join_keys:
        raise RuntimeError(f"Join keys not enough: {join_keys}")

    log_info(f"sent_col={sent_col}, aspect_cols=({l1_col},{l2_col}), join_keys={join_keys}")

    # candidates
    a = extract_terms_from_unmapped(unmapped_xlsx)
    b = extract_terms_from_coverage(coverage_xlsx)
    cand = pd.concat([a, b], ignore_index=True) if (b is not None and len(b)) else a.copy()
    if cand is None or cand.empty:
        raise RuntimeError("No candidates found. Check unmapped_terms.xlsx / aspect_coverage_xxx.xlsx.")

    cand["term"] = cand["term"].astype(str).str.strip()
    cand = cand[cand["term"].map(is_valid_term)]
    cand["df_num"] = pd.to_numeric(cand.get("df", pd.NA), errors="coerce")
    cand["tf_num"] = pd.to_numeric(cand.get("tf", pd.NA), errors="coerce")

    agg = (
        cand.groupby("term", as_index=False)
            .agg(
                df=("df_num", "max"),
                tf=("tf_num", "max"),
                source=("source", lambda x: "|".join(sorted(set(map(str, x)))))
            )
    )

    existing_lex_terms = load_existing_lexicon_terms(lexicon_dir)
    stop_terms = load_stoplist(stoplist_path)
    agg = agg[~agg["term"].isin(existing_lex_terms)]
    agg = agg[~agg["term"].isin(stop_terms)]

    agg["df_rank"] = agg["df"].fillna(-1)
    agg["tf_rank"] = agg["tf"].fillna(-1)
    agg = agg.sort_values(["df_rank", "tf_rank"], ascending=[False, False]).head(int(args.topk)).reset_index(drop=True)

    log_info(f"candidates to process: {len(agg)}")
    log_info("top candidates preview:\n" + agg.head(5).to_string(index=False))

    out_xlsx = out_dir / f"coverage_suggestions_{args.domain}.xlsx"
    partial_parquet = out_dir / f"coverage_suggestions_{args.domain}.partial.parquet"

    # resume
    done_terms = set()
    rows: List[Dict[str, Any]] = []
    if args.resume and partial_parquet.exists():
        try:
            prev = pd.read_parquet(partial_parquet)
            if not prev.empty and "term" in prev.columns:
                done_terms = set(prev["term"].astype(str).tolist())
                rows = prev.to_dict("records")
                log_info(f"resume: loaded {len(done_terms)} done terms from {partial_parquet}")
        except Exception as e:
            log_warn(f"resume load failed: {e}")

    # DuckDB connect
    con = duckdb.connect(database=":memory:")
    duckdb_setup(
        con=con,
        threads=args.threads,
        temp_dir=(out_dir / "_duckdb_tmp"),
        memory_gb=args.duckdb_memory_gb,
        enable_progress=args.enable_duckdb_progress,
        progress_ms=args.progress_ms,
    )

    # build pool_tagged
    build_pool_tagged(
        con=con,
        clean_path=clean_path,
        aspect_path=aspect_path,
        sent_col=sent_col,
        l1_col=l1_col,
        l2_col=l2_col,
        join_keys=join_keys,
        pool_random=int(args.pool_random),
        pool_uncovered=int(args.pool_uncovered),
        uncovered_sample_factor=int(args.uncovered_sample_factor),
        heartbeat_sec=int(args.heartbeat_sec),
    )

    # term loop
    agg2 = agg[~agg["term"].astype(str).isin(done_terms)].reset_index(drop=True)
    log_info(f"remaining terms: {len(agg2)} (skipped {len(done_terms)})")

    t_loop0 = time.time()
    pbar = tqdm(agg2.iterrows(), total=len(agg2), desc="terms", dynamic_ncols=True)

    for j, (_, row) in enumerate(pbar, start=1):
        term = str(row["term"])
        pbar.set_postfix_str(f"term={term[:12]}")

        info = suggest_for_term_from_pool(
            con=con,
            term=term,
            sample_n=int(args.sample_n),
            example_n=int(args.example_n),
        )

        recommendation = "REVIEW"
        reason = ""
        if info["sample_n"] == 0:
            recommendation = "IGNORE"
            reason = "no_sample_in_pool"
        else:
            best_share = info["best_share"]
            best_cnt = info["best_cnt"]
            second_cnt = info["second_cnt"]
            # 这些阈值是工程启发式：让“高一致性”词自动建议 ADD
            if (best_cnt >= 8 and best_share >= 0.55 and (best_cnt - second_cnt) >= 4
                    and info["best_L1"] and info["best_L2"]):
                recommendation = "ADD"
                reason = f"high_consistency(best_share={best_share:.2f}, best_cnt={best_cnt}, gap={best_cnt-second_cnt})"
            elif info["covered_in_sample"] == 0 and info["uncovered_in_sample"] > 0:
                recommendation = "REVIEW"
                reason = "mostly_uncovered_need_manual"
            else:
                recommendation = "REVIEW"
                reason = f"ambiguous(best_share={best_share:.2f}, best_cnt={best_cnt}, second={second_cnt})"

        rows.append({
            "term": term,
            "df": row.get("df", pd.NA),
            "tf": row.get("tf", pd.NA),
            "source": row.get("source", ""),
            "sample_n": info["sample_n"],
            "covered_in_sample": info["covered_in_sample"],
            "uncovered_in_sample": info["uncovered_in_sample"],
            "covered_ratio": info["covered_ratio"],
            "best_L1": info["best_L1"],
            "best_L2": info["best_L2"],
            "best_cnt": info["best_cnt"],
            "best_share_of_covered": info["best_share"],
            "second_cnt": info["second_cnt"],
            "recommendation": recommendation,
            "reason": reason,
            "decision": "",
            "examples": info["examples"]
        })

        # checkpoint
        if args.checkpoint_every > 0 and (j % int(args.checkpoint_every) == 0):
            out_dir.mkdir(parents=True, exist_ok=True)
            df_part = pd.DataFrame(rows)
            df_part.to_parquet(partial_parquet, index=False)
            elapsed = time.time() - t_loop0
            log_info(f"checkpoint: {len(df_part)} rows -> {partial_parquet} | term_progress={j}/{len(agg2)} | elapsed={elapsed/60:.1f} min")

    con.close()

    # final write
    df_out = pd.DataFrame(rows)
    df_out["priority"] = (
        df_out["uncovered_in_sample"].fillna(0) * 2
        + df_out["df"].fillna(0) * 0.001
        + df_out["best_share_of_covered"].fillna(0) * 10
    )
    df_sorted = df_out.sort_values(["recommendation", "priority"], ascending=[True, False])

    out_dir.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        df_sorted.to_excel(w, sheet_name="suggestions", index=False)

    # also write final checkpoint parquet for resume
    df_sorted.to_parquet(partial_parquet, index=False)

    log_info(f"[OK] wrote: {out_xlsx}")
    log_info(f"[OK] wrote: {partial_parquet}")
    log_info("[NEXT] Fill 'decision' with ADD/STOP/IGNORE then run coverage_apply_updates.py")


if __name__ == "__main__":
    main()

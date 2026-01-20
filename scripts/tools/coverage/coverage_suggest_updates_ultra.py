# -*- coding: utf-8 -*-
"""
coverage_suggest_updates_ultra.py

Faster coverage suggestion generator:
- DOES NOT scan/join outputs/<domain>/aspect_sentences.parquet
- Uses current lexicons to "light-tag" only sampled sentences.
- Workflow:
  1) sample pool_random sentences from clean_sentences.parquet
  2) for each candidate uncovered term, collect up to sample_n example sentences from the pool
  3) light-tag these examples by current lexicons -> estimate best (L1,L2)
  4) output suggestions Excel + partial parquet for resume
"""

import argparse
import re
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import deque, Counter, defaultdict

import duckdb
import pandas as pd
from tqdm import tqdm


SENTIMENT_LIKE = {
    "好", "很好", "不错", "一般", "垃圾", "差", "太差", "满意", "不满意",
    "喜欢", "不喜欢", "推荐", "不推荐", "后悔", "值", "不值",
    "香", "拉胯", "翻车", "真香", "绝了", "无语",
}

KEY_CANDIDATES = ["domain", "brand", "model", "doc_id", "sentence_idx"]


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_info(msg: str):
    print(f"[{ts()}][INFO] {msg}", flush=True)


def log_warn(msg: str):
    print(f"[{ts()}][WARN] {msg}", flush=True)


class Heartbeat:
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


def parse_lexicon_filename(name: str) -> Optional[Tuple[str, str]]:
    # file name: L1__L2.txt
    if not name.endswith(".txt"):
        return None
    stem = name[:-4]
    if "__" not in stem:
        return None
    l1, l2 = stem.split("__", 1)
    l1, l2 = l1.strip(), l2.strip()
    if not l1 or not l2:
        return None
    return l1, l2


def load_lexicons(lexicon_dir: Path) -> Tuple[Dict[str, List[Tuple[str, str]]], List[str]]:
    """
    Return:
      - term2aspects: term -> list of (L1,L2) (support overlaps)
      - all_terms: unique term list
    """
    term2aspects: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    if not lexicon_dir.exists():
        return dict(term2aspects), []

    for p in lexicon_dir.glob("*.txt"):
        pair = parse_lexicon_filename(p.name)
        if not pair:
            continue
        l1, l2 = pair
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                t = line.strip()
                if not t:
                    continue
                term2aspects[t].append((l1, l2))
        except Exception:
            continue

    all_terms = sorted(term2aspects.keys(), key=len, reverse=True)
    return dict(term2aspects), all_terms


# -----------------------------
# Aho-Corasick (pure python)
# -----------------------------
class ACAutomaton:
    def __init__(self, patterns: List[str]):
        patterns = [p for p in patterns if p]
        self.next: List[Dict[str, int]] = [dict()]
        self.fail: List[int] = [0]
        self.out: List[List[str]] = [[]]

        for pat in patterns:
            node = 0
            for ch in pat:
                nxt = self.next[node].get(ch)
                if nxt is None:
                    nxt = len(self.next)
                    self.next[node][ch] = nxt
                    self.next.append(dict())
                    self.fail.append(0)
                    self.out.append([])
                node = nxt
            self.out[node].append(pat)

        q = deque()
        for ch, nxt in self.next[0].items():
            self.fail[nxt] = 0
            q.append(nxt)

        while q:
            r = q.popleft()
            for ch, u in self.next[r].items():
                q.append(u)
                v = self.fail[r]
                while v and ch not in self.next[v]:
                    v = self.fail[v]
                self.fail[u] = self.next[v].get(ch, 0)
                if self.out[self.fail[u]]:
                    self.out[u].extend(self.out[self.fail[u]])

    def find_set(self, text: str) -> set:
        node = 0
        found = set()
        for ch in text:
            while node and ch not in self.next[node]:
                node = self.fail[node]
            node = self.next[node].get(ch, 0)
            if self.out[node]:
                found.update(self.out[node])
        return found


# -----------------------------
# DuckDB sampling
# -----------------------------
def duckdb_setup(con: duckdb.DuckDBPyConnection, threads: int):
    con.execute(f"PRAGMA threads={int(threads)};")


def sample_pool_from_clean(clean_path: Path, join_keys: List[str], sent_col: str, pool_random: int, threads: int) -> pd.DataFrame:
    con = duckdb.connect(database=":memory:")
    try:
        duckdb_setup(con, threads=threads)
        keys_sql = ", ".join(join_keys)
        log_info(f"sampling pool_random={pool_random} from clean_sentences ...")
        with Heartbeat("duckdb_sample_pool", interval=10):
            try:
                df = con.execute(f"""
                    SELECT {keys_sql}, {sent_col} AS sentence
                    FROM read_parquet('{clean_path.as_posix()}')
                    USING SAMPLE {int(pool_random)};
                """).df()
            except Exception:
                df = con.execute(f"""
                    SELECT {keys_sql}, {sent_col} AS sentence
                    FROM read_parquet('{clean_path.as_posix()}')
                    LIMIT {int(pool_random)};
                """).df()
        return df
    finally:
        con.close()


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--topk", type=int, default=300)
    ap.add_argument("--sample-n", type=int, default=50)
    ap.add_argument("--example-n", type=int, default=5)
    ap.add_argument("--pool-random", type=int, default=200000)
    ap.add_argument("--threads", type=int, default=8)

    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    out_dir = repo / "outputs" / args.domain

    clean_path = out_dir / "clean_sentences.parquet"
    coverage_xlsx = out_dir / f"aspect_coverage_{args.domain}.xlsx"
    unmapped_xlsx = repo / "aspects" / args.domain / "unmapped_terms.xlsx"

    lexicon_dir = repo / "aspects" / args.domain / "lexicons"
    stoplist_path = repo / "aspects" / args.domain / "stoplist.txt"

    if not clean_path.exists():
        raise FileNotFoundError(f"Missing: {clean_path}")

    # candidate terms
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

    agg2 = agg[~agg["term"].astype(str).isin(done_terms)].reset_index(drop=True)
    log_info(f"remaining terms: {len(agg2)} (skipped {len(done_terms)})")
    if agg2.empty:
        log_info("nothing to do.")
        return

    # Load lexicons once
    term2aspects, lex_terms = load_lexicons(lexicon_dir)
    if not lex_terms:
        raise RuntimeError(f"No lexicon terms found under: {lexicon_dir}")

    log_info(f"loaded lexicon terms: {len(lex_terms)} (unique)")
    lex_ac = ACAutomaton(lex_terms)

    # Determine join keys & sentence col (assume standard)
    join_keys = [k for k in KEY_CANDIDATES]  # clean output already has these
    sent_col = "sentence"

    # Sample pool
    pool_df = sample_pool_from_clean(
        clean_path=clean_path,
        join_keys=join_keys,
        sent_col=sent_col,
        pool_random=int(args.pool_random),
        threads=int(args.threads),
    )
    if pool_df.empty:
        raise RuntimeError("pool is empty; check clean_sentences.parquet")

    log_info(f"pool rows loaded: {len(pool_df)}")

    # Build AC for candidate terms (only remaining terms)
    cand_terms = agg2["term"].astype(str).tolist()
    cand_ac = ACAutomaton(cand_terms)

    # Collect samples per term (scan pool once)
    sample_n = int(args.sample_n)
    samples: Dict[str, List[Dict[str, Any]]] = {t: [] for t in cand_terms}

    log_info("scanning pool once to collect per-term samples ...")
    with Heartbeat("scan_pool_for_term_samples", interval=10):
        for row in tqdm(pool_df.itertuples(index=False), total=len(pool_df), desc="pool_scan", dynamic_ncols=True):
            sentence = getattr(row, "sentence")
            if not isinstance(sentence, str) or not sentence:
                continue
            hits = cand_ac.find_set(sentence)
            if not hits:
                continue

            # only add if still need samples
            for t in hits:
                lst = samples.get(t)
                if lst is None or len(lst) >= sample_n:
                    continue
                rec = {k: getattr(row, k) for k in join_keys}
                rec["sentence"] = sentence
                lst.append(rec)

            # early stop if all filled
            if all(len(samples[t]) >= sample_n for t in cand_terms):
                break

    # Tag cache to avoid repeated lexicon tagging for same sentence key
    def key_of(rec: Dict[str, Any]) -> Tuple:
        return tuple(rec.get(k) for k in join_keys)

    tag_cache: Dict[Tuple, List[Tuple[str, str]]] = {}

    def tag_sentence(rec: Dict[str, Any]) -> List[Tuple[str, str]]:
        k = key_of(rec)
        if k in tag_cache:
            return tag_cache[k]
        s = rec.get("sentence", "")
        found_terms = lex_ac.find_set(s)
        aspects_set = set()
        for ft in found_terms:
            for (l1, l2) in term2aspects.get(ft, []):
                aspects_set.add((l1, l2))
        out = sorted(aspects_set)
        tag_cache[k] = out
        return out

    # Compute suggestions term by term, with checkpointing
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for i, term in enumerate(tqdm(cand_terms, desc="terms", dynamic_ncols=True), start=1):
        term_samples = samples.get(term, [])
        got = len(term_samples)
        if got == 0:
            rows.append({
                "term": term,
                "df": agg2.loc[agg2["term"] == term, "df"].iloc[0] if (agg2["term"] == term).any() else pd.NA,
                "tf": agg2.loc[agg2["term"] == term, "tf"].iloc[0] if (agg2["term"] == term).any() else pd.NA,
                "source": agg2.loc[agg2["term"] == term, "source"].iloc[0] if (agg2["term"] == term).any() else "",
                "sample_n": 0,
                "covered_in_sample": 0,
                "uncovered_in_sample": 0,
                "covered_ratio": 0.0,
                "best_L1": "",
                "best_L2": "",
                "best_cnt": 0,
                "best_share_of_covered": 0.0,
                "second_cnt": 0,
                "recommendation": "IGNORE",
                "reason": "no_sample_in_pool",
                "decision": "",
                "target_L1": "",
                "target_L2": "",
                "examples": ""
            })
            continue

        covered_sent = 0
        dist = Counter()  # (L1,L2) -> count (counted per sentence, per aspect)
        examples = []
        ex_n = int(args.example_n)

        for rec in term_samples:
            aspects = tag_sentence(rec)
            if aspects:
                covered_sent += 1
                for a in aspects:
                    dist[a] += 1
            if len(examples) < ex_n:
                examples.append(rec["sentence"])

        uncovered_sent = got - covered_sent
        covered_ratio = covered_sent / max(got, 1)

        best_L1 = best_L2 = ""
        best_cnt = second_cnt = 0
        best_share = 0.0
        if dist:
            top2 = dist.most_common(2)
            (best_L1, best_L2), best_cnt = top2[0]
            second_cnt = top2[1][1] if len(top2) > 1 else 0
            total_hits = sum(dist.values())
            best_share = (best_cnt / total_hits) if total_hits > 0 else 0.0

        # Heuristic recommendation
        recommendation = "REVIEW"
        reason = ""
        if got == 0:
            recommendation = "IGNORE"
            reason = "no_sample_in_pool"
        else:
            if (best_cnt >= 8 and best_share >= 0.55 and (best_cnt - second_cnt) >= 4 and best_L1 and best_L2):
                recommendation = "ADD"
                reason = f"high_consistency(best_share={best_share:.2f}, best_cnt={best_cnt}, gap={best_cnt-second_cnt})"
            elif covered_sent == 0 and uncovered_sent > 0:
                recommendation = "REVIEW"
                reason = "mostly_uncovered_need_manual"
            else:
                recommendation = "REVIEW"
                reason = f"ambiguous(best_share={best_share:.2f}, best_cnt={best_cnt}, second={second_cnt})"

        # lookup df/tf/source from agg2
        rr = agg2.loc[agg2["term"] == term]
        dfv = rr["df"].iloc[0] if not rr.empty else pd.NA
        tfv = rr["tf"].iloc[0] if not rr.empty else pd.NA
        src = rr["source"].iloc[0] if not rr.empty else ""

        rows.append({
            "term": term,
            "df": dfv,
            "tf": tfv,
            "source": src,
            "sample_n": got,
            "covered_in_sample": covered_sent,
            "uncovered_in_sample": uncovered_sent,
            "covered_ratio": float(covered_ratio),
            "best_L1": str(best_L1),
            "best_L2": str(best_L2),
            "best_cnt": int(best_cnt),
            "best_share_of_covered": float(best_share),
            "second_cnt": int(second_cnt),
            "recommendation": recommendation,
            "reason": reason,
            "decision": "",
            "target_L1": "",
            "target_L2": "",
            "examples": "\n".join(examples),
        })

        if args.checkpoint_every > 0 and (i % int(args.checkpoint_every) == 0):
            df_part = pd.DataFrame(rows)
            df_part.to_parquet(partial_parquet, index=False)
            elapsed = time.time() - t0
            log_info(f"checkpoint: {len(df_part)} rows -> {partial_parquet} | term_progress={i}/{len(cand_terms)} | elapsed={elapsed/60:.1f} min")

    # final write
    df_out = pd.DataFrame(rows)
    df_out["priority"] = (
        df_out["uncovered_in_sample"].fillna(0) * 2
        + df_out["df"].fillna(0) * 0.001
        + df_out["best_share_of_covered"].fillna(0) * 10
    )
    df_sorted = df_out.sort_values(["recommendation", "priority"], ascending=[True, False])

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        df_sorted.to_excel(w, sheet_name="suggestions", index=False)

    df_sorted.to_parquet(partial_parquet, index=False)

    log_info(f"[OK] wrote: {out_xlsx}")
    log_info(f"[OK] wrote: {partial_parquet}")
    log_info("[NEXT] Fill 'decision' with ADD/STOP/IGNORE (optional target_L1/target_L2) then run coverage_apply_updates.py")


if __name__ == "__main__":
    main()

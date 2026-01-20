#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sync Domain Pack configs to legacy flat configs.

Source of truth:
  configs/domains/<domain>/aspects.yaml
  configs/domains/<domain>/domain.yaml

Legacy mirror (compat layer):
  configs/aspects_<domain>.yaml
  configs/domain_<domain>.yaml

Modes:
  --write : write/update legacy files from domain-pack
  --check : verify legacy files are identical to domain-pack (exit code 0/2)

Why:
  Some scripts still reference legacy paths. This tool keeps them in sync while
  we gradually migrate all code to configs/domains/<domain>/... (Module 4).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"
DOMAINS_ROOT = CONFIGS / "domains"


def read_text_smart(p: Path) -> str:
    data = p.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return data.decode(enc)
        except Exception:
            pass
    # last resort
    return data.decode("utf-8", errors="replace")


def norm(s: str) -> str:
    # normalize line endings + trailing spaces
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in s.split("\n")).rstrip() + "\n"


def list_domains() -> Iterable[str]:
    if not DOMAINS_ROOT.exists():
        return []
    return sorted([p.name for p in DOMAINS_ROOT.iterdir() if p.is_dir()])


def pair_paths(domain: str) -> Tuple[Path, Path, Path, Path]:
    src_aspects = DOMAINS_ROOT / domain / "aspects.yaml"
    src_domain = DOMAINS_ROOT / domain / "domain.yaml"
    dst_aspects = CONFIGS / f"aspects_{domain}.yaml"
    dst_domain = CONFIGS / f"domain_{domain}.yaml"
    return src_aspects, src_domain, dst_aspects, dst_domain


def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def sync_one(domain: str, write: bool) -> Tuple[bool, str]:
    src_aspects, src_domain, dst_aspects, dst_domain = pair_paths(domain)

    if not src_aspects.exists() or not src_domain.exists():
        return False, f"[SKIP] domain={domain} missing source files under configs/domains/{domain}/"

    src_a = norm(read_text_smart(src_aspects))
    src_d = norm(read_text_smart(src_domain))

    ok = True
    msgs = []

    for src_txt, dst_path in [(src_a, dst_aspects), (src_d, dst_domain)]:
        dst_txt = None
        if dst_path.exists():
            dst_txt = norm(read_text_smart(dst_path))

        if dst_txt == src_txt:
            msgs.append(f"[OK] {dst_path.as_posix()} in sync")
            continue

        ok = False
        if write:
            ensure_parent(dst_path)
            dst_path.write_text(src_txt, encoding="utf-8", newline="\n")
            msgs.append(f"[WRITE] {dst_path.as_posix()} updated")
        else:
            if dst_txt is None:
                msgs.append(f"[MISSING] {dst_path.as_posix()} (should be generated)")
            else:
                msgs.append(f"[DIFF] {dst_path.as_posix()} differs from domain-pack source")

    return ok, "\n".join(msgs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write/update legacy files from domain-pack")
    ap.add_argument("--check", action="store_true", help="check legacy files are in sync")
    ap.add_argument("--domains", default="", help="comma-separated domains; default all under configs/domains/")
    args = ap.parse_args()

    if (args.write and args.check) or (not args.write and not args.check):
        print("[FATAL] choose exactly one mode: --write or --check", file=sys.stderr)
        return 2

    domains = [d.strip() for d in (args.domains.split(",") if args.domains else list_domains()) if d.strip()]
    if not domains:
        print("[FATAL] no domains found under configs/domains/", file=sys.stderr)
        return 2

    all_ok = True
    for d in domains:
        ok, msg = sync_one(d, write=args.write)
        print(msg)
        all_ok = all_ok and ok

    if args.check and not all_ok:
        print("[FAILED] legacy configs not in sync. Run with --write to update.", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Script Classification (productized runner view)

## Core Entrypoints (keep)
- `scripts/pipeline_e2e.py` — Orchestrates 00 + tag + RouteB steps; invoked by `scripts/domains/*/run_full.ps1` (see run_full.ps1 contents) and documented in README/TECHNICAL_GUIDE.
- `scripts/step00_ingest_json_to_clean_sentences.py` — Called from `pipeline_e2e.py` step "00".
- `scripts/tag_aspects.py` — Called from `pipeline_e2e.py` step "tag".
- `scripts/route_b_sentiment/pipeline.py` — Called from `pipeline_e2e.py` for steps 01..web; resolves configs via `_shared/config_resolver.py`.
- `scripts/route_b_sentiment/sentiment_01~05*.py` + `export_web_tables_l1_11.py` — Invoked only by `route_b_sentiment/pipeline.py` via subprocess (see functions `step01`..`step_web`).
- Domain wrappers `scripts/domains/<domain>/run_full.ps1` / `run_smoke.ps1` — Thin shells over `pipeline_e2e.py` (full) or subset (smoke).

## UI Tools (coverage optimization, potential UI hooks)
- `scripts/build_aspect_candidates.py` — Aspect candidate miner from clean_sentences (standalone argparse, no imports elsewhere; `rg` finds only self).
- `scripts/coverage_suggest_updates_fast.py`, `scripts/coverage_suggest_updates_ultra.py` — Coverage gap discovery; independent argparse tools (no imports into pipeline).
- `scripts/coverage_apply_updates.py` — Apply suggested lexicon/stoplist changes (standalone).
- `scripts/llm_autofill_decisions.py` — LLM helper for coverage decisions; not referenced elsewhere.
- `scripts/export_product_reports.py` — Exports product-level reports from pred_ds; not wired into pipeline (no `rg` hits except file path).

## Debug / Diagnostics (developer-facing; move to `scripts/debug/` or `tools/`)
- `scripts/check_time_fields.py`, `scripts/check_has_time_duckdb.py`, `scripts/print_time_range.py` — Parquet time/ctime inspectors; standalone argparse; not referenced by pipeline.
- `scripts/split_product_excels.py` — Excel splitter utility; no pipeline references.
- `scripts/init_aspects_phone.py` — One-off aspects seeding for phone; not referenced elsewhere.
- `scripts/route_b_sentiment/check_pred_smoke.py`, `check_pred_mapping.py`, `check_pseudolabel_outputs.py`, `check_asc_pred_ds_oneclick.py`, `diagnose_route_b_pred_ds.py`, `inspect_pseudolabel_dist.py`, `build_trainset_v2_from_raw.py` — QA/diagnostic tools around preds/pseudolabels/trainsets; not called by pipeline (only referenced in their own filenames per `rg`).
- `scripts/_ops/audit_repo.ps1` — Ops helper; outside pipeline.

## Legacy / Deprecated (archive/delete after confirmation)
- `review_pipeline/cli.py` and `review_pipeline/*` — Legacy cleaning pipeline superseded by `step00_ingest_json_to_clean_sentences.py`; not referenced by README run paths except as historical baseline. Suggest archive to `archive/review_pipeline` after ensuring no external dependency.
- `scripts/export_product_reports.py` — Produces product reports from pred_ds but not referenced by README/TECHNICAL_GUIDE or any pipeline `rg` hits; keep only if downstream consumers exist, otherwise archive.
- `scripts/init_aspects_phone.py` — One-off initializer for early phone lexicon; no current references; archive after confirming data migration.
- `scripts/split_product_excels.py` — Standalone Excel splitter; not part of current flow; archive if unused.

Deletion guardrails: each candidate above shows zero imports/subprocess calls from core runners (`pipeline_e2e.py`, `route_b_sentiment/pipeline.py`, domain PS scripts) based on `rg` searches; archive instead of hard delete unless downstream dependencies are confirmed absent.

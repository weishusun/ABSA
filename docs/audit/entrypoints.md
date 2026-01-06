# Entrypoint Inventory

Source: repo tree and in-script argparse/Typer help (2026-01-06). Excludes helpers imported only by other scripts.

- `scripts/step00_ingest_json_to_clean_sentences.py` — Step00 ingest raw JSON/JSONL into `outputs/<domain>/clean_sentences.parquet`; writes ingest manifest/stats (`parse_args` in file).
- `scripts/tag_aspects.py` — Tag sentences with aspects using `configs/domains/<domain>/aspects.yaml`, writes `aspect_sentences.parquet` + coverage reports.
- `scripts/pipeline_e2e.py` — One-click 00 + tag + RouteB steps; orchestrates run_root `outputs/<domain>/runs/<run_id>` (docstring and argparse).
- `scripts/route_b_sentiment/pipeline.py` — Wrapper for RouteB steps 01~05 + `export_web_tables_l1_11.py`; resolves configs per domain.
- `scripts/route_b_sentiment/sentiment_01_build_aspect_pairs_and_train_candidates.py` — Build aspect_pairs_ds/train_candidates; supports resume/checkpoint.
- `scripts/route_b_sentiment/sentiment_02_pseudolabel_openai.py` — OpenAI pseudo-labeler for aspect pairs; outputs `train_pseudolabel.parquet`.
- `scripts/route_b_sentiment/sentiment_03_train_asc_lora.py` — LoRA training for ASC classifier.
- `scripts/route_b_sentiment/sentiment_04_infer_asc.py` — Inference over aspect_pairs_ds, produces `asc_pred_ds` shards.
- `scripts/route_b_sentiment/sentiment_05_aggregate_and_build_excels.py` — Aggregate predictions and emit Excel/Parquet summaries.
- `scripts/route_b_sentiment/export_web_tables_l1_11.py` — Build `web_exports` (7d/30d, L1 regrouped to 11 buckets).
- `scripts/route_b_sentiment/check_pred_smoke.py` — Smoke validator for pred_ds shards (prints schema/dist).
- `scripts/route_b_sentiment/check_pred_mapping.py` — Mapping sanity check for aspect mapping vs preds.
- `scripts/route_b_sentiment/check_asc_pred_ds_oneclick.py` — One-click pred_ds QA (schema, nulls, sample rows).
- `scripts/route_b_sentiment/check_pseudolabel_outputs.py` — Inspect pseudo-label outputs; prints label dist (no argparse).
- `scripts/route_b_sentiment/diagnose_route_b_pred_ds.py` — Diagnostics for pred_ds parquet (argparse defined).
- `scripts/route_b_sentiment/inspect_pseudolabel_dist.py` — Dist analyzer for pseudo-label files (argparse).
- `scripts/route_b_sentiment/build_trainset_v2_from_raw.py` — Alternative builder for trainset (argparse).
- `scripts/build_aspect_candidates.py` — Mine candidate aspect terms from clean_sentences (argparse).
- `scripts/coverage_suggest_updates_fast.py` — Coverage gap miner with heartbeat/checkpoints (argparse).
- `scripts/coverage_suggest_updates_ultra.py` — Heavier coverage suggestion tool (argparse).
- `scripts/coverage_apply_updates.py` — Apply coverage suggestions to lexicon/stoplist (argparse).
- `scripts/export_product_reports.py` — Export product-level reports from pred_ds/agg (argparse).
- `scripts/init_aspects_phone.py` — Seed aspects/lexicons for phone domain (argparse).
- `scripts/llm_autofill_decisions.py` — LLM-assisted decision autofill for coverage tools (argparse + OpenAI).
- `scripts/split_product_excels.py` — Split large Excel by product (argparse).
- `scripts/check_time_fields.py` — Inspect parquet time fields (argparse).
- `scripts/check_has_time_duckdb.py` — DuckDB check for ctime presence (argparse).
- `scripts/print_time_range.py` — Quick parquet time range printer (argparse).
- Domain runners `scripts/domains/<domain>/run_full.ps1` — Powershell wrapper calling `pipeline_e2e.py` steps 00..web.
- Domain runners `scripts/domains/<domain>/run_smoke.ps1` — Powershell wrapper for minimal smoke (01,02).
- `review_pipeline/cli.py` (`python -m review_pipeline.cli clean`) — Legacy multi-domain cleaning pipeline (Typer CLI, still usable for baseline cleaning).
- Ops helper `scripts/_ops/audit_repo.ps1` — Repo audit script (not part of data pipeline).

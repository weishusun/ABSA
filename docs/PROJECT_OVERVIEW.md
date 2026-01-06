# ABSA 项目总览（1 页纸）

- 目标：从多域评论 JSON/JSONL（phone/car/laptop/beauty）生成可交付的情感/方面数据，最终交付物是 `outputs/<domain>/runs/<run_id>/web_exports`（近 7 天/30 天分发表）。
- 输入：`data/<domain>/<brand>/<model>/*.json|jsonl`，内容字段自动识别（content/text/comment/评价内容），时间字段自动识别（ctime/create_time 等）。
- 核心步骤：
  1) Step00 摄入：`scripts/step00_ingest_json_to_clean_sentences.py` → `outputs/<domain>/clean_sentences.parquet`（dataset）+ `meta/ingest_manifest.jsonl`。
  2) 方面标注：`scripts/tag_aspects.py`（用 `configs/domains/<domain>/aspects.yaml` 词表）→ `outputs/<domain>/aspect_sentences.parquet` + 覆盖率报表。
  3) RouteB 情感链路（01~05）：`scripts/route_b_sentiment/pipeline.py` → 01 构造训练候选/分片 → 02 OpenAI 伪标注 → 03 LoRA 训练 → 04 全量推理 → 05 聚合/Excel。
  4) Web 导出：`export_web_tables_l1_11.py` → `web_exports/last7d_day`、`web_exports/last1m_week`（L1 归 11 桶）。
- 运行入口：`python -u scripts/pipeline_e2e.py --domain <d> --run-id <id>`（默认 steps=00,tag,01,02,03,04,05,web），或 PowerShell 域脚本 `scripts/domains/<d>/run_full.ps1`。
- run_id 约定：`{YYYYMMDD}_{domain}_v0`（RouteB 默认）或 `{YYYYMMDD}_{domain}_e2e`（e2e 默认），统一落地 `outputs/<domain>/runs/<run_id>/`。
- workspace 约定：代码库与可变数据分离，默认输出到 `repo_root/outputs`；可通过环境变量 `ABSA_WORKSPACE` 指向外部工作目录，脚本用 `Path` 组合而非硬编码绝对路径。
- 设备策略（device=auto）：优先 CUDA（Win/Linux NVIDIA）→ MPS（mac）→ CPU；无 GPU 时自动降级 CPU，并降低推理默认 batch-size（例如 CUDA/MPS=128，CPU=16）。
- 交付物定义：`web_exports` 下包含近 7 天按日、近 30 天按周的 parquet/csv，含 overall/L1/L2/产品维度 POS/NEG/NEU 计数；附加 `meta/run.json` 记录输入、参数、耗时、版本。

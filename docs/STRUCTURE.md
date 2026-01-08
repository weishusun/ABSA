# Route B 目录与运行结构

## 配置布局
- 新路径：`configs/domains/<domain>/aspects.yaml`、`configs/domains/<domain>/domain.yaml`
- 兼容旧路径：`configs/aspects_<domain>.yaml`、`configs/domain_<domain>.yaml`（内容保持一致）
- `config_resolver` 会优先新路径，找不到才回退旧路径。

## 输出布局
- 统一落在 `outputs/<domain>/runs/<run_id>/`
- 约定子目录：
  - `step01_pairs/`：sentiment_01 输出（train_candidates.parquet、aspect_pairs_ds）
  - `step02_pseudo/`：sentiment_02 输出（pseudolabel 子目录、train_pseudolabel.parquet）
  - `step03_model/`：sentiment_03 模型输出
  - `step04_pred/`：sentiment_04 推理产物（asc_pred_ds）
  - `step05_agg/`：sentiment_05 聚合产物（agg parquet/xlsx）
  - `web_exports/`：export_web_tables_l1_11 导出的 Web 表格
  - `meta/run.json`：pipeline 记录 domain/run_id/输入/steps/输出路径

## Pipeline 入口
```powershell
# 最小示例
python -u .\scripts\route_b_sentiment\pipeline.py `
  --domain laptop `
  --run-id 20260105_laptop_v0 `
  --input-aspect-sentences .\outputs\laptop\aspect_sentences.parquet `
  --steps "01,02,03,04,05,web"
```

常用参数：
- `--steps`：逗号分隔的步骤，支持 01~05、web
- `--max-train-rows` / `--train-pool-rows` / `--shard-n` / `--ds-batch-rows`：传给 step01
- `--step02-max-rows`：传给 step02（小样本 smoke）
- `--base-model` / `--fp16`：传给 step03/04
- `--resume`：传递给支持 resume/overwrite 的步骤（01/03/04）

## 域级快捷脚本
- `scripts/domains/<domain>/run_smoke.ps1`：默认 steps=01,02，小样本参数
- `scripts/domains/<domain>/run_full.ps1`：已切到 e2e（00+tag+01..05+web），run_id 默认当日，路径基于仓库根目录执行

## Step00（原始 JSON/JSONL 接入）
- 脚本：`scripts/step00_ingest_json_to_clean_sentences.py`
- 输入：`data/<domain>/<brand>/<model>/**/*.json|jsonl`（brand/model 从路径推断）
- 输出：`outputs/<domain>/clean_sentences.parquet`（dataset 形式，append 友好），日志 `outputs/<domain>/logs/ingest_errors.log`，元数据 `outputs/<domain>/meta/ingest_manifest.jsonl`、`ingest_stats.json`

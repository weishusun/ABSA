# Pipeline Runbook（同未来 Web UI 流程一致）

## 前置检查
- 安装依赖：`python -m venv .venv && .\.venv\Scripts\activate && pip install -r requirements.txt`
- 数据准备：确认 `data/<domain>/<brand>/<model>/*.json|jsonl` 非空，字段包含 content/text/comment/评价内容。
- Workspace：默认输出 `outputs/<domain>/...`；如需单独工作区，设置 `ABSA_WORKSPACE` 并在运行前创建对应目录。

## 快速端到端
```powershell
python -u .\scripts\pipeline_e2e.py --domain phone --run-id 20260106_phone
```
- 默认 steps=`00,tag,01,02,03,04,05,web`。
- 输出根：`outputs/phone/runs/20260106_phone/`，meta：`meta/run.json`。

## 分步运行
1) Step00 摄入  
```powershell
python -u .\scripts\step00_ingest_json_to_clean_sentences.py --domain phone --resume
```
输出：`outputs/phone/clean_sentences.parquet` + `meta/ingest_manifest.jsonl`。

2) 方面标注  
```powershell
python -u .\scripts\tag_aspects.py --input outputs/phone/clean_sentences.parquet `
  --config configs/domains/phone/aspects.yaml `
  --output-dir outputs/phone
```
输出：`aspect_sentences.parquet`，覆盖率报表 `aspect_coverage_phone.xlsx`，未覆盖 top terms。

3) RouteB 链路（可选 steps）  
```powershell
python -u .\scripts\route_b_sentiment\pipeline.py `
  --domain phone `
  --input-aspect-sentences outputs/phone/aspect_sentences.parquet `
  --steps "01,02,03,04,05,web" `
  --run-id 20260106_phone_v0
```
关键输出：  
- `step01_pairs/`（train_candidates.parquet, aspect_pairs_ds/）  
- `step02_pseudo/`（train_pseudolabel.parquet）  
- `step03_model/`（LoRA 模型）  
- `step04_pred/asc_pred_ds/`（分片预测 parquet）  
- `step05_agg/`（聚合 parquet + Excel）  
- `web_exports/`（last7d_day/last1m_week）

## Smoke / Resume
- 小样本：`scripts/domains/<domain>/run_smoke.ps1`（仅 01,02）。
- 断点续跑：Step00/01/04 支持 `--resume`；02/03/05 重复运行会覆盖输出（建议换 run_id）。

## 常见检查点
- Step00：查看 `outputs/<domain>/meta/ingest_stats.json`，确认 `files_processed` > 0。
- tag_aspects：检查 `aspect_coverage_<domain>.xlsx`，关注 `covered_rate`、未覆盖 top terms。
- RouteB：`step05_agg/aspect_sentiment_counts_<domain>.xlsx` 是否生成；`web_exports` 目录是否齐全。

## 错误处理
- 缺列/空文件：确认输入路径正确，数据文件非空。
- OpenAI 限流：在 step02 增加 `--max-rows` 做 smoke 或调整 batch。
- GPU 不可用：device=auto 降级 CPU，酌情调低 step04 推理 batch-size（参考 LOCAL_WEB_UI_DESIGN）。

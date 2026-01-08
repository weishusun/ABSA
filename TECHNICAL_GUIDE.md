# Technical Guide (ABSA Route B E2E)

本指南面向协作者，按仓库现状梳理从原始 JSON/JSONL 到 web_exports 的端到端流程、产物契约、常见问题及扩展建议。

## 目录结构速览
- `scripts/`
  - `step00_ingest_json_to_clean_sentences.py`：原始 JSON/JSONL 摄入
  - `tag_aspects.py`：基于词表的方面标注
  - `pipeline_e2e.py`：端到端编排（00 + tag + 01..05 + web）
  - `route_b_sentiment/`
    - `_shared/config_resolver.py`, `_shared/paths.py`
    - `pipeline.py`：Route B 步骤 01..05 + web 的编排
    - `sentiment_01_build_aspect_pairs_and_train_candidates.py`
    - `sentiment_02_pseudolabel_openai.py`
    - `sentiment_03_train_asc_lora.py`
    - `sentiment_04_infer_asc.py`
    - `sentiment_05_aggregate_and_build_excels.py`
    - `export_web_tables_l1_11.py`
  - `domains/<domain>/run_smoke.ps1`、`run_full.ps1`
- `configs/`
  - 新路径：`configs/domains/<domain>/{aspects.yaml,domain.yaml}`
  - 兼容旧路径：`configs/aspects_<domain>.yaml`、`configs/domain_<domain>.yaml`（内容同新路径）
- `docs/`
  - `STRUCTURE.md`（结构说明）
- `review_pipeline/`：底层清洗切句库（已被 step00 封装调用）

## 端到端流程总览
```
data/<domain>/<brand>/<model>/**/*.json|jsonl
  └─ step00_ingest_json_to_clean_sentences.py
       → outputs/<domain>/clean_sentences.parquet (dataset)
       → logs/meta
  └─ tag_aspects.py (用 configs/domains/<domain>/aspects.yaml)
       → outputs/<domain>/aspect_sentences.parquet
  └─ route_b_sentiment/pipeline.py (steps 01..05 + web)
       → outputs/<domain>/runs/<run_id>/step01_pairs/.../web_exports
```

## 关键脚本与用法
### Step00 摄入（scripts/step00_ingest_json_to_clean_sentences.py）
- 参数：`--domain`（必填）；可选 `--data-root`(默认 `data/<domain>`)、`--output`(默认 `outputs/<domain>/clean_sentences.parquet`)、`--resume`、`--max-files`、`--max-docs`、`--chunk-size`
- 功能：
  - 扫描 `data/<domain>/<brand>/<model>/**/*.json|jsonl`，brand/model 来自路径片段
  - 支持 JSONL（逐行 json）与 JSON（对象或数组），解析失败写 `outputs/<domain>/logs/ingest_errors.log`
  - 内容字段候选：`content,text,comment,review,body,评价内容`
  - 时间字段候选：`ctime,create_time,comment_time,time,date,createdAt,created_at`
  - id 字段候选：`id,review_id,doc_id,comment_id,content_id`，缺失则 `md5(path#index)`
  - 句子切分：中文标点/换行 `SENT_SPLIT_RE`
  - 输出 dataset（append 友好），meta：`meta/ingest_manifest.jsonl`、`meta/ingest_stats.json`

### 方面标注（scripts/tag_aspects.py）
- 输入：`--input outputs/<domain>/clean_sentences.parquet`，`--config configs/domains/<domain>/aspects.yaml`，`--output-dir outputs/<domain>`
- 输出：`outputs/<domain>/aspect_sentences.parquet`，并产出词表冲突等日志。

### E2E 编排（scripts/pipeline_e2e.py）
- 参数：`--domain --run-id --steps --resume`
- 默认 steps：`00,tag,01,02,03,04,05,web`
- 00：调用 step00（若 clean_sentences 已存在且不含 00，可跳过）
- tag：调用 tag_aspects（自动用新路径 config）
- 01..web：调用 `scripts/route_b_sentiment/pipeline.py`，`--input-aspect-sentences` 固定为 `outputs/<domain>/aspect_sentences.parquet`
- 输出：`outputs/<domain>/runs/<run_id>/...`，并写 `meta/run.json`

### Route B 编排（scripts/route_b_sentiment/pipeline.py）
- 参数：`--domain --run-id --input-aspect-sentences --steps --resume ...`
- 默认输出根：`outputs/<domain>/runs/<run_id>/`
- 步骤与核心输出：
  - 01 → `step01_pairs/`：`train_candidates.parquet` + `aspect_pairs_ds/`
  - 02 → `step02_pseudo/`：`pseudolabel/`、`pseudolabel_raw.parquet`、`train_pseudolabel.parquet`
  - 03 → `step03_model/`：LoRA 模型目录
  - 04 → `step04_pred/`：`asc_pred_ds`（分片 parquet，含 p_pos/p_neu/p_neg/confidence/pred_label）
  - 05 → `step05_agg/`：聚合 parquet + Excel
  - web → `web_exports/`：7d/30d Web 表

### 单步脚本（scripts/route_b_sentiment/sentiment_01~05*.py）
- 01：构建 `train_candidates.parquet` + `aspect_pairs_ds`，支持 `--write-ds --resume --shard-n --ds-batch-rows`
- 02：OpenAI 伪标注，输出 `pseudolabel_raw.parquet` / `train_pseudolabel.parquet`，写 `meta.json` 记录 pair_id 策略
- 03：LoRA 训练，输入 `train_pseudolabel.parquet`
- 04：全量推理，输出 `asc_pred_ds`（附 `pred_id/pred_label/confidence/p_neg/p_neu/p_pos`）
- 05：聚合/时间序列/Excel，输入 `asc_pred_ds`，可选 `--pairs-parquet`
- `export_web_tables_l1_11.py`：生成 web_exports（7d/day、30d/week，L1=11 Top10+其他）

### 域级脚本
- `scripts/domains/<domain>/run_smoke.ps1`：小样本 01,02
- `scripts/domains/<domain>/run_full.ps1`：端到端（00+tag+01..05+web），run_id 默认为当天+domain。运行前确保 `data/<domain>/<brand>/<model>/` 有原始 JSON/JSONL。

## 关键产物契约
### clean_sentences.parquet（dataset）
- 列：`domain, brand, model, doc_id, sentence_idx, sentence, ctime, source_path`
- 用途：tag_aspects 输入

### aspect_sentences.parquet
- 列：`domain, brand, model, doc_id, sentence_idx, sentence, aspect_l1, aspect_l2, ctime, ...`
- 用途：step01 输入；aspect_l1/l2 必备；ctime 供下游时间序列

### step01_pairs/aspect_pairs_ds（分片 parquet）
- 列：`domain, brand, model, doc_id, sentence_idx, sentence, aspect_l1, aspect_l2, ctime, shard`
- 用途：step04 推理输入；shard 由脚本计算

### train_candidates.parquet
- 列：同上（不含 shard），候选训练样本

### pseudolabel_raw.parquet / train_pseudolabel.parquet
- 列：`pair_id, label(POS/NEG/NEU), confidence, aspect_l1, aspect_l2, sentence, ctime, ...`
- train_pseudolabel 去重取最高置信；step03 训练输入

### asc_pred_ds（step04 输出，分片）
- 列：上游字段 + `pred_id(0/1/2), pred_label(NEG/NEU/POS), p_neg, p_neu, p_pos, confidence`
- 用途：step05 聚合、web_exports

### step05 聚合 & Excel
- parquet：`aspect_sentiment_agg*.parquet`（全量/硬口径/软权重），`aspect_sentiment_timeseries*.parquet`
- Excel：`aspect_sentiment_counts_{domain}.xlsx`（all/hard/soft 多 sheet）

### web_exports
- 目录：`web_exports/last7d_day`、`web_exports/last1m_week`
- 口径：近 7 天按日、近 30 天按周，L1 映射为 11 段（Top10 + “其他”），产出 overall/产品/L1 维度的 parquet/csv（POS/NEG/NEU 计数）

## 如何运行（最小命令）
### 一键端到端
```powershell
python -u .\scripts\pipeline_e2e.py --domain phone --run-id 20260105_phone
```
（需先放置 `data/phone/<brand>/<model>/*.json|jsonl`）

### 域级入口
```powershell
.\scripts\domains\phone\run_full.ps1   # 全量 00+tag+01..05+web
.\scripts\domains\phone\run_smoke.ps1  # 仅 01,02 小样本
```

### 单步调试示例
```powershell
# 只跑 Step00
python -u .\scripts\step00_ingest_json_to_clean_sentences.py --domain laptop --max-files 2 --max-docs 100
# 只跑 tag
python -u .\scripts\tag_aspects.py --input outputs\laptop\clean_sentences.parquet --config configs\domains\laptop\aspects.yaml --output-dir outputs\laptop
# 只跑 01
python -u .\scripts\route_b_sentiment\sentiment_01_build_aspect_pairs_and_train_candidates.py --input outputs\laptop\aspect_sentences.parquet --output-dir outputs\laptop\sentiment --write-ds --shard-n 4 --ds-batch-rows 10000
```

## 常见错误与排查
- 输入文件不存在/路径错误：确认 `data/<domain>/<brand>/<model>/` 是否有 json/jsonl；`--input-aspect-sentences` 是否指向生成的文件。
- parquet dataset 目录：Step00 输出为目录型；下游脚本接受目录或文件，确保传递目录时带路径末尾。
- pyarrow thrift 限制：Step01/Step00 使用 thrift limit setter；旧版 pyarrow 可能忽略 format_options，已加入 setter 和 duckdb schema fallback。
- DuckDB fallback：Step01 schema 读取失败会回退 duckdb；Step02/05 等依赖 duckdb 读取 parquet，确保路径可访问。
- OpenAI/限流：Step02 需有效 OpenAI key；遇到 429/对齐不足会自动重试。可用 `--max-rows` 做小样本 smoke。
- 覆盖/断点：Step01/04 支持 `--resume`；Step05 Excel 若存在且未 `--overwrite` 会 FATAL；run_full 默认新 run_id 避免覆盖。

## 开发者指南
- 新增 domain：
  - 在 `configs/domains/<domain>/` 准备 `aspects.yaml`、`domain.yaml`（可复制现有模板）
  - 新建 `scripts/domains/<domain>/run_smoke.ps1` / `run_full.ps1`（可参考 phone/laptop）
  - 确保 `configs/aspects_<domain>.yaml`、`configs/domain_<domain>.yaml` 也存在（兼容旧路径，可复制同内容）
- 扩充 aspects：更新 `configs/domains/<domain>/aspects.yaml`，重跑 tag 或 e2e。
- Smoke QC：
  - Step00：`--max-files --max-docs` 小样本；检查 `meta/ingest_stats.json`
  - Step02：`--max-rows` 限制伪标注量；查看 `pseudolabel/meta.json`
  - Step05：生成 `examples.parquet`（加 `--make-examples`）抽检
- 避免大文件入库：`data/`、`outputs/` 应在 `.gitignore`；不要将生成的 parquet/模型提交版本库。

## 当前分支变更摘要
- `restructure-domains`：配置拆分至 `configs/domains/<domain>/`，新增 Route B pipeline 与域级脚本。
- `e2e-ingest`：新增 Step00 摄入与 `pipeline_e2e.py`，域级 run_full 切换到端到端。

## 推荐验收清单
- `python -m compileall scripts` 通过（依赖包的 SyntaxWarning 可忽略）。
- 将一个 json 放入 `data/car/Mercedes/MercedesEQE/`，运行 `scripts/domains/car/run_full.ps1`，至少完成 00+tag，生成 `outputs/car/aspect_sentences.parquet`。
- 跑最小端到端：`python -u scripts/pipeline_e2e.py --domain laptop --steps 00,tag,01 --run-id qa_test`，确认 `outputs/laptop/runs/qa_test/step01_pairs/` 生成。

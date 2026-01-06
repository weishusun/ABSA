# 数据契约（核心表/文件）

## clean_sentences.parquet（Step00 输出，dataset 目录）
- 必备列：`domain`, `brand`, `model`, `doc_id`, `sentence_idx`, `sentence`, `ctime`, `source_path`
- 行级含义：原始评论切句后的句子；brand/model 来自路径片段；doc_id 取优先字段或 md5(path#index)。
- 生产者：`scripts/step00_ingest_json_to_clean_sentences.py`
- 消费者：`scripts/tag_aspects.py`

## aspect_sentences.parquet（tag_aspects 输出）
- 必备列：`sentence`, `aspect_l1`, `aspect_l2`
- 常见附加列：`domain`, `brand`, `model`, `doc_id`, `sentence_idx`, `ctime`, `platform`, `url`
- 生产者：`scripts/tag_aspects.py`
- 消费者：`scripts/route_b_sentiment/sentiment_01_build_aspect_pairs_and_train_candidates.py`
- 附件：`aspect_coverage_<domain>.xlsx`（覆盖率、未覆盖 top terms）、`aspect_counts_<domain>.xlsx`（品牌/型号维度统计）

## aspect_pairs_ds（目录，Step01 输出）
- 结构：分片 parquet，包含 `sentence`, `aspect_l1`, `aspect_l2`, `brand`, `model`, `doc_id`, `sentence_idx`, `ctime`, `shard`
- 生产者：`sentiment_01_build_aspect_pairs_and_train_candidates.py`（`--write-ds`）
- 消费者：`sentiment_04_infer_asc.py`

## train_candidates.parquet / train_pseudolabel.parquet（Step01/02）
- 列：pair 信息 + `label`（Step02 输出带 pseudo label）、`confidence`
- 生产者：Step01 构造候选；Step02 调用 OpenAI 打标签。
- 消费者：`sentiment_03_train_asc_lora.py`

## asc_pred_ds（Step04 输出，目录）
- 列：上游元信息 + `pred_id`(0/1/2), `pred_label`(NEG/NEU/POS), `p_neg`, `p_neu`, `p_pos`, `confidence`
- 生产者：`sentiment_04_infer_asc.py`
- 消费者：`sentiment_05_aggregate_and_build_excels.py`, `export_web_tables_l1_11.py`

## step05 聚合产物
- `aspect_sentiment_agg*.parquet`：总体/硬口径/软权重聚合。
- `aspect_sentiment_timeseries*.parquet`：时间序列。
- `aspect_sentiment_counts_<domain>.xlsx`：多 sheet 计数报表。

## web_exports（最终交付）
- 目录：`runs/<run_id>/web_exports/last7d_day`、`last1m_week`
- 内容：parquet/csv，含 overall/L1/L2/品牌/型号维度 POS/NEG/NEU 计数；L1 归并到 11 个桶。
- 生产者：`export_web_tables_l1_11.py`
- 消费者：Web UI / BI 报表。

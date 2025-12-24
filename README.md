# Review Pipeline

用于多品类评论数据的清理、标准化与切句，可复用到 phone/car/laptop/beauty 等领域。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行

```bash
python -m review_pipeline.cli clean --domain phone --input data/phone --output outputs/phone --config configs/domain_phone.yaml
```

可选参数：
- `--workers`: 并行 worker 数（默认 1）。
- `--force`: 忽略断点续跑状态，强制重跑。

## 数据结构

输入目录：`data/<domain>/<brand>/<model>/*.json|*.jsonl`
输出：
- `clean_sentences.parquet`（或 `clean_sentences.csv`）
- `manifest.json`
- `state.json`（断点续跑）

## 配置说明

以 `configs/domain_phone.yaml` 为例：
- `content_field` 等字段映射支持嵌套路径（用 `.` 分隔）。
- `keep_emoji` / `keep_english` / `min_length` 控制清洗策略。
- `noise_suffix_patterns` / `extra_noise_patterns` 提供领域特定的噪声正则。
- `splitter.min_len` / `splitter.max_len` 控制切句长度。
- `brand_override` / `model_override` 可覆盖路径推断。
- `extra_fields` 会保留到 `extra_json`，避免信息丢失。

## 标准输出 schema（句子级）
- domain, brand, model, doc_id, platform, url, ctime, like_count, reply_count
- content_raw, content_clean
- sentence_idx, sentence
- source_file, source_line
- parse_error, error_msg
- extra_json

## 日志与断点续跑
- 处理进度会以 INFO 级别输出。
- `state.json` 记录文件 hash/mtime/size，未变化文件会被跳过。

## 开发提示
- 默认使用 pandas + pyarrow 写 Parquet，缺失 pyarrow 时自动退回 CSV。
- JSON 解析优先使用 orjson，未安装时自动使用内置 json。

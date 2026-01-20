# Domain Pack Spec（configs/domains/<domain>/）

每个域需具备以下文件，供 Step00/Tag/RouteB 使用：

- `aspects.yaml`  
  - 结构：`l1: [{name, aliases?, l2: [{name, terms: [...]}, ...]}]`  
  - 支持旧版 `aspects: [{l1, l2: {<name>: <lex_path>}}]`，但新包应使用内联 `terms` 或相对路径词表。  
  - 可选 `settings.coverage_gate`：`l1_min_rate`、`l2_min_rate`、`unclassified_max_rate`。  
  - 允许 L1 级 aliases（生成虚拟 `_L1` 命中）。

- `domain.yaml`  
  - 字段映射：`content_field`, `url_field`, `platform_field`, `ctime_field`, `like_field`, `reply_field`。  
  - 清洗策略：`keep_emoji`, `keep_english`, `min_length`、`noise_suffix_patterns`、`extra_noise_patterns`。  
  - 切句：`splitter.min_len`, `splitter.max_len`。  
  - 品牌/型号覆盖：`brand_override`, `model_override`。  
  - 追加字段：`extra_fields`（保留到 `extra_json`）。

- 可选：`stopwords.txt` / `aliases/*.txt` / `lexicons/*.txt`（若词表采用文件存储）。

目录要求：
- 路径：`configs/domains/<domain>/aspects.yaml`、`configs/domains/<domain>/domain.yaml`。
- 兼容旧路径：`configs/aspects_<domain>.yaml`、`configs/domain_<domain>.yaml` 应与新路径内容一致（便于旧脚本）。

扩展新域流程：
1) 复制现有域的 `aspects.yaml`、`domain.yaml` 模板，按新域词表/字段调整。
2) 若有文件型词表，将文件放置在 `aspects/<domain>/lexicons/`（或与 config 相对目录），并在 `aspects.yaml` 使用相对路径。
3) 新建 `scripts/domains/<domain>/run_full.ps1` 与 `run_smoke.ps1`，调用 `pipeline_e2e.py`，修改 `--domain` 与默认 run_id 后缀。
4) 在 README/TECHNICAL_GUIDE 中登记新域入口，确认 `configs/domains/<domain>/` 被 `config_resolver.py` 识别。
5) 跑一次 `scripts/pipeline_e2e.py --domain <domain> --steps 00,tag` 验证 ingest + tag，检查 `aspect_coverage_<domain>.xlsx`。

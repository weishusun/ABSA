# 目录用途说明

- `scripts/` — 产品化入口与工具脚本  
  - `step00_ingest_json_to_clean_sentences.py`：原始 JSON/JSONL 摄入。  
  - `tag_aspects.py`：方面标注 + 覆盖率报表。  
  - `pipeline_e2e.py`：端到端编排（00 + tag + RouteB）。  
  - `route_b_sentiment/`：RouteB 子步骤 01~05 + web 导出及 QA 工具。  
  - `domains/<domain>/run_full.ps1 | run_smoke.ps1`：域级运行入口。  
  - 其他以 coverage/check/inspect 命名的脚本：覆盖优化或诊断工具（详见 `docs/audit/script_classification.md`）。
- `review_pipeline/` — 早期清洗切句库（Typer CLI）；Step00 已封装同类能力，推荐只在极简清洗场景使用或迁移到 archive。
- `configs/` — 配置  
  - 新结构：`configs/domains/<domain>/{aspects.yaml,domain.yaml}`。  
  - 兼容旧路径：`configs/aspects_<domain>.yaml`、`configs/domain_<domain>.yaml`。  
  - aspects.yaml 定义 L1/L2 及词表/别名；domain.yaml 定义清洗字段映射、切句策略等。
- `aspects/` — 领域词表资产（目前仅 phone 细粒度 lexicons/stoplist）。
- `data/` — 示例原始数据，按 `<domain>/<brand>/<model>/*.json|jsonl` 组织。
- `outputs/` — 运行输出（应保持 gitignore）。结构：`outputs/<domain>/clean_sentences.parquet`、`aspect_sentences.parquet`、`runs/<run_id>/step01.../web_exports`。
- `docs/` — 文档与审计产物（本次新增 audit/、PROJECT_OVERVIEW、RUNBOOK 等）。  
- `requirements.txt` — 运行依赖。  
- `.gitignore` — 需刷新为通用 ignore（见 `docs/audit/absolute_path_findings.md`）。

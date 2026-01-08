# 本地 Web UI Runner 设计（架构草案）

## 组件划分
- UI（浏览器）：React/Vue 任意；展示 run 状态、覆盖率、未覆盖词建议、模型选择、日志。
- API（本机服务）：FastAPI/Flask；统一暴露 `/runs/*` REST，封装子进程调用。
- Pipeline Core（状态机）：Python 调度层，按节点驱动 Step00 → tag → coverage review/opt → RouteB → web export，生成 manifest。
- Domain Pack：`configs/domains/<domain>/...` + 词表资产；UI 侧可浏览/编辑词表变更草稿。
- Model Store：`workspace/models/<domain>/<run_id>/`，存放 LoRA/基础模型缓存。
- Workspace：`workspace/<domain>/...`，镜像 `outputs` 结构；repo 只存代码和 configs。

## 状态机节点与交互
1) Preflight：检查数据目录非空 (`data/<domain>`)，校验依赖/设备/端口占用，准备 run_id。  
2) Step00 ingest：子进程跑 `step00_ingest_json_to_clean_sentences.py`。输出 manifest: 输入文件数/行数/耗时/输出路径。  
3) tag_aspects：跑 `tag_aspects.py`，收集覆盖率、未覆盖 top terms（存 `aspect_coverage_<domain>.xlsx` 路径）。  
4) Coverage Review/Optimize Loop：UI 展示覆盖率与未覆盖词；用户可触发 “挖漏→建议→应用→重跑 tag”。实现：  
   - 候选词挖掘：调用 `coverage_suggest_updates_fast.py`（或轻量版）生成 patch 建议。  
   - 归因建议：将建议映射到 L1/L2/stoplist（可用 `llm_autofill_decisions.py`）。  
   - Patch 写入：调用 `coverage_apply_updates.py` 或直接修改 `aspects.yaml`/stoplist，再重跑 tag_aspects。  
5) Model Decision：检测 `workspace/models/<domain>` 是否已有已训练模型（查最新 `step03_model`）。UI 询问 “复用现有模型推理” vs “重新训练”。  
6) RouteB 01~05：按选择运行 `pipeline.py --steps 01..05`（或跳过 03/04 直接用已有模型）。  
7) Export/Report：`pipeline.py --steps web`，同时聚合日志/产物索引，生成最终 manifest。

## 进度与日志采集
- 子进程启动参数：`python -u ...`，stdout/stderr 实时流式到 UI。  
- Heartbeat：包装子进程，若 N 秒无输出则发送心跳到 UI。  
- Manifest：每节点完成后写 `workspace/<domain>/runs/<run_id>/meta/manifest_<step>.json`，记录输入指纹（hash/mtime）、输出路径、行数、耗时、版本（git head + requirements.txt hash）。  
- Resume：读取 manifest/ingest_manifest/step01 checkpoint，允许从任意节点重启（UI 按节点状态启用/禁用按钮）。

## 设备选择策略（device=auto）
- 优先顺序：CUDA (NVIDIA) → MPS (macOS) → CPU。  
- 实现：API 层探测 `torch.cuda.is_available()`、`torch.backends.mps.is_available()`。  
- Batch-size 默认：CUDA/MPS 取 128，CPU 取 16（step04 推理）；UI 可暴露高级设置。  
- 无 GPU 时自动降级 CPU，并在 UI 提示性能预估。

## 单实例锁与端口策略
- 运行锁：`workspace/locks/absa_runner.lock`（包含当前 run_id/step）；重复启动时 UI 提示继续/强制中止。  
- 端口：API 默认 8080；占用则向上递增或提示手动选择。

## 打包与分发
- Repo / Workspace 分离：发布包仅包含 `scripts/`, `configs/`, `docs/`, `requirements.txt`。用户配置 `ABSA_WORKSPACE` 指向可写目录（含 outputs、models、logs）。  
- Windows/macOS：提供 pyinstaller/uvicorn 一键启动脚本；UI 静态资源打包入同一目录。  
- 避免将 `outputs/`、`.venv/`、模型文件纳入分发；运行时自动创建缺失目录。  
- 日志/缓存清理：提供 UI 按钮清理 `workspace/<domain>/runs/*` 中的临时文件（保留 manifest）。

## MVP 拆解
- MVP-1：CLI 封装 + API 壳；提供 Preflight/Step00/tag/RouteB 触发与日志流，device=auto 探测，manifest 写入，每域单 run_id 管理，UI 简单展示进度条/日志。
- MVP-2：覆盖率查看与单次优化回路；UI 展示未覆盖 top terms，支持调用 coverage_suggest + apply patch + 重跑 tag；模型复用检测（提示“复用现有模型推理”）。
- MVP-3：完善断点续跑/锁/并发安全；模型管理面板（列出现有模型，选择复用/重训），批量 run 队列，下载 web_exports/报表按钮，端口/权限配置。

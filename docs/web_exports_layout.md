# Web Exports Layout v1 — 2026-01-08
> 模块 3-1 固化产物：目录规则 + schema（与 `docs/web_exports_schema.md` 配套）

本文件定义 ABSA 工程在 Workspace 中的 `web_exports/` 输出目录结构、版本化策略、以及 Windows 平台下不依赖符号链接的“可审计 + 可原子更新”约定。任何 downstream（Web 后端/前端、ETL、BI、回归测试）必须以本契约为准。

---

## 1. 基线与目标

### 1.1 基准路径（固定约定）
所有 web exports 输出均落在：

`E:\ABSA_WORKSPACE\outputs\<domain>\web_exports\`

其中 `<domain>` ∈ `{phone, car, laptop, beauty}`。

### 1.2 设计目标
1. **网站可直接消费**：网站只需读取 `latest/` 即可获得当前版本的全套标准表。
2. **可追溯与可审计**：每次导出都有不可变快照 `exports/<stamp>/`，并写入 manifest 记录 inputs/outputs。
3. **可重复与可回滚**：`latest.json` 指向快照；发生问题可切换指针回滚到任意历史快照。
4. **Windows 友好**：不依赖 symlink/hardlink；采用“先写快照，再覆盖 latest”的更新语义，尽量避免半更新状态。

---

## 2. 版本化与稳定性（核心约定）

### 2.1 两个视图：exports vs latest
- `exports/<stamp>/`：**不可变快照**（归档/审计/回溯用），一旦成功生成不得修改。
- `latest/`：**可变指针视图**（网站读取用），每次导出成功后可覆盖更新。

### 2.2 `<stamp>` 规则
`<stamp>` 推荐使用本地时间（便于人工排障），格式：

- `YYYYMMDD_HHMMSS`  
  示例：`20260108_153012`

约束：
- 同一 `<domain>` 下 `<stamp>` 必须唯一（即导出任务不应复用同一 stamp 目录）。
- `<stamp>` 仅用于版本标识，不作为业务时间字段（业务时间仍由表内 `day/week_start/week_end` 表达）。

### 2.3 `latest.json` 指针文件
`latest.json` 位于 `web_exports/` 根目录，用于记录当前 `latest/` 对应的快照版本与运行信息，以便调度器/监控/回溯工具快速定位：

路径：`E:\ABSA_WORKSPACE\outputs\<domain>\web_exports\latest.json`

最小建议字段（可扩展但不建议删除/改名）：

```json
{
  "domain": "phone",
  "stamp": "20260108_153012",
  "run_id": "20260108_phone_web",
  "updated_at": "2026-01-08T15:30:12"
}
```

---

## 3. 目录结构定义

### 3.1 标准结构
`web_exports/` 必须包含以下目录/文件（除非导出失败，否则应完整存在）：

| 路径 | 说明 | 稳定性 |
| --- | --- | --- |
| `exports/` | 不可变快照集合（每次导出一个 `<stamp>` 子目录） | 不可变 |
| `exports/<stamp>/tables/` | 网站消费标准表（Parquet），表名与字段遵循 schema v1 | 不可变 |
| `exports/<stamp>/meta/manifest_web.json` | 本次导出的 manifest（inputs/outputs/run 信息） | 不可变 |
| `latest/` | 当前对外暴露视图（网站读取） | 可覆盖 |
| `latest/tables/` | 与快照同名同 schema 的表文件 | 可覆盖 |
| `latest/meta/manifest_web.json` | latest 对应的 manifest（通常与快照一致） | 可覆盖 |
| `latest.json` | 指针文件，指向当前使用的 `<stamp>` 与 `run_id` | 可覆盖 |

### 3.2 `tables/` 内文件集合（schema v1）
`tables/` 内必须按 schema v1 输出固定表集（详见 `docs/web_exports_schema.md`）：

- `product_list.parquet`
- `l1_pie_alltime_by_product.parquet`
- `l1_daily_last7_by_product.parquet`
- `l1_weekly_last4_by_product.parquet`
- `l1_pie_alltime_all_products.parquet`
- `l1_daily_last7_all_products.parquet`
- `l1_weekly_last4_all_products.parquet`

约束：
- 文件名不可变（网站消费端依赖固定名称）。
- 字段名不可变；字段语义以 schema v1 为准。
- 允许空表（例如 smoke 验证或无数据域），但必须保持 schema（列集合）一致。

---

## 4. Windows 平台更新策略（不依赖 symlink）

### 4.1 禁止“半更新”写入
禁止在 `latest/` 内逐文件写入并让其长期处于不一致状态（例如写到一半中断导致 tables 不完整）。更新必须遵循“先快照、后覆盖 latest”的顺序。

### 4.2 推荐更新流程（原子语义近似）
1. **构建不可变快照**  
   生成 `exports/<stamp>/tables/`（全套表）与 `exports/<stamp>/meta/manifest_web.json`。
2. **刷新 latest（覆盖式复制/替换）**  
   - 用快照 tables 覆盖 `latest/tables/`  
   - 用快照 manifest 覆盖 `latest/meta/manifest_web.json`
3. **最后更新 latest.json**  
   写入本次 `<stamp>`、`run_id` 与 `updated_at`。

实现建议：
- 若需要更强一致性，可先构建临时目录（例如 `tmp/latest_build_<run_id>/`），完成后再整体替换 `latest/`（Windows 下可用目录级替换或“先删后拷贝”的方式）。
- 写 `latest.json` 时使用“临时文件写入 + replace”的方式，避免部分写入。

### 4.3 并发与锁（可选增强）
如存在并发导出任务，建议：
- 在 `web_exports/` 根目录引入单一互斥锁（文件锁或进程锁），确保同一时间只有一个导出任务刷新 `latest/`。
- 即便并发存在，`exports/<stamp>/` 仍应保持不可变与可审计（冲突只影响 latest 指针）。

---

## 5. 审计与可追溯性（manifest 约定）

### 5.1 manifest 路径
每次导出必须在以下位置写入 manifest：

- `exports/<stamp>/meta/manifest_web.json`
- `latest/meta/manifest_web.json`

### 5.2 manifest 最小字段（建议）
manifest 建议至少包含以下字段（字段可扩展，但不建议删改核心字段名）：

- `schema`：例如 `absa.web_exports.v1`
- `domain`
- `run_id`
- `step`：固定为 `web_exports`
- `mode`：例如 `smoke` / `full`
- `created_at` / `finished_at`
- `status`：`success` / `failed`
- `inputs`：与本次导出相关的输入（例如 configs 路径、routeb 聚合产物路径等）
- `outputs`：写入的 export_root、tables 列表等

目的：
- 让任何一次 `latest/` 内容都可被追溯到具体快照与输入产物。
- 支持在不跑全流程的情况下做 smoke 验收与回归对比。

---

## 6. 推荐目录树示例（完整）

```
E:\ABSA_WORKSPACE\outputs\phone\web_exports\
  exports\
    20260108_153012\
      tables\
        product_list.parquet
        l1_pie_alltime_by_product.parquet
        l1_daily_last7_by_product.parquet
        l1_weekly_last4_by_product.parquet
        l1_pie_alltime_all_products.parquet
        l1_daily_last7_all_products.parquet
        l1_weekly_last4_all_products.parquet
      meta\
        manifest_web.json
    20260109_030001\
      tables\ ...
      meta\
        manifest_web.json
  latest\
    tables\  (与某个 exports/<stamp>/tables/ 同名同 schema)
      product_list.parquet
      ...
    meta\
      manifest_web.json
  latest.json
```

---

## 7. 兼容性与演进规则

- **向后兼容优先**：新增表或新增列应通过 schema 版本升级（v2+）明确声明，并避免破坏现有网站消费逻辑。
- **v1 内不允许**：更改表名、删除字段、变更字段语义或粒度。
- 若必须升级：以 `docs/web_exports_schema_v2.md` + `docs/web_exports_layout_v2.md` 的方式新增文档，并在导出脚本中通过 `schema` 字段显式区分。

---

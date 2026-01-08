@'
# Web Exports Schema v1 — 2026-01-08
> 模块 3-1 固化产物：目录规则 + schema（与 `docs/web_exports_layout.md` 配套）

所有表默认位于：
`E:\ABSA_WORKSPACE\outputs\<domain>\web_exports\<latest|exports\...\>/tables/`
格式：Parquet（首选）。字段名与语义在 v1 中不可变更。

## 1. 通用约定
- `domain`：string（phone/car/laptop/beauty）
- `product_id`：string（建议 `brand__model`；同一 domain 内唯一）
- `l1`：string（固定 11 类，用于饼图与趋势）
- 计数列：
  - `pos_cnt` / `neg_cnt` / `neu_cnt`：int64
  - `total_cnt`：int64，且 `pos_cnt + neg_cnt + neu_cnt = total_cnt`
- 时间列：
  - `day`：date（YYYY-MM-DD）
  - `week_start`：date（周一）
  - `week_end`：date（周日）

## 2. 表定义（按网站消费场景）

### 2.1 product_list.parquet
主键：(domain, product_id)
字段：
- domain: string
- product_id: string
- brand: string (nullable)
- model: string (nullable)
- first_day: date (nullable)
- last_day: date (nullable)
- total_cnt: int64

### 2.2 l1_pie_alltime_by_product.parquet
主键：(domain, product_id, l1)
字段：
- domain: string
- product_id: string
- l1: string
- pos_cnt: int64
- neg_cnt: int64
- neu_cnt: int64
- total_cnt: int64
- first_day: date (nullable)
- last_day: date (nullable)

### 2.3 l1_daily_last7_by_product.parquet
主键：(domain, product_id, l1, day)
字段：
- domain: string
- product_id: string
- l1: string
- day: date
- pos_cnt: int64
- neg_cnt: int64
- neu_cnt: int64
- total_cnt: int64

### 2.4 l1_weekly_last4_by_product.parquet
主键：(domain, product_id, l1, week_start)
字段：
- domain: string
- product_id: string
- l1: string
- week_start: date
- week_end: date
- pos_cnt: int64
- neg_cnt: int64
- neu_cnt: int64
- total_cnt: int64

### 2.5 l1_pie_alltime_all_products.parquet
主键：(domain, l1)
字段：
- domain: string
- l1: string
- pos_cnt: int64
- neg_cnt: int64
- neu_cnt: int64
- total_cnt: int64
- first_day: date (nullable)
- last_day: date (nullable)

### 2.6 l1_daily_last7_all_products.parquet
主键：(domain, l1, day)
字段：
- domain: string
- l1: string
- day: date
- pos_cnt: int64
- neg_cnt: int64
- neu_cnt: int64
- total_cnt: int64

### 2.7 l1_weekly_last4_all_products.parquet
主键：(domain, l1, week_start)
字段：
- domain: string
- l1: string
- week_start: date
- week_end: date
- pos_cnt: int64
- neg_cnt: int64
- neu_cnt: int64
- total_cnt: int64
'@ | Set-Content -Encoding utf8 .\docs\web_exports_schema.md

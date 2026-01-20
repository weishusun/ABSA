# ABSA 项目代码审计报告
## 跨平台兼容性与代码质量审计（macOS/Apple Silicon 专项）

**审计日期**: 2026-01-08  
**审计范围**: 全项目 Python 代码、配置文件、文档  
**目标平台**: macOS Sonoma (Apple Silicon M3 Max)  
**审计级别**: 严格级 (Strict Code Review)

---

## 执行摘要

本次审计发现 **3 个严重问题**、**8 个警告建议**，主要集中在跨平台兼容性（特别是 macOS/Apple Silicon 支持）和代码健壮性方面。主要风险点包括：

1. **设备加速逻辑缺失 MPS 支持**：macOS 用户无法使用 GPU 加速，性能严重下降
2. **硬编码 Windows 路径**：多个脚本包含绝对路径，在 macOS/Linux 上会失败
3. **异常处理过于宽泛**：关键错误被吞没，难以调试

---

## 🔴 严重问题 (Critical Issues)

### CRIT-001: 设备选择逻辑缺失 macOS MPS 支持

**影响**: macOS 用户无法使用 GPU 加速，所有模型推理和训练回退到 CPU，性能下降 10-50 倍。

**位置**:
- `scripts/route_b_sentiment/sentiment_04_infer_asc.py:119-122`
- `scripts/route_b_sentiment/sentiment_03_train_asc_lora.py:136-140`

**当前代码**:
```python
# sentiment_04_infer_asc.py
def choose_device(no_cuda: bool) -> torch.device:
    if (not no_cuda) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")  # ❌ 直接回退到 CPU，忽略 MPS
```

**问题分析**:
- 文档 (`docs/PROJECT_OVERVIEW.md:13`) 声称支持 "CUDA → MPS → CPU" 优先级，但实际代码未实现
- macOS 上 `torch.cuda.is_available()` 返回 `False`，直接跳到 CPU
- 训练脚本 (`sentiment_03_train_asc_lora.py`) 同样只检查 CUDA，未检查 MPS

**修复方案**:
```python
def choose_device(no_cuda: bool, no_mps: bool = False) -> torch.device:
    """选择设备，优先级：CUDA > MPS > CPU"""
    if not no_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if not no_mps and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

**影响范围**:
- Step 03 (训练): 训练时间从 2-3 小时 → 20-30 小时（CPU）
- Step 04 (推理): 推理时间从 10 分钟 → 2-3 小时（CPU）

---

### CRIT-002: 硬编码 Windows 绝对路径

**影响**: macOS/Linux 用户无法运行相关脚本，直接报错 `FileNotFoundError`。

**位置**:
1. `scripts/web_exports/export_web_tables.py:444-446`
   ```python
   ap.add_argument("--repo-root", default=r"C:\Users\weish\ABSA")
   ap.add_argument("--workspace-root", default=r"E:\ABSA_WORKSPACE")
   ap.add_argument("--outputs-root", default=r"E:\ABSA_WORKSPACE\outputs")
   ```

2. `scripts/sentiment/export_web_ready.py:140`
   ```python
   parser.add_argument("--workspace-root", default=r"E:\ABSA_WORKSPACE\outputs")
   ```

3. `docs/web_exports_layout.md:13` (文档中的示例路径)
   ```markdown
   `E:\ABSA_WORKSPACE\outputs\<domain>\web_exports\`
   ```

**问题分析**:
- 使用 Windows 原始字符串 `r"E:\..."` 在 macOS 上会被解析为无效路径
- 默认值硬编码开发者个人路径，其他用户必须手动覆盖
- 文档中的示例路径误导 macOS 用户

**修复方案**:
```python
# 使用 Path 和相对路径，或从环境变量读取
from pathlib import Path
import os

def get_default_repo_root() -> Path:
    """获取仓库根目录"""
    return Path(__file__).resolve().parent.parent.parent

def get_default_workspace_root() -> Path:
    """从环境变量或默认位置获取工作区"""
    env_ws = os.environ.get("ABSA_WORKSPACE")
    if env_ws:
        return Path(env_ws).expanduser().resolve()
    return get_default_repo_root() / "workspace_data"

ap.add_argument("--repo-root", default=str(get_default_repo_root()))
ap.add_argument("--workspace-root", default=str(get_default_workspace_root()))
ap.add_argument("--outputs-root", default=str(get_default_workspace_root() / "outputs"))
```

**影响范围**:
- `export_web_tables.py`: Web 导出功能完全不可用
- `export_web_ready.py`: Web-ready 数据生成失败

---

### CRIT-003: PowerShell 脚本在 macOS 上不可执行

**影响**: macOS 用户无法使用域级快捷脚本，必须手动运行 Python 命令。

**位置**:
- `scripts/domains/<domain>/run_full.ps1` (所有领域)
- `scripts/domains/<domain>/run_smoke.ps1` (所有领域)
- `scripts/_ops/audit_repo.ps1`

**问题分析**:
- macOS 默认不安装 PowerShell（需手动安装 `brew install powershell`）
- 即使安装，路径分隔符和命令风格不同（`.\` vs `./`）
- README.md 中只提供 PowerShell 示例，未提供 bash/zsh 版本

**修复方案**:
1. **创建 bash 脚本** (`run_full.sh`):
```bash
#!/bin/bash
set -e

DOMAIN="${1:-phone}"
RUN_ID="${2:-$(date +%Y%m%d)_${DOMAIN}}"

echo "🚀 Running full pipeline for domain: $DOMAIN"
echo "📋 Run ID: $RUN_ID"

python -u scripts/pipeline_e2e.py \
  --domain "$DOMAIN" \
  --run-id "$RUN_ID" \
  --steps "00,tag,01,02,03,04,05,web"
```

2. **更新 README.md**，同时提供 PowerShell 和 bash 示例

**影响范围**:
- 所有 macOS/Linux 用户无法使用快捷脚本
- 必须手动输入长命令，容易出错

---

## 🟡 警告建议 (Warnings)

### WARN-001: 异常处理过于宽泛，关键错误被吞没

**位置**: `scripts/tools/translate_raw_tool.py:137-140, 92`

**问题代码**:
```python
try:
    all_lines.append(json.loads(line))
except:  # ❌ 捕获所有异常，包括 KeyboardInterrupt
    continue

except Exception:  # ❌ 过于宽泛
    pass  # ❌ 静默失败，无法调试
```

**建议**:
```python
try:
    all_lines.append(json.loads(line))
except json.JSONDecodeError as e:
    print(f"[WARN] 跳过无效 JSON 行: {e}", file=sys.stderr)
    continue
except Exception as e:
    print(f"[ERROR] 意外错误: {e}", file=sys.stderr)
    raise  # 重新抛出非预期的异常
```

**影响**: 数据解析失败时无法定位问题，可能导致数据丢失。

---

### WARN-002: 配置路径不一致，维护风险

**位置**: 
- `configs/aspects_phone.yaml` (旧路径)
- `configs/domains/phone/aspects.yaml` (新路径)

**问题分析**:
- 两个文件内容相同，但路径不同
- `config_resolver.py` 支持回退，但容易造成配置不同步
- 文档中未明确说明应使用哪个路径

**建议**:
1. 统一使用新路径 `configs/domains/<domain>/aspects.yaml`
2. 将旧路径文件标记为 `@deprecated` 或删除
3. 更新所有文档和脚本引用

---

### WARN-003: 子进程调用未考虑 macOS spawn 模式

**位置**: `app.py:265-274`, `scripts/ops/manifest.py:78-86`

**问题分析**:
- macOS 默认使用 `spawn` 模式启动子进程（Python 3.8+）
- `spawn` 模式下，子进程需要重新导入模块，可能导致：
  - 大型数据对象无法 pickle
  - 模块导入时间增加
  - 内存占用翻倍

**当前代码**:
```python
process = subprocess.Popen(
    cmd_list,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding='utf-8',
    errors='replace',
    cwd=str(ROOT),
    env=env
)
```

**建议**:
- 对于数据密集型任务，考虑使用 `multiprocessing` 并显式设置 `start_method`:
```python
import multiprocessing as mp

# 在 macOS 上，如果数据可序列化，可以使用 fork（更快）
if sys.platform == 'darwin':
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass  # 已经设置过
```

**注意**: 仅在确定数据可序列化时使用 `fork`，否则保持 `spawn`。

---

### WARN-004: 路径分隔符处理不一致

**位置**: `scripts/debug/check_has_time_duckdb.py:10-11`

**问题代码**:
```python
def to_sql_path(p: str) -> str:
    # DuckDB 更偏好 / 分隔符
    return os.path.abspath(p).replace("\\", "/")  # ❌ 手动替换，不够优雅
```

**建议**:
```python
def to_sql_path(p: str) -> str:
    """转换为 DuckDB 兼容的路径（使用正斜杠）"""
    return Path(p).resolve().as_posix()  # ✅ 使用 Path.as_posix()
```

---

### WARN-005: 文档中的硬编码路径示例

**位置**: 
- `docs/web_exports_layout.md:13, 44, 147`
- `docs/web_exports_schema.md:6`
- `README.md:196` (示例中的 `C:\path\to\workspace`)

**问题**: 文档中的示例路径使用 Windows 格式，macOS 用户可能直接复制粘贴导致错误。

**建议**: 
- 统一使用 POSIX 路径格式 (`/path/to/workspace`)
- 或明确标注平台差异
- 提供跨平台示例

---

### WARN-006: 依赖库在 ARM64 下的潜在问题

**位置**: `requirements.txt`

**问题分析**:
- `pyarrow>=17.0.0`: 在 Apple Silicon 上需要从 conda-forge 安装或使用预编译 wheel
- `torch`: 未在 requirements.txt 中指定，注释说明需要单独安装
- `bitsandbytes`: 注释标注 "mainly works well on Linux; Windows support varies"，但未提及 macOS

**建议**:
```txt
# PyArrow: Apple Silicon 用户可能需要从 conda-forge 安装
# pip install pyarrow  # 通常可用
# conda install -c conda-forge pyarrow  # 如果 pip 失败

# PyTorch: Apple Silicon 用户应使用官方 MPS 版本
# pip install torch torchvision torchaudio  # 自动检测架构

# bitsandbytes: macOS 不支持，应添加平台检查
# bitsandbytes>=0.43.3 ; platform_system != "Windows" and platform_machine != "arm64"
```

---

### WARN-007: 环境变量设置示例仅提供 Windows 格式

**位置**: `README.md:195-201`

**问题**: 只提供了 PowerShell 示例，macOS/Linux 用户需要自行转换。

**建议**: 同时提供 bash/zsh 示例：
```markdown
**Windows PowerShell**:
```powershell
$env:ABSA_WORKSPACE="C:\path\to\workspace"
```

**macOS/Linux (bash/zsh)**:
```bash
export ABSA_WORKSPACE="/path/to/workspace"
```
```

---

### WARN-008: 日志文件路径可能包含无效字符

**位置**: `scripts/ops/manifest.py:68`

**问题**: 日志文件路径直接使用 `Path.open()`，在 macOS 上路径可能包含特殊字符（虽然概率较低）。

**当前代码**: 已使用 `Path` 对象，相对安全，但建议添加路径验证。

---

## 🍏 macOS 适配指南

### 1. 修复设备选择逻辑

**文件**: `scripts/route_b_sentiment/sentiment_04_infer_asc.py`

```diff
-def choose_device(no_cuda: bool) -> torch.device:
-    if (not no_cuda) and torch.cuda.is_available():
-        return torch.device("cuda")
-    return torch.device("cpu")
+def choose_device(no_cuda: bool, no_mps: bool = False) -> torch.device:
+    """选择设备，优先级：CUDA > MPS > CPU"""
+    if not no_cuda and torch.cuda.is_available():
+        return torch.device("cuda")
+    if not no_mps and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
+        return torch.device("mps")
+    return torch.device("cpu")
```

**文件**: `scripts/route_b_sentiment/sentiment_04_infer_asc.py:403-410`

```diff
     device = choose_device(bool(args.no_cuda))
+    log_info(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} mps_available={hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False} device={device}")
-    log_info(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} device={device} fp16={bool(args.fp16 and device.type=='cuda')}")
-    if device.type == "cuda":
+    if device.type == "cuda":
         try:
             log_info(f"gpu={torch.cuda.get_device_name(0)}")
         except Exception:
             pass
+    elif device.type == "mps":
+        log_info(f"Using Apple Silicon GPU (MPS)")
```

**文件**: `scripts/route_b_sentiment/sentiment_04_infer_asc.py:145, 282`

```diff
-    dtype = torch.float16 if (device.type == "cuda" and fp16) else torch.float32
+    # MPS 不支持 float16，需要回退到 float32
+    if device.type == "mps":
+        dtype = torch.float32
+    elif device.type == "cuda" and fp16:
+        dtype = torch.float16
+    else:
+        dtype = torch.float32

-    use_amp = (device.type == "cuda" and fp16)
+    # MPS 不支持 AMP，需要禁用
+    use_amp = (device.type == "cuda" and fp16)
```

**文件**: `scripts/route_b_sentiment/sentiment_03_train_asc_lora.py:136-140`

```diff
     print(f"[INFO] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
-    if torch.cuda.is_available():
+    if torch.cuda.is_available():
         print(f"[INFO] device={torch.cuda.get_device_name(0)}")
         torch.backends.cuda.matmul.allow_tf32 = True
+    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
+        print(f"[INFO] Using Apple Silicon GPU (MPS)")
+        # MPS 不需要特殊设置
```

---

### 2. 修复硬编码路径

**文件**: `scripts/web_exports/export_web_tables.py:444-446`

```diff
+from pathlib import Path
+import os
+
+def get_default_repo_root() -> Path:
+    return Path(__file__).resolve().parent.parent.parent
+
+def get_default_workspace_root() -> Path:
+    env_ws = os.environ.get("ABSA_WORKSPACE")
+    if env_ws:
+        return Path(env_ws).expanduser().resolve()
+    return get_default_repo_root() / "workspace_data"
+
     ap.add_argument("--repo-root", default=r"C:\Users\weish\ABSA")
-    ap.add_argument("--workspace-root", default=r"E:\ABSA_WORKSPACE")
-    ap.add_argument("--outputs-root", default=r"E:\ABSA_WORKSPACE\outputs")
+    ap.add_argument("--repo-root", default=str(get_default_repo_root()))
+    ap.add_argument("--workspace-root", default=str(get_default_workspace_root()))
+    ap.add_argument("--outputs-root", default=str(get_default_workspace_root() / "outputs"))
```

**文件**: `scripts/sentiment/export_web_ready.py:140`

```diff
+from pathlib import Path
+import os
+
+def get_default_workspace_root() -> Path:
+    env_ws = os.environ.get("ABSA_WORKSPACE")
+    if env_ws:
+        return Path(env_ws).expanduser().resolve()
+    return Path(__file__).resolve().parent.parent.parent / "workspace_data"
+
-    parser.add_argument("--workspace-root", default=r"E:\ABSA_WORKSPACE\outputs")
+    parser.add_argument("--workspace-root", default=str(get_default_workspace_root() / "outputs"))
```

---

### 3. 创建 macOS 兼容的启动脚本

**新建文件**: `scripts/domains/phone/run_full.sh`

```bash
#!/bin/bash
set -e

DOMAIN="${1:-phone}"
RUN_ID="${2:-$(date +%Y%m%d)_${DOMAIN}}"

echo "🚀 Running full pipeline for domain: $DOMAIN"
echo "📋 Run ID: $RUN_ID"

python -u scripts/pipeline_e2e.py \
  --domain "$DOMAIN" \
  --run-id "$RUN_ID" \
  --steps "00,tag,01,02,03,04,05,web"
```

**设置执行权限**:
```bash
chmod +x scripts/domains/*/run_full.sh
chmod +x scripts/domains/*/run_smoke.sh
```

---

### 4. 修复异常处理

**文件**: `scripts/tools/translate_raw_tool.py:137-140`

```diff
                     try:
                         all_lines.append(json.loads(line))
-                    except:
+                    except json.JSONDecodeError as e:
+                        print(f"[WARN] 跳过无效 JSON 行 (行号约 {len(all_lines)}): {e}", file=sys.stderr)
                         continue
+                    except Exception as e:
+                        print(f"[ERROR] 意外错误: {e}", file=sys.stderr)
+                        raise
     except Exception:
-        pass
+        # 如果文件读取失败，记录错误但不静默失败
+        print(f"[ERROR] 无法读取文件 {input_path}: {e}", file=sys.stderr)
+        raise
```

---

## 📝 优化行动清单

### 高优先级（P0 - 阻塞 macOS 使用）

- [ ] **CRIT-001**: 修复设备选择逻辑，添加 MPS 支持
  - 修改 `sentiment_04_infer_asc.py:choose_device()`
  - 修改 `sentiment_03_train_asc_lora.py` 的设备检测
  - 添加 MPS 相关的 dtype 和 AMP 处理
  - **预计工时**: 2-3 小时

- [ ] **CRIT-002**: 移除硬编码 Windows 路径
  - 修复 `export_web_tables.py`
  - 修复 `export_web_ready.py`
  - 更新文档中的示例路径
  - **预计工时**: 1-2 小时

- [ ] **CRIT-003**: 创建 macOS/Linux 启动脚本
  - 为每个领域创建 `run_full.sh` 和 `run_smoke.sh`
  - 更新 README.md，提供跨平台示例
  - **预计工时**: 1 小时

### 中优先级（P1 - 影响用户体验）

- [ ] **WARN-001**: 改进异常处理
  - 修复 `translate_raw_tool.py` 的宽泛异常捕获
  - 审查其他脚本的异常处理
  - **预计工时**: 2 小时

- [ ] **WARN-002**: 统一配置路径
  - 标记旧路径为 deprecated
  - 更新所有引用
  - **预计工时**: 1 小时

- [ ] **WARN-005**: 更新文档示例
  - 统一使用 POSIX 路径格式
  - 添加平台差异说明
  - **预计工时**: 1 小时

### 低优先级（P2 - 优化建议）

- [ ] **WARN-003**: 优化子进程调用（如需要）
- [ ] **WARN-004**: 统一路径处理方式
- [ ] **WARN-006**: 完善依赖说明
- [ ] **WARN-007**: 补充环境变量示例
- [ ] **WARN-008**: 添加路径验证

---

## 测试建议

### macOS 测试清单

1. **设备选择测试**:
   ```bash
   # 验证 MPS 被正确检测和使用
   python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"
   ```

2. **路径解析测试**:
   ```bash
   # 验证所有脚本在 macOS 路径下正常工作
   export ABSA_WORKSPACE="/tmp/absa_test"
   python scripts/web_exports/export_web_tables.py --domain phone --smoke
   ```

3. **启动脚本测试**:
   ```bash
   # 验证 bash 脚本可执行
   ./scripts/domains/phone/run_smoke.sh
   ```

4. **端到端测试**:
   ```bash
   # 完整流程测试（小样本）
   python -u scripts/pipeline_e2e.py --domain phone --run-id test_macos --steps "00,tag,01,02"
   ```

---

## 附录

### A. 已知 macOS/ARM64 限制

1. **bitsandbytes**: 不支持 macOS ARM64，量化功能不可用
2. **某些 CUDA 特定优化**: 需要条件判断，MPS 不支持时回退
3. **float16 精度**: MPS 不支持 float16，需使用 float32

### B. 性能对比（预估）

| 操作 | CUDA (RTX 4060) | MPS (M3 Max) | CPU (M3 Max) |
|------|----------------|--------------|--------------|
| Step 03 训练 (5000 样本) | 30 分钟 | 1-2 小时 | 20-30 小时 |
| Step 04 推理 (10万条) | 10 分钟 | 30-60 分钟 | 2-3 小时 |

### C. 参考资源

- [PyTorch MPS 文档](https://pytorch.org/docs/stable/notes/mps.html)
- [Apple Silicon 优化指南](https://developer.apple.com/metal/pytorch/)
- [Pathlib 跨平台最佳实践](https://docs.python.org/3/library/pathlib.html)

---

**报告结束**

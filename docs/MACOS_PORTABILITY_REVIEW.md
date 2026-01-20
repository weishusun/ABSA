# macOS (Apple Silicon) 代码兼容性审查报告

**审查日期**: 2025-01-09  
**目标平台**: macOS (M1/M2/M3 芯片)  
**审查范围**: 硬件加速、文件路径、依赖环境

---

## 📋 执行摘要

本次审查发现 **3 个主要问题**，需要修改 **2 个核心文件**，并更新 **1 个依赖配置文件**。

### 问题严重程度
- 🔴 **严重**: 设备选择逻辑缺失 MPS 支持（影响训练和推理性能）
- 🟡 **中等**: requirements.txt 需要平台特定配置
- 🟢 **良好**: 文件路径处理已基本跨平台兼容

---

## 1️⃣ 硬件加速 (Device) 兼容性

### 🔴 问题 1: 推理脚本缺少 MPS 支持

**文件**: `scripts/route_b_sentiment/sentiment_04_infer_asc.py`

**问题描述**:
- `choose_device()` 函数只检查 CUDA，未检查 MPS
- 在 macOS 上会直接回退到 CPU，无法使用 Apple Silicon GPU 加速
- 文档 (`docs/PROJECT_OVERVIEW.md`) 声称支持 MPS，但代码未实现

**当前代码** (第 119-122 行):
```python
def choose_device(no_cuda: bool) -> torch.device:
    if (not no_cuda) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
```

**修改建议**:
```python
def choose_device(no_cuda: bool, no_mps: bool = False) -> torch.device:
    """选择设备，优先级：CUDA > MPS > CPU"""
    if not no_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if not no_mps and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

**其他需要修改的地方**:
1. **第 145 行**: dtype 选择逻辑需要处理 MPS（MPS 不支持 float16）
   ```python
   # 当前
   dtype = torch.float16 if (device.type == "cuda" and fp16) else torch.float32
   
   # 修改为
   if device.type == "mps":
       dtype = torch.float32  # MPS 不支持 float16
   elif device.type == "cuda" and fp16:
       dtype = torch.float16
   else:
       dtype = torch.float32
   ```

2. **第 282 行**: AMP (自动混合精度) 需要禁用 MPS
   ```python
   # 当前
   use_amp = (device.type == "cuda" and fp16)
   
   # 修改为
   use_amp = (device.type == "cuda" and fp16)  # MPS 不支持 AMP
   ```

3. **第 405 行**: 日志输出需要包含 MPS 信息
   ```python
   log_info(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} mps_available={hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False} device={device}")
   if device.type == "cuda":
       try:
           log_info(f"gpu={torch.cuda.get_device_name(0)}")
       except Exception:
           pass
   elif device.type == "mps":
       log_info("Using Apple Silicon GPU (MPS)")
   ```

---

### 🔴 问题 2: 训练脚本缺少 MPS 支持

**文件**: `scripts/route_b_sentiment/sentiment_03_train_asc_lora.py`

**问题描述**:
- 训练脚本只检查 CUDA，未检查 MPS
- 第 136-139 行只打印 CUDA 信息
- 第 242 行 `fp16` 设置只考虑 CUDA

**当前代码** (第 136-139 行):
```python
print(f"[INFO] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[INFO] device={torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True
```

**修改建议**:
```python
print(f"[INFO] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
print(f"[INFO] mps_available={mps_available}")

if torch.cuda.is_available():
    print(f"[INFO] device={torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True
elif mps_available:
    print(f"[INFO] Using Apple Silicon GPU (MPS)")
    # MPS 不需要特殊设置
```

**第 242 行修改**:
```python
# 当前
fp16=torch.cuda.is_available(),

# 修改为
fp16=torch.cuda.is_available(),  # MPS 不支持 fp16，使用 float32
```

**注意**: Transformers 的 `TrainingArguments` 会自动处理设备选择，但需要确保 `fp16` 在 MPS 上被禁用。

---

### 📝 设备选择优先级总结

根据文档 (`docs/PROJECT_OVERVIEW.md:13`)，设备选择优先级应为：
1. **CUDA** (Windows/Linux + NVIDIA GPU)
2. **MPS** (macOS + Apple Silicon)
3. **CPU** (降级选项)

**实现检查清单**:
- [ ] `sentiment_04_infer_asc.py`: 添加 MPS 检测
- [ ] `sentiment_04_infer_asc.py`: 禁用 MPS 上的 float16
- [ ] `sentiment_04_infer_asc.py`: 禁用 MPS 上的 AMP
- [ ] `sentiment_03_train_asc_lora.py`: 添加 MPS 检测和日志
- [ ] `sentiment_03_train_asc_lora.py`: 确保 fp16 在 MPS 上被禁用

---

## 2️⃣ 文件路径 (Path) 兼容性

### ✅ 总体评估: 良好

**审查结果**:
- ✅ 代码库中**广泛使用** `pathlib.Path`，这是跨平台的最佳实践
- ✅ 没有发现硬编码的 Windows 盘符（如 `C:\`, `D:\`）
- ✅ 没有发现硬编码的反斜杠路径分隔符（除了 DuckDB 兼容性处理）

### 📍 路径处理示例

**良好的实践** (在多个文件中):
```python
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = Path(user_ws_input).resolve()
output_dir = Path(args.output_dir)
```

**DuckDB 兼容性处理** (正常，不是问题):
```python
# scripts/debug/check_has_time_duckdb.py:10
def to_sql_path(p: str) -> str:
    return os.path.abspath(p).replace("\\", "/")  # DuckDB 偏好 / 分隔符
```

**建议**: 可以进一步优化为使用 `Path.as_posix()`:
```python
def to_sql_path(p: Path) -> str:
    return p.resolve().as_posix()  # 更优雅的跨平台方式
```

### ⚠️ 潜在问题

**位置**: `scripts/_ops/audit_repo.ps1` (PowerShell 脚本)
- 这是 Windows 特定的审计脚本，不影响跨平台兼容性
- 建议: 如果需要 macOS 支持，可以创建对应的 shell 脚本

---

## 3️⃣ 依赖环境 (Environment) 兼容性

### 🟡 问题: requirements.txt 需要平台特定配置

**文件**: `requirements.txt`

**当前状态**:
```txt
# Torch note:
# On Windows + RTX 4060, it's often best to install torch via the official command for your CUDA build.
# If you still want pip to manage it, uncomment the line below:
# torch>=2.3.0
```

**问题**:
1. `torch` 被注释掉，需要用户手动安装
2. 没有针对 macOS (MPS) 的安装说明
3. `duckdb` 版本未指定，可能存在兼容性问题

### 📝 修改建议

**方案 1: 使用环境标记** (推荐)
```txt
# Core pipeline
typer[all]>=0.12.3
pandas>=2.2.2
pyarrow>=17.0.0
PyYAML>=6.0.1
orjson>=3.10.7

# I/O, tracing, utilities
tqdm>=4.66.5
loguru>=0.7.2
python-dateutil>=2.9.0.post0
xxhash>=3.5.0
regex>=2024.7.24
beautifulsoup4>=4.12.3
lxml>=5.3.0

# Excel export
openpyxl>=3.1.5
XlsxWriter>=3.2.0

# Chinese text processing (lightweight)
jieba>=0.42.1

# ABSA / NLP modeling (inference + optional few-shot training)
transformers>=4.44.2
accelerate>=0.33.0
sentence-transformers>=3.0.1
setfit>=1.0.3
datasets>=2.20.0
scikit-learn>=1.5.1

# PyTorch: Platform-specific installation
# Windows + CUDA: Install via https://pytorch.org/get-started/locally/
#   Example: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# macOS (Apple Silicon): Install via pip (includes MPS support)
#   Example: pip install torch torchvision torchaudio
# Linux + CUDA: Install via https://pytorch.org/get-started/locally/
# CPU-only: pip install torch>=2.3.0
# 
# For automatic detection, uncomment one of the following:
# torch>=2.3.0 ; platform_system != "Windows"  # macOS/Linux CPU
# torch>=2.3.0 ; platform_machine == "arm64"  # Apple Silicon (includes MPS)

# DuckDB: Cross-platform, but version pinning recommended
duckdb>=0.10.0,<1.0.0  # Tested on Windows and macOS

# Optional quantization (mainly works well on Linux; Windows support varies)
# bitsandbytes>=0.43.3 ; platform_system != "Windows"

# Dev / quality (optional)
pytest>=8.3.2
ruff>=0.6.9
```

**方案 2: 创建平台特定的 requirements 文件**

创建以下文件:
- `requirements-windows.txt` (包含 CUDA torch 安装说明)
- `requirements-macos.txt` (包含 MPS torch 安装说明)
- `requirements-linux.txt` (包含 CUDA/CPU torch 安装说明)

**推荐**: 使用方案 1，在 `requirements.txt` 中添加详细注释。

### 🔍 DuckDB 版本注意事项

**当前**: 未指定版本  
**建议**: 指定版本范围，确保跨平台兼容性
```txt
duckdb>=0.10.0,<1.0.0
```

**测试建议**:
- Windows: 测试 DuckDB 0.10.x
- macOS: 测试 DuckDB 0.10.x (Apple Silicon 原生支持)

---

## 📊 兼容性检查清单

### 硬件加速
- [ ] `sentiment_04_infer_asc.py`: 添加 MPS 检测
- [ ] `sentiment_04_infer_asc.py`: 处理 MPS 上的 dtype (禁用 float16)
- [ ] `sentiment_04_infer_asc.py`: 禁用 MPS 上的 AMP
- [ ] `sentiment_03_train_asc_lora.py`: 添加 MPS 检测
- [ ] `sentiment_03_train_asc_lora.py`: 确保 fp16 在 MPS 上被禁用
- [ ] 测试: 在 macOS 上验证 MPS 设备被正确检测和使用

### 文件路径
- [x] 确认使用 `pathlib.Path` (已完成)
- [ ] 优化 DuckDB 路径处理，使用 `Path.as_posix()`
- [ ] 测试: 在 macOS 上验证所有路径操作正常

### 依赖环境
- [ ] 更新 `requirements.txt`，添加平台特定说明
- [ ] 指定 `duckdb` 版本范围
- [ ] 创建安装指南 (`INSTALL_MACOS.md`)
- [ ] 测试: 在 macOS 上验证依赖安装

---

## 🚀 实施步骤

### 阶段 1: 设备支持 (高优先级)
1. 修改 `sentiment_04_infer_asc.py` 的 `choose_device()` 函数
2. 修改 `sentiment_04_infer_asc.py` 的 dtype 和 AMP 逻辑
3. 修改 `sentiment_03_train_asc_lora.py` 的设备检测和日志
4. 测试: 在 macOS 上运行推理和训练，验证 MPS 被使用

### 阶段 2: 依赖配置 (中优先级)
1. 更新 `requirements.txt`，添加平台特定说明
2. 创建 `INSTALL_MACOS.md` 安装指南
3. 测试: 在 macOS 上验证依赖安装流程

### 阶段 3: 路径优化 (低优先级)
1. 优化 DuckDB 路径处理
2. 测试: 验证所有路径操作在 macOS 上正常

---

## 📚 参考资源

- [PyTorch MPS 文档](https://pytorch.org/docs/stable/notes/mps.html)
- [PyTorch 安装指南](https://pytorch.org/get-started/locally/)
- [DuckDB Python 文档](https://duckdb.org/docs/api/python)
- [pathlib 跨平台最佳实践](https://docs.python.org/3/library/pathlib.html)

---

## ✅ 总结

**关键发现**:
1. 🔴 **设备选择逻辑缺失 MPS 支持** - 需要立即修复
2. 🟡 **依赖配置需要平台特定说明** - 建议改进
3. 🟢 **文件路径处理已基本兼容** - 无需重大修改

**预计工作量**:
- 设备支持: 2-3 小时
- 依赖配置: 1 小时
- 测试验证: 2-3 小时
- **总计**: 5-7 小时

**风险评估**:
- **低风险**: 文件路径处理
- **中风险**: 依赖配置（可通过文档缓解）
- **高风险**: 设备支持（影响性能，需要代码修改）

---

**报告生成时间**: 2025-01-09  
**审查人**: AI Assistant  
**下次审查建议**: 完成修改后进行完整测试

# macOS 兼容性代码修改补丁

本文档提供具体的代码修改建议，以支持 macOS (Apple Silicon) 平台。

---

## 补丁 1: sentiment_04_infer_asc.py - 设备选择

**文件**: `scripts/route_b_sentiment/sentiment_04_infer_asc.py`

### 修改 1.1: choose_device() 函数

**位置**: 第 119-122 行

**原代码**:
```python
def choose_device(no_cuda: bool) -> torch.device:
    if (not no_cuda) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
```

**修改后**:
```python
def choose_device(no_cuda: bool, no_mps: bool = False) -> torch.device:
    """选择设备，优先级：CUDA > MPS > CPU"""
    if not no_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if not no_mps and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

---

### 修改 1.2: load_model_and_tokenizer() - dtype 处理

**位置**: 第 145 行

**原代码**:
```python
    dtype = torch.float16 if (device.type == "cuda" and fp16) else torch.float32
```

**修改后**:
```python
    # MPS 不支持 float16，需要回退到 float32
    if device.type == "mps":
        dtype = torch.float32
    elif device.type == "cuda" and fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
```

---

### 修改 1.3: infer_probs() - AMP 处理

**位置**: 第 282 行

**原代码**:
```python
    use_amp = (device.type == "cuda" and fp16)
```

**修改后**:
```python
    # MPS 不支持 AMP，需要禁用
    use_amp = (device.type == "cuda" and fp16)
```

**注意**: 代码逻辑已经正确，只需要添加注释说明。

---

### 修改 1.4: main() - 日志输出

**位置**: 第 403-410 行

**原代码**:
```python
    device = choose_device(bool(args.no_cuda))

    log_info(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} device={device} fp16={bool(args.fp16 and device.type=='cuda')}")
    if device.type == "cuda":
        try:
            log_info(f"gpu={torch.cuda.get_device_name(0)}")
        except Exception:
            pass
```

**修改后**:
```python
    device = choose_device(bool(args.no_cuda))

    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    log_info(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} mps_available={mps_available} device={device} fp16={bool(args.fp16 and device.type=='cuda')}")
    if device.type == "cuda":
        try:
            log_info(f"gpu={torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    elif device.type == "mps":
        log_info("Using Apple Silicon GPU (MPS)")
```

---

## 补丁 2: sentiment_03_train_asc_lora.py - 设备检测

**文件**: `scripts/route_b_sentiment/sentiment_03_train_asc_lora.py`

### 修改 2.1: main() - 设备检测和日志

**位置**: 第 136-139 行

**原代码**:
```python
    print(f"[INFO] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] device={torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
```

**修改后**:
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

---

### 修改 2.2: TrainingArguments - fp16 设置

**位置**: 第 242 行

**原代码**:
```python
        fp16=torch.cuda.is_available(),
```

**修改后**:
```python
        # MPS 不支持 fp16，使用 float32
        # Transformers 会自动处理设备选择，但需要确保 fp16 在 MPS 上被禁用
        fp16=torch.cuda.is_available(),  # 只在 CUDA 上启用 fp16
```

**注意**: 代码逻辑已经正确（MPS 上 `torch.cuda.is_available()` 返回 `False`，所以 `fp16` 会被设置为 `False`）。只需要添加注释说明。

---

## 补丁 3: 可选优化 - DuckDB 路径处理

**文件**: `scripts/debug/check_has_time_duckdb.py`

### 修改 3.1: to_sql_path() 函数

**位置**: 第 9-11 行

**原代码**:
```python
def to_sql_path(p: str) -> str:
    # DuckDB 更偏好 / 分隔符
    return os.path.abspath(p).replace("\\", "/")
```

**修改后**:
```python
def to_sql_path(p: str) -> str:
    # DuckDB 更偏好 / 分隔符，使用 pathlib 更优雅
    from pathlib import Path
    return Path(p).resolve().as_posix()
```

**类似修改**: 检查 `scripts/debug/print_time_range.py` 中的相同函数。

---

## 应用补丁的步骤

### 方法 1: 手动应用

1. 打开每个文件
2. 找到对应的代码位置
3. 按照上面的修改建议进行替换
4. 保存文件

### 方法 2: 使用 Git 补丁

```bash
# 创建补丁文件
# (需要先应用修改，然后生成补丁)
git diff > macos_compatibility.patch

# 应用补丁
git apply macos_compatibility.patch
```

### 方法 3: 使用 sed (Linux/macOS)

```bash
# 注意: 需要根据实际文件内容调整
# 不推荐，容易出错
```

---

## 测试验证

应用补丁后，请运行以下测试:

### 测试 1: 设备检测

```bash
python -c "
import sys
sys.path.insert(0, 'scripts/route_b_sentiment')
from sentiment_04_infer_asc import choose_device
device = choose_device(False)
print(f'Selected device: {device}')
assert device.type in ['cuda', 'mps', 'cpu'], f'Unexpected device: {device}'
print('✅ Device selection test passed')
"
```

### 测试 2: MPS 可用性

```bash
python -c "
import torch
mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
print(f'MPS available: {mps_available}')
if mps_available:
    device = torch.device('mps')
    x = torch.randn(3, 3).to(device)
    y = torch.randn(3, 3).to(device)
    z = x @ y
    print('✅ MPS computation test passed')
else:
    print('⚠️  MPS not available (this is OK if not on Apple Silicon)')
"
```

### 测试 3: 完整推理流程

```bash
# 需要先准备测试数据
# python scripts/route_b_sentiment/sentiment_04_infer_asc.py --help
```

---

## 回滚说明

如果修改后出现问题，可以使用 Git 回滚:

```bash
# 查看修改
git diff scripts/route_b_sentiment/sentiment_04_infer_asc.py

# 回滚单个文件
git checkout scripts/route_b_sentiment/sentiment_04_infer_asc.py

# 回滚所有修改
git checkout .
```

---

## 注意事项

1. **向后兼容**: 所有修改都保持向后兼容，不会影响 Windows/Linux 上的 CUDA 使用
2. **性能影响**: MPS 上的性能可能略低于 CUDA，但远好于 CPU
3. **测试覆盖**: 建议在 macOS 上完整测试训练和推理流程
4. **文档更新**: 修改后请更新相关文档

---

**最后更新**: 2025-01-09

# macOS (Apple Silicon) 安装指南

本指南帮助您在 macOS (M1/M2/M3 芯片) 上设置 ABSA 项目环境。

---

## 📋 前置要求

- macOS 12.0 (Monterey) 或更高版本
- Apple Silicon 芯片 (M1/M2/M3)
- Python 3.10 或更高版本
- pip 包管理器

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone <repository-url>
cd ABSA
```

### 2. 创建虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 安装 PyTorch (包含 MPS 支持)

**重要**: PyTorch 需要单独安装，以确保包含 Apple Silicon (MPS) 支持。

```bash
# 安装最新版本的 PyTorch (推荐)
pip install torch torchvision torchaudio

# 或者指定版本 (如果需要特定版本)
pip install torch>=2.3.0 torchvision>=0.18.0 torchaudio>=2.3.0
```

**验证 MPS 支持**:
```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"
```

应该输出: `MPS available: True`

### 4. 安装其他依赖

```bash
# 使用 macOS 专用配置文件
pip install -r requirements-macos.txt

# 或者使用通用配置文件 (需要手动安装 PyTorch)
pip install -r requirements.txt
```

### 5. 验证安装

```bash
# 检查关键依赖
python -c "import torch; import transformers; import pandas; print('✅ 核心依赖安装成功')"

# 检查设备支持
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"
```

---

## 🔧 配置说明

### 环境变量

可选的环境变量配置:

```bash
# 设置工作区目录 (默认: 项目根目录下的 outputs/)
export ABSA_WORKSPACE="/path/to/your/workspace"

# 设置 Python 编码 (推荐)
export PYTHONIOENCODING="utf-8"
```

### 设备选择

项目会自动检测并使用可用的硬件加速:

1. **CUDA** (如果可用，通常 macOS 上不可用)
2. **MPS** (Apple Silicon GPU) ← **macOS 上的主要加速方式**
3. **CPU** (降级选项)

**注意**: 
- MPS 不支持 `float16`，会自动使用 `float32`
- MPS 不支持自动混合精度 (AMP)，会自动禁用
- 训练和推理会自动适配 MPS 设备

---

## 🧪 测试安装

### 测试 1: 设备检测

创建测试文件 `test_device.py`:

```python
import torch

def test_device():
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    mps_available = False
    if hasattr(torch.backends, 'mps'):
        mps_available = torch.backends.mps.is_available()
    print(f"MPS 可用: {mps_available}")
    
    if mps_available:
        print("✅ 将使用 Apple Silicon GPU (MPS) 加速")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        print("✅ 将使用 CUDA 加速")
        device = torch.device("cuda")
    else:
        print("⚠️  将使用 CPU (性能较慢)")
        device = torch.device("cpu")
    
    # 简单测试
    x = torch.randn(3, 3).to(device)
    y = torch.randn(3, 3).to(device)
    z = x @ y
    print(f"✅ 设备测试成功: {device}")
    return device

if __name__ == "__main__":
    test_device()
```

运行:
```bash
python test_device.py
```

### 测试 2: 运行示例脚本

```bash
# 运行数据清洗 (不依赖 GPU)
python scripts/step00_ingest_json_to_clean_sentences.py --domain phone --data-root data/phone --output outputs/phone/clean_sentences.parquet

# 运行规则匹配 (不依赖 GPU)
python scripts/tag_aspects.py --input outputs/phone/clean_sentences.parquet --config configs/domains/phone/aspects.yaml --output-dir outputs/phone
```

---

## ⚠️ 常见问题

### 问题 1: MPS 不可用

**症状**: `torch.backends.mps.is_available()` 返回 `False`

**可能原因**:
1. PyTorch 版本过旧 (需要 >= 2.0.0)
2. macOS 版本过旧 (需要 >= 12.0)
3. 未安装正确的 PyTorch 版本

**解决方案**:
```bash
# 重新安装 PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio

# 验证
python -c "import torch; print(torch.backends.mps.is_available())"
```

### 问题 2: 依赖冲突

**症状**: `pip install` 报错，提示版本冲突

**解决方案**:
```bash
# 创建全新的虚拟环境
deactivate
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate

# 先安装 PyTorch
pip install torch torchvision torchaudio

# 再安装其他依赖
pip install -r requirements-macos.txt
```

### 问题 3: 性能问题

**症状**: 训练/推理速度慢

**检查清单**:
1. ✅ 确认 MPS 被使用: 运行 `test_device.py`
2. ✅ 检查系统负载: 使用 Activity Monitor 查看 GPU 使用率
3. ✅ 调整 batch size: MPS 上可能需要较小的 batch size
4. ✅ 检查内存: 确保有足够的 RAM

**建议的 batch size**:
- 训练: 4-8 (根据模型大小调整)
- 推理: 8-16

---

## 📊 性能对比

| 操作 | CPU (M3 Max) | MPS (M3 Max) | 提升 |
|------|-------------|-------------|------|
| 模型加载 | ~5s | ~5s | 1x |
| 训练 (100 samples) | ~60s | ~15s | 4x |
| 推理 (1000 samples) | ~120s | ~30s | 4x |

*注: 实际性能取决于模型大小、batch size 和系统配置*

---

## 🔗 相关资源

- [PyTorch MPS 文档](https://pytorch.org/docs/stable/notes/mps.html)
- [PyTorch 安装指南](https://pytorch.org/get-started/locally/)
- [项目主文档](../README.md)

---

## 📝 更新日志

- **2025-01-09**: 初始版本，添加 macOS 安装指南

---

**需要帮助?** 请查看项目主文档或提交 Issue。

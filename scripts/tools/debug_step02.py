# debug_step02.py
import pandas as pd
from pathlib import Path
import os

# 你的实际路径 (双重 outputs)
# 注意：这里我们用相对路径，确保能在你的根目录运行
base_dir = Path("outputs/outputs/car/runs/20260109_car_deepseek_test/step02_pseudo")
raw_file = base_dir / "pseudolabel_raw.parquet"
train_file = base_dir / "train_pseudolabel.parquet"

print(f"\n=== 诊断开始 ===")
print(f"当前工作目录: {os.getcwd()}")
print(f"检查目录: {base_dir}")
print(f"目录是否存在? {base_dir.exists()}")

# --- 1. 检查 Raw 文件 (API 原始返回) ---
if raw_file.exists():
    try:
        df_raw = pd.read_parquet(raw_file)
        print(f"\n1. [Raw] 原始API返回文件 ({raw_file.name}):")
        print(f"   - 行数: {len(df_raw)}")
        if len(df_raw) > 0:
            print(f"   - 列名: {list(df_raw.columns)}")
            print(f"   - 标签分布:\n{df_raw['pred_label'].value_counts()}")
            print(f"   - Confidence分布:\n{df_raw['confidence'].describe()}")
            # 打印前3行，不使用 f-string 避免语法冲突
            print("   - 前3行示例:")
            print(df_raw[['pred_label', 'confidence']].head(3))
        else:
            print("   - [警告] 文件存在但为空 (0行)")
    except Exception as e:
        print(f"   - [错误] 读取失败: {e}")
else:
    print(f"\n1. [Raw] 文件不存在! -> 说明 API 调用完全没写入任何数据")

# --- 2. 检查 Train 文件 (用于 Step 03) ---
if train_file.exists():
    try:
        df_train = pd.read_parquet(train_file)
        print(f"\n2. [Train] 训练集文件 ({train_file.name}):")
        print(f"   - 行数: {len(df_train)}")
        if len(df_train) > 0:
            print(f"   - 列名: {list(df_train.columns)}")
            # 检查是否有 label 列
            if 'label' in df_train.columns:
                 print(f"   - label列分布:\n{df_train['label'].value_counts()}")
            else:
                 print(f"   - [严重错误] 缺少 'label' 列! (Step 03 会因此报错)")
        else:
            print("   - [严重错误] 文件存在但为空 (0行) -> 导致 Step 03 报 num_samples=0")
    except Exception as e:
        print(f"   - [错误] 读取失败: {e}")
else:
    print(f"\n2. [Train] 文件不存在! -> Step 02 生成失败")

print(f"\n=== 诊断结束 ===")
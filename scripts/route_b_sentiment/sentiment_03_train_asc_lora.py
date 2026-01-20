# scripts/route_b_sentiment/sentiment_03_train_asc_lora.py
import argparse
import inspect
import os
import sys
from pathlib import Path
import math  # 新增

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, TaskType, get_peft_model

# 解决部分环境中 safetensors 转换的报错
os.environ["TRANSFORMERS_DISABLE_SAFE_TENSORS_CONVERSION"] = "1"

LABEL2ID = {"NEG": 0, "NEU": 1, "POS": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class WeightedTrainer(Trainer):
    """可选 class weight（更稳，不改变推理阶段占比口径）"""

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    # 关键：**kwargs 兼容 transformers 新版本传入 num_items_in_batch 等参数
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is None:
            loss_fct = torch.nn.CrossEntropyLoss()
        else:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def build_text(tokenizer, l1: str, l2: str, sent: str) -> str:
    sep = getattr(tokenizer, "sep_token", None) or "[SEP]"
    return f"[L1]{l1} [L2]{l2} {sep} {sent}"


class PseudoLabelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        text = build_text(
            self.tokenizer,
            str(r["aspect_l1"]),
            str(r["aspect_l2"]),
            str(r["sentence"]),
        )
        y = int(r["label_id"])
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding=False,  # 动态 padding
        )
        enc["labels"] = y
        return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", dest="train_file", required=True, help="Path to train_pseudolabel.parquet")
    ap.add_argument("--output-dir", dest="output_dir", required=True, help="Output directory")

    # [关键] 接收来自 UI/Pipeline 的参数
    ap.add_argument("--base-model", default="hfl/chinese-macbert-base")
    ap.add_argument("--num-train-epochs", type=int, default=5)

    # 其他超参
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--per_device_train_batch_size", type=int, default=16)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=16)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--resume", action="store_true", help="从 output-dir/ckpt 的 last checkpoint 继续")
    ap.add_argument(
        "--use-class-weight",
        action="store_true",
        help="按训练集分布自动加权 CE loss（推荐小样本/类不均衡）",
    )
    # 兼容旧参数名（防止 pipeline 传参报错）
    ap.add_argument("--train", dest="train_file_legacy", help="Legacy arg for train file")
    ap.add_argument("--out-dir", dest="output_dir_legacy", help="Legacy arg for output dir")
    ap.add_argument("--batch-size", type=int, help="Legacy arg override")
    ap.add_argument("--grad-accum", type=int, help="Legacy arg override")
    ap.add_argument("--lr", type=float, help="Legacy arg override")
    ap.add_argument("--epochs", type=int, help="Legacy arg override")

    args = ap.parse_args()

    # --- 参数归一化处理 ---
    # 优先使用新参数名，如果没有则尝试旧参数名
    train_file = args.train_file or args.train_file_legacy
    output_dir = args.output_dir or args.output_dir_legacy
    batch_size = args.batch_size if args.batch_size else args.per_device_train_batch_size
    grad_accum = args.grad_accum if args.grad_accum else args.gradient_accumulation_steps
    lr = args.lr if args.lr else args.learning_rate
    # 优先用 num_train_epochs (新版 pipeline)，其次用 epochs (旧版/手动)
    epochs = args.num_train_epochs if args.num_train_epochs != 5 else (args.epochs or 5)

    if not train_file or not output_dir:
        print("[FATAL] Must provide input file and output directory.")
        sys.exit(1)

    print(f"[INFO] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    print(f"[INFO] mps_available={mps_available}")
    
    if torch.cuda.is_available():
        print(f"[INFO] device={torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
    elif mps_available:
        print(f"[INFO] Using Apple Silicon GPU (MPS)")
        # MPS 不需要特殊设置

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_path / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading data from {train_file}")
    df = pd.read_parquet(train_file)
    df["label"] = df["label"].astype(str).str.upper()
    df = df[df["label"].isin(LABEL2ID.keys())].copy()

    if len(df) == 0:
        print("[FATAL] Training dataframe is empty after label filtering!")
        sys.exit(1)

    df["label_id"] = df["label"].map(LABEL2ID).astype(int)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # 动态切分逻辑
    total_rows = len(df)
    if total_rows < 500:
        n_valid = int(total_rows * 0.2)
    else:
        n_valid = max(200, int(total_rows * 0.05))

    df_valid = df.iloc[:n_valid].copy()
    df_train = df.iloc[n_valid:].copy()

    print(f"[INFO] Total: {total_rows} | Train: {len(df_train)} | Valid: {len(df_valid)}")
    print(f"[INFO] Base Model: {args.base_model} | Epochs: {epochs} | Batch: {batch_size} | Accum: {grad_accum}")

    if len(df_train) == 0:
        print("[WARN] Train set is empty. Fallback: using all data for training.")
        df_train = df.copy()
        df_valid = df.iloc[:0].copy()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # 加载模型
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=3,
        ignore_mismatched_sizes=True
    )

    # LoRA 配置
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["query", "value", "key", "query_key_value", "q_proj", "v_proj", "k_proj"],  # 覆盖常见 attention 层名称
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = PseudoLabelDataset(df_train, tokenizer, args.max_len)
    valid_ds = PseudoLabelDataset(df_valid, tokenizer, args.max_len)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    class_weights = None
    if args.use_class_weight:
        counts = df_train["label_id"].value_counts().to_dict()
        w = []
        for i in range(3):
            c = max(1, int(counts.get(i, 1)))
            w.append(1.0 / c)
        w = np.array(w, dtype=np.float32)
        w = w / w.mean()
        class_weights = torch.tensor(w, dtype=torch.float32)
        print("[INFO] class_weights:", w.tolist())

    # --- [关键修改] 计算更频繁的存档步数 ---
    # 计算总步数
    num_update_steps_per_epoch = len(df_train) // batch_size // grad_accum
    total_steps = int(num_update_steps_per_epoch * epochs)

    # 策略：至少每50步存一次，或者每10%进度存一次（取最大值），但不要超过500步存一次
    # 这样小数据存的快，大数据也不会存太慢
    save_steps = max(10, min(50, int(total_steps * 0.1)))
    print(f"[INFO] Save Strategy: Saving checkpoint every {save_steps} steps (Total steps: {total_steps})")

    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        logging_steps=10,

        # --- 核心修改：改为按 step 存档 ---
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=1,  # 只保留最近的一个，节省空间

        # 验证策略必须与存档策略同步
        eval_strategy="steps" if len(df_valid) > 0 else "no",
        eval_steps=save_steps,
        load_best_model_at_end=True if len(df_valid) > 0 else False,
        # -------------------------------

        # MPS 不支持 fp16，使用 float32
        # Transformers 会自动处理设备选择，但需要确保 fp16 在 MPS 上被禁用
        fp16=torch.cuda.is_available(),  # 只在 CUDA 上启用 fp16
        report_to="none",
        seed=args.seed,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds if len(df_valid) > 0 else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        class_weights=class_weights,
    )

    resume_ckpt = None
    if args.resume:
        resume_ckpt = get_last_checkpoint(str(ckpt_dir))
        if resume_ckpt:
            print(f"[INFO] Resuming from checkpoint: {resume_ckpt}")
        else:
            print("[INFO] Resume requested but no checkpoint found. Starting from scratch.")

    trainer.train(resume_from_checkpoint=resume_ckpt)

    # 保存最终模型和 Tokenizer
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    print(f"[SUCCESS] Model saved to {out_path}")


if __name__ == "__main__":
    main()
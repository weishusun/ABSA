# scripts/route_b_sentiment/sentiment_03_train_asc_lora.py
import argparse
import inspect
from pathlib import Path

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
import os
os.environ["TRANSFORMERS_DISABLE_SAFE_TENSORS_CONVERSION"] = "1"

LABEL2ID = {"NEG": 0, "NEU": 1, "POS": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="train_pseudolabel(.parquet) or balanced version")
    ap.add_argument("--out-dir", required=True, help="outputs/phone_v2/models/asc_lora_v1")
    ap.add_argument("--base-model", default="hfl/chinese-macbert-base")  # 走网络下载/缓存
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--resume", action="store_true", help="从 out-dir/ckpt 的 last checkpoint 继续")
    ap.add_argument(
        "--use-class-weight",
        dest="use_class_weight",
        action="store_true",
        help="按训练集分布自动加权 CE loss（推荐小样本/类不均衡）",
    )
    ap.add_argument("--gradient-checkpointing", action="store_true", help="进一步省显存（会稍慢）")
    args = ap.parse_args()

    print(f"[INFO] torch={torch.__version__} cuda_available={torch.cuda.is_available()} cuda={torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"[INFO] device={torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.train)
    df["label"] = df["label"].astype(str).str.upper()
    df = df[df["label"].isin(LABEL2ID.keys())].copy()
    df["label_id"] = df["label"].map(LABEL2ID).astype(int)

    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_valid = max(200, int(len(df) * 0.05))
    df_valid = df.iloc[:n_valid].copy()
    df_train = df.iloc[n_valid:].copy()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    hf_token = os.environ.get("HF_TOKEN")
    token_kwargs = {"token": hf_token} if hf_token else {}

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=3,
        ignore_mismatched_sizes=True,
        use_safetensors=False,  # 关键：别触发 safetensors auto-conversion / discussions 探测
        **token_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        **token_kwargs,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["query", "value"],
    )
    model = get_peft_model(model, lora)

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

    ta_kwargs = dict(
        output_dir=str(ckpt_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, int(args.grad_accum)),
        learning_rate=args.lr,
        logging_steps=20,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=args.seed,
        load_best_model_at_end=False,
    )

    sig = inspect.signature(TrainingArguments).parameters
    if "eval_strategy" in sig:
        ta_kwargs["eval_strategy"] = "epoch"
    else:
        ta_kwargs["evaluation_strategy"] = "epoch"
    ta_kwargs["save_strategy"] = "epoch"

    training_args = TrainingArguments(**ta_kwargs)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"acc": float((preds == labels).mean())}

    trainer_cls = WeightedTrainer if args.use_class_weight else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        class_weights=class_weights if args.use_class_weight else None,
    )

    resume_ckpt = None
    if args.resume:
        resume_ckpt = get_last_checkpoint(str(ckpt_dir))
        if resume_ckpt:
            print(f"[INFO] resume from: {resume_ckpt}")

    trainer.train(resume_from_checkpoint=resume_ckpt)

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"[OK] saved model+tokenizer to: {out_dir}")
    print("[NEXT] 运行 sentiment_04_infer_asc.py 做全量推理。")


if __name__ == "__main__":
    main()

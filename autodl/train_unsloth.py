# -*- coding: utf-8 -*-
"""
Unsloth 训练脚本（备选方案）。

注意：Qwythos 是 Qwen3.5 混合注意力架构（Gated DeltaNet 线性注意力），
Unsloth 可能尚未完全支持。若 from_pretrained 报架构不支持，请改用
LLaMA-Factory 方案（见 AUTODL_GUIDE.md 与 llamafactory_train.yaml）。
"""

import json
import random

from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel, is_bfloat16_supported

MODEL = "/root/autodl-tmp/models/qwythos-9b"
DATA = "/root/autodl-tmp/data/train.jsonl"
OUT = "/root/autodl-tmp/output/mathpkg_v1_unsloth"
MAX_SEQ = 4096

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=MAX_SEQ,
    load_in_4bit=True,          # 4bit QLoRA；若 bitsandbytes 报 sm_120 错误，改用 load_in_4bit=False（bf16）
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)


def build_dataset(path, limit=0):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            user_content = r["instruction"].strip()
            if r.get("input", "").strip():
                user_content += "\n" + r["input"].strip()
            rows.append(
                {
                    "user": user_content,
                    "assistant": r["output"],
                }
            )
            if limit and len(rows) >= limit:
                break
    random.shuffle(rows)
    return rows


rows = build_dataset(DATA)
ds = Dataset.from_list(rows)
train_val = ds.train_test_split(test_size=0.01, seed=42)


def fmt(examples):
    texts = []
    for u, a in zip(examples["user"], examples["assistant"]):
        messages = [
            {"role": "user", "content": u},
            {"role": "assistant", "content": a},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}


train_ds = train_val["train"].map(fmt, batched=True)
eval_ds = train_val["test"].map(fmt, batched=True)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    args=SFTConfig(
        output_dir=OUT,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        bf16=is_bfloat16_supported(),
        gradient_checkpointing=True,
        max_seq_length=MAX_SEQ,
        dataset_num_proc=4,
        packing=False,
    ),
)

trainer.train()
trainer.save_model(OUT)
model.save_pretrained_merged(OUT + "_merged", tokenizer, save_method="merged_16bit")

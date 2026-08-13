# -*- coding: utf-8 -*-
"""
Qwythos-9B 数学 SFT 训练脚本（TRL 直连，绕过 LLaMA-Factory）。

在 AutoDL 服务器上运行：
    python /root/autodl-tmp/train_trl.py > /root/autodl-tmp/train_trl.log 2>&1

全量训练版本：SMOKE = False（跑全部 4 万条）
"""

import json
import random

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL_ID = "/root/autodl-tmp/models/qwythos-9b"
DATA_PATH = "/root/autodl-tmp/data/train.jsonl"
OUT_DIR = "/root/autodl-tmp/output/mathpkg_trl"
MAX_LEN = 1024  # 样本平均约 945 token，1024 够用；调小可避免长样本显存峰值
SMOKE = False  # 全量训练：全部 4 万条

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(
    model, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="SEQ_2_SEQ_LM")
)

rows = []
with open(DATA_PATH, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        rows.append(
            {
                "text": tok.apply_chat_template(
                    r["messages"], tokenize=False, add_generation_prompt=False
                )
                + tok.eos_token
            }
        )
        if SMOKE and len(rows) >= 200:
            break

random.seed(42)
random.shuffle(rows)
ds = Dataset.from_list(rows).train_test_split(test_size=0.01, seed=42)

trainer = SFTTrainer(
    model=model,
    processing_class=tok,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    args=SFTConfig(
        output_dir=OUT_DIR,
        max_length=MAX_LEN,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        dataloader_num_workers=4,
        num_train_epochs=1.0,
        learning_rate=2.0e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_only_model=True,
        bf16=True,
        gradient_checkpointing=True,
        report_to="tensorboard",
    ),
)
trainer.train()
trainer.save_model(OUT_DIR)
print("TRAINING-DONE")

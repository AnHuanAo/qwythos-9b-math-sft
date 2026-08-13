# -*- coding: utf-8 -*-
"""把 LoRA 适配器合并进完整模型，产出可直接部署的合并模型。"""

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

BASE = "/root/autodl-tmp/models/qwythos-9b"
ADAPTER = "/root/autodl-tmp/output/mathpkg_trl"
OUT = "/root/autodl-tmp/output/mathpkg_trl_merged"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda:0")
model = PeftModel.from_pretrained(model, ADAPTER)
model = model.merge_and_unload()
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print("MERGED-OK")

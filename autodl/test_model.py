# -*- coding: utf-8 -*-
"""
微调前后对比测试：同一批证明题，分别用 base 模型和微调后的模型回答。
用法：
    python /root/autodl-tmp/test_model.py --mode base
    python /root/autodl-tmp/test_model.py --mode finetuned
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

BASE = "/root/autodl-tmp/models/qwythos-9b"
ADAPTER = "/root/autodl-tmp/output/mathpkg_trl"

QUESTIONS = [
    "请证明以下定理：设 G 是有限群，p 是整除 |G| 的素数，则 G 中存在 p 阶元。",
    "请证明以下定理：若两个链复形 C 和 D 链等价，则它们的张量积 C ⊗ D 也链等价。",
    "请证明以下定理：每个有限生成阿贝尔群都同构于某个自由阿贝尔群与有限挠群的直和。",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["base", "finetuned"], required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    if args.mode == "finetuned":
        model = PeftModel.from_pretrained(model, ADAPTER)
        model = model.merge_and_unload()
    model.eval()

    print(f"########## MODE = {args.mode} ##########")
    for q in QUESTIONS:
        text = tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
        )
        enc = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                repetition_penalty=1.05,
            )
        ans = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print("\n" + "=" * 80)
        print("题目:", q)
        print("回答:", ans[:2000])


if __name__ == "__main__":
    main()

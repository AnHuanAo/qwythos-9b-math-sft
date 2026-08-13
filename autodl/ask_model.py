# -*- coding: utf-8 -*-
"""对话式做题：给合并后的模型出一道题，它回答。
用法：python /root/autodl-tmp/ask_model.py 请证明：1+1=2
"""

import argparse

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL = "/root/autodl-tmp/output/mathpkg_trl_merged"


def generate(q: str) -> str:
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()
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
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    args = ap.parse_args()
    q = " ".join(args.question) or input("题目：")
    print(generate(q))


if __name__ == "__main__":
    main()

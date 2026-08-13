# -*- coding: utf-8 -*-
"""把 v1_sft.jsonl 转成 LLaMA-Factory 可用的格式（默认 sharegpt，可 --format alpaca）。"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=r"E:\My_Project\mathpkg\v1训练数据\v1_sft.jsonl")
    ap.add_argument("--out", default=r"E:\My_Project\mathpkg\v1训练数据\autodl\train.jsonl")
    ap.add_argument("--task", default="all", choices=["all", "proof", "explain"])
    ap.add_argument("--max", type=int, default=0, help="只取前 N 条（冒烟测试用）")
    ap.add_argument("--format", default="sharegpt", choices=["sharegpt", "alpaca"])
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n = 0
    with open(args.input, encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            r = json.loads(line)
            if args.task != "all" and r["task"] != args.task:
                continue
            if args.max and n >= args.max:
                break
            user_content = r["instruction"].strip()
            if r.get("input", "").strip():
                user_content += "\n" + r["input"].strip()
            if args.format == "alpaca":
                rec = {
                    "instruction": r["instruction"],
                    "input": r.get("input", ""),
                    "output": r["output"],
                }
            else:
                rec = {
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": r["output"]},
                    ]
                }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"written {n} samples -> {args.out}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Lean 编译验证 pass@k 评估 —— 验证闭环的核心脚本。

目标：在**整书留出的测试集**上，测基座 vs 微调模型把自然语言定理翻译成
Lean 4 证明的**编译通过率**（pass@k）。编译器通过 = 真金白银的正确性证据，
替代只能看风格的 eval_loss / token 准确率。

两种模式：
  1. 生成+编译（默认）：
      模型对每个定理采样 n 个 Lean 证明 → 写入临时 .lean 文件 →
      用 lean/lake 编译 → 统计 pass@k。
  2. --check-corpus：只编译一个目录下已有的 .lean 文件（比如朋友流水线
     lean/MathPkg/MathPkg/Pipeline/ 的 480 个文件），得出基线通过率。

用法（AutoDL / 有 GPU 与 Lean 的机器）：
    # 从测试集提取证明题陈述
    python - <<'EOF'
    import json
    with open("data/clean/clean_test.jsonl", encoding="utf-8") as f, \
         open("data/clean/test_proofs.jsonl", "w", encoding="utf-8") as g:
        for line in f:
            r = json.loads(line)
            if r["task"] == "proof":
                g.write(json.dumps({"id": r["meta"]["concept_id"],
                                    "statement": r["input"]}) + "\n")
    EOF

    # 基座模型
    python eval/lean_passk.py --model /root/autodl-tmp/models/qwythos-9b \
        --statements data/clean/test_proofs.jsonl --n 8 --k 1 5 8 \
        --lean-project /root/autodl-tmp/lean/MathPkg --out reports/base_passk.json

    # 微调模型（加载 LoRA 适配器）
    python eval/lean_passk.py --model /root/autodl-tmp/models/qwythos-9b \
        --adapter /root/autodl-tmp/output/mathpkg_trl \
        --statements data/clean/test_proofs.jsonl --n 8 --k 1 5 8 \
        --lean-project /root/autodl-tmp/lean/MathPkg --out reports/ft_passk.json

    # 已有语料基线（不需要 GPU）
    python eval/lean_passk.py --check-corpus lean/MathPkg/MathPkg/Pipeline \
        --lean-project lean/MathPkg

报告：--out 指向 JSON；同时打印 Markdown 表格（可直接贴进博客/论文）。

数据飞轮（下一步）：把"编译失败的报错"喂回模型做修正（参考 mathpkg 的
self_heal 模式），再把修正成功的 (statement, proof) 加入训练集——这就是
"生成 → 编译 → 修正"循环。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROMPT = "请把下面的数学定理陈述翻译成 Lean 4 证明（用 import Mathlib 开头，直接给出可编译的定理与证明）：\n\n{statement}"


def comb(n: int, k: int) -> int:
    return math.comb(n, k) if n >= k else 0


def pass_at_k(n: int, c: int, k: int) -> float:
    """标准 pass@k（Codex 论文公式）：1 - C(n-c, k)/C(n, k)。n=采样数, c=成功数。"""
    if n == 0:
        return 0.0
    if c == 0:
        return 0.0
    return 1.0 - comb(n - c, k) / comb(n, k)


# ── 编译 ──────────────────────────────────────────────────────────────────────

def compile_file(lean_file: Path, lean_project: Path, timeout: int = 120) -> tuple[bool, str]:
    """
    在 lean_project 的 lake 环境里编译单个 .lean 文件。
    返回 (是否通过, 错误摘要首 500 字符)。lake env lean 会复用 Mathlib 的缓存 olean。
    """
    cmd = ["lake", "env", "lean", str(lean_file)]
    try:
        proc = subprocess.run(
            cmd, cwd=str(lean_project), capture_output=True, text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        raise SystemExit("未找到 lake/lean 命令。请先安装 elan + Lean 4（https://lean-lang.org/lean4/doc/quickstart.html）")
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT >{timeout}s"
    if proc.returncode == 0:
        return True, ""
    err = proc.stderr.strip() or proc.stdout.strip()
    return False, err[:500]


# ── 生成 ──────────────────────────────────────────────────────────────────────

def load_model(model_path: str, adapter: str | None, quantize: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    if quantize == "4bit":
        from transformers import BitsAndBytesConfig
        qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        qcfg = None
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        quantization_config=qcfg, device_map="cuda:0",
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    return model, tok


def generate(model, tok, statement: str, n: int, temperature: float,
             max_new_tokens: int) -> list[str]:
    import torch
    text = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT.format(statement=statement)}],
        tokenize=False, add_generation_prompt=True,
    )
    enc = tok(text, return_tensors="pt").to("cuda")
    outs = model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=True,
        temperature=temperature, top_p=0.95, top_k=20, repetition_penalty=1.05,
        num_return_sequences=n,
    )
    return [
        tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for o in outs
    ]


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Lean 编译验证 pass@k 评估")
    ap.add_argument("--model", help="基座模型路径")
    ap.add_argument("--adapter", default=None, help="LoRA 适配器路径（评估微调模型时）")
    ap.add_argument("--quantize", default="none", choices=["none", "4bit"])
    ap.add_argument("--statements", help="JSONL：每行 {\"id\": ..., \"statement\": ...}")
    ap.add_argument("--n", type=int, default=8, help="每个定理采样数")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 5, 8], help="pass@k 的 k 列表")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--lean-project", required=True, help="Lean lake 项目目录（含 lakefile.toml）")
    ap.add_argument("--work-dir", help="临时 .lean 文件目录（默认系统临时目录）")
    ap.add_argument("--out", default="reports/passk_report.json", help="JSON 报告输出路径")
    ap.add_argument("--max-statements", type=int, default=0, help="最多评估 N 条定理（0=全部）")
    ap.add_argument("--check-corpus", help="只编译该目录下所有 .lean 文件并报告通过率（不需要 GPU）")
    args = ap.parse_args()

    if args.check_corpus:
        corpus = Path(args.check_corpus)
        files = sorted(corpus.rglob("*.lean"))
        ok, fail = 0, 0
        for i, f in enumerate(files):
            passed, err = compile_file(f, Path(args.lean_project))
            if passed:
                ok += 1
            else:
                fail += 1
            if (i + 1) % 20 == 0 or i == len(files) - 1:
                print(f"[corpus] {i + 1}/{len(files)}  ok={ok} fail={fail}")
        report = {
            "mode": "check-corpus",
            "corpus": args.check_corpus,
            "total": len(files), "ok": ok, "fail": fail,
            "pass_rate": ok / len(files) if files else 0.0,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    if not args.model or not args.statements:
        raise SystemExit("生成模式需要 --model 与 --statements")

    statements = []
    with open(args.statements, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            statements.append((r["id"], r["statement"]))
    if args.max_statements > 0:
        statements = statements[: args.max_statements]
    print(f"[eval] {len(statements)} theorems, n={args.n}, k={args.k}")

    model, tok = load_model(args.model, args.adapter, args.quantize)

    work_dir = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="lean_passk_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "imports.lean").write_text("import Mathlib\n", encoding="utf-8")

    results: dict[str, dict] = {}
    t0 = time.time()
    for idx, (sid, statement) in enumerate(statements):
        samples = generate(model, tok, statement, args.n, args.temperature,
                           args.max_new_tokens)
        passed = []
        for j, code in enumerate(samples):
            fname = work_dir / f"eval_{idx:04d}_{j:02d}.lean"
            fname.write_text("import Mathlib\n\n" + code, encoding="utf-8")
            ok, err = compile_file(fname, Path(args.lean_project))
            passed.append(ok)
            if not ok:
                (work_dir / f"eval_{idx:04d}_{j:02d}.err.txt").write_text(err, encoding="utf-8")
        c = sum(passed)
        results[sid] = {
            "statement": statement,
            "samples": args.n,
            "passed": c,
            "pass@1": pass_at_k(args.n, c, 1),
            **{f"pass@{k}": pass_at_k(args.n, c, k) for k in args.k},
            "successful_samples": [j for j, ok in enumerate(passed) if ok],
        }
        agg = {f"pass@{k}": sum(r[f"pass@{k}"] for r in results.values()) / len(results)
               for k in args.k}
        agg["pass@1"] = sum(r["pass@1"] for r in results.values()) / len(results)
        print(f"[eval] {idx + 1}/{len(statements)}  {sid}: "
              f"passed={c}/{args.n}  " + "  ".join(f"{kk}={agg[kk]:.3f}" for kk in ["pass@1", *[f'pass@{k}' for k in args.k]]))

    report = {
        "mode": "generate+compile",
        "model": args.model,
        "adapter": args.adapter,
        "n": args.n, "k": args.k, "temperature": args.temperature,
        "num_statements": len(results),
        "num_compiles": len(results) * args.n,
        "elapsed_sec": round(time.time() - t0, 1),
        "aggregate": agg,
        "per_statement": results,
        "work_dir": str(work_dir),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown 表格输出
    print("\n## pass@k 汇总")
    print("| metric | value |")
    print("|---|---|")
    for k in ["pass@1", *[f"pass@{k}" for k in args.k]]:
        print(f"| {k} | {agg[k]:.4f} |")
    print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()

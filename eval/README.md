# 验证闭环：Lean 编译 pass@k 评估

## 为什么必须有这一步

v1 训练的 `eval_loss 0.6905 / token 准确率 0.8119` 只是在**同分布 1% 切分**上的
风格指标——它证明不了证明正确。唯一可信的正确性信号是 **Lean 编译器**：
编译通过 = 命题类型正确 + 证明项可构造，二者都是机器可核验的。

`lean_passk.py` 把这条闭环落地：对每条定理采样 n 个证明 → 逐个 `lake env lean`
编译 → 用 Codex 论文的标准公式算 `pass@k`。

## 前置条件（AutoDL 上一次性配置）

```bash
# Lean 工具链（elan + lake + mathlib，约 20 分钟首次构建，之后有缓存）
curl -fsSL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | bash -s -- -y
source ~/.elan/env

# 拉取朋友流水线的 Lean 项目（或自建最小项目）
# 最小项目：lakefile.toml 里 require mathlib v4.31.0，见 ../lean/MathPkg
cd /root/autodl-tmp && lake new MathPkg && cd MathPkg
# 编辑 lakefile.toml 加 mathlib 依赖后：
lake update mathlib && lake build MathPkg   # 构建 Mathlib 缓存
```

## 用法

```bash
# 0) 从整书留出的测试集提取证明题陈述
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

# 1) 基线：朋友流水线现有语料的编译通过率（无需 GPU）
python eval/lean_passk.py --check-corpus ../lean/MathPkg/MathPkg/Pipeline \
    --lean-project ../lean/MathPkg --out reports/corpus_baseline.json

# 2) 基座模型
python eval/lean_passk.py --model /root/autodl-tmp/models/qwythos-9b \
    --statements data/clean/test_proofs.jsonl --n 8 --k 1 5 8 \
    --lean-project /root/autodl-tmp/lean/MathPkg --out reports/base_passk.json

# 3) 微调模型（对比）
python eval/lean_passk.py --model /root/autodl-tmp/models/qwythos-9b \
    --adapter /root/autodl-tmp/output/mathpkg_trl \
    --statements data/clean/test_proofs.jsonl --n 8 --k 1 5 8 \
    --lean-project /root/autodl-tmp/lean/MathPkg --out reports/ft_passk.json
```

参数速查：`--n` 每定理采样数（越大越稳，显存/时间翻倍）；`--k` pass@k 列表；
`--quantize 4bit` 显存不够时用；`--max-statements 20` 先小范围冒烟。

## 数据飞轮（这就是论文的"编译器反馈"）

1. 用评估结果收集**编译失败样本**（`work_dir/*.err.txt` 有报错原文）；
2. 把 (定理, 失败证明, 报错) 喂给模型做修正（可复用 mathpkg 的 self-heal 提示词模式）；
3. 修正后**编译通过**的 (定理 → 证明) 加入下一轮训练集；
4. 重训 → 重测 pass@k → 记录通过率曲线。

每轮训练成本 ≈ 5.5 小时/张 5090，通过率提升 5–10% 即可构成论文的 solid contribution。

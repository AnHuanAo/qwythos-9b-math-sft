# mathpkg-sft — Qwythos-9B 数学证明 SFT 实验仓库

> 基于 mathpkg 数学知识库（61 本 GTM 教材、27,741 个概念、13,436 条证明），对
> Qwythos-9B（Qwen3.5 架构、9.4B、1M 上下文）做 QLoRA 后训练的实验仓库。
> 本仓库是 **公开、可复现** 的部分；mathpkg 数据/形式化仓库与它分开管理（可私有）。

## 现状摘要（v1，2026-08）

- **数据**：40,443 条 SFT 样本（13,277 证明题 + 27,166 知识讲解），由 `build_sft.py`
  从 mathpkg 发布数据自动生成；`clean_data.py` 进一步做截断过滤、跨书近似去重、整书留出。
- **训练**：RTX 5090（32GB）× QLoRA 4bit × r=16/alpha=32，约 5.5 小时 / 5005 步。
  `train_loss 0.7176`，`eval_loss 0.6905`，`eval_mean_token_accuracy 0.8119`。
- **结论（诚实版）**：模型学会了"教材证明的皮"（风格迁移可量化，见 `博客/`），
  **但证明正确性未解决**——下一步必须接上 Lean 编译验证闭环（脚本见 `eval/`）。

## 目录结构

```
mathpkg-sft/
├── build_sft.py          # mathpkg 发布数据 → SFT JSONL（相对路径，可移植）
├── clean_data.py         # 数据清洗：截断过滤 + MinHash 跨书近似去重 + 整书留出测试集
├── prep_data.py          # SFT JSONL → LLaMA-Factory/Unsloth 训练格式
├── requirements.txt      # 训练环境依赖（cu128）
├── autodl/               # AutoDL 训练/合并/测试脚本 + 保姆级指南
├── eval/                 # Lean 编译验证 pass@k 评估（验证闭环，核心下一步）
├── bench/                # vLLM 推理 benchmark（TTFT/TPOT/吞吐/显存）
├── 博客/                 # 训练实录博客（含诚实结论）
├── 训练结果/             # base vs finetuned 三题对比
└── data/                 # （本地，不入库）生成的 JSONL 与清洗产物
```

## 快速开始

```bash
# 0. 环境（AutoDL RTX 5090 或本地等效；torch 必须 cu128）
pip install -r requirements.txt

# 1. 从 mathpkg 发布数据重建 SFT 数据集（默认输入 ../mathpkg/math_pkg_release）
python build_sft.py --out data/v1_sft.jsonl

# 2. 清洗 + 去重 + 整书留出测试集
python clean_data.py --input data/v1_sft.jsonl --out-dir data/clean \
    --holdout-count 6

# 3. 转训练格式 + 训练（详见 autodl/AUTODL_GUIDE.md）
python prep_data.py --input data/clean/clean_train.jsonl --out autodl/train.jsonl

# 4. 验证闭环：基座 vs 微调模型在留出集上的 Lean 编译 pass@k
python eval/lean_passk.py --model /path/to/qwythos-9b \
    --statements data/clean/clean_test.jsonl --k 1 5 10

# 5. 推理 benchmark
python bench/vllm_bench.py --model /path/to/merged --mode offline
```

## 数据与版权

- 数据由 LLM 从 **OCR 版 Springer 教材** 抽取，版权风险高：**原始 JSONL 不入库、不公开**，
  需要时用 `build_sft.py` 自行重建；公开内容仅限脚本/文档/评测报告。
- 训练数据中存在流水线残次样本（"the proof section is truncated..." 等），
  `clean_data.py` 会过滤（v1 原始语料中约 570/40,443 条）。
- 模型权重仅发布 LoRA 适配器（`adapter_config.json` 入库存档；大权重建议 Git LFS 或 HuggingFace）。

## 论文定位（目标）

首次以 **Lean 编译器反馈** 作为监督信号，对 9B 级通用推理模型做 Autoformalization
后训练；构建"生成 → 编译 → 报错修正"数据飞轮，逐步逼近编译器验证通过率。

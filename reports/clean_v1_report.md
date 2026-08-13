# v1 数据清洗报告（2026-08-09 生成，clean_data.py 运行结果）

| 阶段 | 记录数 | 说明 |
|---|---|---|
| 原始输入 | 40,443 | v1_sft.jsonl |
| 截断/残次过滤 | −533 | "the proof section is truncated" / "reconstruct" / "source text" 等（已泄漏进 v1 模型输出） |
| 长度过滤 | −880 | output < 100 字符（856）、input < 10 字符（24） |
| **跨书近似去重** | **−1,134** | MinHash-LSH（32 签名 × 8 bands），归一化 Jaccard ≥ 0.85 → 证实跨书重复污染真实存在 |
| 清洗后总数 | 37,896 | |

## 整书留出（防评估污染）

按记录数最多的 **6 本书** 作为测试集（与训练集完全不相交）：

| 数据集 | 总量 | proof | explain |
|---|---|---|---|
| clean_train | 30,557 | 9,902 | 20,655 |
| clean_test（gtm003/023/027/037/040/054） | 7,339 | 2,466 | 4,873 |

> 评估教训（v1）：原始 1% 随机行级切分在同分布数据上测出的
> eval_loss 0.6905 / token acc 0.8119 **高估了真实泛化**——既有截断残次样本
> 混入训练，又有 1,134 条跨书近似重复横跨训练/验证。

## 复现

```bash
python clean_data.py \
    --input  ../data/v1_sft.jsonl \
    --out-dir ../data/clean \
    --holdout-count 6 \
    --dedup-threshold 0.85
```

产物：`data/clean/clean_train.jsonl`、`clean_test.jsonl`、`clean_stats.json`、
`dropped_samples.jsonl`（被过滤样本抽样，供人工抽检）。

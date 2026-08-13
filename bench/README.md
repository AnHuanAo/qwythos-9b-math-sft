# vLLM 推理 Benchmark

对标《岗位训练计划.md》阶段 1/2：用 vLLM 部署微调模型，量化 TTFT / TPOT / 吞吐 / 显存，
产出可写进简历/博客的 benchmark 报告。

## offline 模式（进程内，单机快速对比）

```bash
# 先合并 LoRA（或直接用 merged 模型目录）
python bench/vllm_bench.py --model /root/autodl-tmp/output/mathpkg_v1_merged \
    --mode offline \
    --prompts-file data/clean/test_proofs.jsonl --max-statements 30 \
    --out reports/bench_offline.json
```

## online 模式（贴近线上，并发扫描）

```bash
# 终端 1：起服务（微调模型 vs 基座各起一个端口，做对比）
vllm serve /root/autodl-tmp/output/mathpkg_v1_merged --max-model-len 65536 --port 8000
vllm serve /root/autodl-tmp/models/qwythos-9b      --max-model-len 65536 --port 8001

# 终端 2：并发扫描
python bench/vllm_bench.py --mode online --server http://localhost:8000/v1 \
    --concurrency 1 4 8 16 --out reports/bench_ft_online.json
python bench/vllm_bench.py --mode online --server http://localhost:8001/v1 \
    --concurrency 1 4 8 16 --out reports/bench_base_online.json
```

## 建议的对比矩阵（每项一个实验，配置→数据→结论三段式写报告）

| 维度 | 变量 | 指标 |
|---|---|---|
| 长上下文 | max-model-len 4096/16K/64K | 显存曲线、TTFT |
| 量化 | bf16 vs AWQ/GPTQ/GGUF | 吞吐↑ vs pass@k 质量损失 |
| 并发 | 1/4/8/16/32 | 吞吐拐点、P95 延迟 |
| 投机解码 | 开/关 | 加速比（对证明生成长文本尤其明显） |

注意：benchmark 要同时报告 **推理指标** 与 **质量指标**（接 `eval/lean_passk.py`），
否则"快了但错了"没有意义。

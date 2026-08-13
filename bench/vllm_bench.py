# -*- coding: utf-8 -*-
"""
vLLM 推理 benchmark：TTFT / TPOT / 吞吐 / 显存。

两种模式：
  offline —— 进程内加载模型（vllm.LLM），串行压测，适合单机快速对比
  online  —— 对已启动的 vllm serve 端点发请求，并发扫描，贴近线上

用法：
    # offline（AutoDL 上）
    python bench/vllm_bench.py --model /root/autodl-tmp/output/mathpkg_v1_merged \
        --mode offline --prompts-file data/clean/test_proofs.jsonl \
        --max-statements 30 --out reports/bench_offline.json

    # online（先另开终端：vllm serve <merged> --max-model-len 65536 --port 8000）
    python bench/vllm_bench.py --mode online --server http://localhost:8000/v1 \
        --concurrency 1 4 8 16 --out reports/bench_online.json

输出：JSON 报告 + Markdown 表格；--out 缺省打印到 stdout。
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

QUESTIONS = [
    "请证明以下定理：设 G 是有限群，p 是整除 |G| 的素数，则 G 中存在 p 阶元。",
    "请证明以下定理：每个有限生成阿贝尔群都同构于某个自由阿贝尔群与有限挠群的直和。",
]


def load_prompts(path: str, max_n: int) -> list[str]:
    if not path:
        return QUESTIONS
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            stmt = r.get("input") or r.get("statement") or ""
            if stmt:
                out.append("请证明以下定理：\n" + stmt)
            if max_n and len(out) >= max_n:
                break
    return out or QUESTIONS


# ── offline ───────────────────────────────────────────────────────────────────

def run_offline(model_path: str, prompts: list[str], max_new_tokens: int,
                temperature: float) -> dict:
    import time as _t
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, max_model_len=65536, enforce_eager=True)
    params = SamplingParams(max_tokens=max_new_tokens, temperature=temperature,
                            top_p=0.95)
    t0 = _t.perf_counter()
    outputs = llm.generate(prompts, params)
    dt = _t.perf_counter() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    # TTFT 尽量从 metrics 取；不同 vLLM 版本字段名不同，取不到就置 None
    ttfts = []
    for o in outputs:
        try:
            m = o.metrics
            first = getattr(m, "first_token_time", None) or getattr(m, "first_ts", None)
            arr = getattr(m, "arrival_time", None) or getattr(m, "arrival_ts", None)
            if first is not None and arr is not None:
                ttfts.append(first - arr)
        except Exception:
            pass
    ttft_p50 = statistics.median(ttfts) if ttfts else None

    try:
        import torch
        free_before, _ = torch.cuda.mem_get_info()
        free_after, total = torch.cuda.mem_get_info()
        vram_used = (free_before - free_after) / 1e9
        vram_total = total / 1e9
    except Exception:
        vram_used, vram_total = None, None

    return {
        "mode": "offline",
        "model": model_path,
        "num_requests": len(prompts),
        "total_sec": round(dt, 2),
        "throughput_req_per_s": round(len(prompts) / dt, 2),
        "throughput_tok_per_s": round(total_tokens / dt, 2),
        "total_output_tokens": total_tokens,
        "ttft_p50_sec": ttft_p50,
        "vram_used_gb": vram_used,
        "vram_total_gb": vram_total,
    }


# ── online ────────────────────────────────────────────────────────────────────

def _call_once(server: str, prompt: str, max_new_tokens: int,
               temperature: float) -> tuple[float, float, int, str]:
    """返回 (ttft_sec, total_sec, output_tokens, error)。TTFT 用流式首包测量。"""
    import requests
    payload = {
        "model": "/",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "stream": True,
    }
    t0 = time.perf_counter()
    ttft = None
    ntok = 0
    try:
        with requests.post(server + "/chat/completions", json=payload,
                           stream=True, timeout=600) as resp:
            if resp.status_code != 200:
                return float("nan"), float("nan"), 0, f"HTTP {resp.status_code}"
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data: "):
                    continue
                chunk = raw[6:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    data = json.loads(chunk)
                    delta = data["choices"][0].get("delta", {}).get("content", "")
                    if delta:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        ntok += len(delta.split())
                except json.JSONDecodeError:
                    continue
        dt = time.perf_counter() - t0
        return (ttft if ttft is not None else float("nan")), dt, ntok, ""
    except Exception as e:  # noqa: BLE001
        return float("nan"), float("nan"), 0, str(e)


def run_online(server: str, prompts: list[str], concurrency: list[int],
               max_new_tokens: int, temperature: float) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    def sweep(c: int) -> dict:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=c) as ex:
            futs = [ex.submit(_call_once, server, p, max_new_tokens, temperature)
                    for p in prompts]
            results = [f.result() for f in futs]
        dt = time.perf_counter() - t0
        totals = [r[1] for r in results if r[1] == r[1]]
        ttfts = [r[0] for r in results if r[0] == r[0]]
        ntok = sum(r[2] for r in results)
        return {
            "concurrency": c,
            "num_requests": len(results),
            "total_sec": round(dt, 2),
            "throughput_req_per_s": round(len(results) / dt, 2),
            "throughput_tok_per_s": round(ntok / dt, 2),
            "ttft_p50_sec": round(statistics.median(ttfts), 3) if ttfts else None,
            "ttft_p95_sec": round(sorted(ttfts)[int(len(ttfts) * 0.95) - 1], 3)
            if ttfts else None,
            "p50_latency_sec": round(statistics.median(totals), 3) if totals else None,
            "p95_latency_sec": round(sorted(totals)[int(len(totals) * 0.95) - 1], 3)
            if totals else None,
            "errors": sum(1 for r in results if r[3]),
        }

    rows = [sweep(c) for c in concurrency]
    return {"mode": "online", "server": server, "rows": rows}


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="vLLM 推理 benchmark")
    ap.add_argument("--mode", choices=["offline", "online"], required=True)
    ap.add_argument("--model", help="offline 模式：模型路径")
    ap.add_argument("--server", default="http://localhost:8000/v1", help="online 模式：服务端点")
    ap.add_argument("--prompts-file", help="JSONL（取 input/statement 字段）或留空用内置 2 题")
    ap.add_argument("--max-statements", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--out", help="JSON 报告路径（缺省打印 stdout）")
    args = ap.parse_args()

    prompts = load_prompts(args.prompts_file, args.max_statements)
    if args.mode == "offline":
        if not args.model:
            raise SystemExit("offline 模式需要 --model")
        report = run_offline(args.model, prompts, args.max_new_tokens, args.temperature)
    else:
        report = run_online(args.server, prompts, args.concurrency,
                            args.max_new_tokens, args.temperature)

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()

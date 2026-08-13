# -*- coding: utf-8 -*-
"""
SFT 数据清洗：截断样本过滤 + 跨书近似去重 + 整书留出测试集。

针对 v1 数据已知问题：
  1. 流水线残次样本：证明正文里出现 "the proof section is truncated, I will
     reconstruct..." 之类措辞（v1 原始语料约 570/40,443 条，且已泄漏进模型输出）。
  2. 跨书近似重复：同一（或几乎同一）定理在不同教材中重复出现，若不做近似去重，
     训练/评估同分布切分会互相污染。
  3. 评估集污染：留出集必须按"整本书"划分，而不是随机行级切分。

用法：
    python clean_data.py --input ../data/v1_sft.jsonl --out-dir ../data/clean \
        --holdout-count 6
    python clean_data.py --input ... --holdout-books gtm004,gtm005 \
        --dedup-threshold 0.85

输出（out-dir）：
    clean_train.jsonl / clean_test.jsonl   清洗 + 拆分后的数据
    clean_stats.json                       各阶段统计
    dropped_samples.jsonl                  被过滤样本抽样（前 200 条，供人工抽检）
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

# ── 1. 截断/残次样本特征词（大小写不敏感，命中即丢弃该样本的 output）──
TRUNCATION_PATTERNS = [
    r"the proof section is truncated",
    r"the statement is truncated",
    r"the source (text|material) is (truncated|fragmentary|missing)",
    r"fragmentary nature",
    r"i will reconstruct",
    r"reconstruct(ed|ing)? (a|the|this) (complete|concise|full)",
    r"source text",
    r"due to (the )?truncat",
    r"is truncated",
    r"not (fully )?(available|provided|included)",
    r"missing (part|section|portion|text|content)",
    r"\bomitted\b",
    r"placeholder",
    r"TODO",
    r"unavailable",
    r"\[unclear\]",
    r"\[illegible\]",
    r"\[missing\]",
]

# 归一化：小写、仅保留字母数字、压缩空白（用于去重与长度比较）
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    t = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return _WS.sub(" ", t).strip()


def shingles(text: str, k: int = 16):
    """返回去重后的字符级 shingle 集合（md5 int）。"""
    out = set()
    for i in range(len(text) - k + 1):
        out.add(int(hashlib.md5(text[i : i + k].encode("utf-8")).hexdigest()[:8], 16))
    return out


_M = (1 << 32) - 1
_PERMS = [(3 * i + 1, 7 * i + 5) for i in range(64)]  # 固定伪随机置换 (a,b)


def minhash(sig_len: int, text: str, max_chars: int = 4000) -> list[int]:
    """对归一化文本做 32 个随机线性置换下的 MinHash 签名。"""
    sh = shingles(text[:max_chars])
    sig = []
    for a, b in _PERMS[:sig_len]:
        best = min(((a * x + b) % _M) for x in sh) if sh else 0
        sig.append(best)
    return sig


def jaccard(a: str, b: str) -> float:
    sa, sb = shingles(a), shingles(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def flag_truncated(output: str) -> str | None:
    for pat in TRUNCATION_PATTERNS:
        if re.search(pat, output, re.IGNORECASE):
            return pat
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="SFT 数据清洗")
    ap.add_argument("--input",
                    default=str(Path(__file__).resolve().parent / "data" / "v1_sft.jsonl"))
    ap.add_argument("--out-dir",
                    default=str(Path(__file__).resolve().parent / "data" / "clean"))
    ap.add_argument("--holdout-books", default="",
                    help="逗号分隔的测试书 ID，如 gtm004,gtm005")
    ap.add_argument("--holdout-count", type=int, default=0,
                    help="把记录数最多的 N 本书作为测试集（整书留出，防污染）")
    ap.add_argument("--dedup-threshold", type=float, default=0.85,
                    help="归一化后 Jaccard 相似度 ≥ 阈值即判为近似重复（默认 0.85）")
    ap.add_argument("--min-out-len", type=int, default=100, help="output 最小字符数")
    ap.add_argument("--min-in-len", type=int, default=10, help="input 最小字符数")
    args = ap.parse_args()

    src = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = collections.Counter()
    dropped: list[dict] = []

    # ── 加载 ──────────────────────────────────────────────────────
    records = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                stats["json_parse_fail"] += 1
                continue
            records.append(r)
    stats["total_in"] = len(records)

    # ── 阶段 A：截断/残次过滤 ────────────────────────────────────
    kept = []
    for r in records:
        pat = flag_truncated(r.get("output", ""))
        if pat:
            stats["dropped_truncated"] += 1
            if len(dropped) < 200:
                dropped.append({"reason": f"truncated:{pat}", "record": r})
            continue
        kept.append(r)
    records = kept

    # ── 阶段 B：长度过滤 ──────────────────────────────────────────
    kept = []
    for r in records:
        o, i = r.get("output", ""), r.get("input", "")
        if len(o) < args.min_out_len:
            stats["dropped_out_too_short"] += 1
            continue
        if len(i) < args.min_in_len:
            stats["dropped_in_too_short"] += 1
            continue
        kept.append(r)
    records = kept
    stats["after_filter"] = len(records)

    # ── 阶段 C：跨书近似去重（MinHash-LSH，按任务分组、先到先留）──
    drop_idx: set[int] = set()
    if args.dedup_threshold > 0:
        for task in ("proof", "explain"):
            group = [i for i, r in enumerate(records) if r.get("task") == task]
            if len(group) < 2:
                continue
            norms = {i: normalize(records[i]["output"]) for i in group}
            sigs = {i: minhash(32, norms[i]) for i in group}
            # LSH：8 bands × 4 rows
            bands: dict[tuple, list[int]] = collections.defaultdict(list)
            for i in group:
                for b in range(8):
                    bands[(b, tuple(sigs[i][b * 4 : b * 4 + 4]))].append(i)
            for i in group:  # 按原顺序，先到先留
                if i in drop_idx:
                    continue
                for b in range(8):
                    for j in bands[(b, tuple(sigs[i][b * 4 : b * 4 + 4]))]:
                        if j == i or j in drop_idx:
                            continue
                        ln_i, ln_j = len(norms[i]), len(norms[j])
                        if ln_i == 0 or ln_j == 0:
                            continue
                        if max(ln_i, ln_j) / min(ln_i, ln_j) > 1.35:
                            continue
                        if jaccard(norms[i], norms[j]) >= args.dedup_threshold:
                            drop_idx.add(j)
                            stats["dropped_near_dup"] += 1
                            if len(dropped) < 200:
                                dropped.append({
                                    "reason": f"near_dup:{args.dedup_threshold:.2f}",
                                    "keep": records[i]["meta"],
                                    "drop": records[j]["meta"],
                                })
    records = [r for idx, r in enumerate(records) if idx not in drop_idx]
    stats["after_dedup"] = len(records)

    # ── 阶段 D：整书留出测试集 ────────────────────────────────────
    holdout_books: set[str] = set()
    if args.holdout_books:
        holdout_books = {b.strip() for b in args.holdout_books.split(",") if b.strip()}
    elif args.holdout_count > 0:
        by_book = collections.Counter(r["meta"]["book"] for r in records)
        holdout_books = {b for b, _ in by_book.most_common(args.holdout_count)}

    train, test = [], []
    for r in records:
        if r["meta"].get("book") in holdout_books:
            test.append(r)
        else:
            train.append(r)
    stats["holdout_books"] = sorted(holdout_books)
    stats["train"] = len(train)
    stats["test"] = len(test)
    stats["train_proof"] = sum(1 for r in train if r["task"] == "proof")
    stats["train_explain"] = sum(1 for r in train if r["task"] == "explain")
    stats["test_proof"] = sum(1 for r in test if r["task"] == "proof")
    stats["test_explain"] = sum(1 for r in test if r["task"] == "explain")

    # ── 写出 ──────────────────────────────────────────────────────
    def dump(path: Path, rows: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump(out_dir / "clean_train.jsonl", train)
    dump(out_dir / "clean_test.jsonl", test)
    with open(out_dir / "dropped_samples.jsonl", "w", encoding="utf-8") as f:
        for d in dropped:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    stats_path = out_dir / "clean_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(dict(stats), f, ensure_ascii=False, indent=2)

    print(json.dumps(dict(stats), ensure_ascii=False, indent=2))
    print(f"\ntrain -> {out_dir / 'clean_train.jsonl'}")
    print(f"test  -> {out_dir / 'clean_test.jsonl'}")
    print(f"stats -> {stats_path}")


if __name__ == "__main__":
    main()

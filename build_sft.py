# -*- coding: utf-8 -*-
"""
从 mathpkg 发布数据构建 SFT 数据集（JSONL）。v2：相对路径 + CLI 参数，可移植。

两类任务：
  1. proof   （证明题）：instruction=请证明以下定理，input=定理陈述(theorem.tex)，output=自然语言证明(proof_*.md)
  2. explain （知识讲解）：instruction=请详细讲解以下数学概念，input=概念信息，output=概念讲解(introduce.en.md)

用法：
    python build_sft.py                                # 默认：../mathpkg/math_pkg_release → ./data/v1_sft.jsonl
    python build_sft.py --release <路径> --out <路径>
    python build_sft.py --task proof --max 200         # 只要证明题（冒烟）
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import re
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RELEASE = SCRIPT_DIR.parent / "mathpkg" / "math_pkg_release"
DEFAULT_OUT = SCRIPT_DIR / "data" / "v1_sft.jsonl"

LONG = "\\\\?\\"


def lp(path) -> str:
    """Windows 长路径前缀，绕过 260 字符限制。"""
    a = os.path.abspath(str(path))
    return LONG + a if not a.startswith(LONG) else a


def read_text(path) -> str:
    with open(lp(path), "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def parse_frontmatter(block: str) -> dict:
    meta = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip().strip("\"'")
    return meta


def split_frontmatter(text: str):
    """返回 (frontmatter dict, body)。没有 frontmatter 时返回 ({}, 全文)。"""
    t = text.lstrip("\ufeff \t\r\n")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", t, re.S)
    if m:
        return parse_frontmatter(m.group(1)), m.group(2).strip()
    return {}, t.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="mathpkg 发布数据 → SFT JSONL")
    ap.add_argument("--release", default=str(DEFAULT_RELEASE),
                    help="mathpkg 发布数据根目录（含 domain/book/.../concept.yaml）")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="输出 JSONL 路径")
    ap.add_argument("--task", default="all", choices=["all", "proof", "explain"],
                    help="只输出指定任务")
    ap.add_argument("--max", type=int, default=0, help="最多输出 N 条（冒烟测试）")
    args = ap.parse_args()

    release = Path(args.release)
    if not release.is_dir():
        raise SystemExit(f"release 目录不存在: {release}（可用 --release 指定）")

    out_path = Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = collections.Counter()
    seen = set()
    n_dups = 0

    # ── 1. 收集概念 ──────────────────────────────────────────────
    concept_yamls = glob.glob(str(release / "**" / "concept.yaml"), recursive=True)
    concepts = {}          # full rel path -> dict
    by_book_id = collections.defaultdict(list)  # (book, cid) -> [paths]
    for y in concept_yamls:
        d = Path(y).parent
        rel = d.relative_to(release)
        parts = rel.parts
        domain, book = parts[0], parts[1]
        cid = d.name
        try:
            yd = yaml.safe_load(read_text(d / "concept.yaml")) or {}
        except Exception:
            yd = {}
            stats["yaml_parse_fail"] += 1
        if not isinstance(yd, dict):
            yd = {}

        intro_body = ""
        if (d / "introduce.en.md").exists():
            _, intro_body = split_frontmatter(read_text(d / "introduce.en.md"))
        tex_body = ""
        if (d / "theorem.tex").exists():
            tex_body = read_text(d / "theorem.tex").strip()

        concept = {
            "path": str(rel).replace(os.sep, "/"),
            "id": cid,
            "domain": domain,
            "book": book,
            "section": "/".join(parts[2:-1]) if len(parts) > 3 else "",
            "title_en": yd.get("title_en", ""),
            "title_zh": yd.get("title_zh", ""),
            "type": yd.get("type", ""),
            "subdomain": yd.get("subdomain", ""),
            "difficulty": yd.get("difficulty", ""),
            "tags": yd.get("tags", []),
            "intro_body": intro_body,
            "tex_body": tex_body,
        }
        concepts[str(rel)] = concept
        by_book_id[(book, cid)].append(rel)

    stats["concepts_total"] = len(concepts)
    stats["concepts_with_intro"] = sum(1 for c in concepts.values() if c["intro_body"])
    stats["concepts_with_tex"] = sum(1 for c in concepts.values() if c["tex_body"])

    # ── 2. 收集证明文件，并按 (book, id) 配对概念 ───────────────
    proof_files = glob.glob(str(release / "**" / "proof_*.md"), recursive=True)
    proof_pairs = []  # (rel, fm, body, book, cid)
    for pf in proof_files:
        d = Path(pf).parent
        rel = d.relative_to(release)
        parts = rel.parts
        if len(parts) < 2:
            continue
        book = parts[1]
        cid = d.name
        fm, body = split_frontmatter(read_text(pf))
        proof_pairs.append((rel, fm, body, book, cid))
    stats["proof_files"] = len(proof_files)

    def find_concept(book: str, cid: str):
        cands = by_book_id.get((book, cid), [])
        return concepts[str(cands[0])] if cands else None

    # ── 3. 生成记录 ──────────────────────────────────────────────
    records = []

    for rel, fm, body, book, cid in proof_pairs:
        concept = find_concept(book, fm.get("of_concept", cid))
        if concept is None:
            stats["proof_no_concept"] += 1
            continue
        if len(body) < 30:
            stats["proof_body_too_short"] += 1
            continue
        statement = concept["tex_body"]
        if len(statement) < 10:
            statement = concept["title_en"]
        if len(statement) < 2:
            stats["proof_no_statement"] += 1
            continue
        meta = {
            "task": "proof",
            "concept_id": concept["id"],
            "title_en": concept["title_en"],
            "title_zh": concept["title_zh"],
            "type": concept["type"],
            "domain": concept["domain"],
            "book": concept["book"],
            "chapter": fm.get("source_chapter", ""),
            "section": fm.get("source_section", ""),
        }
        rec = {
            "task": "proof",
            "instruction": "请证明以下定理。",
            "input": statement,
            "output": body,
            "meta": meta,
        }
        h = hashlib.sha1(("proof|" + str(concept["path"]) + "|" + body).encode("utf-8")).hexdigest()
        if h in seen:
            n_dups += 1
            continue
        seen.add(h)
        if args.task in ("all", "proof"):
            records.append(rec)

    for rel, c in concepts.items():
        if len(c["intro_body"]) < 30:
            stats["intro_body_too_short"] += 1
            continue
        info = c["title_en"]
        if c["title_zh"]:
            info += f"（{c['title_zh']}）"
        if c["type"]:
            info += f"\n类型：{c['type']}"
        if c["domain"]:
            info += f"\n领域：{c['domain']}"
        if c["difficulty"]:
            info += f"\n难度：{c['difficulty']}"
        meta = {
            "task": "explain",
            "concept_id": c["id"],
            "title_en": c["title_en"],
            "title_zh": c["title_zh"],
            "type": c["type"],
            "domain": c["domain"],
            "book": c["book"],
            "section": c["section"],
            "difficulty": c["difficulty"],
        }
        rec = {
            "task": "explain",
            "instruction": "请详细讲解以下数学概念。",
            "input": info,
            "output": c["intro_body"],
            "meta": meta,
        }
        h = hashlib.sha1(("explain|" + rel + "|" + c["intro_body"]).encode("utf-8")).hexdigest()
        if h in seen:
            n_dups += 1
            continue
        seen.add(h)
        if args.task in ("all", "explain"):
            records.append(rec)

    if args.max > 0:
        records = records[: args.max]

    stats["records_total"] = len(records)
    stats["records_proof"] = sum(1 for r in records if r["task"] == "proof")
    stats["records_explain"] = sum(1 for r in records if r["task"] == "explain")
    stats["duplicates_removed"] = n_dups

    by_domain = collections.Counter()
    for r in records:
        by_domain[r["meta"]["domain"]] += 1
    stats["by_domain"] = dict(sorted(by_domain.items()))

    # ── 4. 写出 ──────────────────────────────────────────────────
    with open(lp(out_path), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats_path = out_path.with_suffix(".stats.json")
    with open(lp(stats_path), "w", encoding="utf-8") as f:
        json.dump(dict(stats), f, ensure_ascii=False, indent=2)

    print(f"written {stats['records_total']:,} records -> {out_path}")
    print(json.dumps(dict(stats), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

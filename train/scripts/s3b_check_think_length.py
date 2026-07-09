#!/usr/bin/env python3
"""
s3b_check_think_length.py — SID Routing think token 长度预检
=============================================================

在 Stage 3b 训练启动前运行，用实际 tokenizer 测量 sid_routing_think 的
token 长度分布。如果 max > 128，自动输出建议值 (max + 30)。

用法 (由 run_train.sh 自动调用):
    python3 train/scripts/s3b_check_think_length.py \
        --data_path  data/processed/Beauty/training_sid_routing_train.parquet \
        --model_path model/merged_beauty_model

输出:
    - 终端打印完整统计
    - 最后一行输出推荐 think_max_tokens 值（供 shell 捕获）
    - JSON 报告写入 data_path 同目录下 think_length_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="SID Routing think token 长度预检")
    parser.add_argument("--data_path", type=str, required=True, help="routing train parquet")
    parser.add_argument("--model_path", type=str, required=True, help="tokenizer 所在模型目录")
    parser.add_argument("--default_max", type=int, default=128, help="当前默认 think_max_tokens")
    parser.add_argument("--margin", type=int, default=30, help="超限时在 max 基础上追加的余量")
    args = parser.parse_args()

    # ── 加载 tokenizer ──────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # ── 加载数据 ────────────────────────────────────────────────────
    df = pd.read_parquet(args.data_path)
    total = len(df)

    if "sid_routing_think" not in df.columns:
        print("ERROR: sid_routing_think 列不存在", file=sys.stderr)
        print(str(args.default_max))  # fallback
        sys.exit(1)

    if "cot_steps" in df.columns:
        non_empty = df[df["cot_steps"] > 0]
    else:
        non_empty = df[df["sid_routing_think"].apply(
            lambda x: x is not None and str(x).strip() not in ("", "<think>\n\n</think>")
        )]

    n_empty = total - len(non_empty)
    n_non_empty = len(non_empty)

    print("=" * 60)
    print("  SID Routing Think Token 长度预检")
    print("=" * 60)
    print(f"  数据文件: {args.data_path}")
    print(f"  总样本数: {total}")
    print(f"  空 think:  {n_empty} ({100 * n_empty / total:.1f}%)")
    print(f"  非空 think: {n_non_empty} ({100 * n_non_empty / total:.1f}%)")

    if n_non_empty == 0:
        print("\n  所有样本均为空 think，无需调整")
        print(f"  推荐 think_max_tokens: {args.default_max}")
        print("=" * 60)
        # 写报告
        _write_report(args, total, n_empty, n_non_empty, lengths=[], recommended=args.default_max)
        print(str(args.default_max))
        return

    # ── Tokenize 非空 think ─────────────────────────────────────────
    think_texts = non_empty["sid_routing_think"].astype(str).tolist()
    lengths = []
    for text in think_texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        lengths.append(len(tokens))

    lengths_arr = np.array(lengths)

    max_len = int(lengths_arr.max())
    p99 = int(np.percentile(lengths_arr, 99))
    p95 = int(np.percentile(lengths_arr, 95))
    p50 = int(np.percentile(lengths_arr, 50))
    mean_len = float(lengths_arr.mean())
    over_128 = int((lengths_arr > args.default_max).sum())

    print(f"\n  Token 长度统计 (非空 think, n={n_non_empty}):")
    print(f"    max:  {max_len}")
    print(f"    p99:  {p99}")
    print(f"    p95:  {p95}")
    print(f"    p50:  {p50}")
    print(f"    mean: {mean_len:.1f}")
    print(f"    > {args.default_max}: {over_128} ({100 * over_128 / n_non_empty:.2f}%)")

    # ── 分桶直方图 ──────────────────────────────────────────────────
    bins = [0, 20, 40, 60, 80, 100, 128, 160, 200, 300, 999]
    hist, _ = np.histogram(lengths_arr, bins=bins)
    print(f"\n  分布直方图:")
    for i in range(len(bins) - 1):
        bar = "█" * max(1, int(40 * hist[i] / max(hist.max(), 1)))
        print(f"    [{bins[i]:>3}-{bins[i + 1]:>3}) {hist[i]:>5}  {bar}")

    # ── 推荐值 ──────────────────────────────────────────────────────
    if max_len > args.default_max:
        recommended = max_len + args.margin
        print(f"\n  ⚠ max ({max_len}) > 默认限制 ({args.default_max})")
        print(f"  → 自动调整 think_max_tokens = {max_len} + {args.margin} = {recommended}")
    else:
        recommended = args.default_max
        print(f"\n  ✅ max ({max_len}) ≤ 默认限制 ({args.default_max}), 无需调整")

    print(f"  推荐 think_max_tokens: {recommended}")
    print("=" * 60)

    _write_report(args, total, n_empty, n_non_empty, lengths, recommended)

    # ── 最后一行: 供 shell 捕获 ─────────────────────────────────────
    print(str(recommended))


def _write_report(args, total, n_empty, n_non_empty, lengths, recommended):
    report = {
        "data_path": args.data_path,
        "total_samples": total,
        "empty_think": n_empty,
        "non_empty_think": n_non_empty,
        "default_max": args.default_max,
        "margin": args.margin,
        "recommended_think_max_tokens": recommended,
    }
    if lengths:
        lengths_arr = np.array(lengths)
        report.update({
            "max_tokens": int(lengths_arr.max()),
            "p99_tokens": int(np.percentile(lengths_arr, 99)),
            "p95_tokens": int(np.percentile(lengths_arr, 95)),
            "p50_tokens": int(np.percentile(lengths_arr, 50)),
            "mean_tokens": round(float(lengths_arr.mean()), 2),
            "over_default": int((lengths_arr > args.default_max).sum()),
        })

    report_path = Path(args.data_path).parent / "think_length_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

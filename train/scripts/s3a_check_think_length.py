#!/usr/bin/env python3
"""
s3a_check_think_length.py — RA think token 长度预检
====================================================

RA 任务的 think 文本由 `s3a_train_ra.py` 构造,形态固定:
    "The user is likely to buy items in {categories} category"
(当 row['title'] is None 时,think 为空字符串。)

本脚本读取 `training_RA_train.parquet`,按上述模板**模拟** RA think 的
token 长度分布,输出统计报告 + 推荐 think_max_tokens 值,供 RA CoT
评估或生成配置参考。

设计同 s3b_check_think_length.py:
  - 输出 JSON 报告到 `data/processed/<cat>/think_length_report_ra.json`
  - 终端最后一行打印推荐值,供 shell 捕获
  - 当 max > default_max 时,自动建议 (max + margin)

注意:这里测的是 *S3 监督标签* 的 think 长度,不是模型实际生成长度。
该值只是生成上限的经验参考,不是行为约束。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


RA_THINK_TEMPLATE = "The user is likely to buy items in {categories} category"


def construct_ra_think(row) -> str:
    """复刻 s3a_train_ra.py 中的 think 文本构造逻辑。"""
    title = row.get("title")
    if title is None or (isinstance(title, float) and pd.isna(title)):
        return ""
    categories = row.get("categories", "")
    return RA_THINK_TEMPLATE.format(categories=categories)


def main() -> None:
    parser = argparse.ArgumentParser(description="RA think token 长度预检")
    parser.add_argument("--data_path", type=str, required=True,
                        help="training_RA_train.parquet 路径")
    parser.add_argument("--model_path", type=str, required=True,
                        help="tokenizer 所在模型目录")
    parser.add_argument("--default_max", type=int, default=64,
                        help="RA 默认 think_max_tokens(注意比 routing 的 128 更小)")
    parser.add_argument("--margin", type=int, default=20,
                        help="超限时在 max 基础上追加的余量")
    parser.add_argument("--report_path", type=str, default=None,
                        help="可选;不指定则写入 data_path 同目录")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    df = pd.read_parquet(args.data_path)
    total = len(df)

    if "categories" not in df.columns:
        print("ERROR: training_RA_train.parquet 缺少 categories 列", file=sys.stderr)
        print(str(args.default_max))
        sys.exit(1)

    has_title = "title" in df.columns
    n_empty_label = 0
    think_texts: list[str] = []
    for _, row in df.iterrows():
        think = construct_ra_think(row)
        if think == "":
            n_empty_label += 1
        else:
            think_texts.append(think)

    n_non_empty = len(think_texts)

    print("=" * 60)
    print("  RA Think Token 长度预检")
    print("=" * 60)
    print(f"  数据文件: {args.data_path}")
    print(f"  总样本数: {total}")
    print(f"  has_title 列: {has_title}")
    if has_title:
        print(f"  空 think (title is None):  {n_empty_label} ({100 * n_empty_label / total:.2f}%)")
    print(f"  非空 think: {n_non_empty} ({100 * n_non_empty / total:.2f}%)")

    if n_non_empty == 0:
        print("\n  所有 label think 均为空,使用默认上限")
        print(f"  推荐 think_max_tokens: {args.default_max}")
        print("=" * 60)
        _write_report(args, total, n_empty_label, n_non_empty, lengths=[],
                      recommended=args.default_max)
        print(str(args.default_max))
        return

    # 用 tokenizer 编码,与 s4_prepare_data.encode_prompt_ids 一致
    lengths = []
    for text in think_texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        lengths.append(len(ids))
    arr = np.array(lengths)

    max_len = int(arr.max())
    p99 = int(np.percentile(arr, 99))
    p95 = int(np.percentile(arr, 95))
    p50 = int(np.percentile(arr, 50))
    mean_len = float(arr.mean())
    over_default = int((arr > args.default_max).sum())

    print(f"\n  Token 长度统计 (非空 think, n={n_non_empty}):")
    print(f"    max:  {max_len}")
    print(f"    p99:  {p99}")
    print(f"    p95:  {p95}")
    print(f"    p50:  {p50}")
    print(f"    mean: {mean_len:.1f}")
    print(f"    > {args.default_max}: {over_default} ({100 * over_default / n_non_empty:.2f}%)")

    bins = [0, 10, 20, 30, 40, 50, 64, 80, 100, 128, 160, 999]
    hist, _ = np.histogram(arr, bins=bins)
    print(f"\n  分布直方图:")
    for i in range(len(bins) - 1):
        bar = "█" * max(1, int(40 * hist[i] / max(hist.max(), 1)))
        print(f"    [{bins[i]:>3}-{bins[i + 1]:>3}) {hist[i]:>5}  {bar}")

    if max_len > args.default_max:
        recommended = max_len + args.margin
        print(f"\n  ⚠ max ({max_len}) > 默认 ({args.default_max})")
        print(f"  → 调整 think_max_tokens = {max_len} + {args.margin} = {recommended}")
    else:
        # 即使 max <= default,也保留 p99 + margin 作为一个参考(更紧的上限)
        suggested_tight = int(p99) + args.margin
        recommended = min(args.default_max, suggested_tight)
        # 取两者较紧者,但不小于 max + 10
        recommended = max(recommended, max_len + 10)
        print(f"\n  ✅ max ({max_len}) ≤ 默认 ({args.default_max})")
        print(f"  使用紧贴 p99 的上限: max({max_len}+10, p99({p99})+{args.margin}, ≤{args.default_max}) = {recommended}")

    print(f"  推荐 think_max_tokens: {recommended}")
    print("=" * 60)

    _write_report(args, total, n_empty_label, n_non_empty, lengths, recommended)
    print(str(recommended))


def _write_report(args, total, n_empty, n_non_empty, lengths, recommended):
    report = {
        "format": "ra_think_length_report_v1",
        "data_path": args.data_path,
        "total_samples": total,
        "empty_think_label": n_empty,
        "empty_think_label_ratio": round(n_empty / max(1, total), 6),
        "non_empty_think_label": n_non_empty,
        "default_max": args.default_max,
        "margin": args.margin,
        "recommended_think_max_tokens": int(recommended),
    }
    if lengths:
        arr = np.array(lengths)
        report.update({
            "max_tokens": int(arr.max()),
            "p99_tokens": int(np.percentile(arr, 99)),
            "p95_tokens": int(np.percentile(arr, 95)),
            "p50_tokens": int(np.percentile(arr, 50)),
            "mean_tokens": round(float(arr.mean()), 2),
            "over_default": int((arr > args.default_max).sum()),
        })

    if args.report_path:
        report_path = Path(args.report_path)
    else:
        report_path = Path(args.data_path).parent / "think_length_report_ra.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")


if __name__ == "__main__":
    main()

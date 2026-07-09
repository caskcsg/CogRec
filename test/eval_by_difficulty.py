#!/usr/bin/env python3
"""
test/eval_by_difficulty.py — 统一分层评估（strict row-aligned）
================================================================

功能:
  1. 兼容 no-CoT / CoT 两类日志格式
  2. 兼容 CoT 多行 TARGET 与多段 THINKING k/n
  3. 按日志出现顺序与 row-aligned difficulty labels 对齐
  4. 对缺失 target / 缺失 candidates 的样本给出显式统计

说明:
  - 只接受 row-aligned difficulty 文件。
  - 不再保留 legacy difficulty 兼容逻辑。
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("eval_by_difficulty")

SID_RE = re.compile(r"<\|sid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><s_d_\d+><\|sid_end\|>")
SAMPLE_RE = re.compile(r"SAMPLE\s+(\d+)")
COT_SAMPLE_RE = re.compile(r"CoT-ENHANCED\s+SAMPLE\s+(\d+)")
RANK_RE = re.compile(r"Rank\s+(\d+):\s*score=([\d.eE+-]+).*?(<\|sid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><s_d_\d+><\|sid_end\|>)")
THINK_INLINE_RE = re.compile(r"THINK:\s*(.*)")
THINK_BLOCK_START_RE = re.compile(r"THINKING\s+(\d+)/(\d+):")


# ══════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════
def extract_sid(text: str | None) -> Optional[str]:
    if not text:
        return None
    m = SID_RE.search(text)
    return m.group(0) if m else None


def normalize_text(s: str | None) -> str:
    return "" if s is None else s.replace("\r\n", "\n").replace("\r", "\n")


# ══════════════════════════════════════════════════════════════════════
# 指标工具
# ══════════════════════════════════════════════════════════════════════
def compute_metrics(topk_list: List[List[int]], k_values: List[int]) -> Dict[str, float]:
    n = len(topk_list)
    if n == 0:
        return {f"hit@{k}": 0.0 for k in k_values} | {f"ndcg@{k}": 0.0 for k in [5, 10]}

    results: Dict[str, float] = {}
    for k in k_values:
        hit_sum = sum(any(rank <= k for rank in ranks) for ranks in topk_list)
        results[f"hit@{k}"] = hit_sum / n

    for k in [5, 10]:
        ndcg_sum = 0.0
        for ranks in topk_list:
            valid = [r for r in ranks if r <= k]
            if valid:
                best_rank = min(valid)
                ndcg_sum += 1.0 / np.log2(best_rank + 1)
        results[f"ndcg@{k}"] = ndcg_sum / n
    return results


# ══════════════════════════════════════════════════════════════════════
# 日志解析
# ══════════════════════════════════════════════════════════════════════
def _new_sample(sample_idx: int) -> Dict[str, Any]:
    return {
        "sample_idx": sample_idx,
        "target": None,
        "target_raw": None,
        "candidates": [],
        "thinking_blocks": [],
        "meta": {
            "has_target": False,
            "has_candidates": False,
            "has_nonempty_think": False,
        },
    }


def parse_eval_log(log_file: str) -> List[dict]:
    """
    兼容以下两种格式：
      1) no-CoT:
           ----- SAMPLE 0 -----
           ...
           Rank 1: score=... <SID>
           TARGET: <SID>

      2) CoT:
           ----- CoT-ENHANCED SAMPLE 0 -----
           THINKING 1/5:
           ...
           THINKING 2/5:
           ...
           UNIQUE TOP-10 SID_CANDIDATES:
           Rank 1: score=... -> <SID>
           TARGET:
           <SID>
    """
    samples: List[dict] = []
    current: Optional[dict] = None

    collecting_target = False
    collecting_think = False
    current_think_lines: List[str] = []

    def flush_think_block() -> None:
        nonlocal collecting_think, current_think_lines, current
        if current is None or not collecting_think:
            current_think_lines = []
            collecting_think = False
            return
        text = "\n".join(current_think_lines).strip()
        current["thinking_blocks"].append(text)
        if text:
            current["meta"]["has_nonempty_think"] = True
        current_think_lines = []
        collecting_think = False

    def flush_sample() -> None:
        nonlocal current
        if current is None:
            return
        flush_think_block()
        current["meta"]["has_target"] = current.get("target") is not None
        current["meta"]["has_candidates"] = len(current.get("candidates", [])) > 0
        samples.append(current)
        current = None

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            # 新样本开始
            m = COT_SAMPLE_RE.search(line) or SAMPLE_RE.search(line)
            if m:
                flush_sample()
                current = _new_sample(int(m.group(1)))
                collecting_target = False
                continue

            if current is None:
                continue

            # 如果上一行是裸 TARGET:，则优先在后续行里找 SID
            if collecting_target:
                sid = extract_sid(line)
                if sid:
                    current["target"] = sid
                    current["target_raw"] = line.strip()
                    collecting_target = False
                    continue
                # 遇到空行 / 分隔线 / 下一段标题时，放弃继续等待 target
                if (not line.strip()) or ("-----" in line) or THINK_BLOCK_START_RE.search(line) or ("UNIQUE TOP-10 SID_CANDIDATES" in line):
                    collecting_target = False

            # TARGET: 单行或多行
            if "TARGET:" in line:
                sid = extract_sid(line)
                if sid:
                    current["target"] = sid
                    current["target_raw"] = line.strip()
                    collecting_target = False
                else:
                    # 兼容下一行才写 SID 的格式
                    collecting_target = True
                flush_think_block()
                continue

            # THINKING k/n: 块起始
            if THINK_BLOCK_START_RE.search(line):
                flush_think_block()
                collecting_think = True
                current_think_lines = []
                continue

            # THINK: inline 格式
            m_inline = THINK_INLINE_RE.search(line)
            if m_inline:
                flush_think_block()
                text = (m_inline.group(1) or "").strip()
                current["thinking_blocks"].append(text)
                if text:
                    current["meta"]["has_nonempty_think"] = True
                continue

            # 候选 Rank 行
            m_rank = RANK_RE.search(line)
            if m_rank:
                flush_think_block()
                current["candidates"].append(
                    {
                        "rank": int(m_rank.group(1)),
                        "score": float(m_rank.group(2)),
                        "sid": m_rank.group(3),
                    }
                )
                continue

            # 终止 think block 的一些标题
            if collecting_think and (
                ("UNIQUE TOP-10 SID_CANDIDATES" in line)
                or ("USER INPUT:" in line)
                or ("PROGRESS REPORT" in line)
                or ("Final CoT Hit Rate Results" in line)
            ):
                flush_think_block()
                continue

            # 正在收集 think 文本
            if collecting_think:
                current_think_lines.append(line)

    flush_sample()

    n_no_target = sum(1 for s in samples if not s["meta"]["has_target"])
    n_no_cands = sum(1 for s in samples if not s["meta"]["has_candidates"])
    n_nonempty_think = sum(1 for s in samples if s["meta"]["has_nonempty_think"])

    log.info("解析日志: %d 个样本", len(samples))
    log.info("  无 target 样本: %d", n_no_target)
    log.info("  无 candidates 样本: %d", n_no_cands)
    log.info("  非空 think 样本: %d", n_nonempty_think)
    return samples


def compute_topk_from_parsed(parsed_samples: List[dict]) -> List[List[int]]:
    topk_list: List[List[int]] = []
    for sample in parsed_samples:
        target = sample.get("target")
        if not target:
            topk_list.append([])
            continue
        ranks = [cand["rank"] for cand in sample.get("candidates", []) if cand.get("sid") == target]
        topk_list.append(ranks if ranks else [])
    return topk_list


# ══════════════════════════════════════════════════════════════════════
# label 加载
# ══════════════════════════════════════════════════════════════════════

def load_difficulty_labels(difficulty_file: str, test_parquet: str) -> Dict[int, dict]:
    payload = json.loads(Path(difficulty_file).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("difficulty_file 必须是 JSON object，且格式为 row_aligned")

    fmt = payload.get("format")
    if fmt != "row_aligned":
        raise ValueError(f"只支持 row_aligned difficulty 文件，当前 format={fmt!r}")

    samples = payload.get("samples", [])
    df = pd.read_parquet(test_parquet)

    if len(samples) != len(df):
        raise ValueError(
            f"row_aligned 标签与测试集长度不一致: labels={len(samples)} vs parquet={len(df)}"
        )

    labels: Dict[int, dict] = {}
    for idx, item in enumerate(samples):
        label_sid = str(item.get("sample_id")) if item.get("sample_id") is not None else None
        if "sample_id" in df.columns and label_sid is not None:
            row_sid = str(df.iloc[idx]["sample_id"])
            if row_sid != label_sid:
                raise ValueError(f"sample_id 对齐失败: row={idx} parquet={row_sid} label={label_sid}")

        labels[idx] = {
            "sample_id": label_sid,
            "difficulty": item.get("difficulty", "unknown"),
            "cot_steps": int(item.get("cot_steps", -1)),
            "anchor_sid": item.get("anchor_sid"),
            "routing_total_cost": int(item.get("routing_total_cost", 0)),
        }
    return labels


# ══════════════════════════════════════════════════════════════════════
# 分层报告
# ══════════════════════════════════════════════════════════════════════
def _align_topk_with_labels(topk_list: List[List[int]], labels: Dict[int, dict]) -> List[List[int]]:
    expected = len(labels)
    actual = len(topk_list)
    if actual == expected:
        return topk_list
    if actual < expected:
        log.warning("解析到的样本数少于 labels：parsed=%d labels=%d；自动补空样本", actual, expected)
        return topk_list + ([[]] * (expected - actual))
    log.warning("解析到的样本数多于 labels：parsed=%d labels=%d；自动截断到 labels 长度", actual, expected)
    return topk_list[:expected]


def generate_stratified_report(topk_list: List[List[int]], labels: Dict[int, dict], output_dir: str | None = None) -> dict:
    k_values = [1, 5, 10]
    topk_list = _align_topk_with_labels(topk_list, labels)

    difficulty_groups: Dict[str, List[List[int]]] = defaultdict(list)
    step_groups: Dict[int, List[List[int]]] = defaultdict(list)

    for idx, topk in enumerate(topk_list):
        label = labels.get(idx, {"difficulty": "unknown", "cot_steps": -1})
        difficulty_groups[label["difficulty"]].append(topk)
        step_groups[int(label["cot_steps"])].append(topk)

    report = {
        "total_samples": len(topk_list),
        "by_difficulty": {},
        "by_cot_steps": {},
    }

    print("\n" + "=" * 80)
    print("  分层评估报告（正式版）")
    print("=" * 80)

    print(f"\n{'难度':<10} {'样本数':>8} ", end="")
    for k in k_values:
        print(f"  {'hit@'+str(k):>8}", end="")
    for k in [5, 10]:
        print(f"  {'ndcg@'+str(k):>9}", end="")
    print()
    print("-" * 80)

    for diff in ["easy", "medium", "hard", "plain", "unknown", "error"]:
        if diff not in difficulty_groups:
            continue
        group = difficulty_groups[diff]
        metrics = compute_metrics(group, k_values)
        report["by_difficulty"][diff] = {"n": len(group), **metrics}
        print(f"{diff:<10} {len(group):>8} ", end="")
        for k in k_values:
            print(f"  {metrics[f'hit@{k}']:>8.4f}", end="")
        for k in [5, 10]:
            print(f"  {metrics[f'ndcg@{k}']:>9.4f}", end="")
        print()

    total_metrics = compute_metrics(topk_list, k_values)
    report["total_metrics"] = total_metrics
    print("-" * 80)
    print(f"{'ALL':<10} {len(topk_list):>8} ", end="")
    for k in k_values:
        print(f"  {total_metrics[f'hit@{k}']:>8.4f}", end="")
    for k in [5, 10]:
        print(f"  {total_metrics[f'ndcg@{k}']:>9.4f}", end="")
    print()

    print(f"\n{'CoT步数':<10} {'样本数':>8} ", end="")
    for k in k_values:
        print(f"  {'hit@'+str(k):>8}", end="")
    for k in [5, 10]:
        print(f"  {'ndcg@'+str(k):>9}", end="")
    print()
    print("-" * 80)
    for ns in sorted(step_groups.keys()):
        if ns < 0:
            continue
        group = step_groups[ns]
        metrics = compute_metrics(group, k_values)
        report["by_cot_steps"][str(ns)] = {"n": len(group), **metrics}
        print(f"{(str(ns)+'步'):<10} {len(group):>8} ", end="")
        for k in k_values:
            print(f"  {metrics[f'hit@{k}']:>8.4f}", end="")
        for k in [5, 10]:
            print(f"  {metrics[f'ndcg@{k}']:>9.4f}", end="")
        print()
    print("=" * 80)

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        report_file = out_path / "stratified_metrics.json"
        report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("保存分层报告: %s", report_file)
    return report


# ══════════════════════════════════════════════════════════════════════
# Think 长度分析
# ══════════════════════════════════════════════════════════════════════
def analyze_think_lengths(parsed_samples: List[dict], labels: Dict[int, dict], output_dir: str | None = None) -> dict:
    """
    基于 parse_eval_log 的结果分析 think 长度。

    定义三个长度统计：
      - max_len:    单样本多条 thinking 中最长的一条
      - mean_len:   单样本多条 thinking 的平均长度
      - nonempty_n: 单样本非空 thinking 条数

    重要修复（2026-04-24）：
      - nonempty_n 只是文本解析得到的“非空 THINKING block 数”，不能作为
        fig4_4 的 canonical 思考步数使用。
      - fig4_4 需要的横轴应与 stratified_metrics.json['by_cot_steps'] 使用
        同一套 row-aligned 标签，即 labels[idx]['cot_steps']。
      - 因此本函数在每个 difficulty bucket 中额外写出 cot_steps_values，
        且与 mean_len_values / max_len_values 严格逐样本对齐。
      - 同时额外写出 think_length_by_cot_steps.json，方便直接审计。
    """
    n_expected = len(labels)
    samples = parsed_samples[:n_expected] + ([None] * max(0, n_expected - len(parsed_samples)))

    grouped_max: Dict[str, List[int]] = defaultdict(list)
    grouped_mean: Dict[str, List[float]] = defaultdict(list)
    grouped_nonempty: Dict[str, List[int]] = defaultdict(list)
    grouped_cot_steps: Dict[str, List[int]] = defaultdict(list)
    grouped_sample_idx: Dict[str, List[int]] = defaultdict(list)
    grouped_sample_id: Dict[str, List[str]] = defaultdict(list)
    grouped_blocks: Dict[str, List[int]] = defaultdict(list)

    step_grouped_max: Dict[int, List[int]] = defaultdict(list)
    step_grouped_mean: Dict[int, List[float]] = defaultdict(list)
    step_grouped_nonempty: Dict[int, List[int]] = defaultdict(list)
    step_grouped_difficulty: Dict[int, List[str]] = defaultdict(list)
    step_grouped_sample_idx: Dict[int, List[int]] = defaultdict(list)
    step_grouped_sample_id: Dict[int, List[str]] = defaultdict(list)
    step_grouped_blocks: Dict[int, List[int]] = defaultdict(list)

    n_invalid_cot_steps = 0

    def _as_int(value: Any, default: int = -1) -> int:
        try:
            if value is None:
                return default
            return int(value)
        except Exception:
            return default

    def _stats_numeric(values: List[Any], *, integer: bool = False) -> Dict[str, Any]:
        if not values:
            return {"mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0 if integer else 0.0}
        arr = np.array(values, dtype=np.int32 if integer else np.float32)
        out = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "max": int(arr.max()) if integer else float(arr.max()),
        }
        return out

    for idx in range(n_expected):
        label_obj = labels.get(idx, {})
        label = str(label_obj.get("difficulty", "unknown"))
        cot_steps = _as_int(label_obj.get("cot_steps", -1), -1)
        sample_id = label_obj.get("sample_id")
        sample_id_str = "" if sample_id is None else str(sample_id)

        sample = samples[idx]
        thinks = [] if sample is None else [normalize_text(t).strip() for t in sample.get("thinking_blocks", [])]
        lengths = [len(t) for t in thinks]
        nonempty_lengths = [x for x in lengths if x > 0]

        max_len = max(lengths) if lengths else 0
        mean_len = float(np.mean(lengths)) if lengths else 0.0
        nonempty_n = len(nonempty_lengths)

        grouped_max[label].append(int(max_len))
        grouped_mean[label].append(float(mean_len))
        grouped_nonempty[label].append(int(nonempty_n))
        grouped_cot_steps[label].append(int(cot_steps))
        grouped_sample_idx[label].append(int(idx))
        grouped_sample_id[label].append(sample_id_str)
        grouped_blocks[label].extend(int(x) for x in lengths)

        # Canonical step grouping for fig4_4 and audit.  Keep invalid labels out
        # of step-grouped plots/reports rather than falling back to parser blocks.
        if cot_steps >= 0:
            step_grouped_max[cot_steps].append(int(max_len))
            step_grouped_mean[cot_steps].append(float(mean_len))
            step_grouped_nonempty[cot_steps].append(int(nonempty_n))
            step_grouped_difficulty[cot_steps].append(label)
            step_grouped_sample_idx[cot_steps].append(int(idx))
            step_grouped_sample_id[cot_steps].append(sample_id_str)
            step_grouped_blocks[cot_steps].extend(int(x) for x in lengths)
        else:
            n_invalid_cot_steps += 1

    print("\n" + "=" * 72)
    print("  模型实际生成 Think 长度分布（row-aligned cot_steps 修复版）")
    print("=" * 72)
    print(f"{'难度':<10} {'样本数':>8}  {'max均值':>10}  {'max中位':>10}  {'max-p90':>10}  {'非空均值':>10}  {'cot均值':>10}")
    print("-" * 72)

    report: Dict[str, Any] = {}
    for diff in ["easy", "medium", "hard", "plain", "unknown", "error"]:
        if diff not in grouped_max or not grouped_max[diff]:
            continue

        arr_cot = np.array(grouped_cot_steps[diff], dtype=np.int32)
        valid_cot = arr_cot[arr_cot >= 0]
        cot_stats = _stats_numeric(valid_cot.tolist(), integer=True) if valid_cot.size else {
            "mean": 0.0, "median": 0.0, "p90": 0.0, "max": -1,
        }

        stats = {
            "n": int(len(grouped_max[diff])),
            "max_len": _stats_numeric(grouped_max[diff], integer=True),
            "mean_len": _stats_numeric(grouped_mean[diff], integer=False),
            "nonempty_count": _stats_numeric(grouped_nonempty[diff], integer=True),
            # Canonical row-aligned CoT steps from difficulty labels.
            # This is the field fig4_4 should use.  It is deliberately separate
            # from nonempty_count_values, which is only a parser-derived block count.
            "cot_steps": cot_stats,
            # Full empirical distributions for paper figures.
            # All *_values below are one value per sample and row-aligned within
            # this difficulty bucket.
            "sample_idx_values": [int(x) for x in grouped_sample_idx[diff]],
            "sample_id_values": [str(x) for x in grouped_sample_id[diff]],
            "max_len_values": [int(x) for x in grouped_max[diff]],
            "mean_len_values": [float(x) for x in grouped_mean[diff]],
            "nonempty_count_values": [int(x) for x in grouped_nonempty[diff]],
            "cot_steps_values": [int(x) for x in grouped_cot_steps[diff]],
            "block_len_values": [int(x) for x in grouped_blocks.get(diff, [])],
        }
        report[diff] = stats
        print(
            f"{diff:<10} {stats['n']:>8}"
            f"  {stats['max_len']['mean']:>10.1f}"
            f"  {stats['max_len']['median']:>10.0f}"
            f"  {stats['max_len']['p90']:>10.0f}"
            f"  {stats['nonempty_count']['mean']:>10.2f}"
            f"  {stats['cot_steps']['mean']:>10.2f}"
        )
    print("=" * 72)

    # Direct step-grouped payload.  This is not required by older analysis code,
    # but it gives an auditable canonical source for fig4_4 and prevents future
    # confusion between cot_steps and nonempty_count.
    by_step_report: Dict[str, Any] = {}
    if step_grouped_mean:
        print("\n" + "=" * 72)
        print("  Think 长度按 canonical cot_steps 分组")
        print("=" * 72)
        print(f"{'cot_steps':<10} {'样本数':>8}  {'mean均值':>10}  {'mean中位':>10}  {'mean-p90':>10}  {'max均值':>10}")
        print("-" * 72)
        for step in sorted(step_grouped_mean.keys()):
            rec = {
                "n": int(len(step_grouped_mean[step])),
                "max_len": _stats_numeric(step_grouped_max[step], integer=True),
                "mean_len": _stats_numeric(step_grouped_mean[step], integer=False),
                "nonempty_count": _stats_numeric(step_grouped_nonempty[step], integer=True),
                "sample_idx_values": [int(x) for x in step_grouped_sample_idx[step]],
                "sample_id_values": [str(x) for x in step_grouped_sample_id[step]],
                "difficulty_values": [str(x) for x in step_grouped_difficulty[step]],
                "max_len_values": [int(x) for x in step_grouped_max[step]],
                "mean_len_values": [float(x) for x in step_grouped_mean[step]],
                "nonempty_count_values": [int(x) for x in step_grouped_nonempty[step]],
                "block_len_values": [int(x) for x in step_grouped_blocks.get(step, [])],
            }
            by_step_report[str(step)] = rec
            print(
                f"{str(step):<10} {rec['n']:>8}"
                f"  {rec['mean_len']['mean']:>10.1f}"
                f"  {rec['mean_len']['median']:>10.1f}"
                f"  {rec['mean_len']['p90']:>10.1f}"
                f"  {rec['max_len']['mean']:>10.1f}"
            )
        print("=" * 72)

    if n_invalid_cot_steps > 0:
        log.warning("存在无效 cot_steps 标签样本: %d；这些样本不会进入 think_length_by_cot_steps.json", n_invalid_cot_steps)

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        out_file = out_path / "think_length_analysis.json"
        out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("保存 think 长度分析: %s", out_file)

        step_file = out_path / "think_length_by_cot_steps.json"
        step_file.write_text(json.dumps(by_step_report, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("保存按 cot_steps 分组的 think 长度分析: %s", step_file)

        audit_file = out_path / "think_length_alignment_debug.json"
        audit_payload = {
            "label_count": int(n_expected),
            "parsed_samples": int(len(parsed_samples)),
            "invalid_cot_steps": int(n_invalid_cot_steps),
            "difficulty_counts": {k: len(v) for k, v in grouped_mean.items()},
            "cot_step_counts": {str(k): len(v) for k, v in step_grouped_mean.items()},
            "note": "cot_steps_values are copied from row-aligned difficulty labels and are aligned with mean_len_values within each difficulty bucket.",
        }
        audit_file.write_text(json.dumps(audit_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("保存 think 长度对齐调试信息: %s", audit_file)

    return report


# ══════════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="按难度 / cot_steps 分层评估 (统一版)")
    parser.add_argument("--eval_log", type=str, required=True)
    parser.add_argument("--difficulty_file", type=str, required=True)
    parser.add_argument("--test_parquet", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--analyze_think", action="store_true")
    parser.add_argument("--strict_length_check", action="store_true", help="若日志样本数与 labels 不一致则报错")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    labels = load_difficulty_labels(args.difficulty_file, args.test_parquet)
    parsed = parse_eval_log(args.eval_log)

    if args.strict_length_check and len(parsed) != len(labels):
        raise ValueError(f"日志样本数与 labels 长度不一致: parsed={len(parsed)} labels={len(labels)}")

    topk_list = compute_topk_from_parsed(parsed)
    report = generate_stratified_report(topk_list, labels, args.output_dir)

    # 保存解析调试信息，方便后续排查
    if args.output_dir:
        out_path = Path(args.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        debug_payload = {
            "parsed_samples": len(parsed),
            "label_count": len(labels),
            "samples_without_target": sum(1 for s in parsed if not s["meta"]["has_target"]),
            "samples_without_candidates": sum(1 for s in parsed if not s["meta"]["has_candidates"]),
            "samples_with_nonempty_think": sum(1 for s in parsed if s["meta"]["has_nonempty_think"]),
        }
        (out_path / "parse_debug.json").write_text(json.dumps(debug_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("保存解析调试信息: %s", out_path / "parse_debug.json")

    if args.analyze_think:
        analyze_think_lengths(parsed, labels, args.output_dir)


if __name__ == "__main__":
    main()



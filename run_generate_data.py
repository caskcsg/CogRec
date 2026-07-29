#!/usr/bin/env python3
"""
run_generate_data.py — 离线数据主入口
======================================
统一负责正式版所需的全部离线资产生成。

阶段:
  download   下载原始数据
  sid        raw_to_sids.py      — review/meta → full sequence → Embedding → RQ-kMeans 产码
  hnsw       hnsw_and_splits.py  — 认知地图 → 训练切片 → Routing full data + CoT subset

说明:
  - eval trie 不再在本脚本中构建
  - eval trie 改为训练/评估阶段按需生成，保存在 data/processed/<category>/ 下
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.config import (  # type: ignore
    DATASETS,
    RAW_DIR,
    PROC_DIR,
    LOG_DIR,
    SID_LAYER_PREFIX,
    category_processed_dir,
)

PHASES = ["download", "sid", "hnsw"]

PHASE_CONFIG = {
    "download": {"script": "data/download_amazon_dataset.py", "desc": "下载原始数据"},
    "sid": {"script": "data/raw_to_sids.py", "desc": "review/meta → full sequence → Embedding → RQ-kMeans 产码"},
    "hnsw": {"script": "data/hnsw_and_splits.py", "desc": "认知地图 → 训练切片 → Routing full data + CoT subset"},
}

PHASE_PREREQUISITES = {
    "download": [],
    "sid": [],
    "hnsw": [
        "codebooks.npz",
        "sid_codes.npy",
        "item_embeddings.npy",
        "item_mapping.json",
        "{cat}.pretrain.json",
        "{cat}_sequential_data.txt",
    ],
}

PHASE_OUTPUTS = {
    "download": [],
    "sid": [
        "user_sequences.jsonl",
        "all_item_seqs.json",
        "id_mapping.json",
        "items_for_embedding.jsonl",
        "item_mapping.json",
        "item_embeddings.npy",
        "codebooks.npz",
        "sid_codes.npy",
        "sid_quality_report.json",
        "dataset_alignment_report.json",
        "{cat}.pretrain.json",
        "{cat}_sequential_data.txt",
    ],
    "hnsw": [
        "cognitive_map.json",
        "training_align_data_train.parquet",
        "training_prediction_sid_data_train.parquet",
        "training_RA_train.parquet",
        "training_sid_routing_train.parquet",
        "training_RA_test_cot_eval.parquet",
        "training_sid_routing_test_cot_eval.parquet",
        "test_difficulty_labels.json",
        "test_difficulty_labels_cot_eval.json",
        "routing_quality_report.json",
        "cot_eval_subset_ids.json",
        "hnsw_pipeline_report.json",
    ],
}


def setup_logging(log_name: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{log_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
    root.addHandler(console)

    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)-8s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    root.addHandler(fh)
    return log_file


log = logging.getLogger("run_generate_data")
_file_handler = None


def _get_file_handler():
    global _file_handler
    if _file_handler is None:
        for h in logging.getLogger().handlers:
            if isinstance(h, logging.FileHandler):
                _file_handler = h
                break
    return _file_handler


def run_script(script: str, extra_args: list | None = None, description: str = "") -> float:
    cmd = [sys.executable, "-u", str(ROOT / script)]
    if extra_args:
        cmd.extend(extra_args)

    log.info("")
    log.info("▶ %s", description)
    log.info("  脚本: %s", script)
    log.debug("  命令: %s", " ".join(cmd))

    fh = _get_file_handler()
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    assert proc.stdout is not None
    line_count = 0
    for line in proc.stdout:
        line = line.rstrip("\n\r")
        print(f"  │ {line}", flush=True)
        if fh:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fh.stream.write(f"[{ts}] [CHILD   ] {line}\n")
            fh.stream.flush()
        line_count += 1

    proc.wait()
    elapsed = time.time() - t0
    if proc.returncode != 0:
        log.error("  ✗ 失败 (exit=%d, %.1fs, %d 行输出)", proc.returncode, elapsed, line_count)
        raise RuntimeError(f"脚本失败: {script} (exit code {proc.returncode})")

    log.info("  ✔ 完成 (%.1fs, %d 行输出)", elapsed, line_count)
    return elapsed


def _expand_pattern(pattern: str, category: str) -> str:
    return pattern.replace("{cat}", category)


def check_prerequisites(phase: str, categories: list[str]) -> bool:
    required = PHASE_PREREQUISITES.get(phase, [])
    if not required:
        return True

    missing = []
    for cat in categories:
        cat_dir = category_processed_dir(cat)
        for pattern in required:
            fname = _expand_pattern(pattern, cat)
            if not (cat_dir / fname).exists():
                missing.append(f"{cat}/{fname}")

    if missing:
        log.error("前置文件缺失 (%d 个):", len(missing))
        for m in missing[:20]:
            log.error("  ✗ data/processed/%s", m)
        return False

    log.info("  前置检查通过")
    return True


def verify_outputs(phase: str, categories: list[str]):
    expected = PHASE_OUTPUTS.get(phase, [])
    if not expected:
        return

    log.info("  产物验证:")
    total_size = 0
    all_ok = True
    for cat in categories:
        cat_dir = category_processed_dir(cat)
        for pattern in expected:
            fname = _expand_pattern(pattern, cat)
            path = cat_dir / fname
            if path.exists():
                sz = path.stat().st_size
                total_size += sz
                log.debug("    ✓ %s/%s (%.1f MB)", cat, fname, sz / 1e6)
            else:
                log.warning("    ✗ %s/%s 不存在", cat, fname)
                all_ok = False

    if all_ok:
        log.info("    ✓ 全部产物验证通过 (%.1f MB)", total_size / 1e6)
    else:
        log.warning("    ⚠ 部分产物缺失")


def verify_dataset_stats(categories: list[str]) -> None:
    log.info("  benchmark 统计校验:")
    all_ok = True

    for cat in categories:
        cfg = DATASETS[cat]
        cat_dir = category_processed_dir(cat)

        id_mapping_path = cat_dir / "id_mapping.json"
        all_item_seqs_path = cat_dir / "all_item_seqs.json"
        pretrain_path = cat_dir / f"{cat}.pretrain.json"
        sequential_path = cat_dir / f"{cat}_sequential_data.txt"
        align_path = cat_dir / "dataset_alignment_report.json"

        required = [id_mapping_path, all_item_seqs_path, pretrain_path, sequential_path, align_path]
        missing = [p for p in required if not p.exists()]
        if missing:
            all_ok = False
            log.warning("    ✗ %s 缺少校验文件: %s", cat, ", ".join(p.name for p in missing))
            continue

        with open(id_mapping_path, "r", encoding="utf-8") as f:
            id_mapping = json.load(f)
        with open(all_item_seqs_path, "r", encoding="utf-8") as f:
            all_item_seqs = json.load(f)
        with open(pretrain_path, "r", encoding="utf-8") as f:
            pretrain = json.load(f)
        with open(align_path, "r", encoding="utf-8") as f:
            align = json.load(f)

        users = len(id_mapping["user2id"]) - 1
        items = len(id_mapping["item2id"]) - 1
        interactions = sum(len(seq) for seq in all_item_seqs.values())

        seq_users = 0
        seq_interactions = 0
        with open(sequential_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    seq_users += 1
                    seq_interactions += max(len(parts) - 1, 0)

        ok = True
        if users != cfg["expected_users"]:
            ok = False
            log.warning("    ✗ %s users=%d != expected=%d", cat, users, cfg["expected_users"])
        if items != cfg["expected_items"]:
            ok = False
            log.warning("    ✗ %s items=%d != expected=%d", cat, items, cfg["expected_items"])
        if interactions != cfg["expected_interactions"]:
            ok = False
            log.warning("    ✗ %s interactions=%d != expected=%d", cat, interactions, cfg["expected_interactions"])
        if len(pretrain) != cfg["expected_items"]:
            ok = False
            log.warning("    ✗ %s pretrain_items=%d != expected=%d", cat, len(pretrain), cfg["expected_items"])
        if seq_users != cfg["expected_users"]:
            ok = False
            log.warning("    ✗ %s sequential_users=%d != expected=%d", cat, seq_users, cfg["expected_users"])
        if seq_interactions != cfg["expected_interactions"]:
            ok = False
            log.warning("    ✗ %s sequential_interactions=%d != expected=%d", cat, seq_interactions, cfg["expected_interactions"])

        if align.get("sequence_mode") != "full_sequence":
            ok = False
            log.warning("    ✗ %s dataset_alignment_report sequence_mode != full_sequence", cat)
        if align.get("truncated") is not False:
            ok = False
            log.warning("    ✗ %s dataset_alignment_report truncated != False", cat)

        if ok:
            avg_len = interactions / max(users, 1)
            log.info("    ✓ %s 对齐通过 | users=%d, items=%d, interactions=%d, avg=%.2f", cat, users, items, interactions, avg_len)
        else:
            all_ok = False

    if all_ok:
        log.info("    ✓ 全部类目统计校验通过")
    else:
        raise RuntimeError("benchmark 统计校验失败")


def _build_args(phase: str, args) -> list[str]:
    extra = []
    if args.category != "all":
        extra.extend(["--category", args.category])

    if phase == "sid":
        if args.skip_embedding:
            extra.append("--skip-embedding")
    elif phase == "hnsw":
        extra.extend([
            "--cot-subset-ratio", str(args.cot_subset_ratio),
            "--cot-subset-size", str(args.cot_subset_size),
            "--cot-subset-seed", str(args.cot_subset_seed),
        ])
        if args.force_routing_data:
            extra.append("--force-routing-data")
    return extra


def dry_run_check(categories: list[str]):
    log.info("╔══════════════════════════════════════════════╗")
    log.info("║      DRY RUN — 仅检查文件状态               ║")
    log.info("╚══════════════════════════════════════════════╝")

    for cat in categories:
        log.info("")
        log.info("━━━ %s ━━━", cat)
        cat_dir = category_processed_dir(cat)
        raw_dir = RAW_DIR / cat

        sections = [
            ("Raw", list(raw_dir.glob("*.json.gz")) if raw_dir.exists() else []),
            ("SID", [
                cat_dir / "user_sequences.jsonl",
                cat_dir / "all_item_seqs.json",
                cat_dir / "id_mapping.json",
                cat_dir / "item_mapping.json",
                cat_dir / "item_embeddings.npy",
                cat_dir / "codebooks.npz",
                cat_dir / "sid_quality_report.json",
                cat_dir / "dataset_alignment_report.json",
                cat_dir / f"{cat}.pretrain.json",
                cat_dir / f"{cat}_sequential_data.txt",
            ]),
            ("HNSW", [
                cat_dir / "cognitive_map.json",
                cat_dir / f"edges_{SID_LAYER_PREFIX[0]}.json",
                cat_dir / "routing_quality_report.json",
                cat_dir / "cot_eval_subset_ids.json",
                cat_dir / "hnsw_pipeline_report.json",
            ]),
            ("Splits", [
                cat_dir / "training_align_data_train.parquet",
                cat_dir / "training_prediction_sid_data_train.parquet",
                cat_dir / "training_RA_train.parquet",
                cat_dir / "training_sid_routing_train.parquet",
                cat_dir / "training_RA_test_cot_eval.parquet",
                cat_dir / "training_sid_routing_test_cot_eval.parquet",
            ]),
        ]

        for label, files in sections:
            for f in files:
                if f.exists():
                    log.info("  ✓ [%-6s] %s (%.1f MB)", label, f.name, f.stat().st_size / 1e6)
                else:
                    log.info("  ✗ [%-6s] %s (missing)", label, f.name)


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def _disk_free_gb(path: Path) -> float:
    total, used, free = shutil.disk_usage(str(path))
    return free / (1024 ** 3)


def run_phase(phase: str, categories: list[str], args) -> float:
    cfg = PHASE_CONFIG[phase]
    log.info("")
    log.info("━" * 72)
    log.info("  阶段: %s — %s", phase.upper(), cfg["desc"])
    log.info("━" * 72)

    if not check_prerequisites(phase, categories):
        raise RuntimeError(f"阶段 [{phase}] 前置条件不满足")

    elapsed = run_script(cfg["script"], _build_args(phase, args), cfg["desc"])
    verify_outputs(phase, categories)

    if phase == "sid":
        verify_dataset_stats(categories)

    return elapsed


def parse_args():
    p = argparse.ArgumentParser(
        description="数据处理全流程一键运行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
阶段:
  download   下载原始数据
  sid        raw_to_sids.py      — review/meta → full sequence → Embedding → 产码
  hnsw       hnsw_and_splits.py  — 认知地图 → 切片 → Routing full data + CoT subset
""",
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument("--phase", choices=PHASES, help="仅运行指定阶段")
    group.add_argument("--from", dest="from_phase", choices=PHASES, help="从指定阶段开始运行到结束")

    p.add_argument("--category", choices=["Beauty", "Sports", "Toys", "all"], default="all")
    p.add_argument("--skip-embedding", action="store_true", help="跳过 Embedding 生成 (传递给 raw_to_sids.py)")
    p.add_argument("--cot-subset-ratio", type=float, default=0.1, help="CoT eval 子集比例")
    p.add_argument("--cot-subset-size", type=int, default=-1, help="固定 CoT eval 子集大小；>0 时覆盖 ratio")
    p.add_argument("--cot-subset-seed", type=int, default=42, help="CoT eval 子集随机种子")
    p.add_argument("--force-routing-data", action="store_true", help="强制重建 routing full data、subset")
    p.add_argument("--dry-run", action="store_true", help="仅检查文件状态")
    return p.parse_args()


def main():
    args = parse_args()
    if args.phase:
        phases_to_run = [args.phase]
    elif args.from_phase:
        phases_to_run = PHASES[PHASES.index(args.from_phase):]
    else:
        phases_to_run = list(PHASES)

    categories = list(DATASETS.keys()) if args.category == "all" else [args.category]
    label = args.phase or args.from_phase or "full"
    log_file = setup_logging(f"run_generate_data_{label}")

    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║                    数据处理流水线                          ║")
    log.info("╠══════════════════════════════════════════════════════════════╣")
    log.info("║  序列模式:       %-42s ║", "full sequence / no truncate")
    log.info("║  阶段:           %-42s ║", " → ".join(phases_to_run))
    log.info("║  类目:           %-42s ║", ", ".join(categories))
    log.info("║  subset_ratio:   %-42s ║", f"{args.cot_subset_ratio:.0%}")
    log.info("║  subset_size:    %-42s ║", str(args.cot_subset_size))
    log.info("║  subset_seed:    %-42s ║", str(args.cot_subset_seed))
    log.info("║  force routing:  %-42s ║", str(args.force_routing_data))
    log.info("║  跳过 EMB:       %-42s ║", str(args.skip_embedding))
    log.info("║  日志文件:       %-42s ║", log_file.name)
    log.info("╚══════════════════════════════════════════════════════════════╝")

    if args.dry_run:
        dry_run_check(categories)
        return

    t_total = time.time()
    phase_times = {}
    for phase in phases_to_run:
        t_phase = time.time()
        run_phase(phase, categories, args)
        phase_times[phase] = time.time() - t_phase

    total_elapsed = time.time() - t_total
    log.info("")
    log.info("═" * 72)
    log.info("  执行报告")
    log.info("═" * 72)
    log.info("  ┌─── 阶段耗时 ───")
    for phase, t in phase_times.items():
        log.info("  │ %-12s %6.1fs  (%4.1f min)", phase, t, t / 60)
    log.info("  │ %-12s %6.1fs  (%4.1f min)", "TOTAL", total_elapsed, total_elapsed / 60)
    log.info("  └─────────────────")

    log.info("  ┌─── 磁盘占用 ───")
    total_data_mb = 0
    for cat in categories:
        sz = _dir_size_mb(category_processed_dir(cat))
        total_data_mb += sz
        log.info("  │ %-12s %8.1f MB", cat, sz)
    log.info("  │ %-12s %8.1f MB", "数据合计", total_data_mb)
    log.info("  │ %-12s %8.1f GB", "磁盘剩余", _disk_free_gb(ROOT))
    log.info("  └─────────────────")

    log.info("  ┌─── 产物统计 ───")
    for cat in categories:
        cat_dir = category_processed_dir(cat)
        n_pkl = len(list(cat_dir.glob("exact_trie_*.pkl")))
        log.info(
            "  │ %s: %d parquet, %d json, %d npy, %d trie",
            cat,
            len(list(cat_dir.glob("training_*.parquet"))),
            len(list(cat_dir.glob("*.json"))),
            len(list(cat_dir.glob("*.npy"))),
            n_pkl,
        )
    log.info("  └─────────────────")

    log.info("")
    log.info("  日志文件: %s", log_file)
    log.info("  下一步:")
    for cat in categories:
        log.info("    bash run_train.sh --stage all --category %s --gpus 2", cat)
    log.info("🎉 全部完成!")


if __name__ == "__main__":
    main()

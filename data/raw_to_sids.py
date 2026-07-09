#!/usr/bin/env python3
"""
data/raw_to_sids.py — 原始数据 → SID 产码全流程
==============================================
当前协议基于 Amazon2014 数据。

流程:
  Step 1: review → full sequence + reference ID
  Step 2: meta → 字段提取 + embedding 文本
  Step 3: Embedding 生成
  Step 4: RQ-kMeans 产码
  Step 5: 产物组装
"""

from __future__ import annotations

import argparse
import ast
import gzip
import html
import json
import logging
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import sys
import torch
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.config import (  # type: ignore
    DATASETS, RAW_DIR, PROC_DIR,
    K_CORE,
    NUM_SID_LAYERS, CODEBOOK_SIZE, RQ_N_ITER,
    SID_LAYER_PREFIX, SID_BEGIN_TOKEN, SID_END_TOKEN,
    EMB_MODEL_NAME, EMB_BATCH_SIZE, EMB_MAX_SEQ_LENGTH,
    setup_logging, format_sid,
)

log = logging.getLogger("raw_to_sids")


def step_banner(step_num: int, total: int, title: str):
    log.info("")
    log.info("━" * 72)
    log.info("  Step %d/%d: %s", step_num, total, title)
    log.info("━" * 72)


def _elapsed(start_time: float) -> float:
    return time.time() - start_time


def _round_seconds(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 3)


def _format_timing_table(timing_dict: dict, indent: str = "    ") -> list[str]:
    lines = []
    for key, value in timing_dict.items():
        if isinstance(value, dict):
            lines.append(f"{indent}{key}:")
            lines.extend(_format_timing_table(value, indent=indent + "  "))
        else:
            lines.append(f"{indent}{key}: {value}")
    return lines


def iter_gz_records(path: Path):
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidate = (
                line.replace("true", "True")
                    .replace("false", "False")
                    .replace("null", "None")
            )
            try:
                yield ast.literal_eval(candidate)
            except Exception:
                continue


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)

    text = html.unescape(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_scalar_text(value: Any) -> str:
    return _clean_text(value)


def _normalize_description(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        parts = [_clean_text(x) for x in value]
        return " ".join([p for p in parts if p]).strip()
    return _clean_text(value)


def _normalize_categories(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, list):
        if len(value) == 0:
            return ""

        if isinstance(value[0], list):
            for path in value:
                cleaned = [_clean_text(x) for x in path if _clean_text(x)]
                if cleaned:
                    return " > ".join(cleaned).strip()
            return ""

        cleaned = [_clean_text(x) for x in value if _clean_text(x)]
        return " > ".join(cleaned).strip()

    return _clean_text(value)


def _compute_item_field_stats(
    items_data: dict,
    valid_item_count: int,
    meta_items_found: int | None,
    meta_missing_items: int | None,
    source: str,
) -> dict:
    title_empty = 0
    description_empty = 0
    categories_empty = 0
    all_fields_empty = 0
    no_embedding_text = 0

    for fields in items_data.values():
        title = _normalize_scalar_text(fields.get("title", ""))
        description = _normalize_scalar_text(fields.get("description", ""))
        categories = _normalize_scalar_text(fields.get("categories", ""))

        if not title:
            title_empty += 1
        if not description:
            description_empty += 1
        if not categories:
            categories_empty += 1
        if not title and not description and not categories:
            all_fields_empty += 1
        if not any([title, description, categories]):
            no_embedding_text += 1

    stats = {
        "source": source,
        "valid_items": int(valid_item_count),
        "items_with_records": int(len(items_data)),
        "meta_items_found": None if meta_items_found is None else int(meta_items_found),
        "meta_missing_items": None if meta_missing_items is None else int(meta_missing_items),
        "title_empty_items": int(title_empty),
        "description_empty_items": int(description_empty),
        "categories_empty_items": int(categories_empty),
        "all_fields_empty_items": int(all_fields_empty),
        "no_embedding_text_items": int(no_embedding_text),
        "title_nonempty_items": int(len(items_data) - title_empty),
        "description_nonempty_items": int(len(items_data) - description_empty),
        "categories_nonempty_items": int(len(items_data) - categories_empty),
    }

    denom = max(valid_item_count, 1)
    stats["title_empty_rate"] = round(title_empty / denom, 6)
    stats["description_empty_rate"] = round(description_empty / denom, 6)
    stats["categories_empty_rate"] = round(categories_empty / denom, 6)
    stats["all_fields_empty_rate"] = round(all_fields_empty / denom, 6)
    stats["no_embedding_text_rate"] = round(no_embedding_text / denom, 6)
    if meta_missing_items is not None:
        stats["meta_missing_rate"] = round(meta_missing_items / denom, 6)

    return stats


def _log_item_field_stats(stats: dict):
    valid_items = max(int(stats.get("valid_items", 0)), 1)
    log.info("  字段质量统计:")
    if stats.get("meta_items_found") is not None and stats.get("meta_missing_items") is not None:
        log.info(
            "    Meta 命中: %d | 无 Meta: %d (%.2f%%)",
            stats["meta_items_found"],
            stats["meta_missing_items"],
            stats.get("meta_missing_rate", 0.0) * 100,
        )
    log.info(
        "    title 空: %d/%d (%.2f%%)",
        stats["title_empty_items"], valid_items, stats["title_empty_rate"] * 100,
    )
    log.info(
        "    description 空: %d/%d (%.2f%%)",
        stats["description_empty_items"], valid_items, stats["description_empty_rate"] * 100,
    )
    log.info(
        "    categories 空: %d/%d (%.2f%%)",
        stats["categories_empty_items"], valid_items, stats["categories_empty_rate"] * 100,
    )
    log.info(
        "    三字段全空: %d/%d (%.2f%%)",
        stats["all_fields_empty_items"], valid_items, stats["all_fields_empty_rate"] * 100,
    )
    log.info(
        "    embedding 回退文本: %d/%d (%.2f%%)",
        stats["no_embedding_text_items"], valid_items, stats["no_embedding_text_rate"] * 100,
    )


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def step1_build_sequences_and_ids(category: str) -> tuple[dict, dict, dict]:
    step_banner(1, 5, "review → full sequence + ID")
    t0 = time.time()
    timings = {}

    cfg = DATASETS[category]
    review_path = RAW_DIR / category / cfg["review_filename"]
    out_dir = PROC_DIR / category
    out_dir.mkdir(parents=True, exist_ok=True)

    user_seq_path = out_dir / "user_sequences.jsonl"
    all_item_seqs_path = out_dir / "all_item_seqs.json"
    id_mapping_path = out_dir / "id_mapping.json"

    if not review_path.exists():
        raise FileNotFoundError(f"Review 文件不存在: {review_path}")

    t_load = time.time()
    reviews = []
    for rec in iter_gz_records(review_path):
        user = rec.get("reviewerID", "")
        item = rec.get("asin", "")
        ts = rec.get("unixReviewTime", 0)
        if not user or not item:
            continue
        try:
            ts = int(ts)
        except Exception:
            ts = 0
        reviews.append((user, item, ts))
    timings["load_reviews_sec"] = _round_seconds(_elapsed(t_load))
    log.info("  加载 reviews: %d 条", len(reviews))

    t_group = time.time()
    item_seqs_with_time = defaultdict(list)
    item_counter = Counter()

    for user, item, ts in reviews:
        item_seqs_with_time[user].append((item, ts))
        item_counter[item] += 1

    all_item_seqs = {}
    for user, item_time in item_seqs_with_time.items():
        item_time.sort(key=lambda x: x[1])
        all_item_seqs[user] = [x[0] for x in item_time]

    timings["group_and_sort_sec"] = _round_seconds(_elapsed(t_group))
    log.info("  构建 full sequence: %d 用户", len(all_item_seqs))

    t_verify = time.time()
    user_len_counter = {u: len(seq) for u, seq in all_item_seqs.items()}
    min_user_deg = min(user_len_counter.values()) if user_len_counter else 0
    min_item_deg = min(item_counter.values()) if item_counter else 0

    if min_user_deg < K_CORE:
        raise ValueError(f"{category}: user min degree={min_user_deg} < K_CORE={K_CORE}")
    if min_item_deg < K_CORE:
        raise ValueError(f"{category}: item min degree={min_item_deg} < K_CORE={K_CORE}")

    timings["verify_kcore_sec"] = _round_seconds(_elapsed(t_verify))
    log.info("  K-core 校验通过: min_user_deg=%d, min_item_deg=%d", min_user_deg, min_item_deg)

    t_remap = time.time()
    id_mapping = {
        "user2id": {"[PAD]": 0},
        "item2id": {"[PAD]": 0},
        "id2user": ["[PAD]"],
        "id2item": ["[PAD]"],
    }

    for user, items in all_item_seqs.items():
        if user not in id_mapping["user2id"]:
            id_mapping["user2id"][user] = len(id_mapping["id2user"])
            id_mapping["id2user"].append(user)

        for item in items:
            if item not in id_mapping["item2id"]:
                id_mapping["item2id"][item] = len(id_mapping["id2item"])
                id_mapping["id2item"].append(item)

    timings["remap_ids_sec"] = _round_seconds(_elapsed(t_remap))

    n_users = len(id_mapping["user2id"]) - 1
    n_items = len(id_mapping["item2id"]) - 1
    n_interactions = sum(len(seq) for seq in all_item_seqs.values())

    log.info("  Remap 完成: users=%d, items=%d, interactions=%d", n_users, n_items, n_interactions)

    t_bench = time.time()
    expected_users = cfg.get("expected_users")
    expected_items = cfg.get("expected_items")
    expected_interactions = cfg.get("expected_interactions")

    if expected_users is not None and n_users != expected_users:
        raise ValueError(f"{category}: users={n_users} != expected_users={expected_users}")
    if expected_items is not None and n_items != expected_items:
        raise ValueError(f"{category}: items={n_items} != expected_items={expected_items}")
    if expected_interactions is not None and n_interactions != expected_interactions:
        raise ValueError(f"{category}: interactions={n_interactions} != expected_interactions={expected_interactions}")

    timings["benchmark_check_sec"] = _round_seconds(_elapsed(t_bench))
    log.info("  统计校验通过")

    t_save = time.time()
    with open(user_seq_path, "w", encoding="utf-8") as f:
        for raw_user in id_mapping["id2user"][1:]:
            f.write(json.dumps({
                "user_id": raw_user,
                "sequence": all_item_seqs[raw_user],
            }, ensure_ascii=False) + "\n")

    _save_json(all_item_seqs_path, all_item_seqs)
    _save_json(id_mapping_path, id_mapping)

    timings["write_outputs_sec"] = _round_seconds(_elapsed(t_save))
    timings["total_sec"] = _round_seconds(_elapsed(t0))

    stats = {
        "n_users": int(n_users),
        "n_items": int(n_items),
        "n_interactions": int(n_interactions),
        "min_user_deg": int(min_user_deg),
        "min_item_deg": int(min_item_deg),
        "expected_users": int(expected_users),
        "expected_items": int(expected_items),
        "expected_interactions": int(expected_interactions),
        "timing": timings,
    }

    log.info("  输出: %s, %s, %s", user_seq_path.name, all_item_seqs_path.name, id_mapping_path.name)
    log.info("  Step 1 耗时细分:")
    for line in _format_timing_table(timings):
        log.info(line)

    return all_item_seqs, id_mapping, stats


def load_step1_artifacts(category: str) -> tuple[dict, dict]:
    out_dir = PROC_DIR / category
    all_item_seqs_path = out_dir / "all_item_seqs.json"
    id_mapping_path = out_dir / "id_mapping.json"

    if not all_item_seqs_path.exists():
        raise FileNotFoundError(f"缺少 Step 1 产物: {all_item_seqs_path}")
    if not id_mapping_path.exists():
        raise FileNotFoundError(f"缺少 Step 1 产物: {id_mapping_path}")

    all_item_seqs = _load_json(all_item_seqs_path)
    id_mapping = _load_json(id_mapping_path)
    return all_item_seqs, id_mapping


def step2_extract_items(category: str, all_item_seqs: dict, id_mapping: dict) -> tuple[list[str], dict, dict, dict, dict]:
    step_banner(2, 5, "meta → 字段提取")
    t0 = time.time()
    timings = {}

    cfg = DATASETS[category]
    meta_path = RAW_DIR / category / cfg["meta_filename"]
    out_dir = PROC_DIR / category
    emb_path = out_dir / "items_for_embedding.jsonl"

    if not meta_path.exists():
        raise FileNotFoundError(f"Meta 文件不存在: {meta_path}")

    item2id = id_mapping["item2id"]
    valid_items = set(item2id.keys()) - {"[PAD]"}
    sorted_asins = id_mapping["id2item"][1:]
    asin2intid = {asin: int(item2id[asin]) for asin in valid_items}

    t_scan = time.time()
    log.info("  扫描 meta 文件...")
    items_data = {}
    seen_meta_items = set()
    duplicate_meta_records = 0

    for rec in iter_gz_records(meta_path):
        asin = rec.get("asin", "")
        if asin not in valid_items:
            continue
        if asin in seen_meta_items:
            duplicate_meta_records += 1

        title = _normalize_scalar_text(rec.get("title"))
        description = _normalize_description(rec.get("description"))
        categories = _normalize_categories(rec.get("categories"))

        items_data[asin] = {
            "title": title,
            "description": description,
            "categories": categories,
        }
        seen_meta_items.add(asin)

    timings["scan_meta_sec"] = _round_seconds(_elapsed(t_scan))

    t_fill = time.time()
    missing = valid_items - set(items_data.keys())
    if missing:
        log.warning("  %d 个商品无 Meta 信息, 使用空字段回填", len(missing))
        for asin in missing:
            items_data[asin] = {
                "title": "",
                "description": "",
                "categories": "",
            }
    timings["fill_missing_meta_sec"] = _round_seconds(_elapsed(t_fill))

    field_stats = _compute_item_field_stats(
        items_data=items_data,
        valid_item_count=len(valid_items),
        meta_items_found=len(seen_meta_items),
        meta_missing_items=len(missing),
        source="meta_scan",
    )
    _log_item_field_stats(field_stats)

    if set(items_data.keys()) != valid_items:
        raise ValueError("Step 2 items_data keys 与 id_mapping.item2id 不一致")

    t_write = time.time()
    with open(emb_path, "w", encoding="utf-8") as f:
        for asin in sorted_asins:
            d = items_data[asin]
            parts = [p for p in [d["title"], d["description"], d["categories"]] if p]
            text = " ".join(parts).strip() or "No embedding text available."
            f.write(json.dumps({
                "item_id": asin,
                "text": text,
            }, ensure_ascii=False) + "\n")
    timings["write_embedding_text_sec"] = _round_seconds(_elapsed(t_write))
    timings["total_sec"] = _round_seconds(_elapsed(t0))

    log.info("  提取完成: %d items", len(items_data))
    if duplicate_meta_records:
        log.info("  Meta 重复记录: %d (按最后一次覆盖)", duplicate_meta_records)
    log.info("  输出: %s", emb_path.name)
    log.info("  Step 2 耗时细分:")
    for line in _format_timing_table(timings):
        log.info(line)

    step2_stats = {
        "timing": timings,
        "duplicate_meta_records": int(duplicate_meta_records),
        "field_stats": field_stats,
        "n_items": int(len(sorted_asins)),
    }
    return sorted_asins, items_data, asin2intid, field_stats, step2_stats


def recover_step2_from_pretrain(category: str, id_mapping: dict) -> tuple[list[str], dict, dict, dict]:
    out_dir = PROC_DIR / category
    pretrain_path = out_dir / f"{category}.pretrain.json"
    if not pretrain_path.exists():
        raise FileNotFoundError(f"缺少 Step 2 恢复所需文件: {pretrain_path}")

    pretrain = _load_json(pretrain_path)
    sorted_asins = id_mapping["id2item"][1:]
    item2id = id_mapping["item2id"]
    asin2intid = {asin: int(item2id[asin]) for asin in sorted_asins}

    items_data = {}
    for asin in sorted_asins:
        iid = str(asin2intid[asin])
        fields = pretrain.get(iid, {})
        items_data[asin] = {
            "title": fields.get("title", ""),
            "description": fields.get("description", ""),
            "categories": fields.get("categories", ""),
        }

    field_stats = _compute_item_field_stats(
        items_data=items_data,
        valid_item_count=len(sorted_asins),
        meta_items_found=None,
        meta_missing_items=None,
        source="recovered_from_pretrain",
    )
    return sorted_asins, items_data, asin2intid, field_stats


def step3_embeddings(category: str, sorted_asins: list[str]) -> tuple[np.ndarray, dict]:
    step_banner(3, 5, "Embedding 生成")
    t0 = time.time()
    timings = {}

    out_dir = PROC_DIR / category
    emb_matrix_path = out_dir / "item_embeddings.npy"
    texts_path = out_dir / "items_for_embedding.jsonl"

    t_load = time.time()
    item_ids, texts = [], []
    with open(texts_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            item_ids.append(rec["item_id"])
            texts.append(rec["text"])
    timings["load_texts_sec"] = _round_seconds(_elapsed(t_load))

    if item_ids != sorted_asins:
        raise ValueError("Embedding 文本顺序与 Step 2 / id_mapping 不一致")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("  文本条数: %d | 模型: %s | 设备: %s | batch=%d",
             len(texts), EMB_MODEL_NAME, device, EMB_BATCH_SIZE)

    t_model = time.time()
    model = SentenceTransformer(EMB_MODEL_NAME, device=device)
    model.max_seq_length = EMB_MAX_SEQ_LENGTH
    if device == "cuda":
        model.half()
    timings["model_load_sec"] = _round_seconds(_elapsed(t_model))

    t_encode = time.time()
    embeddings = model.encode(
        texts,
        batch_size=EMB_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    timings["encode_sec"] = _round_seconds(_elapsed(t_encode))

    t_save = time.time()
    np.save(emb_matrix_path, embeddings)
    timings["save_embeddings_sec"] = _round_seconds(_elapsed(t_save))
    timings["total_sec"] = _round_seconds(_elapsed(t0))

    log.info("  输出: %s (shape=%s, %.1f MB)",
             emb_matrix_path.name, embeddings.shape, emb_matrix_path.stat().st_size / 1e6)
    log.info("  Step 3 耗时细分:")
    for line in _format_timing_table(timings):
        log.info(line)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embeddings, {"timing": timings}


def step3_load_existing(category: str) -> tuple[np.ndarray, dict]:
    t0 = time.time()
    emb_path = PROC_DIR / category / "item_embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(f"--skip-embedding 但找不到 {emb_path}")
    embeddings = np.load(emb_path)
    timings = {
        "load_existing_embeddings_sec": _round_seconds(_elapsed(t0)),
        "total_sec": _round_seconds(_elapsed(t0)),
    }
    log.info("  [skip-embedding] 加载已有: %s (shape=%s)", emb_path.name, embeddings.shape)
    return embeddings, {"timing": timings}


def _run_rq_kmeans(embeddings: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[dict]]:
    n_items, dim = embeddings.shape
    codes = np.zeros((n_items, NUM_SID_LAYERS), dtype=np.int32)
    codebooks = []
    residuals = embeddings.astype(np.float32).copy()
    layer_stats = []

    for layer in range(NUM_SID_LAYERS):
        log.info(
            "    Layer %d/%d: K=%d, dim=%d, iter=%d ...",
            layer + 1, NUM_SID_LAYERS, CODEBOOK_SIZE, dim, RQ_N_ITER,
        )
        t_layer = time.time()

        kmeans = faiss.Kmeans(
            d=dim,
            k=CODEBOOK_SIZE,
            niter=RQ_N_ITER,
            verbose=False,
            gpu=False,
            spherical=False,
            seed=42,
        )
        kmeans.train(residuals)
        _, I = kmeans.index.search(residuals, 1)

        code = I.flatten()
        codes[:, layer] = code
        centroids = kmeans.centroids.copy()
        codebooks.append(centroids)
        residuals = residuals - centroids[code]

        unique_count = len(np.unique(code))
        layer_seconds = _round_seconds(_elapsed(t_layer))
        layer_stats.append({
            "layer": int(layer + 1),
            "seconds": layer_seconds,
            "utilized_codes": int(unique_count),
            "utilization_rate": round(unique_count / CODEBOOK_SIZE, 6),
        })
        log.info(
            "    Layer %d/%d: %.1fs | 利用率 %d/%d (%.1f%%)",
            layer + 1, NUM_SID_LAYERS, layer_seconds,
            unique_count, CODEBOOK_SIZE, unique_count / CODEBOOK_SIZE * 100,
        )

    return codes, codebooks, layer_stats


def _compute_quality_report(
    category: str,
    embeddings: np.ndarray,
    codes: np.ndarray,
    codebooks: list[np.ndarray],
    n_users: int,
    n_items_total: int,
    field_stats: dict | None = None,
    timing_summary: dict | None = None,
    quantization_layer_timings: list[dict] | None = None,
) -> dict:
    n_items = embeddings.shape[0]
    report = {
        "category": category,
        "n_items": int(n_items),
        "n_users": int(n_users),
        "n_items_total": int(n_items_total),
        "embedding_dim": int(embeddings.shape[1]),
        "num_layers": NUM_SID_LAYERS,
        "codebook_size": CODEBOOK_SIZE,
        "rq_n_iter": RQ_N_ITER,
        "sid_format": f"{SID_BEGIN_TOKEN}<s_a_X><s_b_Y><s_c_Z><s_d_W>{SID_END_TOKEN}",
        "layers": [],
    }

    for layer in range(NUM_SID_LAYERS):
        unique, counts = np.unique(codes[:, layer], return_counts=True)
        arr = counts.astype(float)
        n = len(arr)
        gini = 0.0
        if n > 0 and arr.sum() > 0:
            sorted_arr = np.sort(arr)
            index = np.arange(1, n + 1)
            gini = float((2 * (index * sorted_arr).sum() / (n * arr.sum())) - (n + 1) / n)

        report["layers"].append({
            "layer": int(layer + 1),
            "prefix": SID_LAYER_PREFIX[layer],
            "utilized_codes": int(len(unique)),
            "utilization_rate": round(len(unique) / CODEBOOK_SIZE, 4),
            "max_cluster_size": int(counts.max()),
            "min_cluster_size": int(counts.min()),
            "mean_cluster_size": round(float(counts.mean()), 1),
            "std_cluster_size": round(float(counts.std()), 1),
            "gini_coefficient": round(gini, 4),
        })

    emb_f32 = embeddings.astype(np.float32)
    recon = np.zeros_like(emb_f32, dtype=np.float32)
    for layer in range(NUM_SID_LAYERS):
        recon += codebooks[layer][codes[:, layer]]

    mse = float(np.mean(np.sum((emb_f32 - recon) ** 2, axis=1)))
    emb_norm = np.linalg.norm(emb_f32, axis=1)
    recon_norm = np.linalg.norm(recon, axis=1)
    cosine_num = np.sum(emb_f32 * recon, axis=1)
    cosines = cosine_num / (emb_norm * recon_norm + 1e-8)

    report["reconstruction"] = {
        "mse": round(mse, 6),
        "cosine_mean": round(float(cosines.mean()), 4),
        "cosine_std": round(float(cosines.std()), 4),
        "cosine_min": round(float(cosines.min()), 4),
        "cosine_p5": round(float(np.percentile(cosines, 5)), 4),
        "cosine_p50": round(float(np.percentile(cosines, 50)), 4),
        "cosine_p95": round(float(np.percentile(cosines, 95)), 4),
        "sample_size": int(n_items),
        "full_scan": True,
    }

    sid_tuples = [tuple(codes[i].tolist()) for i in range(n_items)]
    sid_counter = Counter(sid_tuples)
    n_unique = len(sid_counter)
    collision_sizes = [c for c in sid_counter.values() if c > 1]

    report["collision"] = {
        "unique_sids": int(n_unique),
        "collision_rate": round(1.0 - n_unique / n_items, 4),
        "max_collision_size": int(max(sid_counter.values())),
        "collision_groups": int(len(collision_sizes)),
        "items_in_collision": int(sum(collision_sizes)),
    }

    theoretical_capacity = CODEBOOK_SIZE ** NUM_SID_LAYERS
    report["capacity"] = {
        "theoretical": int(theoretical_capacity),
        "actual_items": int(n_items),
        "utilization": round(n_items / theoretical_capacity, 6),
    }

    if field_stats is not None:
        report["metadata_quality"] = field_stats
    if timing_summary is not None:
        report["timing"] = timing_summary
    if quantization_layer_timings is not None:
        report["quantization_timing"] = quantization_layer_timings

    return report


def step4_quantize(category: str, embeddings: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], dict]:
    step_banner(4, 5, "RQ-kMeans 产码")
    t0 = time.time()

    out_dir = PROC_DIR / category
    log.info("  输入: %d items, %d 维", embeddings.shape[0], embeddings.shape[1])
    log.info("  配置: %d 层 × %d 聚类 × %d 轮迭代", NUM_SID_LAYERS, CODEBOOK_SIZE, RQ_N_ITER)

    t_rq = time.time()
    codes, codebooks, layer_stats = _run_rq_kmeans(embeddings)
    rq_seconds = _round_seconds(_elapsed(t_rq))

    t_save = time.time()
    cb_path = out_dir / "codebooks.npz"
    np.savez(cb_path, **{SID_LAYER_PREFIX[i]: codebooks[i] for i in range(NUM_SID_LAYERS)})

    codes_path = out_dir / "sid_codes.npy"
    np.save(codes_path, codes)
    save_seconds = _round_seconds(_elapsed(t_save))

    stats = {
        "timing": {
            "rq_kmeans_sec": rq_seconds,
            "save_codebooks_and_codes_sec": save_seconds,
            "total_sec": _round_seconds(_elapsed(t0)),
        },
        "layer_timing": layer_stats,
    }

    log.info("  输出: %s, %s", cb_path.name, codes_path.name)
    log.info("  Step 4 耗时细分:")
    for line in _format_timing_table(stats["timing"]):
        log.info(line)

    return codes, codebooks, stats


def step5_assemble(
    category: str,
    all_item_seqs: dict,
    id_mapping: dict,
    sorted_asins: list[str],
    items_data: dict,
    asin2intid: dict,
    codes: np.ndarray,
    codebooks: list[np.ndarray],
    embeddings: np.ndarray,
    field_stats: dict | None = None,
    timing_summary: dict | None = None,
    quantization_layer_timings: list[dict] | None = None,
) -> dict:
    step_banner(5, 5, "组装最终产物")
    t0 = time.time()
    step5_timing = {}

    out_dir = PROC_DIR / category
    item2id = id_mapping["item2id"]
    user2id = id_mapping["user2id"]
    id2user = id_mapping["id2user"]
    id2item = id_mapping["id2item"]

    expected_n_users = len(id2user) - 1
    expected_n_items = len(id2item) - 1
    expected_n_interactions = sum(len(seq) for seq in all_item_seqs.values())

    t_mapping = time.time()
    item_mapping = {}
    for i, asin in enumerate(sorted_asins):
        sid_str = format_sid(codes[i].tolist())
        item_mapping[asin] = {
            "int_id": int(asin2intid[asin]),
            "sid": sid_str,
        }

    mapping_path = out_dir / "item_mapping.json"
    _save_json(mapping_path, item_mapping)
    step5_timing["write_item_mapping_sec"] = _round_seconds(_elapsed(t_mapping))
    log.info("  %s: %d 条", mapping_path.name, len(item_mapping))

    t_pretrain = time.time()
    pretrain = {}
    for asin in sorted_asins:
        int_id = asin2intid[asin]
        fields = items_data[asin]
        pretrain[str(int_id)] = {
            "title": fields["title"],
            "description": fields["description"],
            "categories": fields["categories"],
            "sid": item_mapping[asin]["sid"],
        }

    pretrain_path = out_dir / f"{category}.pretrain.json"
    _save_json(pretrain_path, pretrain)
    step5_timing["write_pretrain_sec"] = _round_seconds(_elapsed(t_pretrain))
    log.info("  %s: %d items", pretrain_path.name, len(pretrain))

    t_seq = time.time()
    seq_out_path = out_dir / f"{category}_sequential_data.txt"

    sequential_user_count = 0
    sequential_interactions = 0

    with open(seq_out_path, "w", encoding="utf-8") as fout:
        for raw_user in id2user[1:]:
            uid = int(user2id[raw_user])
            raw_seq = all_item_seqs[raw_user]
            int_seq = [str(item2id[asin]) for asin in raw_seq]
            fout.write(f"{uid} " + " ".join(int_seq) + "\n")
            sequential_user_count += 1
            sequential_interactions += len(int_seq)

    step5_timing["write_sequential_data_sec"] = _round_seconds(_elapsed(t_seq))
    log.info("  %s: %d 用户, %d 交互, 平均 %.2f 条/用户",
             seq_out_path.name,
             sequential_user_count,
             sequential_interactions,
             sequential_interactions / max(sequential_user_count, 1))

    t_align = time.time()
    cfg = DATASETS[category]
    alignment_report = {
        "category": category,
        "sequence_mode": "full_sequence",
        "truncated": False,
        "sampling_applied": False,
        "reference_ids_reused": True,
        "expected": {
            "users": int(cfg["expected_users"]),
            "items": int(cfg["expected_items"]),
            "interactions": int(cfg["expected_interactions"]),
        },
        "actual": {
            "users_from_id_mapping": int(expected_n_users),
            "items_from_id_mapping": int(expected_n_items),
            "interactions_from_all_item_seqs": int(expected_n_interactions),
            "pretrain_items": int(len(pretrain)),
            "sequential_users": int(sequential_user_count),
            "sequential_interactions": int(sequential_interactions),
        },
        "id_sources": {
            "user_ids": "id_mapping.json -> user2id / id2user",
            "item_ids": "id_mapping.json -> item2id / id2item",
            "pretrain_keys": "item2id",
            "sequence_first_column": "user2id",
            "sequence_item_ids": "item2id",
        },
    }

    if expected_n_users != cfg["expected_users"]:
        raise ValueError(f"{category}: id_mapping users={expected_n_users} != expected={cfg['expected_users']}")
    if expected_n_items != cfg["expected_items"]:
        raise ValueError(f"{category}: id_mapping items={expected_n_items} != expected={cfg['expected_items']}")
    if expected_n_interactions != cfg["expected_interactions"]:
        raise ValueError(f"{category}: all_item_seqs interactions={expected_n_interactions} != expected={cfg['expected_interactions']}")

    if len(pretrain) != cfg["expected_items"]:
        raise ValueError(f"{category}: pretrain items={len(pretrain)} != expected={cfg['expected_items']}")
    if sequential_user_count != cfg["expected_users"]:
        raise ValueError(f"{category}: sequential users={sequential_user_count} != expected={cfg['expected_users']}")
    if sequential_interactions != cfg["expected_interactions"]:
        raise ValueError(f"{category}: sequential interactions={sequential_interactions} != expected={cfg['expected_interactions']}")

    alignment_report_path = out_dir / "dataset_alignment_report.json"
    _save_json(alignment_report_path, alignment_report)
    step5_timing["write_alignment_report_sec"] = _round_seconds(_elapsed(t_align))
    log.info("  %s: 校验通过", alignment_report_path.name)

    t_report = time.time()
    merged_timing = dict(timing_summary or {})
    merged_timing["step5"] = {k: v for k, v in step5_timing.items()}
    merged_timing["step5"]["total_sec"] = _round_seconds(_elapsed(t0))

    report = _compute_quality_report(
        category=category,
        embeddings=embeddings,
        codes=codes,
        codebooks=codebooks,
        n_users=expected_n_users,
        n_items_total=len(sorted_asins),
        field_stats=field_stats,
        timing_summary=merged_timing,
        quantization_layer_timings=quantization_layer_timings,
    )

    report_path = out_dir / "sid_quality_report.json"
    _save_json(report_path, report)

    step5_timing["write_quality_report_sec"] = _round_seconds(_elapsed(t_report))
    step5_timing["total_sec"] = _round_seconds(_elapsed(t0))

    log.info("")
    log.info("  ┌─── %s 产码摘要 ────────────────────────────────", category)
    log.info("  │ users=%d | items=%d | interactions=%d",
             expected_n_users, expected_n_items, expected_n_interactions)
    log.info("  │ pretrain items=%d | sequential users=%d | sequential interactions=%d",
             len(pretrain), sequential_user_count, sequential_interactions)
    log.info("  │ SID 唯一数: %d | 碰撞率 %.2f%%",
             report["collision"]["unique_sids"],
             report["collision"]["collision_rate"] * 100)
    log.info("  │ full sequence | no truncate | no sampling")
    log.info("  └──────────────────────────────────────────────")

    log.info("  Step 5 耗时细分:")
    for line in _format_timing_table(step5_timing):
        log.info(line)

    return {
        "timing": step5_timing,
        "report": report,
        "alignment": alignment_report,
    }


def process_category(category: str, skip_embedding: bool = False, from_step: int = 1):
    log.info("")
    log.info("═" * 72)
    log.info("  类目: %s | skip_embedding=%s | from_step=%d", category, skip_embedding, from_step)
    log.info("═" * 72)

    t_total = time.time()
    stage_timing = {}
    field_stats = None

    if from_step <= 1:
        all_item_seqs, id_mapping, step1_stats = step1_build_sequences_and_ids(category)
        stage_timing["step1"] = step1_stats["timing"]
    else:
        log.info("  [跳过 Step 1, 从缓存恢复]")
        all_item_seqs, id_mapping = load_step1_artifacts(category)
        stage_timing["step1"] = {"skipped": True, "source": "loaded_from_cache"}

    if from_step <= 2:
        sorted_asins, items_data, asin2intid, field_stats, step2_stats = step2_extract_items(
            category,
            all_item_seqs,
            id_mapping,
        )
        stage_timing["step2"] = step2_stats["timing"]
    else:
        log.info("  [跳过 Step 2, 从 pretrain 恢复]")
        sorted_asins, items_data, asin2intid, field_stats = recover_step2_from_pretrain(category, id_mapping)
        stage_timing["step2"] = {"skipped": True, "source": "recovered_from_pretrain"}
        _log_item_field_stats(field_stats)

    if skip_embedding:
        log.info("")
        log.info("━" * 72)
        log.info("  Step 3/5: [跳过 — 使用已有 Embedding]")
        log.info("━" * 72)
        embeddings, step3_stats = step3_load_existing(category)
    elif from_step <= 3:
        embeddings, step3_stats = step3_embeddings(category, sorted_asins)
    else:
        log.info("  [跳过 Step 3, 从缓存加载]")
        embeddings, step3_stats = step3_load_existing(category)
    stage_timing["step3"] = step3_stats["timing"]

    if embeddings.shape[0] != len(sorted_asins):
        raise ValueError(
            f"Embedding 行数 ({embeddings.shape[0]}) != item 数 ({len(sorted_asins)})"
        )

    if from_step <= 4:
        codes, codebooks, step4_stats = step4_quantize(category, embeddings)
    else:
        log.info("  [跳过 Step 4, 从缓存加载]")
        out_dir = PROC_DIR / category
        codes = np.load(out_dir / "sid_codes.npy")
        cb = np.load(out_dir / "codebooks.npz")
        codebooks = [cb[SID_LAYER_PREFIX[i]] for i in range(NUM_SID_LAYERS)]
        step4_stats = {"timing": {"skipped": True}, "layer_timing": []}
    stage_timing["step4"] = step4_stats["timing"]

    stage_timing["total_before_step5_sec"] = _round_seconds(_elapsed(t_total))
    step5_stats = step5_assemble(
        category=category,
        all_item_seqs=all_item_seqs,
        id_mapping=id_mapping,
        sorted_asins=sorted_asins,
        items_data=items_data,
        asin2intid=asin2intid,
        codes=codes,
        codebooks=codebooks,
        embeddings=embeddings,
        field_stats=field_stats,
        timing_summary=stage_timing,
        quantization_layer_timings=step4_stats.get("layer_timing", []),
    )
    stage_timing["step5"] = step5_stats["timing"]

    total = _elapsed(t_total)
    stage_timing["total_pipeline_sec"] = _round_seconds(total)

    log.info("")
    log.info("  全流程耗时摘要:")
    for line in _format_timing_table(stage_timing):
        log.info(line)

    log.info("")
    log.info(" 🎉 %s 全流程完成！总耗时: %.1f 秒 (%.1f 分钟)", category, total, total / 60)


def parse_args():
    p = argparse.ArgumentParser(
        description="原始数据 → SID 产码全流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python data/raw_to_sids.py
  python data/raw_to_sids.py --category Sports
  python data/raw_to_sids.py --category Sports --skip-embedding
  python data/raw_to_sids.py --category Sports --from-step 4
"""
    )
    p.add_argument("--category", choices=["Beauty", "Sports", "Toys", "all"], default="all")
    p.add_argument(
        "--skip-embedding",
        action="store_true",
        help="跳过 Embedding 生成（使用已有的 item_embeddings.npy）",
    )
    p.add_argument(
        "--from-step",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="从第 N 步开始 (1=review→seq+id, 2=meta字段提取, 3=embedding, 4=产码, 5=组装)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging("raw_to_sids")

    categories = list(DATASETS.keys()) if args.category == "all" else [args.category]

    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║           原始数据 → SID 产码                              ║")
    log.info("╠══════════════════════════════════════════════════════════════╣")
    log.info("║  类目:     %-44s ║", ", ".join(categories))
    log.info("║  跳过EMB:  %-44s ║", str(args.skip_embedding))
    log.info("║  起始步骤: %-44s ║", f"Step {args.from_step}")
    log.info("║  序列模式: %-44s ║", "full sequence / no truncate")
    log.info("╚══════════════════════════════════════════════════════════════╝")

    t_all = time.time()
    for cat in categories:
        process_category(cat, skip_embedding=args.skip_embedding, from_step=args.from_step)

    total = time.time() - t_all
    log.info("")
    log.info("═" * 72)
    log.info(" 🎉 全部完成！总耗时: %.1f 秒 (%.1f 分钟)", total, total / 60)

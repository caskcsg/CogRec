#!/usr/bin/env python3
"""
data/hnsw_and_splits.py — 离线切片与 Routing 资产构建
====================================================

承接 raw_to_sids.py 的产物:
  codebooks.npz, sid_codes.npy, item_embeddings.npy,
  item_mapping.json, {cat}.pretrain.json, {cat}_sequential_data.txt

流水线步骤:
  Step 1: 认知地图        codebooks → 层内横向边 + s_d 层 HNSW
  Step 2: 训练切片        pretrain.json + sequential → 9 个 parquet
  Step 3: Routing 统一资产  RA 基底 → routing full parquet + difficulty + CoT subset

最终产物 (每类目):
  训练数据:
    ├── training_align_data_{train,val,test}.parquet
    ├── training_prediction_sid_data_{train,val,test}.parquet
    ├── training_RA_{train,val,test}.parquet
    ├── training_sid_routing_{train,val,test}.parquet
    ├── training_RA_test_cot_eval.parquet
    └── training_sid_routing_test_cot_eval.parquet
  Routing 辅助文件:
    ├── test_difficulty_labels.json
    ├── test_difficulty_labels_cot_eval.json
    ├── routing_quality_report.json
    ├── hnsw_pipeline_report.json
    └── cot_eval_subset_ids.json
  认知地图:
    ├── cognitive_map.json
    ├── edges_s_{a,b,c,d}.json + adjacency_s_{a,b,c,d}.npz
    └── hnsw_s_d/

说明:
  - Routing full data / CoT subset / row-aligned difficulty 全部由 Step 3 统一生成。
  - CoT eval 子集固定抽样并与 Routing difficulty 行对齐。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
from scipy import sparse
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.config import (  # type: ignore
    DATASETS,
    PROC_DIR,
    NUM_SID_LAYERS,
    CODEBOOK_SIZE,
    SID_LAYER_PREFIX,
    GRAPH_TOP_K,
    GRAPH_SIM_THRESH,
    HNSW_M,
    HNSW_EF_CONSTRUCTION,
    HARD_SAMPLE_PREFIX_MATCH_LAYERS,
    HARD_SAMPLE_GRAPH_DIST_THRESH,
    MIN_CLUSTER_SIZE_FOR_HNSW,
    SID_PATTERN,
    setup_logging,
    category_processed_dir,
)

log = logging.getLogger("hnsw_and_splits")


def parse_sid(sid_str: str) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(sid_str, str):
        return None
    m = SID_PATTERN.search(sid_str)
    if m:
        return (int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
    tokens = re.findall(r"<s_([a-d])_(\d+)>", sid_str)
    if len(tokens) >= 4:
        layer_map = {"a": 0, "b": 1, "c": 2, "d": 3}
        codes = [0, 0, 0, 0]
        for ch, val in tokens[:4]:
            codes[layer_map[ch]] = int(val)
        return tuple(codes)  # type: ignore[return-value]
    return None


def extract_sid_strings(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return [m.group(1) for m in SID_PATTERN.finditer(text)]


def _make_sample_id(user_id: object, groundtruth: str, description: str) -> str:
    raw = f"{user_id}\t{groundtruth}\t{description}".encode("utf-8", errors="ignore")
    return hashlib.md5(raw).hexdigest()


def _ensure_sample_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sample_id" not in out.columns:
        out["sample_id"] = [
            _make_sample_id(row.get("user_id", f"user_{idx}"), str(row["groundtruth"]), str(row["description"]))
            for idx, row in out.iterrows()
        ]
    return out


def _step_banner(step: int, total: int, title: str) -> None:
    log.info("")
    log.info("━" * 72)
    log.info("  Step %d/%d: %s", step, total, title)
    log.info("━" * 72)


def _round_seconds(x: float) -> float:
    return round(float(x), 3)


def _elapsed(start: float) -> float:
    return time.time() - start


def _format_timing_table(timing_dict: dict, indent: str = "    ") -> list[str]:
    lines = []
    for key, value in timing_dict.items():
        if isinstance(value, dict):
            lines.append(f"{indent}{key}:")
            lines.extend(_format_timing_table(value, indent=indent + "  "))
        else:
            lines.append(f"{indent}{key}: {value}")
    return lines


def _load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def _load_codebooks(category: str) -> List[np.ndarray]:
    cb_path = category_processed_dir(category) / "codebooks.npz"
    cb = np.load(cb_path)
    return [cb[k] for k in list(SID_LAYER_PREFIX)]


def _build_layer_edges(centroids: np.ndarray, layer_idx: int, top_k: int, threshold: float):
    prefix = SID_LAYER_PREFIX[layer_idx]
    k = centroids.shape[0]

    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = centroids / norms
    sim_matrix = normed @ normed.T
    np.fill_diagonal(sim_matrix, 0.0)

    edges = []
    rows, cols, vals = [], [], []
    for i in range(k):
        top_indices = np.argsort(sim_matrix[i])[::-1][:top_k]
        for j in top_indices:
            w = float(sim_matrix[i][j])
            if w >= threshold:
                edges.append({"src": int(i), "dst": int(j), "weight": round(w, 4)})
                rows.append(i)
                cols.append(int(j))
                vals.append(w)

    adj = sparse.csr_matrix((vals, (rows, cols)), shape=(k, k))
    degrees = np.array(adj.sum(axis=1)).flatten()
    stats = {
        "layer_prefix": prefix,
        "n_nodes": k,
        "n_edges": len(edges),
        "avg_degree": round(float(degrees.mean()), 2) if len(degrees) else 0.0,
        "max_degree": int(degrees.max()) if len(degrees) else 0,
        "min_degree": int(degrees.min()) if len(degrees) else 0,
        "avg_weight": round(float(np.mean(vals)), 4) if vals else 0.0,
        "weight_range": [round(float(np.min(vals)), 4), round(float(np.max(vals)), 4)] if vals else [0, 0],
        "isolated_nodes": int(np.sum(degrees == 0)) if len(degrees) else 0,
    }
    return edges, adj, stats


def _build_hnsw(embeddings: np.ndarray, codes: np.ndarray, category: str):
    prefix_d = SID_LAYER_PREFIX[3]
    out_dir = category_processed_dir(category) / f"hnsw_{prefix_d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sd_codes = codes[:, 3]
    unique_clusters = np.unique(sd_codes)
    dim = embeddings.shape[1]
    built, skipped = 0, 0
    hnsw_stats = []

    for cid in unique_clusters:
        indices = np.where(sd_codes == cid)[0]
        if len(indices) < MIN_CLUSTER_SIZE_FOR_HNSW:
            skipped += 1
            continue
        cluster_embs = embeddings[indices].astype(np.float32)
        faiss.normalize_L2(cluster_embs)
        index = faiss.IndexHNSWFlat(dim, HNSW_M)
        index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        index.add(cluster_embs)
        faiss.write_index(index, str(out_dir / f"hnsw_cluster_{cid}.index"))
        np.save(out_dir / f"mapping_cluster_{cid}.npy", indices)
        hnsw_stats.append({"cluster_id": int(cid), "n_items": len(indices)})
        built += 1

    with open(out_dir / "hnsw_meta.json", "w", encoding="utf-8") as f:
        json.dump(hnsw_stats, f, indent=2)

    log.info("  HNSW: %d 簇已建图, %d 簇跳过 (size < %d)", built, skipped, MIN_CLUSTER_SIZE_FOR_HNSW)
    return hnsw_stats


def step1_cognitive_map(category: str) -> dict:
    _step_banner(1, 3, f"认知地图构建 [{category}]")
    t0 = time.time()
    timings = {}

    out_dir = category_processed_dir(category)

    t_load = time.time()
    codebooks = _load_codebooks(category)
    embeddings = np.load(out_dir / "item_embeddings.npy")
    codes = np.load(out_dir / "sid_codes.npy")
    timings["load_inputs_sec"] = _round_seconds(_elapsed(t_load))

    log.info("  Items: %d, Dim: %d", embeddings.shape[0], embeddings.shape[1])
    log.info("")
    log.info("  ┌─── 横向边构建 (%d 层 × %d 聚类中心) ───", NUM_SID_LAYERS, CODEBOOK_SIZE)

    layer_timings = {}
    all_stats = []
    for idx in range(NUM_SID_LAYERS):
        prefix = SID_LAYER_PREFIX[idx]
        t_layer = time.time()
        log.info("  │ %s 层: 计算相似度...", prefix)
        edges, adj, stats = _build_layer_edges(codebooks[idx], idx, GRAPH_TOP_K, GRAPH_SIM_THRESH)
        with open(out_dir / f"edges_{prefix}.json", "w", encoding="utf-8") as f:
            json.dump(edges, f, indent=2)
        sparse.save_npz(out_dir / f"adjacency_{prefix}.npz", adj)
        all_stats.append(stats)
        layer_timings[prefix] = _round_seconds(_elapsed(t_layer))
        log.info("  │ %s 层: %d 条边 | 平均度 %.1f | 权重 [%.3f, %.3f] | 孤立 %d | %.2fs", prefix, stats["n_edges"], stats["avg_degree"], stats["weight_range"][0], stats["weight_range"][1], stats["isolated_nodes"], layer_timings[prefix])
    timings["layer_edge_build_sec"] = layer_timings
    log.info("  └────────────────────────────────────")

    log.info("")
    log.info("  ┌─── %s 层 HNSW 近邻图 ───", SID_LAYER_PREFIX[3])
    t_hnsw = time.time()
    hnsw_stats = _build_hnsw(embeddings, codes, category)
    timings["build_hnsw_sec"] = _round_seconds(_elapsed(t_hnsw))
    log.info("  └────────────────────────────────────")

    cognitive_map = {
        "category": category,
        "n_items": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "num_layers": NUM_SID_LAYERS,
        "codebook_size": CODEBOOK_SIZE,
        "layer_prefixes": SID_LAYER_PREFIX,
        "graph_config": {
            "top_k": GRAPH_TOP_K,
            "sim_threshold": GRAPH_SIM_THRESH,
            "hnsw_M": HNSW_M,
            "hnsw_ef_construction": HNSW_EF_CONSTRUCTION,
            "min_cluster_size_for_hnsw": MIN_CLUSTER_SIZE_FOR_HNSW,
        },
        "layer_graphs": all_stats,
        "hnsw_clusters_built": len(hnsw_stats),
        "files": {
            "edges": [f"edges_{p}.json" for p in SID_LAYER_PREFIX],
            "adjacency": [f"adjacency_{p}.npz" for p in SID_LAYER_PREFIX],
            "hnsw_dir": f"hnsw_{SID_LAYER_PREFIX[3]}/",
        },
    }
    t_write = time.time()
    with open(out_dir / "cognitive_map.json", "w", encoding="utf-8") as f:
        json.dump(cognitive_map, f, indent=2, ensure_ascii=False)
    timings["write_cognitive_map_sec"] = _round_seconds(_elapsed(t_write))
    timings["total_sec"] = _round_seconds(_elapsed(t0))

    log.info("")
    log.info("  认知地图完成: 边 %s | HNSW %d 簇 | 耗时 %.1fs", "+".join(str(s["n_edges"]) for s in all_stats), len(hnsw_stats), timings["total_sec"])
    log.info("  Step 1 耗时细分:")
    for line in _format_timing_table(timings):
        log.info(line)

    return {
        "timing": timings,
        "stats": {
            "n_items": int(embeddings.shape[0]),
            "embedding_dim": int(embeddings.shape[1]),
            "layer_edges": {s["layer_prefix"]: int(s["n_edges"]) for s in all_stats},
            "hnsw_clusters_built": int(len(hnsw_stats)),
            "hnsw_clusters": hnsw_stats,
        },
    }


def _load_pretrain(category: str) -> dict:
    with open(category_processed_dir(category) / f"{category}.pretrain.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _load_sequential(category: str) -> List[Tuple[str, List[str]]]:
    data = []
    with open(category_processed_dir(category) / f"{category}_sequential_data.txt", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                data.append((parts[0], parts[1:]))
    return data


def _split_align(items: dict, sequences: List[Tuple[str, List[str]]]) -> Dict[str, pd.DataFrame]:
    def _build_desc(item_ids: List[str]) -> List[str]:
        descs = []
        for iid in item_ids:
            info = items.get(iid)
            if info and info.get("sid") and info.get("title") and info.get("categories"):
                descs.append(f'{info["sid"]}, its title is "{info["title"]}", its categories are "{info["categories"]}"')
        return descs

    splits = {"train": [], "val": [], "test": []}
    for uid, iids in sequences:
        full = _build_desc(iids)
        if full:
            splits["test"].append({"user_id": uid, "description": "The user has purchased the following items: " + "; ".join(full) + ";"})
        if len(iids) > 1:
            val = _build_desc(iids[:-1])
            if val:
                splits["val"].append({"user_id": uid, "description": "The user has purchased the following items: " + "; ".join(val) + ";"})
        if len(iids) > 2:
            train = _build_desc(iids[:-2])
            if train:
                splits["train"].append({"user_id": uid, "description": "The user has purchased the following items: " + "; ".join(train) + ";"})
    return {k: pd.DataFrame(v) for k, v in splits.items()}


def _split_sid_prediction(items: dict, sequences: List[Tuple[str, List[str]]]) -> Dict[str, pd.DataFrame]:
    def _extract_sids(item_ids: List[str]) -> List[str]:
        return [items[i]["sid"] for i in item_ids if items.get(i, {}).get("sid")]

    def _entry(sids: List[str], uid: str, tail_remove: int):
        cand = sids[:len(sids) - tail_remove] if tail_remove > 0 else sids
        if len(cand) < 2:
            return None
        return {"user_id": uid, "description": "The user has purchased the following items: " + "; ".join(cand[:-1]) + ";", "groundtruth": cand[-1]}

    splits = {"train": [], "val": [], "test": []}
    for uid, iids in sequences:
        sids = _extract_sids(iids)
        if not sids:
            continue
        for name, tail in [("train", 2), ("val", 1), ("test", 0)]:
            e = _entry(sids, uid, tail)
            if e:
                splits[name].append(e)
    return {k: pd.DataFrame(v) for k, v in splits.items()}


def _split_ra(items: dict, sequences: List[Tuple[str, List[str]]]) -> Dict[str, pd.DataFrame]:
    def _extract_seq(item_ids: List[str]) -> List[dict]:
        return [{"sid": items[i]["sid"], "title": items[i].get("title", ""), "categories": items[i].get("categories", "")} for i in item_ids if items.get(i, {}).get("sid")]

    def _entry(seq: List[dict], uid: str, tail_remove: int):
        cand = seq[:len(seq) - tail_remove] if tail_remove > 0 else seq
        if len(cand) < 2:
            return None
        gt = cand[-1]
        history = cand[:-1]
        descs = [f'{h["sid"]}, its title is "{h["title"]}", its categories are "{h["categories"]}"' for h in history]
        return {"user_id": uid, "description": "The user has purchased the following items: " + "; ".join(descs) + ";", "groundtruth": gt["sid"], "title": gt["title"], "categories": gt["categories"]}

    splits = {"train": [], "val": [], "test": []}
    for uid, iids in sequences:
        seq = _extract_seq(iids)
        if not seq:
            continue
        for name, tail in [("train", 2), ("val", 1), ("test", 0)]:
            e = _entry(seq, uid, tail)
            if e:
                splits[name].append(e)
    return {k: pd.DataFrame(v) for k, v in splits.items()}


def step2_training_splits(category: str) -> dict:
    _step_banner(2, 3, f"训练数据切片 [{category}]")
    t0 = time.time()
    timings = {}

    t_load = time.time()
    items = _load_pretrain(category)
    sequences = _load_sequential(category)
    out_dir = category_processed_dir(category)
    timings["load_inputs_sec"] = _round_seconds(_elapsed(t_load))
    log.info("  pretrain.json: %d items | 序列: %d 用户", len(items), len(sequences))

    generators = [("align", "training_align_data", _split_align), ("prediction", "training_prediction_sid_data", _split_sid_prediction), ("RA", "training_RA", _split_ra)]
    split_counts = {}
    generator_timings = {}
    for stage_name, file_prefix, gen_fn in generators:
        t_stage = time.time()
        log.info("  生成 %s 数据...", stage_name)
        dfs = gen_fn(items, sequences)
        generator_timings[stage_name] = _round_seconds(_elapsed(t_stage))
        split_counts[stage_name] = {}
        for split_name, df in dfs.items():
            out_path = out_dir / f"{file_prefix}_{split_name}.parquet"
            df.to_parquet(out_path, engine="pyarrow", index=False)
            split_counts[stage_name][split_name] = int(len(df))
            log.info("    %s %s: %d 条", stage_name, split_name, len(df))
    timings["build_and_write_sec"] = generator_timings
    timings["total_sec"] = _round_seconds(_elapsed(t0))

    log.info("  9 个 parquet 生成完成 (%.1fs)", timings["total_sec"])
    log.info("  Step 2 耗时细分:")
    for line in _format_timing_table(timings):
        log.info(line)

    return {
        "timing": timings,
        "stats": {
            "n_items": int(len(items)),
            "n_sequences": int(len(sequences)),
            "split_counts": split_counts,
        },
    }


def _load_edge_adj(category: str, layer_idx: int) -> Dict[int, List[dict]]:
    prefix = SID_LAYER_PREFIX[layer_idx]
    path = category_processed_dir(category) / f"edges_{prefix}.json"
    if not path.exists():
        raise FileNotFoundError(f"边文件不存在: {path}")
    edges = json.loads(path.read_text(encoding="utf-8"))
    adj: Dict[int, List[dict]] = defaultdict(list)
    for e in edges:
        adj[int(e["src"])].append({"dst": int(e["dst"]), "weight": float(e.get("weight", 0.0))})
    return dict(adj)


def _load_edge_adj_set(category: str, layer_idx: int) -> Dict[int, set[int]]:
    raw = _load_edge_adj(category, layer_idx)
    return {src: {int(item["dst"]) for item in neighs} for src, neighs in raw.items()}


def _bfs_dist(src: int, dst: int, adj: Dict[int, set[int]], max_d: int = 4) -> int:
    if src == dst:
        return 0
    q: deque[Tuple[int, int]] = deque([(src, 0)])
    seen = {src}
    while q:
        node, depth = q.popleft()
        if depth >= max_d:
            continue
        for nxt in adj.get(node, set()):
            if nxt == dst:
                return depth + 1
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, depth + 1))
    return max_d + 1


def _classify_sample(history_sids: List[str], target_sid: str, sa_adj: Dict[int, set[int]]) -> dict:
    tc = parse_sid(target_sid)
    if not tc:
        return {"difficulty": "error", "anchor_sid": None, "anchor_match_depth": 0, "s_a_min_distance": 999}

    best_depth, best_anchor, min_dist = 0, None, 999
    for h in history_sids:
        hc = parse_sid(h)
        if not hc:
            continue
        depth = 0
        for i in range(NUM_SID_LAYERS):
            if hc[i] == tc[i]:
                depth += 1
            else:
                break
        if depth > best_depth:
            best_depth, best_anchor = depth, h
        min_dist = min(min_dist, _bfs_dist(hc[0], tc[0], sa_adj))

    if best_anchor is None and history_sids:
        best_anchor = history_sids[-1]

    if best_depth >= HARD_SAMPLE_PREFIX_MATCH_LAYERS:
        difficulty = "easy"
    elif best_depth >= 1 or min_dist <= HARD_SAMPLE_GRAPH_DIST_THRESH:
        difficulty = "medium"
    else:
        difficulty = "hard"

    return {"difficulty": difficulty, "anchor_sid": best_anchor, "anchor_match_depth": int(best_depth), "s_a_min_distance": int(min_dist)}


def _construct_nav_path(anchor_sid: Optional[str], target_sid: str, layer_adjs: Dict[int, Dict[int, List[dict]]]) -> Optional[dict]:
    if not anchor_sid:
        return None
    ac = parse_sid(anchor_sid)
    tc = parse_sid(target_sid)
    if not ac or not tc:
        return None

    steps: List[dict] = []
    total_cost = 0
    for idx in range(NUM_SID_LAYERS):
        prefix = SID_LAYER_PREFIX[idx]
        src, dst = ac[idx], tc[idx]
        if src == dst:
            steps.append({"layer": prefix, "action": "match", "from": src, "to": dst, "cost": 0})
            continue
        edge_info = next((n for n in layer_adjs.get(idx, {}).get(src, []) if n["dst"] == dst), None)
        if edge_info:
            steps.append({"layer": prefix, "action": "lateral_jump", "from": src, "to": dst, "weight": float(edge_info.get("weight", 0.0)), "cost": 1})
            total_cost += 1
        else:
            steps.append({"layer": prefix, "action": "explore", "from": src, "to": dst, "cost": 2})
            total_cost += 2
    return {"path_steps": steps, "total_cost": int(total_cost)}


def _format_step_text(step: dict) -> str:
    prefix = step["layer"]
    action = step["action"]
    src, dst = step["from"], step["to"]
    if action == "match":
        return f"{prefix}: Match Code {dst}, direct match with history."
    if action == "lateral_jump":
        w = step.get("weight", 0.0)
        return f"{prefix}: Lateral jump from cluster {src} to {dst} via edge (similarity={w:.4f})."
    return f"{prefix}: Explore from cluster {src} to {dst} (no direct edge, inferred from user interest pattern)."


def _build_cot_text(navigation: Optional[dict]) -> Tuple[str, int]:
    if not navigation or "path_steps" not in navigation:
        return "<think>\n\n</think>", 0
    non_trivial = [s for s in navigation["path_steps"] if s["action"] != "match"]
    if not non_trivial:
        return "<think>\n\n</think>", 0
    return "<think>\n" + "\n".join(_format_step_text(s) for s in non_trivial) + "\n</think>", len(non_trivial)


def _build_split_from_ra_base(category: str, split: str, sa_adj: Dict[int, set[int]], layer_adjs: Dict[int, Dict[int, List[dict]]]) -> pd.DataFrame:
    base_path = category_processed_dir(category) / f"training_RA_{split}.parquet"
    if not base_path.exists():
        raise FileNotFoundError(f"Stage-2b RA 基底不存在: {base_path}")
    df = _ensure_sample_id(pd.read_parquet(base_path))
    log.info("  [%s] 基底样本: %d", split, len(df))

    rows = []
    for idx, row in df.iterrows():
        description = str(row["description"])
        groundtruth = str(row["groundtruth"])
        history_sids = extract_sid_strings(description)
        cls = _classify_sample(history_sids, groundtruth, sa_adj)
        nav = _construct_nav_path(cls.get("anchor_sid"), groundtruth, layer_adjs) if cls.get("anchor_sid") else None
        cot_text, cot_steps = _build_cot_text(nav)
        rows.append({
            "sample_id": row["sample_id"],
            "user_id": row.get("user_id", f"user_{idx}"),
            "description": description,
            "groundtruth": groundtruth,
            "sid_routing_think": cot_text,
            "difficulty": cls.get("difficulty", "unknown"),
            "cot_steps": int(cot_steps),
            "anchor_sid": cls.get("anchor_sid"),
            "anchor_match_depth": int(cls.get("anchor_match_depth", 0)),
            "s_a_min_distance": int(cls.get("s_a_min_distance", 999)),
            "routing_total_cost": int(nav.get("total_cost", 0)) if nav else 0,
        })
    out_df = pd.DataFrame(rows)
    log.info("  [%s] 完成 | 空 think=%d | 非空 think=%d | 步数分布=%s", split, int((out_df["cot_steps"] == 0).sum()), int((out_df["cot_steps"] > 0).sum()), dict(Counter(out_df["cot_steps"].tolist())))
    return out_df




def _save_row_aligned_difficulty(df_test: pd.DataFrame, out_file: Path) -> None:
    payload = {
        "format": "row_aligned",
        "split": "test",
        "n": int(len(df_test)),
        "samples": [
            {
                "sample_id": row["sample_id"],
                "difficulty": row["difficulty"],
                "cot_steps": int(row["cot_steps"]),
                "anchor_sid": row.get("anchor_sid"),
                "anchor_match_depth": int(row.get("anchor_match_depth", 0)),
                "s_a_min_distance": int(row.get("s_a_min_distance", 999)),
                "routing_total_cost": int(row.get("routing_total_cost", 0)),
            }
            for _, row in df_test.iterrows()
        ],
    }
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _verify_alignment(df_ra: pd.DataFrame, df_routing: pd.DataFrame) -> None:
    if len(df_ra) != len(df_routing):
        raise ValueError(f"RA / Routing test 行数不一致: {len(df_ra)} vs {len(df_routing)}")
    for col in ["sample_id", "groundtruth", "description"]:
        if col not in df_ra.columns:
            continue
        if col not in df_routing.columns:
            raise ValueError(f"Routing 缺少列: {col}")
        if not df_ra[col].astype(str).equals(df_routing[col].astype(str)):
            bad = next((i for i, (a, b) in enumerate(zip(df_ra[col].astype(str), df_routing[col].astype(str))) if a != b), None)
            raise ValueError(f"RA / Routing 在列 {col} 上未对齐，首个不一致行: {bad}")


def _subset_difficulty(diff_file: Path, keep_indices: List[int], out_file: Path) -> None:
    payload = json.loads(diff_file.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    sub_samples = [samples[i] for i in keep_indices]
    out_payload = {**{k: v for k, v in payload.items() if k != "samples"}, "split": "test_cot_eval", "n": len(sub_samples), "source_n": len(samples), "keep_indices": keep_indices, "samples": sub_samples}
    out_file.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_cot_eval_subset(category: str, ratio: float, sample_size: int, seed: int, force: bool = False) -> dict:
    t0 = time.time()

    cat_dir = category_processed_dir(category)
    subset_ids_file = cat_dir / "cot_eval_subset_ids.json"
    ra_out = cat_dir / "training_RA_test_cot_eval.parquet"
    routing_out = cat_dir / "training_sid_routing_test_cot_eval.parquet"
    diff_out = cat_dir / "test_difficulty_labels_cot_eval.json"
    diff_file = cat_dir / "test_difficulty_labels.json"

    if (not force) and all(p.exists() for p in [subset_ids_file, ra_out, routing_out, diff_out]):
        log.info("  [subset] 已存在，跳过")

        meta = _load_json_if_exists(subset_ids_file) or {}
        source_n = int(meta.get("source_n", 0))
        subset_n = int(meta.get("subset_n", 0))
        keep_indices = meta.get("keep_indices", [])
        sample_ids = meta.get("sample_ids", [])

        return {
            "timing": {
                "skipped": True,
                "total_sec": _round_seconds(_elapsed(t0)),
            },
            "stats": {
                "source_n": source_n,
                "subset_n": subset_n,
                "keep_indices": keep_indices,
                "sample_ids": sample_ids,
            },
        }

    t_load = time.time()
    df_ra = _ensure_sample_id(pd.read_parquet(cat_dir / "training_RA_test.parquet"))
    df_routing = pd.read_parquet(cat_dir / "training_sid_routing_test.parquet")
    _verify_alignment(df_ra, df_routing)
    load_sec = _round_seconds(_elapsed(t_load))

    t_select = time.time()
    n = len(df_ra)
    if sample_size is not None and sample_size > 0:
        k = min(sample_size, n)
    else:
        k = max(1, int(round(n * ratio)))

    rng = np.random.default_rng(seed)
    keep_indices = sorted(rng.choice(np.arange(n), size=k, replace=False).tolist())
    select_sec = _round_seconds(_elapsed(t_select))

    t_write = time.time()
    sub_ra = df_ra.iloc[keep_indices].reset_index(drop=True)
    sub_routing = df_routing.iloc[keep_indices].reset_index(drop=True)

    sub_ra.to_parquet(ra_out, index=False)
    sub_routing.to_parquet(routing_out, index=False)
    _subset_difficulty(diff_file, keep_indices, diff_out)

    payload = {
        "format": "cot_eval_subset",
        "category": category,
        "seed": seed,
        "ratio": ratio,
        "sample_size": sample_size,
        "source_n": n,
        "subset_n": len(keep_indices),
        "keep_indices": keep_indices,
        "sample_ids": sub_ra["sample_id"].astype(str).tolist() if "sample_id" in sub_ra.columns else [],
    }
    subset_ids_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_sec = _round_seconds(_elapsed(t_write))

    log.info("  [subset] 完成 | %d/%d 样本", len(keep_indices), n)

    return {
        "timing": {
            "load_inputs_sec": load_sec,
            "select_subset_sec": select_sec,
            "write_outputs_sec": write_sec,
            "total_sec": _round_seconds(_elapsed(t0)),
        },
        "stats": {
            "source_n": int(n),
            "subset_n": int(len(keep_indices)),
            "keep_indices": keep_indices,
            "sample_ids": payload["sample_ids"],
        },
    }


def step3_routing_assets(category: str, subset_ratio: float = 0.1, subset_size: int = -1, subset_seed: int = 42, force: bool = False) -> dict:
    _step_banner(3, 3, f"Routing 统一资产 [{category}]")
    t0 = time.time()
    timings = {}
    cat_dir = category_processed_dir(category)
    train_out = cat_dir / "training_sid_routing_train.parquet"
    val_out = cat_dir / "training_sid_routing_val.parquet"
    test_out = cat_dir / "training_sid_routing_test.parquet"
    diff_out = cat_dir / "test_difficulty_labels.json"
    report_out = cat_dir / "routing_quality_report.json"

    if (not force) and all(p.exists() for p in [train_out, val_out, test_out, diff_out, report_out]):
        log.info("  [routing-full] 已存在全部产物，跳过 full build")
        timings["build_routing_full_sec"] = {"skipped": True}
        report = _load_json_if_exists(report_out) or {}
    else:
        t_build = time.time()
        sa_adj = _load_edge_adj_set(category, 0)
        layer_adjs = {idx: _load_edge_adj(category, idx) for idx in range(NUM_SID_LAYERS)}
        log.info("  加载邻接完成 | s_a 有边节点=%d", len(sa_adj))

        t_train = time.time()
        df_train = _build_split_from_ra_base(category, "train", sa_adj, layer_adjs)
        train_sec = _round_seconds(_elapsed(t_train))
        t_val = time.time()
        df_val = _build_split_from_ra_base(category, "val", sa_adj, layer_adjs)
        val_sec = _round_seconds(_elapsed(t_val))
        t_test = time.time()
        df_test = _build_split_from_ra_base(category, "test", sa_adj, layer_adjs)
        test_sec = _round_seconds(_elapsed(t_test))

        t_write = time.time()
        df_train.to_parquet(train_out, index=False)
        df_val.to_parquet(val_out, index=False)
        df_test.to_parquet(test_out, index=False)
        _save_row_aligned_difficulty(df_test, diff_out)
        write_sec = _round_seconds(_elapsed(t_write))

        report = {"category": category, "subset_ratio": subset_ratio, "subset_size": subset_size, "subset_seed": subset_seed, "splits": {}, "elapsed_sec": round(time.time() - t0, 2)}
        for split, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
            report["splits"][split] = {
                "n": int(len(df)),
                "difficulty": {k: int(v) for k, v in Counter(df["difficulty"].tolist()).items()},
                "cot_steps": {str(k): int(v) for k, v in Counter(df["cot_steps"].tolist()).items()},
                "empty_think": int((df["cot_steps"] == 0).sum()),
                "non_empty_think": int((df["cot_steps"] > 0).sum()),
                "avg_total_cost": round(float(df["routing_total_cost"].mean()), 4) if len(df) else 0.0,
            }
        timings["build_routing_full_sec"] = {
            "train_split_sec": train_sec,
            "val_split_sec": val_sec,
            "test_split_sec": test_sec,
            "write_outputs_sec": write_sec,
            "total_sec": _round_seconds(_elapsed(t_build)),
        }
        report["timing"] = dict(timings["build_routing_full_sec"])
        report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("  [routing-full] 已写出 routing full parquet + difficulty + report")

    subset_stats = _build_cot_eval_subset(category, subset_ratio, subset_size, subset_seed, force=force)
    timings["build_cot_subset_sec"] = subset_stats["timing"]
    timings["total_sec"] = _round_seconds(_elapsed(t0))
    log.info("  Routing 统一资产完成 [%s] | %.1fs", category, timings["total_sec"])
    log.info("  Step 3 耗时细分:")
    for line in _format_timing_table(timings):
        log.info(line)

    return {
        "timing": timings,
        "stats": {
            "routing_report": report,
            "subset": subset_stats["stats"],
        },
    }


def process_category(category: str, from_step: int = 1, cot_subset_ratio: float = 0.1, cot_subset_size: int = -1, cot_subset_seed: int = 42, force_routing_data: bool = False) -> dict:
    log.info("")
    log.info("═" * 72)
    log.info("  类目: %s | from_step=%d | subset_ratio=%s | subset_size=%s | force=%s", category, from_step, f"{cot_subset_ratio:.2f}", cot_subset_size, force_routing_data)
    log.info("═" * 72)

    t_total = time.time()
    step_reports = {}
    if from_step <= 1:
        step_reports["step1"] = step1_cognitive_map(category)
    else:
        log.info("  [跳过 Step 1]")
        step_reports["step1"] = {"timing": {"skipped": True}, "stats": {}}
    if from_step <= 2:
        step_reports["step2"] = step2_training_splits(category)
    else:
        log.info("  [跳过 Step 2]")
        step_reports["step2"] = {"timing": {"skipped": True}, "stats": {}}
    step_reports["step3"] = step3_routing_assets(category, subset_ratio=cot_subset_ratio, subset_size=cot_subset_size, subset_seed=cot_subset_seed, force=force_routing_data)

    total = time.time() - t_total
    cat_dir = category_processed_dir(category)
    parquets = sorted(cat_dir.glob("training_*.parquet"))
    pipeline_report = {
        "category": category,
        "from_step": int(from_step),
        "subset_ratio": float(cot_subset_ratio),
        "subset_size": int(cot_subset_size),
        "subset_seed": int(cot_subset_seed),
        "force_routing_data": bool(force_routing_data),
        "timing": {
            "step1": step_reports["step1"]["timing"],
            "step2": step_reports["step2"]["timing"],
            "step3": step_reports["step3"]["timing"],
            "total_sec": _round_seconds(total),
        },
        "stats": {
            "step1": step_reports["step1"].get("stats", {}),
            "step2": step_reports["step2"].get("stats", {}),
            "step3": step_reports["step3"].get("stats", {}),
            "artifact_counts": {
                "training_parquet_files": int(len(parquets)),
                "json_files": int(len(list(cat_dir.glob("*.json")))),
                "npz_files": int(len(list(cat_dir.glob("*.npz")))),
            },
        },
    }
    pipeline_report_path = cat_dir / "hnsw_pipeline_report.json"
    pipeline_report_path.write_text(json.dumps(pipeline_report, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("")
    log.info("  ┌─── %s 产物汇总 ───", category)
    log.info("  │ Parquet 文件: %d 个", len(parquets))
    for p in parquets:
        df = pd.read_parquet(p)
        log.info("  │   %s: %d 条, 列=%s", p.name, len(df), list(df.columns))
    log.info("  │ 附加文件: difficulty + subset ids + routing report + hnsw pipeline report")
    log.info("  │ 认知地图: cognitive_map.json + 边文件 + HNSW")
    log.info("  │ 总耗时: %.1fs (%.1f min)", total, total / 60)
    log.info("  └────────────────────────────────────────────")
    log.info("  ✓ %s 完成", category)
    return pipeline_report


def parse_args():
    p = argparse.ArgumentParser(description="认知地图构建 + 训练数据切片 + Routing 离线资产全流程", formatter_class=argparse.RawDescriptionHelpFormatter, epilog="""
示例:
  python data/hnsw_and_splits.py --category Beauty
  python data/hnsw_and_splits.py --category Beauty --from-step 3
  python data/hnsw_and_splits.py --category Beauty --from-step 3 --force-routing-data
  python data/hnsw_and_splits.py --category Beauty --cot-subset-ratio 0.1 --cot-subset-seed 42
""")
    p.add_argument("--category", choices=["Beauty", "Sports", "Toys", "all"], default="all")
    p.add_argument("--from-step", type=int, default=1, choices=[1, 2, 3], help="从第 N 步开始 (1=认知地图, 2=训练切片, 3=Routing 统一资产)")
    p.add_argument("--cot-subset-ratio", type=float, default=0.1, help="CoT eval 子集比例，默认 0.1")
    p.add_argument("--cot-subset-size", type=int, default=-1, help="固定 CoT eval 子集大小；>0 时覆盖 ratio")
    p.add_argument("--cot-subset-seed", type=int, default=42, help="CoT eval 子集固定随机种子")
    p.add_argument("--force-routing-data", action="store_true", help="强制重建 Routing full data + CoT subset")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging("hnsw_and_splits")
    categories = list(DATASETS.keys()) if args.category == "all" else [args.category]

    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║      正式版: 认知地图 + 训练切片 + Routing 统一资产      ║")
    log.info("╠══════════════════════════════════════════════════════════════╣")
    log.info("║  类目:         %-44s ║", ", ".join(categories))
    log.info("║  起始步骤:     %-44s ║", f"Step {args.from_step}")
    log.info("║  subset_ratio: %-44s ║", f"{args.cot_subset_ratio:.0%}")
    log.info("║  subset_size:  %-44s ║", str(args.cot_subset_size))
    log.info("║  subset_seed:  %-44s ║", str(args.cot_subset_seed))
    log.info("║  force:        %-44s ║", str(args.force_routing_data))
    log.info("╚══════════════════════════════════════════════════════════════╝")

    t_all = time.time()
    reports = []
    for cat in categories:
        reports.append(process_category(cat, from_step=args.from_step, cot_subset_ratio=args.cot_subset_ratio, cot_subset_size=args.cot_subset_size, cot_subset_seed=args.cot_subset_seed, force_routing_data=args.force_routing_data))
    elapsed = time.time() - t_all
    log.info("")
    log.info("═" * 72)
    log.info("  总流程耗时摘要:")
    for report in reports:
        log.info("    %s: step1=%s | step2=%s | step3=%s | total=%s", report["category"], report["timing"]["step1"].get("total_sec", "skip"), report["timing"]["step2"].get("total_sec", "skip"), report["timing"]["step3"].get("total_sec", "skip"), report["timing"].get("total_sec", 0.0))
    log.info("🎉 全部完成！总耗时: %.1fs (%.1f min)", elapsed, elapsed / 60)


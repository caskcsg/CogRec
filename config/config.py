"""
config/config.py — 项目统一配置与路径注册
========================================
当前正式版协议:
  - Amazon 2014 三个 benchmark 类目
  - full sequence
  - 不采样
  - 不截断
"""

from pathlib import Path
import logging
from datetime import datetime
import re

ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "data"
MODEL_DIR  = ROOT_DIR / "model"
RAW_DIR    = DATA_DIR / "raw"
PROC_DIR   = DATA_DIR / "processed"
RESULT_DIR = DATA_DIR / "results"
FIG_DIR    = ROOT_DIR / "outputs" / "figures"
STATS_DIR  = ROOT_DIR / "outputs" / "stats"
LOG_DIR    = ROOT_DIR / "logs"

BASE_MODEL_DIR   = MODEL_DIR / "Qwen3-1-7B"
EXPAND_MODEL_DIR = MODEL_DIR / "Qwen3-1-7B-expand"

for d in [RAW_DIR, PROC_DIR, FIG_DIR, STATS_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SNAP_2014_BASE = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles"

DATASETS = {
    "Beauty": {
        "label": "Beauty",
        "color": "#E91E8C",
        "source_category": "Beauty",
        "review_filename": "reviews_Beauty_5.json.gz",
        "meta_filename": "meta_Beauty.json.gz",
        "review": f"{SNAP_2014_BASE}/reviews_Beauty_5.json.gz",
        "meta": f"{SNAP_2014_BASE}/meta_Beauty.json.gz",
        "expected_users": 22363,
        "expected_items": 12101,
        "expected_interactions": 198502,
        "expected_avg_interactions_per_user": 8.88,
    },
    "Sports": {
        "label": "Sports & Outdoors",
        "color": "#00B4D8",
        "source_category": "Sports_and_Outdoors",
        "review_filename": "reviews_Sports_and_Outdoors_5.json.gz",
        "meta_filename": "meta_Sports_and_Outdoors.json.gz",
        "review": f"{SNAP_2014_BASE}/reviews_Sports_and_Outdoors_5.json.gz",
        "meta": f"{SNAP_2014_BASE}/meta_Sports_and_Outdoors.json.gz",
        "expected_users": 35598,
        "expected_items": 18357,
        "expected_interactions": 296337,
        "expected_avg_interactions_per_user": 8.32,
    },
    "Toys": {
        "label": "Toys & Games",
        "color": "#F4A261",
        "source_category": "Toys_and_Games",
        "review_filename": "reviews_Toys_and_Games_5.json.gz",
        "meta_filename": "meta_Toys_and_Games.json.gz",
        "review": f"{SNAP_2014_BASE}/reviews_Toys_and_Games_5.json.gz",
        "meta": f"{SNAP_2014_BASE}/meta_Toys_and_Games.json.gz",
        "expected_users": 19412,
        "expected_items": 11924,
        "expected_interactions": 167597,
        "expected_avg_interactions_per_user": 8.63,
    },
}

BASE_URL = SNAP_2014_BASE
HF_REPO = "amazon-reviews-2014-compat"
HF_MIRRORS = {"snap": SNAP_2014_BASE}
HF_FILENAMES = {
    "Beauty": "Beauty",
    "Sports": "Sports_and_Outdoors",
    "Toys": "Toys_and_Games",
}
APPROX_GB = {
    "Beauty": {"review": 0.0, "meta": 0.0},
    "Sports": {"review": 0.0, "meta": 0.0},
    "Toys": {"review": 0.0, "meta": 0.0},
}

REPORTED_STATS = {
    "Beauty": {"n_users": 22363, "n_items": 12101, "n_ratings": 198502, "avg_seq_len": 8.88},
    "Sports": {"n_users": 35598, "n_items": 18357, "n_ratings": 296337, "avg_seq_len": 8.32},
    "Toys":   {"n_users": 19412, "n_items": 11924, "n_ratings": 167597, "avg_seq_len": 8.63},
}

K_CORE = 5
MAX_SEQ_LEN = 10**9
SPLIT_RATIO = (0.8, 0.1, 0.1)

NUM_SID_LAYERS = 4
CODEBOOK_SIZE  = 256
RQ_N_ITER      = 30

SID_LAYER_PREFIX = ["s_a", "s_b", "s_c", "s_d"]
SID_BEGIN_TOKEN  = "<|sid_begin|>"
SID_END_TOKEN    = "<|sid_end|>"

GRAPH_TOP_K          = 16
GRAPH_SIM_THRESH     = 0.15
HNSW_M               = 32
HNSW_EF_CONSTRUCTION = 200

HARD_SAMPLE_PREFIX_MATCH_LAYERS = 2
HARD_SAMPLE_GRAPH_DIST_THRESH   = 2

STYLE = {
    "bg":     "#0D1117",
    "panel":  "#161B22",
    "text":   "#E6EDF3",
    "muted":  "#8B949E",
    "grid":   "#21262D",
    "accent": "#58A6FF",
}
FIGSIZE_WIDE = (18, 10)
FIGSIZE_TALL = (14, 16)
FIGSIZE_SQ   = (12, 12)
DPI          = 150

EMB_MODEL_NAME       = "BAAI/bge-base-en-v1.5"
EMB_BATCH_SIZE       = 64
EMB_MAX_SEQ_LENGTH   = 512
MIN_SEQ_LEN          = 1

MIN_CLUSTER_SIZE_FOR_HNSW = 10
MIN_SEQ_LEN_FOR_SPLIT     = 1

SID_PATTERN = re.compile(
    r"(<\|sid_begin\|>"
    r"<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)><s_d_(\d+)>"
    r"<\|sid_end\|>)"
)

CATEGORIES = list(DATASETS.keys())
_THEME = {
    "bg": "#0D1117", "panel": "#161B22", "text": "#E6EDF3",
    "muted": "#8B949E", "grid": "#21262D", "accent": "#58A6FF",
}
CAT_COLORS = {cat: DATASETS[cat]["color"] for cat in CATEGORIES}

PHASES = ["download", "sid", "hnsw"]
PHASE_CONFIG = {
    "download": {"script": "data/download_amazon_dataset.py", "desc": "下载原始数据"},
    "sid": {"script": "data/raw_to_sids.py", "desc": "review/meta → full sequence → Embedding → RQ-kMeans 产码"},
    "hnsw": {"script": "data/hnsw_and_splits.py", "desc": "认知地图 → 训练切片 → Routing full data + CoT subset"},
}
PHASE_PREREQUISITES = {
    "download": [],
    "sid": [],
    "hnsw": ["codebooks.npz", "sid_codes.npy", "item_embeddings.npy", "item_mapping.json", "{cat}.pretrain.json", "{cat}_sequential_data.txt"],
}
PHASE_OUTPUTS = {
    "download": [],
    "sid": ["user_sequences.jsonl", "all_item_seqs.json", "id_mapping.json", "items_for_embedding.jsonl", "item_mapping.json", "item_embeddings.npy", "codebooks.npz", "sid_codes.npy", "sid_quality_report.json", "dataset_alignment_report.json", "{cat}.pretrain.json", "{cat}_sequential_data.txt"],
    "hnsw": ["cognitive_map.json", "training_align_data_train.parquet", "training_prediction_sid_data_train.parquet", "training_RA_train.parquet", "training_sid_routing_train.parquet", "training_RA_test_cot_eval.parquet", "training_sid_routing_test_cot_eval.parquet", "test_difficulty_labels.json", "test_difficulty_labels_cot_eval.json", "routing_quality_report.json", "cot_eval_subset_ids.json", "hnsw_pipeline_report.json"],
}

MAX_USERS = None
MAX_ITEMS = None

PAPER_PALETTE = {
    "blue":   "#2563EB",
    "red":    "#DC2626",
    "green":  "#059669",
    "orange": "#D97706",
    "purple": "#7C3AED",
    "pink":   "#EC4899",
    "gray":   "#9CA3AF",
}
METRIC_STYLES = {
    "hit@1":   {"color": PAPER_PALETTE["red"],    "marker": "o", "label": "Hit@1"},
    "hit@5":   {"color": PAPER_PALETTE["blue"],   "marker": "s", "label": "Hit@5"},
    "hit@10":  {"color": PAPER_PALETTE["green"],  "marker": "^", "label": "Hit@10"},
    "ndcg@5":  {"color": PAPER_PALETTE["orange"], "marker": "D", "label": "NDCG@5"},
    "ndcg@10": {"color": PAPER_PALETTE["purple"], "marker": "v", "label": "NDCG@10"},
}


def category_processed_dir(category: str) -> Path:
    path = PROC_DIR / category
    path.mkdir(parents=True, exist_ok=True)
    return path


def routing_reconstruct_dir(category: str, epoch: int) -> Path:
    path = category_processed_dir(category) / "reconstruct" / "routing" / f"epoch_{epoch}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ra_reconstruct_dir(category: str, epoch: int) -> Path:
    path = category_processed_dir(category) / "reconstruct" / "ra" / f"epoch_{epoch}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def stage3_reconstruct_dir(category: str, task: str, epoch: int) -> Path:
    task = task.strip().lower()
    if task == "routing":
        return routing_reconstruct_dir(category, epoch)
    if task == "ra":
        return ra_reconstruct_dir(category, epoch)
    raise ValueError(f"未知 Stage 3 重构任务: {task}")


def category_merged_model_dir(category: str) -> Path:
    return MODEL_DIR / f"merged_{category.lower()}_model"


def category_train_result_dir(category: str) -> Path:
    return ROOT_DIR / "train" / "results" / category.lower()


def eval_trie_file(category: str, task: str) -> Path:
    task = task.strip().lower()
    mapping = {
        "rec": "exact_trie_sid_test_full.pkl",
        "ra": "exact_trie_ra_test_full.pkl",
        "routing": "exact_trie_routing_test_full.pkl",
    }
    if task not in mapping:
        raise ValueError(f"未知 eval trie task: {task}")
    return category_processed_dir(category) / mapping[task]


def sid_token_name(layer: int, code: int) -> str:
    return f"<{SID_LAYER_PREFIX[layer - 1]}_{code}>"


def format_sid(codes: list) -> str:
    tokens = "".join(f"<{SID_LAYER_PREFIX[i]}_{codes[i]}>" for i in range(NUM_SID_LAYERS))
    return f"{SID_BEGIN_TOKEN}{tokens}{SID_END_TOKEN}"


def get_all_sid_tokens() -> list:
    tokens = [SID_BEGIN_TOKEN, SID_END_TOKEN]
    for layer in range(NUM_SID_LAYERS):
        for code in range(CODEBOOK_SIZE):
            tokens.append(f"<{SID_LAYER_PREFIX[layer]}_{code}>")
    return tokens


def setup_logging(name: str = "project") -> Path:
    log_file = LOG_DIR / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if not root.handlers:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
        root.addHandler(console)
        fh = logging.FileHandler(str(log_file), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)-8s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        root.addHandler(fh)
    logging.getLogger().info("日志文件: %s", log_file)
    return log_file

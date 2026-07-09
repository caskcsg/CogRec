#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging
import hashlib
import pickle
import random
import time
import numpy as np
import re
import pickle
from pathlib import Path
from config.config import PROC_DIR
import statistics

USER_PREFIX = "The user has purchased the following items: "

# ====================== 项目配置 ======================
def extract_all_sids_from_text(text):
    sid_pattern = r'<\|sid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><s_d_\d+><\|sid_end\|>'
    matches = re.findall(sid_pattern, text)
    return matches

def extract_sid_from_text(text):
    sid_pattern = r'<\|sid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><s_d_\d+><\|sid_end\|>'
    match = re.search(sid_pattern, text)
    if match:
        return match.group(0)
    return text.strip()

def parse_args():
    parser = argparse.ArgumentParser(description="Optimized CoT Model Hit Rate Test (Single & Multi-GPU Ready)")

    parser.add_argument("--category", type=str, default="Beauty",
                        choices=["Beauty", "Sports", "Toys"],
                        help="Dataset category for auto-pathing")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--merged_model_path", type=str, required=True, help="Merged model path")
    parser.add_argument("--additional_lora_path", type=str, default=None, help="Optional additional LoRA path")

    parser.add_argument("--test_parquet_file", type=str, default=None,
                        help="Test parquet file path (leave empty to auto-resolve via category)")

    parser.add_argument("--test_batch_size", type=int, default=1, help="Test batch size")
    parser.add_argument("--num_beams", type=int, default=20, help="Number of beams for beam search")
    parser.add_argument("--sample_num", type=int, default=-1, help="test sample number, -1 represents using all test data")
    parser.add_argument("--sample_offset", type=int, default=0, help="sample offset for multi-GPU parallel processing")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID for logging purposes")
    parser.add_argument("--metrics", type=str, default="hit@1,hit@5,hit@10,ndcg@5,ndcg@10", help="test metrics, separate by comma")

    parser.add_argument("--think_max_tokens", type=int, default=128, help="max new tokens for thinking stage")
    parser.add_argument("--sid_max_tokens", type=int, default=20, help="max new tokens for SID generation stage")

    parser.add_argument("--think_temperature", type=float, default=0.8, help="temperature for CoT thinking generation")
    parser.add_argument("--think_top_p", type=float, default=0.95, help="top_p for CoT thinking generation")

    parser.add_argument("--sid_temperature", type=float, default=0.6, help="temperature for SID generation")
    parser.add_argument("--sid_top_p", type=float, default=0.9, help="top_p for SID generation")
    
    parser.add_argument("--num_thinking_samples", type=int, default=4, help="number of thinking samples to generate")
    parser.add_argument("--num_beams_per_sample", type=int, default=4, help="number of beams for each thinking sample")

    parser.add_argument("--print_generations", action="store_true", default=False, help="print prompts, think, and response candidates")
    parser.add_argument("--log_file", type=str, default="./logs/cot_optimized_test.log", help="all output log file path")
    parser.add_argument("--global_trie_file", type=str, required=True, help="离线预计算的 exact trie 文件 (必填)")
    parser.add_argument(
        "--trie_source_parquet_file",
        type=str,
        default=None,
        help="用于校验 trie metadata 的源 parquet。subset 评估时传 full parquet；不传则默认用 test_parquet_file。"
    )
    parser.add_argument("--system_prompt", type=str,
                        default="You are a professional recommendation expert who needs to recommend the next possible purchase for users based on their purchase history. Please predict the most likely next product that the user will purchase based on the user's historical purchase information.",
                        help="System prompt (must match training)")
    parser.add_argument("--serial_sid", action="store_true", default=False, help="Force serial SID generation (Low VRAM mode)")
    parser.add_argument("--max_history_items", type=int, default=0,
                    help="Maximum number of history items to keep in eval prompt. 0 means no truncation.")
    parser.add_argument("--history_truncation_side", type=str, default="tail",
                    choices=["tail", "head"],
                    help="Keep tail (most recent) or head (earliest) history items.")
    
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False



def compute_parquet_fingerprint(df):
    cols = [c for c in ['sample_id', 'description', 'groundtruth'] if c in df.columns]
    if not cols:
        raise ValueError("Parquet 缺少 fingerprint 所需列；至少需要 sample_id/description/groundtruth 之一")
    h = hashlib.sha256()
    h.update(f"rows={len(df)}|cols={'|'.join(cols)}".encode('utf-8'))
    for col in cols:
        series = df[col].fillna('').astype(str)
        for value in series.tolist():
            h.update(value.encode('utf-8', errors='ignore'))
            h.update(b"\n")
    return {
        'row_count': int(len(df)),
        'columns': cols,
        'sha256': h.hexdigest(),
    }


def validate_trie_metadata(trie_data, parquet_df, parquet_file, logger):
    meta = trie_data.get('source_metadata')
    if not meta:
        raise ValueError(f"Trie 文件缺少 source_metadata: {parquet_file}")
    actual = compute_parquet_fingerprint(parquet_df)
    expected_rows = int(meta.get('row_count', -1))
    expected_hash = meta.get('sha256')
    if expected_rows != actual['row_count']:
        raise ValueError(
            f"Trie 样本数不匹配: trie={expected_rows}, parquet={actual['row_count']}, file={parquet_file}"
        )
    if expected_hash != actual['sha256']:
        raise ValueError(
            f"Trie fingerprint 不匹配: trie={expected_hash}, parquet={actual['sha256']}, file={parquet_file}"
        )
    logger.info(f"✅Trie metadata verified: rows={actual['row_count']}, sha256={actual['sha256']}")


def setup_logging(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('cot_test')
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def format_chat_prompt_think_stage(user_content, system_message):
    return f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"

def extract_thinking_content(generated_text, user_content):
    if "<think>" in generated_text:
        if user_content in generated_text:
            assistant_part = generated_text.split(user_content)[-1]
        else:
            assistant_part = generated_text
        if "<think>" in assistant_part:
            think_part = assistant_part.split("<think>")[-1]
            if "</think>" in think_part:
                think_part = think_part.split("</think>")[0]
            return think_part.strip()
    return ""

def format_chat_prompt_sid_stage(user_content, thinking_content, system_message):
    return f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n<think>\n{thinking_content}\n</think>\n"

def split_history_items(user_text: str):
    if not isinstance(user_text, str) or not user_text.startswith(USER_PREFIX):
        return []

    body = user_text[len(USER_PREFIX):].strip()
    if not body:
        return []

    # 只在“; + 下一个 SID item 开始”处分割，避免把字段内部文本误切开
    parts = re.split(r';\s*(?=<\|sid_begin\|>)', body)
    parts = [p.strip().rstrip(';').strip() for p in parts if p.strip()]
    return parts

def rebuild_user_text(items):
    if not items:
        return USER_PREFIX
    return USER_PREFIX + "; ".join(items) + ";"

def truncate_history_by_items(user_text: str, max_history_items: int, side: str = "tail") -> str:
    if max_history_items <= 0:
        return user_text

    items = split_history_items(user_text)
    if not items or len(items) <= max_history_items:
        return user_text

    if side == "tail":
        items = items[-max_history_items:]
    else:
        items = items[:max_history_items]

    return rebuild_user_text(items)

def count_history_items(user_text: str) -> int:
    return len(split_history_items(user_text))
    
def load_merged_model(model_path, additional_lora_path=None, logger=None):
    logger.info(f"Loading merged model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    if torch.cuda.is_available():
        device = f"cuda:{torch.cuda.current_device()}"
        logger.info(f"🔧 Forcing model to GPU: {device}")
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
        # Verify GPU placement
        first_param_device = next(model.parameters()).device
        if 'cuda' in str(first_param_device):
            logger.info(f"✅ Confirmed: Model is on {first_param_device}")
        else:
            logger.error(f"❌ Failed: Model is still on {first_param_device}")
            raise RuntimeError("Failed to move model to GPU")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    
    logger.info(f"Merged model loaded successfully, tokenizer vocab size: {tokenizer.vocab_size}")
    
    if additional_lora_path and os.path.exists(additional_lora_path):
        logger.info(f"Loading additional LoRA from: {additional_lora_path}")
        model = PeftModel.from_pretrained(model, additional_lora_path)
    
    return model, tokenizer

class ParquetTestDataset(Dataset):
    def __init__(self, parquet_file, sample_num=-1, sample_offset=0, source_parquet_file=None, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.parquet_file = parquet_file
        self.source_parquet_file = source_parquet_file or parquet_file

        self.logger.info(f"Loading eval parquet: {self.parquet_file}")
        eval_df = pd.read_parquet(self.parquet_file)

        # 用于 trie metadata 校验的“源 parquet”
        # - full eval：source == eval
        # - subset eval：source = full parquet，eval = subset parquet
        if self.source_parquet_file == self.parquet_file:
            self.source_df = eval_df.copy()
        else:
            self.logger.info(f"Loading trie-source parquet for validation: {self.source_parquet_file}")
            self.source_df = pd.read_parquet(self.source_parquet_file)

        # 真正参与本 GPU 评估的 shard
        self.df = eval_df
        if sample_offset > 0:
            self.df = self.df.iloc[sample_offset:].reset_index(drop=True)
        if sample_num > 0 and sample_num < len(self.df):
            self.df = self.df.iloc[:sample_num].reset_index(drop=True)

        self.logger.info(
            f"Loaded eval samples={len(self.df)} | trie-source samples={len(self.source_df)}"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'input_ids': row['description'],
            'labels': row['groundtruth'],
            'user_id': row.get('user_id', f'user_{idx}')
        }

    def get_prefix_allowed_tokens_fn(self, tokenizer, global_trie_file=None):
        if not global_trie_file or not os.path.exists(global_trie_file):
            raise FileNotFoundError(f"Global trie file not found: {global_trie_file}")

        self.logger.info(f"Loading pre-computed exact trie from: {global_trie_file}")
        with open(global_trie_file, 'rb') as f:
            trie_data = pickle.load(f)

        # 关键修复：始终按“源 parquet”校验 trie，而不是按当前 shard / subset 校验
        validate_trie_metadata(
            trie_data,
            self.source_df,
            self.source_parquet_file,
            self.logger
        )

        trie_type = trie_data.get('trie_type', None)
        if trie_type is not None and trie_type != 'exact':
            raise ValueError(f"Expected exact trie file, but got trie_type='{trie_type}'")

        allowed_tokens = trie_data['exact_trie']
        sep = tokenizer("</think>\n", add_special_tokens=False)["input_ids"]

        def find_last_sublist(lst, sub):
            if not sub:
                return None
            n, m = len(lst), len(sub)
            for start in range(n - m, -1, -1):
                if lst[start:start + m] == sub:
                    return start
            return None

        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()
            pos = find_last_sublist(sentence, sep)
            if pos is None:
                return list(tokenizer.get_vocab().values())

            sid_pos = len(sentence) - (pos + len(sep))
            if sid_pos == 0:
                return list(allowed_tokens.get(0, {}).keys()) or list(tokenizer.get_vocab().values())

            if sid_pos > 0 and len(sentence[pos + len(sep):]) >= sid_pos:
                prev_token = sentence[pos + len(sep):][sid_pos - 1]
                prev_pos = sid_pos - 1
                if prev_pos in allowed_tokens and prev_token in allowed_tokens[prev_pos]:
                    return allowed_tokens[prev_pos][prev_token]

            return list(tokenizer.get_vocab().values())

        prefix_allowed_tokens_fn._allowed_tokens = allowed_tokens
        prefix_allowed_tokens_fn._sep_tokens = sep
        return prefix_allowed_tokens_fn

class TestCollator:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        # CRITICAL FIX: Restore pad_token_id fallback
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"
    
    def __call__(self, batch):
        return {
            "user_contents": [d["input_ids"] for d in batch],
            "targets": [d["labels"] for d in batch]
        }

def batch_generate_thinking_optimized(model, tokenizer, user_contents, args, logger):
    logger.info("🚀 Optimized batch thinking generation started...")
    all_think_prompts, batch_mapping = [], []
    for sample_idx, user_content in enumerate(user_contents):
        think_prompt = format_chat_prompt_think_stage(user_content, args.system_prompt)
        for thinking_idx in range(args.num_thinking_samples):
            all_think_prompts.append(think_prompt)
            batch_mapping.append((sample_idx, thinking_idx))

    enc_think_batch = tokenizer(all_think_prompts, return_tensors="pt", padding=True, truncation=True, max_length=tokenizer.model_max_length)
    enc_think_batch = {k: v.to(model.device) for k, v in enc_think_batch.items()}

    think_outputs = model.generate(
        input_ids=enc_think_batch["input_ids"],
        attention_mask=enc_think_batch.get("attention_mask", None),
        max_new_tokens=args.think_max_tokens,
        num_beams=1,
        do_sample=True,
        temperature=args.think_temperature,
        top_p=args.think_top_p,
        return_dict_in_generate=True,
        output_scores=False,
        early_stopping=False,
        use_cache=True,
        output_hidden_states=False
    )

    think_decoded_all = tokenizer.batch_decode(think_outputs["sequences"], skip_special_tokens=True)
    all_thinking_contents = [[] for _ in range(len(user_contents))]
    for i, (sample_idx, thinking_idx) in enumerate(batch_mapping):
        thinking = extract_thinking_content(think_decoded_all[i], user_contents[sample_idx])
        all_thinking_contents[sample_idx].append(thinking)
    return all_thinking_contents

def batch_generate_sid(model, tokenizer, user_contents, all_thinking_contents, prefix_allowed_tokens_fn, args, logger):
    bs, num_thinking, num_beams = len(user_contents), args.num_thinking_samples, args.num_beams_per_sample
    all_sid_prompts = []
    for sample_idx in range(bs):
        for thinking_idx in range(num_thinking):
            all_sid_prompts.append(format_chat_prompt_sid_stage(user_contents[sample_idx], all_thinking_contents[sample_idx][thinking_idx], args.system_prompt))

    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc_all = tokenizer(all_sid_prompts, return_tensors="pt", padding=True, truncation=True, max_length=tokenizer.model_max_length)
    enc_all = {k: v.to(model.device) for k, v in enc_all.items()}
    padded_len = enc_all["input_ids"].shape[1]
    tokenizer.padding_side = orig_padding_side

    generate_kwargs = {
        "input_ids": enc_all["input_ids"],
        "attention_mask": enc_all["attention_mask"],
        "max_new_tokens": args.sid_max_tokens,
        "num_beams": num_beams,
        "num_return_sequences": num_beams,
        "output_scores": True,
        "return_dict_in_generate": True,
        "early_stopping": True,
        "use_cache": True,
    }

    if prefix_allowed_tokens_fn is not None:
        allowed_tokens = getattr(prefix_allowed_tokens_fn, '_allowed_tokens', None)
        if allowed_tokens is not None:
            all_tokens_list = list(range(len(tokenizer)))
            first_sid_tokens = list(allowed_tokens.get(0, {}).keys()) or all_tokens_list

            def fast_prefix_fn(batch_id, sentence):
                sid_pos = len(sentence) - padded_len
                if sid_pos <= 0: return first_sid_tokens
                prev_pos = sid_pos - 1
                prev_token = int(sentence[padded_len + prev_pos])
                if prev_pos in allowed_tokens and prev_token in allowed_tokens[prev_pos]:
                    return allowed_tokens[prev_pos][prev_token]
                return all_tokens_list
            generate_kwargs["prefix_allowed_tokens_fn"] = fast_prefix_fn
        else:
            generate_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn

    try:
        output = model.generate(**generate_kwargs)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning(f"⚠️ Batch SID OOM. Falling back to serial.")
            torch.cuda.empty_cache()
            return _serial_sid_fallback(model, tokenizer, user_contents, all_thinking_contents, prefix_allowed_tokens_fn, args, logger)
        raise

    all_decoded = tokenizer.batch_decode(output["sequences"], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    scores_out = output.get("sequences_scores", None)
    all_scores = [float(s) for s in (scores_out.detach().cpu().tolist() if hasattr(scores_out, 'detach') else scores_out)] if scores_out is not None else [0.0] * len(all_decoded)
    
    return all_decoded, all_scores

def _serial_sid_fallback(model, tokenizer, user_contents, all_thinking_contents, prefix_allowed_tokens_fn, args, logger):
    all_decoded, all_scores = [], []
    torch.cuda.empty_cache()
    for si in range(len(user_contents)):
        for ti in range(args.num_thinking_samples):
            out, _ = generate_sid_standard(model, tokenizer, user_contents[si], all_thinking_contents[si][ti], prefix_allowed_tokens_fn, args, logger)
            dec = tokenizer.batch_decode(out["sequences"], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            sc = out.get("sequences_scores", None)
            all_scores.extend([float(s) for s in (sc.detach().cpu().tolist() if hasattr(sc, 'detach') else sc)] if sc is not None else [0.0] * len(dec))
            all_decoded.extend(dec)
            del out
            torch.cuda.empty_cache()
    return all_decoded, all_scores

def generate_sid_standard(model, tokenizer, user_content, thinking_content, prefix_allowed_tokens_fn, args, logger):
    sid_prompt = format_chat_prompt_sid_stage(user_content, thinking_content, args.system_prompt)
    enc_sid = tokenizer([sid_prompt], return_tensors="pt", padding=True, truncation=True, max_length=tokenizer.model_max_length)
    enc_sid = {k: v.to(model.device) for k, v in enc_sid.items()}
    num_beams = args.num_beams_per_sample
    
    generate_kwargs = {
        "input_ids": enc_sid["input_ids"],
        "attention_mask": enc_sid.get("attention_mask", None),
        "max_new_tokens": args.sid_max_tokens,
        "num_beams": num_beams,
        "num_return_sequences": num_beams,
        "output_scores": True,
        "return_dict_in_generate": True,
        "early_stopping": True,
        "use_cache": True,
    }
    if prefix_allowed_tokens_fn is not None:
        generate_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn
    
    try:
        output = model.generate(**generate_kwargs)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            logger.warning(f"CUDA OOM with beam={num_beams}. Reducing beam size.")
            num_beams -= 1
            if num_beams < 1: raise RuntimeError("Beam search OOM even with beam=1") from e
            torch.cuda.empty_cache()
            generate_kwargs["num_beams"] = num_beams
            generate_kwargs["num_return_sequences"] = num_beams
            output = model.generate(**generate_kwargs)
        else: raise
    return output, num_beams

def process_unique_top10_candidates(predictions, scores, effective_num_beams):
    batch_size = len(predictions) // effective_num_beams
    new_predictions, new_scores = [], []
    for b in range(batch_size):
        start, end = b * effective_num_beams, (b + 1) * effective_num_beams
        batch_preds, batch_scores = predictions[start:end], scores[start:end]
        sid_to_score = {}
        for pred, score in zip(batch_preds, batch_scores):
            sid = extract_sid_from_text(pred.split("</think>")[-1].strip().replace(" ", ""))
            sid_to_score[sid] = max(sid_to_score.get(sid, score), score)
        
        top10_items = sorted(sid_to_score.items(), key=lambda x: x[1], reverse=True)[:10]
        sample_preds, sample_scores = [], []
        for sid, score in top10_items:
            full_pred = (batch_preds[0].split("</think>")[0] + "</think>\n" + sid) if batch_preds else sid
            sample_preds.append(full_pred)
            sample_scores.append(score)
            
        while len(sample_preds) < 10:
            if sample_preds:
                sample_preds.append(sample_preds[-1])
                sample_scores.append(sample_scores[-1] - ((len(sample_preds) - len(top10_items)) * 0.1))
            else:
                sample_preds.append("")
                sample_scores.append(-1000.0)
        new_predictions.extend(sample_preds)
        new_scores.extend(sample_scores)
    return new_predictions, new_scores

def get_topk_results(predictions, scores, targets, k):
    results = []
    predictions = [extract_sid_from_text(_.split("</think>")[-1].strip().replace(" ", "")) for _ in predictions]
    for b in range(len(targets)):
        batch_seqs, batch_scores = predictions[b * k: (b + 1) * k], scores[b * k: (b + 1) * k]
        sorted_pairs = sorted(zip(batch_seqs, batch_scores), key=lambda x: x[1], reverse=True)
        target_item = extract_sid_from_text(targets[b])
        results.append([1 if pred_seq == target_item else 0 for pred_seq, _ in sorted_pairs])
    return results

def get_metrics_results(topk_results, metrics):
    res = {}
    for m in metrics:
        k = int(m.split("@")[1])
        if m.lower().startswith("hit"):
            res[m] = sum(1 for row in topk_results if len(row) >= k and max(row[:k]) == 1) / len(topk_results)
        elif m.lower().startswith("ndcg"):
            res[m] = sum(sum(1.0 / np.log2(i + 2) for i in range(min(k, len(row))) if row[i] == 1) / (1.0 / np.log2(2)) for row in topk_results) / len(topk_results)
    return res

def run_evaluation(args):
    set_seed(args.seed)
    logger = setup_logging(args.log_file)
    logger.info(f"🚀 Starting CoT-Enhanced Model Hit Rate Evaluation [GPU {args.gpu_id}]")

    # CRITICAL FIX: Restore Configuration Logging
    logger.info("=" * 80)
    logger.info("📋 EVALUATION CONFIGURATION")
    logger.info("=" * 80)
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    logger.info("=" * 80)

    if args.test_parquet_file is None:
        args.test_parquet_file = str(Path(PROC_DIR) / args.category / "training_prediction_sid_data_test.parquet")
        logger.info(f"Auto path resolved: {args.test_parquet_file}")

    final_model, tokenizer = load_merged_model(args.merged_model_path, args.additional_lora_path, logger)
    final_model.eval()

    test_dataset = ParquetTestDataset(
        args.test_parquet_file,
        args.sample_num,
        args.sample_offset,
        source_parquet_file=args.trie_source_parquet_file,
        logger=logger,
    )
    prefix_allowed_tokens_fn = test_dataset.get_prefix_allowed_tokens_fn(tokenizer, args.global_trie_file)

    collator = TestCollator(args, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, collate_fn=collator, shuffle=False, num_workers=0, pin_memory=True)
    
    metrics = args.metrics.split(",")
    all_topk_results = []
    total = 0
    start_time = time.time()

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="CoT Testing")
        for step, batch in enumerate(progress_bar):
            user_contents, targets = batch["user_contents"], batch["targets"]
            bs = len(targets)

            # user_contents 构造完成后，紧接着插入
            orig_item_counts = [count_history_items(x) for x in user_contents]
            
            if args.max_history_items > 0:
                user_contents = [
                    truncate_history_by_items(x, args.max_history_items, args.history_truncation_side)
                    for x in user_contents
                ]
            
            new_item_counts = [count_history_items(x) for x in user_contents]
            truncated_count = sum(int(a != b) for a, b in zip(orig_item_counts, new_item_counts))
            
            logger.info(
                f"History truncation: max_history_items={args.max_history_items}, "
                f"side={args.history_truncation_side}, truncated_samples={truncated_count}/{len(user_contents)}"
            )
            logger.info(
                f"History item count stats | "
                f"before: mean={statistics.mean(orig_item_counts):.2f}, max={max(orig_item_counts)} | "
                f"after: mean={statistics.mean(new_item_counts):.2f}, max={max(new_item_counts)}"
            )
            
            # CRITICAL FIX: Restore ETA Tracking
            current_step, total_steps = step + 1, len(test_loader)
            elapsed = time.time() - start_time
            if current_step > 0:
                avg_time = elapsed / current_step
                remaining_time = avg_time * (total_steps - current_step)
                elapsed_str, remaining_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}", f"{int(remaining_time//60):02d}:{int(remaining_time%60):02d}"
                progress_pct = current_step / total_steps
                bar = '█' * int(progress_pct * 10) + '░' * (10 - int(progress_pct * 10))
                logger.info(f"Testing: {progress_pct*100:3.0f}%|{bar}| {current_step}/{total_steps} [{elapsed_str}<{remaining_str}, {avg_time:.2f}s/it]")

            all_thinking_contents = batch_generate_thinking_optimized(final_model, tokenizer, user_contents, args, logger)

            # 释放 think 阶段的 KV cache，为 SID beam search 腾出显存
            torch.cuda.empty_cache()

            if args.serial_sid:
                logger.info("🎯 Stage 2: Serial SID generation...")
                all_decoded, all_scores = _serial_sid_fallback(final_model, tokenizer, user_contents, all_thinking_contents, prefix_allowed_tokens_fn, args, logger)
            else:
                logger.info("🎯 Stage 2: Batch SID generation...")
                all_decoded, all_scores = batch_generate_sid(final_model, tokenizer, user_contents, all_thinking_contents, prefix_allowed_tokens_fn, args, logger)

            effective_num_beams = args.num_thinking_samples * args.num_beams_per_sample
            decoded, scores_list = process_unique_top10_candidates(all_decoded, all_scores, effective_num_beams)
            effective_num_beams = 10
            
            # CRITICAL FIX: Restore Print Generations Logging
            if args.print_generations:
                for i in range(bs):
                    start, end = i * effective_num_beams, i * effective_num_beams + effective_num_beams
                    cands, cand_scores = decoded[start:end], scores_list[start:end]
                    
                    logger.info(f"----- CoT-ENHANCED SAMPLE {step*bs + i} -----")
                    logger.info(f"USER INPUT:\n{user_contents[i]}\n")
                    for t_idx in range(min(5, args.num_thinking_samples)):
                        logger.info(f"THINKING {t_idx+1}/{args.num_thinking_samples}:\n{all_thinking_contents[i][t_idx]}\n")
                    logger.info("UNIQUE TOP-10 SID_CANDIDATES:")
                    for j, (c, sc) in enumerate(zip(cands, cand_scores)):
                        logger.info(f"  Rank {j+1}: score={sc:.4f} → {extract_sid_from_text(c.split('</think>')[-1])}")
                    logger.info(f"\nTARGET:\n{targets[i]}\n" + "-" * 80)

            all_topk_results.extend(get_topk_results(decoded, scores_list, targets, effective_num_beams))
            total += bs

            # CRITICAL FIX: Restore 20-step intermediate metrics
            if (step + 1) % 20 == 0:
                temp_metrics = get_metrics_results(all_topk_results, metrics)
                logger.info("=" * 50)
                logger.info(f"📊 PROGRESS REPORT - Step {step+1}/{len(test_loader)} (Samples: {total})")
                for metric, value in temp_metrics.items():
                    logger.info(f"  {metric:>10}: {value:.4f}")
                logger.info("=" * 50)

    final_metrics_results = get_metrics_results(all_topk_results, metrics)
    logger.info("=" * 60 + "\n🎯 Final CoT Hit Rate Results:\n" + "=" * 60)
    for metric, value in final_metrics_results.items():
        logger.info(f"{metric:>10}: {value:.4f}")
    logger.info("=" * 60 + "\n✅ Evaluation completed successfully!")
    return final_metrics_results

if __name__ == "__main__":
    success = True
    try:
        run_evaluation(parse_args())
    except Exception as e:
        import traceback
        traceback.print_exc()
        success = False
    sys.exit(0 if success else 1)

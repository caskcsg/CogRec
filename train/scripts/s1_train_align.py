#!/usr/bin/env python3
"""
s1_train_align.py — 统一单卡 & 多卡 Stage 1 脚本（已支持 category 自适应）
支持：
  --single_gpu_mode True   → A800 单卡优化（bf16 + padding=False + debug）
  --single_gpu_mode False  → DeepSpeed 多卡原版（默认）
"""

import torch
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from datasets import Dataset
from typing import Optional
from peft import TrainableTokensConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from config.config import PROC_DIR

@dataclass
class ScriptArguments:
    model_dir: str = "../model/Qwen3-1-7B-expand"
    category: str = field(default="Beauty", metadata={"help": "Beauty / Sports / Toys"})
    train_data_path: Optional[str] = field(default=None)
    val_data_path: Optional[str] = field(default=None)
    single_gpu_mode: bool = False   # 单卡优化开关


def prepare_dataset(data_path, sample_size=None, local_rank=0):
    if local_rank == 0:
        print(f"Loading parquet file: {data_path}")
    data_pq = pd.read_parquet(data_path)
    if local_rank == 0:
        print(f"Data shape: {data_pq.shape}")
        print(f"Columns: {list(data_pq.columns)}")

    if sample_size is not None and len(data_pq) > sample_size:
        if local_rank == 0:
            print(f"Sampling {sample_size} samples...")
        data_pq = data_pq.head(sample_size)

    texts = data_pq['description'].tolist()
    if local_rank == 0:
        print(f"Total texts: {len(texts)}")
        print("\nFirst 3 text examples:")
        for i, text in enumerate(texts[:3]):
            print(f"  [{i}]: {text}")
    
    return Dataset.from_dict({'text': texts})


def tokenize_function(examples, tokenizer, padding_strategy):
    tokenized = tokenizer(
        examples['text'],
        padding=padding_strategy,
        truncation=True,
        max_length=4096,
        add_special_tokens=True,
        return_attention_mask=True,
    )
    return tokenized


def get_special_tokens():
    special_tokens = ['<|sid_begin|>', '<|sid_end|>']
    for prefix in ['s_a', 's_b', 's_c', 's_d']:
        for i in range(256):
            special_tokens.append(f'<{prefix}_{i}>')
    return special_tokens


def debug_inspect_data(tokenizer, train_dataset, num_samples=4):
    """单卡专用调试（长度、SID token 统计、截断警告）"""
    print("\n" + "=" * 60)
    print("  DEBUG: Tokenization Inspection (Single-GPU Mode)")
    print("=" * 60)

    sid_begin_id = tokenizer.convert_tokens_to_ids('<|sid_begin|>')
    sid_end_id = tokenizer.convert_tokens_to_ids('<|sid_end|>')

    for i in range(min(num_samples, len(train_dataset))):
        sample = train_dataset[i]
        input_ids = sample['input_ids']
        attention_mask = sample['attention_mask']

        print(f"\n--- Sample {i} ---")
        print(f"  input_ids length (true): {len(input_ids)}")
        print(f"  non-pad tokens: {sum(attention_mask)}")

        if len(input_ids) >= 4096:
            print(f"  ⚠️  WARNING: Sequence at max_length=4096, likely truncated!")
        else:
            print(f"  ✅ No truncation")

        first_n = input_ids[:150]
        decoded = tokenizer.decode(first_n, skip_special_tokens=False)
        print(f"  First 150 tokens: {decoded[:600]}")

        sid_begin_count = input_ids.count(sid_begin_id)
        sid_end_count = input_ids.count(sid_end_id)
        print(f"  SID count: {sid_begin_count} begin, {sid_end_count} end")

        sid_comp_count = sum(1 for tid in input_ids 
                             if tokenizer.convert_ids_to_tokens(tid).startswith('<s_'))
        print(f"  SID components: {sid_comp_count} (expected: {sid_begin_count * 4})")

    # 统计
    n_check = min(100, len(train_dataset))
    lengths = [len(train_dataset[i]['input_ids']) for i in range(n_check)]
    print(f"\n--- Sequence length stats (first {n_check} samples) ---")
    print(f"  Min: {min(lengths)}, Max: {max(lengths)}, Mean: {sum(lengths)/len(lengths):.0f}")
    truncated = sum(1 for l in lengths if l >= 4096)
    print(f"  Truncated: {truncated}/{n_check}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.label_names = ["labels"]

    local_rank = training_args.local_rank

   # ====================== 数据路径自适应 ======================
    if script_args.train_data_path is None:
        script_args.train_data_path = str(PROC_DIR / script_args.category / "training_align_data_train.parquet")
    if script_args.val_data_path is None:
        script_args.val_data_path = str(PROC_DIR / script_args.category / "training_align_data_val.parquet")

    model_dir = Path(script_args.model_dir).resolve()
    train_data_path = Path(script_args.train_data_path).resolve()
    val_data_path = Path(script_args.val_data_path).resolve()

    if not model_dir.exists() or not train_data_path.exists() or not val_data_path.exists():
        raise FileNotFoundError("Model or data path not found")

    if local_rank == 0:
        print(f"Mode: {'Single-GPU (A800 optimized)' if script_args.single_gpu_mode else 'Multi-GPU / DeepSpeed'}")
        print(f"Category: {script_args.category}")
        print(f"Using model_dir: {model_dir}")

    # ====================== 单卡 vs 多卡切换 ======================
    if script_args.single_gpu_mode:
        # 单卡适配（A800 专用）
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        padding_strategy = False
        do_debug = True
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()
    else:
        # 多卡 / DeepSpeed
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        padding_strategy = False #由于数据分布不均，这里将原OneRec的longgest改为false加速但不影响训练结果
        do_debug = False
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    tokenizer.pad_token = tokenizer.eos_token

    if local_rank == 0:
        print(f"Model dtype: {next(model.parameters()).dtype}")
        print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # ====================== PEFT ======================
    special_tokens = get_special_tokens()
    tokenized_special_tokens = tokenizer.convert_tokens_to_ids(special_tokens)

    valid_special_token_ids = [
        tid for tid in tokenized_special_tokens 
        if tid != tokenizer.unk_token_id
    ]

    if local_rank == 0:
        print(f"Training {len(valid_special_token_ids)} special SID tokens")

    lora_config = TrainableTokensConfig(
        token_indices=valid_special_token_ids,
        target_modules=["embed_tokens"],
        init_weights=True
    )
    
    model = get_peft_model(model, lora_config)
    if local_rank == 0:
        model.print_trainable_parameters()

    # ====================== 数据处理 ======================
    if local_rank == 0:
        print("\nLoading datasets...")
    train_dataset = prepare_dataset(train_data_path, local_rank=local_rank)
    val_dataset = prepare_dataset(val_data_path, local_rank=local_rank)

    if local_rank == 0:
        print("Tokenizing...")

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, padding_strategy),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train"
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer, padding_strategy),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing val"
    )

    # 单卡调试
    if do_debug and local_rank == 0:
        debug_inspect_data(tokenizer, train_dataset)

    # ====================== Trainer ======================
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    if local_rank == 0:
        print("\nStarting training...")

    trainer.train()

    if training_args.local_rank in [-1, 0]:
        print("\n" + "=" * 60)
        print("  Training completed!")
        print("  Best model selected by shell save_best_model().")
        print("  Skipping final evaluate/save to avoid DeepSpeed deadlock.")
        print("=" * 60)
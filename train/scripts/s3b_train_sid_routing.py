#!/usr/bin/env python3
"""
s3b_train_sid_routing.py — 统一 Routing 严格训练脚本（单卡 / 多卡）
============================================================================

统一版说明:
  1. 只使用 sid_routing_think / 空 think，不允许 fallback 到 RA template
  2. 默认训练集为 training_sid_routing_train.parquet
  3. 默认验证集为 training_sid_routing_val.parquet

正式版训练约定：
  - ChatML 格式
  - max_length=4096
  - loss 从 <|im_start|>user 起算
  - padding 区域不额外 mask（对齐 OneRec 协议）
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from config.config import PROC_DIR


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    data_path: Optional[str] = field(default=None)
    category: str = field(default="Beauty", metadata={"help": "Beauty / Sports / Toys"})
    use_lora: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=64)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    sample_size: Optional[int] = field(default=None)


SYSTEM_MESSAGE = (
    "You are a professional recommendation expert who needs to recommend "
    "the next possible purchase for users based on their purchase history. "
    "Please predict the most likely next product that the user will purchase "
    "based on the user's historical purchase information. "
    "Express your prediction as a Semantic ID by navigating the SID hierarchy."
)


def get_special_tokens() -> List[str]:
    tokens = ["<|sid_begin|>", "<|sid_end|>"]
    for prefix in ["s_a", "s_b", "s_c", "s_d"]:
        for i in range(256):
            tokens.append(f"<{prefix}_{i}>")
    return tokens


def _normalize_think(value: object) -> str:
    if value is None:
        return "<think>\n\n</think>"
    text = str(value).strip()
    if not text:
        return "<think>\n\n</think>"
    return text


def prepare_chat_dataset(data_path: str, sample_size: Optional[int] = None, local_rank: int = 0) -> Dataset:
    if local_rank == 0:
        print(f"Loading: {data_path}")
    data_pq = pd.read_parquet(data_path)
    if local_rank == 0:
        print(f"Shape: {data_pq.shape}")

    if sample_size and len(data_pq) > sample_size:
        data_pq = data_pq.head(sample_size)

    texts: List[str] = []
    cot_steps = data_pq["cot_steps"].tolist() if "cot_steps" in data_pq.columns else None

    for _, row in data_pq.iterrows():
        think_content = _normalize_think(row.get("sid_routing_think", None))
        assistant_content = f"{think_content}\n{row['groundtruth']}"
        formatted = (
            f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n"
            f"<|im_start|>user\n{row['description']}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_content}<|im_end|>\n"
        )
        texts.append(formatted)

    if local_rank == 0:
        print(f"Total samples: {len(texts)}")
        n_empty = sum(1 for t in texts if "<think>\n\n</think>" in t)
        n_non_empty = len(texts) - n_empty
        print(f"  Empty think: {n_empty} | Non-empty think: {n_non_empty}")
        if cot_steps is not None:
            from collections import Counter
            print(f"  cot_steps: {dict(Counter(cot_steps))}")
        for i, t in enumerate(texts[:2]):
            print(f"\n  Sample [{i}] ({len(t)} chars): {t[:600]}...")

    return Dataset.from_dict({"text": texts})


def tokenize_function(examples: Dict[str, List[str]], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    print(f"Tokenizing batch of {len(examples['text'])} samples...")
    sample = tokenizer(
        examples["text"],
        padding="longest",
        truncation=True,
        max_length=4096,
        add_special_tokens=True,
        return_attention_mask=True,
    )
    print(f"  Tokenized input_ids shape: {len(sample['input_ids'])} x {len(sample['input_ids'][0])}")
    return sample


class CustomDataCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        max_len = max(len(ids) for ids in input_ids)

        padded_ids, padded_mask, labels = [], [], []
        user_start_tokens = self.tokenizer.encode("<|im_start|>user", add_special_tokens=False)

        for ids, mask in zip(input_ids, attention_mask):
            pad_len = max_len - len(ids)
            p_ids = ids + [self.tokenizer.pad_token_id] * pad_len
            p_mask = mask + [0] * pad_len
            label = p_ids.copy()

            found = False
            for j in range(len(ids) - len(user_start_tokens) + 1):
                if ids[j:j + len(user_start_tokens)] == user_start_tokens:
                    label[:j] = [-100] * j
                    found = True
                    break
            if not found:
                label = [-100] * len(label)


            padded_ids.append(p_ids)
            padded_mask.append(p_mask)
            labels.append(label)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    training_args.label_names = ["labels"]
    local_rank = training_args.local_rank

    if model_args.data_path is None:
        model_args.data_path = str(PROC_DIR / model_args.category / "training_sid_routing_train.parquet")

    if "ROUTING_VAL_DATA" in os.environ and os.environ["ROUTING_VAL_DATA"]:
        val_data_path = os.environ["ROUTING_VAL_DATA"]
    else:
        if model_args.category:
            category = model_args.category
        else:
            train_path = Path(model_args.data_path)
            category = train_path.parent.name
        val_data_path = str(PROC_DIR / category / "training_sid_routing_val.parquet")

    if local_rank in [-1, 0]:
        print(f"Loading model: {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    special_tokens = get_special_tokens()
    valid_ids = [tokenizer.convert_tokens_to_ids(t) for t in special_tokens]
    valid_ids = [tid for tid in valid_ids if tid != tokenizer.unk_token_id]

    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules.split(","),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            trainable_token_indices={"embed_tokens": valid_ids},
        )
        model = get_peft_model(model, lora_config)
    else:
        if local_rank in [-1, 0]:
            print("Full parameter training (Routing)")

    if local_rank in [-1, 0]:
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
        else:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100 * trainable / total:.4f}")

    train_dataset = prepare_chat_dataset(model_args.data_path, sample_size=model_args.sample_size, local_rank=local_rank)
    val_dataset = prepare_chat_dataset(val_data_path, sample_size=None, local_rank=local_rank)

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"],
        desc="Tokenize train",
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"],
        desc="Tokenize val",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=CustomDataCollator(tokenizer),
    )

    if local_rank in [-1, 0]:
        print("Starting training...")
    trainer.train()

    if training_args.local_rank in [-1, 0]:
        print("\n" + "=" * 60)
        print("  Training completed!")
        print("  Routing trainer finished successfully.")
        print("=" * 60)



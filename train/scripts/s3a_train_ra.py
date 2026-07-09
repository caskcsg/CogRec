#!/usr/bin/env python3
"""
s3a_train_ra.py — 统一 RA 训练脚本（单卡 / 多卡）
"""

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)
import pandas as pd
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
import os
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
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


def prepare_chat_dataset(data_path, sample_size=None, local_rank=0):
    if local_rank == 0:
        print(f"Loading parquet file: {data_path}")
    data_pq = pd.read_parquet(data_path)
    if local_rank == 0:
        print(f"Data shape: {data_pq.shape}")

    if sample_size:
        data_pq = data_pq.head(sample_size)

    texts = []
    system_message = (
        "You are a professional recommendation expert who needs to recommend "
        "the next possible purchase for users based on their purchase history. "
        "Please predict the most likely next product that the user will purchase "
        "based on the user's historical purchase information."
    )

    for _, row in data_pq.iterrows():
        if 'title' in row and row['title'] is not None:
            assistant_content = (
                f"<think>\nThe user is likely to buy items in {row['categories']} category\n</think>\n"
                f"{row['groundtruth']}"
            )
        else:
            assistant_content = f"<think>\n\n</think>\n{row['groundtruth']}"

        formatted_text = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{row['description']}<|im_end|>
<|im_start|>assistant
{assistant_content}<|im_end|>
"""
        texts.append(formatted_text)

    if local_rank == 0:
        print(f"Total texts: {len(texts)}")
    return Dataset.from_dict({'text': texts})


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples['text'],
        padding='longest',
        truncation=True,
        max_length=4096,
        add_special_tokens=True,
        return_attention_mask=True,
    )


def get_special_tokens():
    special_tokens = ['<|sid_begin|>', '<|sid_end|>']
    for prefix in ['s_a', 's_b', 's_c', 's_d']:
        for i in range(256):
            special_tokens.append(f'<{prefix}_{i}>')
    return special_tokens


class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        max_length = max(len(ids) for ids in input_ids)
        padded_input_ids, padded_attention_mask, labels = [], [], []

        for ids, mask in zip(input_ids, attention_mask):
            padding_length = max_length - len(ids)
            padded_ids = ids + [self.tokenizer.pad_token_id] * padding_length
            padded_mask = mask + [0] * padding_length
            label = padded_ids.copy()

            text = self.tokenizer.decode(ids, skip_special_tokens=False)
            user_start_pos = text.find("<|im_start|>user")
            if user_start_pos != -1:
                user_start_tokens = self.tokenizer.encode("<|im_start|>user", add_special_tokens=False)
                for j in range(len(ids) - len(user_start_tokens) + 1):
                    if ids[j:j + len(user_start_tokens)] == user_start_tokens:
                        for k in range(j):
                            label[k] = -100
                        break
                else:
                    label = [-100] * len(label)
            else:
                label = [-100] * len(label)

            padded_input_ids.append(padded_ids)
            padded_attention_mask.append(padded_mask)
            labels.append(label)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    training_args.label_names = ["labels"]

    local_rank = training_args.local_rank

    # ====================== 数据路径自适应 ======================
    if model_args.data_path is None:
        model_args.data_path = str(PROC_DIR / model_args.category / "training_RA_train.parquet")

    # val 路径智能推导（兼容 run_train.sh 的 RA_VAL_DATA）
    if 'RA_VAL_DATA' in os.environ and os.environ['RA_VAL_DATA']:
        val_data_path = os.environ['RA_VAL_DATA']
    else:
        train_path = Path(model_args.data_path)
        category = train_path.parent.name
        val_data_path = str(PROC_DIR / category / "training_RA_val.parquet")

    # ====================== 模型加载 ======================
    if local_rank == 0:
        print(f"Loading model: {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # ====================== Special Tokens + PEFT ======================
    special_tokens = get_special_tokens()
    valid_special_token_ids = [
        tid for tid in tokenizer.convert_tokens_to_ids(special_tokens)
        if tid != tokenizer.unk_token_id
    ]

    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules.split(","),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            trainable_token_indices={'embed_tokens': valid_special_token_ids},
        )
        model = get_peft_model(model, lora_config)
    else:
        if local_rank == 0:
            print("Full parameter training (embed_tokens will be trained)")

    if local_rank == 0:
        if hasattr(model, 'print_trainable_parameters'):
            model.print_trainable_parameters()
        else:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100 * trainable / total:.4f}")

    # ====================== 数据集 ======================
    train_dataset = prepare_chat_dataset(model_args.data_path, local_rank=local_rank)
    val_dataset = prepare_chat_dataset(val_data_path, local_rank=local_rank)

    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=['text'])
    val_dataset = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=['text'])

    # ====================== Trainer ======================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=CustomDataCollator(tokenizer),
    )

    if local_rank == 0:
        print(f"Starting training...")

    trainer.train()

    if training_args.local_rank in [-1, 0]:
        print("\n" + "=" * 60)
        print("  Training completed!")
        print("  Best model selected by shell save_best_model().")
        print("  Skipping final evaluate/save to avoid DeepSpeed deadlock.")
        print("=" * 60)
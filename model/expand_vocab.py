#!/usr/bin/env python3
"""
basemodel/expand_vocab.py
--------------------------
为基座模型扩充 1026 个 SID 特殊 Token

Usage:
    python basemodel/expand_vocab.py
    python basemodel/expand_vocab.py --base_model_dir basemodel/Qwen3-1.7B --save_dir basemodel/Qwen3-1.7B-expand
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def get_special_tokens(max_range: int = 256) -> list[str]:
    special_tokens: list[str] = []
    special_tokens.append("<|sid_begin|>")
    special_tokens.append("<|sid_end|>")
    for prefix in ["s_a", "s_b", "s_c", "s_d"]:
        for idx in range(max_range):
            special_tokens.append(f"<{prefix}_{idx}>")
    return special_tokens


def round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("multiple must be a positive integer")
    return ((value + multiple - 1) // multiple) * multiple


def expand_vocabulary(base_model_dir: Path, save_dir: Path) -> None:
    print(f"Loading model config from: {base_model_dir}")
    config = AutoConfig.from_pretrained(base_model_dir)

    print("Loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(base_model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        model = model.to("cuda")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)

    new_tokens = get_special_tokens(max_range=256)
    print(f"Preparing to add {len(new_tokens)} special tokens.")

    tokens_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": new_tokens}, replace_additional_special_tokens=False
    )
    print(f"Successfully added {tokens_added} tokens.")

    updated_vocab_size = len(tokenizer)
    target_vocab_size = round_up_to_multiple(updated_vocab_size, 256)

    print(f"Vocab: {updated_vocab_size} → {target_vocab_size} (aligned to 256)")
    model.resize_token_embeddings(target_vocab_size)
    config.vocab_size = target_vocab_size

    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {save_dir}")
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    config.save_pretrained(save_dir)

    # 验证
    sample = "<|sid_begin|><s_a_0><s_b_0><s_c_0><s_d_0><|sid_end|>"
    ids = tokenizer.encode(sample, return_tensors="pt")
    print(f"Verify: '{sample}' → {ids.shape[1]} tokens")
    print("Done!")


def main():
    parser = argparse.ArgumentParser()
    base_dir = Path(__file__).resolve().parent
    parser.add_argument("--base_model_dir", type=str, default=str(base_dir / "Qwen3-1-7B"))
    parser.add_argument("--save_dir", type=str, default=str(base_dir / "Qwen3-1-7B-expand"))
    args = parser.parse_args()

    expand_vocabulary(
        base_model_dir=Path(args.base_model_dir),
        save_dir=Path(args.save_dir),
    )


if __name__ == "__main__":
    main()

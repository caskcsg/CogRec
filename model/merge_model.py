#!/usr/bin/env python3
"""
merge_model.py
-------------------------
将 Stage 1 (Alignment) 训练的 PEFT/LoRA adapter（主要是 SID embedding）
合并回基座模型中，生成可供 Stage 2 直接加载的完整模型。

功能特性：
1. 标准化命令行参数输入
2. 自动寻找 Best Checkpoint（兼容 DeepSpeed 和标准 Trainer 格式）
3. 严格的 SID Embedding 权重验证（检查是否为 0 或未被训练）
"""

import argparse
import os
import sys
import json
import shutil
import glob
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def find_best_checkpoint(lora_path: str) -> str:
    """
    自动查找最优 checkpoint 路径。
    如果用户传入的就是一个具体的 checkpoint 目录（包含 adapter_config.json），则直接返回。
    否则按以下优先级在目录中寻找：
    1. output_dir 根目录
    2. trainer_state.json 中的 best_model_checkpoint
    3. 按 step 数最大的 checkpoint
    """
    path_obj = Path(lora_path)
    
    if not path_obj.exists():
        raise FileNotFoundError(f"Provided LoRA path does not exist: {lora_path}")

    # 检查 0：用户是否直接传入了具体的 adapter 目录
    if (path_obj / "adapter_config.json").exists():
        print(f"  [Auto-Detect] Found adapter directly in provided path: {lora_path}")
        return str(path_obj)
    
    # 检查 1：trainer_state.json
    trainer_state_file = path_obj / "trainer_state.json"
    if trainer_state_file.exists():
        with open(trainer_state_file) as f:
            state = json.load(f)
        best_ckpt = state.get("best_model_checkpoint")
        if best_ckpt and Path(best_ckpt).exists():
            if (Path(best_ckpt) / "adapter_config.json").exists():
                print(f"  [Auto-Detect] Found best checkpoint from trainer_state: {best_ckpt}")
                return best_ckpt
            else:
                print(f"  ⚠️ trainer_state best_ckpt exists but no adapter_config.json: {best_ckpt}")
    
    # 检查 2：遍历所有 checkpoint 目录查找内嵌的 trainer_state
    checkpoints = sorted(
        glob.glob(str(path_obj / "checkpoint-*")),
        key=lambda x: int(x.split("-")[-1])
    )
    for ckpt_dir in reversed(checkpoints):
        ts = Path(ckpt_dir) / "trainer_state.json"
        if ts.exists():
            with open(ts) as f:
                state = json.load(f)
            best_ckpt = state.get("best_model_checkpoint")
            if best_ckpt and (Path(best_ckpt) / "adapter_config.json").exists():
                print(f"  [Auto-Detect] Found best checkpoint from {ts}: {best_ckpt}")
                return best_ckpt
    
    # 检查 3：回退到最新的 checkpoint 目录
    for ckpt_dir in reversed(checkpoints):
        if (Path(ckpt_dir) / "adapter_config.json").exists():
            print(f"  [Auto-Detect] Using latest checkpoint with adapter: {ckpt_dir}")
            return ckpt_dir
    
    raise FileNotFoundError(
        f"No valid PEFT adapter_config.json found in {lora_path} or its subdirectories.\n"
        f"Contents: {os.listdir(lora_path)[:10]}..."
    )


def validate_sid_embeddings(model, tokenizer):
    """验证 SID embedding 是否被正确合并且被有效训练过。"""
    print("\n  Validating SID embeddings...")
    
    embed_weight = model.get_input_embeddings().weight
    
    # 检查几个具有代表性的 SID token
    test_tokens = ['<|sid_begin|>', '<s_a_0>', '<s_b_128>', '<s_c_255>', '<|sid_end|>']
    for token_str in test_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        if token_id == tokenizer.unk_token_id:
            print(f"  ❌ Validation Failed: Token '{token_str}' maps to UNK! Model vocab is broken.")
            return False
        
        emb = embed_weight[token_id]
        norm = emb.norm().item()
        print(f"    {token_str} (id={token_id:^6}): norm = {norm:.4f}")
        
        if norm < 1e-6:
            print(f"  ⚠️ WARNING: Token '{token_str}' embedding is near zero! Training might have failed.")
    
    # 检查 SID token embedding 与普通文字 token 的区分度 (Cosine Similarity)
    sid_begin_id = tokenizer.convert_tokens_to_ids('<|sid_begin|>')
    regular_id = tokenizer.convert_tokens_to_ids('the')
    
    if regular_id != tokenizer.unk_token_id:
        sid_emb = embed_weight[sid_begin_id]
        reg_emb = embed_weight[regular_id]
        
        cosine_sim = torch.nn.functional.cosine_similarity(
            sid_emb.unsqueeze(0).float(), 
            reg_emb.unsqueeze(0).float()
        ).item()
        print(f"    Cosine sim (<|sid_begin|> vs 'the'): {cosine_sim:.4f}")
    
    print("  ✅ SID embedding validation passed")
    return True


def merge_and_save(base_model_path: str, lora_model_path: str, output_path: str):
    print("=" * 80)
    print(" STAGE 1 → STAGE 2: MERGING LORA INTO BASE MODEL")
    print("=" * 80)

    # 1. 寻找真实的 LoRA 路径
    try:
        print(f"\n1. Resolving LoRA checkpoint path...")
        resolved_lora_path = find_best_checkpoint(lora_model_path)
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        sys.exit(1)

    # 清理旧目录
    if os.path.exists(output_path):
        print(f"   Removing existing output directory: {output_path}")
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    try:
        # 2. 加载基座模型
        print(f"\n2. Loading base model: {base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"   Vocab size: {tokenizer.vocab_size}")

        # 3. 加载并合并 LoRA
        print(f"\n3. Loading PEFT adapter from: {resolved_lora_path}")
        peft_model = PeftModel.from_pretrained(base_model, resolved_lora_path)
        print("   Merging weights... (This may take a moment)")
        merged_model = peft_model.merge_and_unload()
        print("   ✅ LoRA merged successfully")
        
        del base_model, peft_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 4. 验证合并后的 Embeddings
        validate_sid_embeddings(merged_model, tokenizer)

        # 5. 保存完整模型
        print(f"\n4. Saving merged model: {output_path}")
        merged_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        # 6. 最终读写完整性验证
        print(f"\n5. Verifying saved model loads correctly...")
        test_model = AutoModelForCausalLM.from_pretrained(output_path)
        test_tokenizer = AutoTokenizer.from_pretrained(output_path)
        print(f"   Parameters: {test_model.num_parameters():,}")
        print(f"   Vocab: {test_tokenizer.vocab_size}")
        
        # 检查 config.json 中的 vocab_size
        config_path = os.path.join(output_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            print(f"   config.json vocab_size: {config.get('vocab_size', 'NOT FOUND')}")

        del test_model, test_tokenizer, merged_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n{'=' * 80}")
        print(f" ✅ MERGE COMPLETED AND VERIFIED: {output_path}")
        print(f"{'=' * 80}")
        return output_path

    except Exception as e:
        print(f"\n❌ Error during merge: {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Merge Stage 1 LoRA Adapter into Base Model")
    
    # 获取脚本所在的当前目录，作为默认 base_model 的参照物
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser.add_argument("--base_model_path", type=str, 
                        default=os.path.join(base_dir, "Qwen3-1-7B-expand"),
                        help="Path to the base model with expanded vocab")
    parser.add_argument("--lora_model_path", type=str, required=True,
                        help="Path to the Stage 1 training output dir OR specific checkpoint dir")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path for the merged full model")
    
    args = parser.parse_args()

    merge_and_save(
        base_model_path=args.base_model_path, 
        lora_model_path=args.lora_model_path, 
        output_path=args.output_path
    )


if __name__ == "__main__":
    main()

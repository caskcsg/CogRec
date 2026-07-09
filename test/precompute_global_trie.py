#!/usr/bin/env python3
"""
预计算评估阶段使用的 exact trie，并写出与源 parquet 强绑定的元数据。

正式协议:
  - trie 必须离线生成
  - trie 必须与评估 parquet 一一对应
  - 运行时通过 parquet fingerprint 做严格校验
"""

import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


SID_PATTERN = re.compile(r'<\|sid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><s_d_\d+><\|sid_end\|>')


def extract_all_sids_from_text(text):
    return SID_PATTERN.findall(str(text or ''))


def extract_sid_from_text(text):
    text = str(text or '').strip()
    match = SID_PATTERN.search(text)
    return match.group(0) if match else text


def compute_parquet_fingerprint(df: pd.DataFrame) -> dict:
    cols = [c for c in ['sample_id', 'description', 'groundtruth'] if c in df.columns]
    if not cols:
        raise ValueError('Parquet 缺少 fingerprint 所需列；至少需要 sample_id/description/groundtruth 之一')

    h = hashlib.sha256()
    h.update(f"rows={len(df)}|cols={'|'.join(cols)}".encode('utf-8'))
    for col in cols:
        series = df[col].fillna('').astype(str)
        for value in series.tolist():
            h.update(value.encode('utf-8', errors='ignore'))
            h.update(b"\n")

    preview = {col: df[col].fillna('').astype(str).head(5).tolist() for col in cols}
    return {
        'row_count': int(len(df)),
        'columns': cols,
        'sha256': h.hexdigest(),
        'preview_head': preview,
    }


def build_global_trie(test_parquet_file, model_path, output_file):
    parquet_path = Path(test_parquet_file)
    output_path = Path(output_file)

    print(f'Loading test data from: {parquet_path}')
    df = pd.read_parquet(parquet_path)
    print(f'Total samples in test set: {len(df)}')

    fingerprint = compute_parquet_fingerprint(df)
    print(f"Parquet fingerprint: {fingerprint['sha256']}")

    print(f'Loading tokenizer from: {model_path}')
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print('Extracting all SIDs from test set (description + groundtruth)...')
    valid_sids = set()
    for _, row in df.iterrows():
        for sid in extract_all_sids_from_text(row.get('description', '')):
            if sid:
                valid_sids.add(sid)
        gt_sid = extract_sid_from_text(row.get('groundtruth', ''))
        if gt_sid and '<|sid_begin|>' in gt_sid and '<|sid_end|>' in gt_sid:
            valid_sids.add(gt_sid)

    print(f'Found {len(valid_sids)} unique valid SIDs in test set')
    print('Converting SIDs to token sequences...')
    sid_token_sequences = [tokenizer.encode(sid, add_special_tokens=False) for sid in valid_sids]
    print(f'Converted {len(sid_token_sequences)} SIDs to token sequences')

    from collections import defaultdict
    exact_trie = defaultdict(lambda: defaultdict(set))
    max_length = max((len(seq) for seq in sid_token_sequences), default=0)
    print(f'Maximum SID token length: {max_length}')

    for seq in sid_token_sequences:
        for pos in range(len(seq)):
            current_token = seq[pos]
            if pos + 1 < len(seq):
                exact_trie[pos][current_token].add(seq[pos + 1])
            else:
                eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
                exact_trie[pos][current_token].add(eos_id)

    final_exact_trie = {
        pos: {token_id: list(next_tokens) for token_id, next_tokens in token_map.items()}
        for pos, token_map in exact_trie.items()
    }

    print('Built exact trie tree:')
    print(f'  Total unique SIDs: {len(valid_sids)}')
    print(f'  Search space size: {len(valid_sids):,} (exact match only)')
    print(f'  Trie depth: {max_length}')
    for pos in range(min(6, max_length)):
        print(f"  Position {pos}: {len(final_exact_trie.get(pos, {}))} possible tokens")

    trie_data = {
        'exact_trie': final_exact_trie,
        'valid_sids': valid_sids,
        'valid_sid_tokens': sid_token_sequences,
        'tokenizer_name': model_path,
        'total_samples': len(df),
        'search_space_size': len(valid_sids),
        'max_length': max_length,
        'trie_type': 'exact',
        'source_metadata': {
            'parquet_path': str(parquet_path),
            'parquet_name': parquet_path.name,
            **fingerprint,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(trie_data, f)

    meta_path = output_path.with_suffix(output_path.suffix + '.meta.json')
    meta_payload = {
        'trie_file': str(output_path),
        'trie_type': 'exact',
        'tokenizer_name': model_path,
        'search_space_size': len(valid_sids),
        'max_length': max_length,
        'source_metadata': trie_data['source_metadata'],
    }
    meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'Exact trie saved to: {output_path}')
    print(f'Metadata sidecar saved to: {meta_path}')
    return trie_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Precompute exact trie for offline evaluation')
    parser.add_argument('--test_parquet_file', type=str, required=True, help='Test parquet file')
    parser.add_argument('--model_path', type=str, required=True, help='Tokenizer/model path')
    parser.add_argument('--output_file', type=str, required=True, help='Output pickle file')
    args = parser.parse_args()
    build_global_trie(args.test_parquet_file, args.model_path, args.output_file)

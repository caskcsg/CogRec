# CogRec Anonymous Implementation

This repository provides an anonymized implementation of CogRec, a structure-grounded fast-and-slow reasoning framework for Semantic ID based generative recommendation.

## Scope

This release focuses on the supervised reproducibility path used by the main paper:

- Amazon Reviews 2014 data download
- Semantic ID construction
- SID cognitive-map construction and train/eval split generation
- Stage 1 token alignment
- Stage 2 direct SID recommendation
- Stage 3a natural-language reasoning activation
- Stage 3b SID Routing
- constrained SID decoding and evaluation

Visualization utilities and preference-optimization/GRPO code are not part of this anonymous release because they are not required for the main supervised reproduction pipeline.

## Environment

Python 3.10 or newer is required. The recommended setup is:

```bash
bash setup_env.sh
conda activate torch
```

Alternatively, install dependencies manually:

```bash
pip install -r requirements.txt
```

Install PyTorch with the CUDA wheel that matches your machine if the default wheel is not suitable. Stage 3 reconstruction uses vLLM.

## Data

The experiments use Amazon Reviews 2014 5-core categories:

- Beauty
- Sports and Outdoors
- Toys and Games

Raw and processed datasets are not redistributed. Download data with the provided script, for example:

```bash
python data/download_amazon_dataset.py --category Beauty --type both
```

See `docs/DATA.md` for details.

## Base Model

The base model weights are not included. Download Qwen3-1.7B into `model/Qwen3-1-7B`:

```bash
python model/download_basemodel.py
```

## Paper Reproduction Protocol

The training and evaluation protocol uses seed 42. Stage 2 SID recommendation is full-parameter fine-tuning; LoRA remains available only as an explicitly requested alternative. Training, direct/No-CoT evaluation, and the default subset-CoT evaluation use complete histories without truncation.

The training script defaults to 2 GPUs. The canonical paper workflow below uses 4 GPUs and DeepSpeed ZeRO-2.

### Data Preparation

Prepare one category with:

```bash
python run_generate_data.py --category Beauty --phase download
python run_generate_data.py --category Beauty --phase sid
python run_generate_data.py --category Beauty --phase hnsw
```

Replace `Beauty` with `Sports` or `Toys`. To prepare all categories, run `python run_generate_data.py --category all`.

### Beauty Workflow

`all` runs canonical training and the default lightweight evaluation flow: full-test No-CoT plus the predefined subset-CoT evaluation. The subset-CoT path does not truncate history. Full-test CoT is intentionally separate because 5 reasoning samples x 10 beams can require substantially more memory.

```bash
bash run_train.sh --stage all --category Beauty --gpus 4

bash run_train.sh --stage eval_sid_routing_ckpt --category Beauty --gpus 4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export COT_MAX_HISTORY_ITEMS=21
export COT_HISTORY_TRUNCATION_SIDE=tail
bash run_train.sh --stage eval_ra_full_cot --category Beauty --gpus 4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export COT_MAX_HISTORY_ITEMS=21
export COT_HISTORY_TRUNCATION_SIDE=tail
bash run_train.sh --stage eval_sid_routing_full_cot --category Beauty --gpus 4
```

`eval_sid_routing_ckpt` evaluates and selects the intended Stage 3b routing checkpoint. The checkpoints reported in the paper are Beauty epoch 2, Sports epoch 2, and Toys epoch 3.

### Full-Test CoT Memory Controls

These values are approximate P95 history lengths. They are memory-control settings only for explicit full-test CoT commands; they are not training or evaluation protocol truncation defaults.

| Dataset | Full-CoT max history | Side |
| ------- | -------------------: | ---- |
| Beauty  |                   21 | tail |
| Sports  |                   20 | tail |
| Toys    |                   18 | tail |

Use the following template for Sports or Toys:

```bash
CATEGORY=Sports
MAX_HISTORY_ITEMS=20
# For Toys, use: CATEGORY=Toys and MAX_HISTORY_ITEMS=18

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export COT_MAX_HISTORY_ITEMS="${MAX_HISTORY_ITEMS}"
export COT_HISTORY_TRUNCATION_SIDE=tail
bash run_train.sh --stage eval_ra_full_cot --category "${CATEGORY}" --gpus 4
bash run_train.sh --stage eval_sid_routing_full_cot --category "${CATEGORY}" --gpus 4
```

If `COT_MAX_HISTORY_ITEMS` is unset, full-test CoT keeps complete histories and emits an OOM warning; the code does not silently select a dataset-specific value.

The released evaluation code emits Hit@K/NDCG@K metrics and supports stratified difficulty and CoT-step statistics. Final paper-rendering and plotting utilities are intentionally excluded from this anonymous release.

## Outputs

Generated data, model checkpoints, logs, figures, and merged model weights are intentionally excluded from version control. Typical output locations are:

- `data/raw/`
- `data/processed/`
- `train/results/`
- `model/Qwen3-1-7B/`
- `model/Qwen3-1-7B-expand/`
- `model/merged_*_model/`
- `logs/`
- `outputs/`

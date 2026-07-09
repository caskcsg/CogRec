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

## Beauty Reproduction Example

The following commands run the complete supervised pipeline for one dataset. The default example assumes 2 GPUs:

```bash
python run_generate_data.py --category Beauty --phase download
python run_generate_data.py --category Beauty --phase sid
python run_generate_data.py --category Beauty --phase hnsw

bash run_train.sh --stage expand --category Beauty --gpus 2
bash run_train.sh --stage align --category Beauty --gpus 2
bash run_train.sh --stage merge --category Beauty --gpus 2
bash run_train.sh --stage rec --category Beauty --gpus 2
bash run_train.sh --stage eval_rec --category Beauty --gpus 2
bash run_train.sh --stage ra --category Beauty --gpus 2
bash run_train.sh --stage eval_ra --category Beauty --gpus 2
bash run_train.sh --stage sid_routing --category Beauty --gpus 2
bash run_train.sh --stage eval_sid_routing --category Beauty --gpus 2
```

You can also run the supervised training/evaluation stages through:

```bash
bash run_train.sh --stage all --category Beauty --gpus 2
```

The script uses DeepSpeed ZeRO-2 when `--gpus` is greater than 1. For a single-GPU smoke run, set `--gpus 1`.

## Other Categories

Replace `Beauty` with `Sports` or `Toys` in the commands above. To generate offline data for all categories:

```bash
python run_generate_data.py --category all
```

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

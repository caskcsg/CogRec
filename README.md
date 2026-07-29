# CogRec

Official implementation of [CogRec: Structure-Cognitive Fast-and-Slow Reasoning for Generative Recommendation](https://arxiv.org/abs/2607.24402).

CogRec is a structure-grounded fast-and-slow reasoning framework for Semantic-ID-based generative recommendation. It augments hierarchical Semantic IDs with intra-layer semantic graphs and item-level neighborhoods, and represents recommendation reasoning with layer-wise SID Routing operations.

## Scope

This release focuses on the supervised reproduction path used in the paper:

- Amazon Reviews 2014 data download
- four-layer Semantic ID construction
- SID topology construction and chronological train/validation/test splits
- Stage 1 token alignment
- Stage 2 direct SID recommendation
- Stage 3a natural-language reasoning activation
- Stage 3b SID Routing
- trie-constrained SID decoding and evaluation

Visualization utilities and preference-optimization/GRPO code are not included because they are not required for the main supervised reproduction pipeline.

## Environment

Python 3.10 and an NVIDIA CUDA environment are recommended. The setup script creates a `torch` Conda environment and installs PyTorch, DeepSpeed, the data-processing dependencies, and vLLM:

```bash
git clone https://github.com/caskcsg/CogRec.git
cd CogRec
bash setup_env.sh
conda activate torch
```

Alternatively, install dependencies manually:

```bash
pip install -r requirements.txt
```

Install the PyTorch wheel that matches the local CUDA driver when using the manual path. Stage 3a reconstruction uses vLLM.

## Data and Model

The experiments use Amazon Reviews 2014 5-core categories:

- Beauty
- Sports and Outdoors
- Toys and Games

Raw and processed datasets are not redistributed. Prepare Beauty with one command:

```bash
python run_generate_data.py --category Beauty
```

This executes the required preprocessing order:

1. `download`: download the SNAP Amazon Reviews 2014 review and metadata files.
2. `sid`: build chronological sequences, metadata embeddings, and four-layer SIDs.
3. `hnsw`: construct the SID topology, data splits, difficulty labels, and CoT subset.

Use `--category Sports`, `--category Toys`, or `--category all` for the other settings. Individual phases can be resumed with `--phase download`, `--phase sid`, or `--phase hnsw`. See `docs/DATA.md` for the generated assets.

The base model weights are not included. Download Qwen3-1.7B into `model/Qwen3-1-7B`:

```bash
python model/download_basemodel.py
```

## Reproduction Protocol

The implementation follows the paper protocol summarized below.

| Stage | Initialization | Main setting |
| --- | --- | --- |
| Stage 0 | Qwen3-1.7B | 4 SID layers, 256 codes per layer, 30 RQ-K-means iterations |
| Stage 1: alignment | expanded Stage 0 model | LR `1e-4`, up to 15 epochs, effective batch 64, SID-token embeddings only |
| Stage 2: direct SID | merged Stage 1 model | LR `1e-5`, 6 epochs, effective batch 16, full-parameter tuning |
| Stage 3a: natural-language reasoning | Stage 2 checkpoint | 2 iterative epochs with reconstruction between epochs |
| Stage 3b: SID Routing | the same Stage 2 checkpoint | LR `1e-5`, 3 epochs, effective batch 16, full-parameter tuning |

All controlled stages use seed 42. Stage 2, Stage 3a, and Stage 3b use AdamW, a `0.1` warmup ratio, `0.01` weight decay, and `constant_with_warmup`. Stage 3a and Stage 3b are sibling branches: running Stage 3a first does not initialize Stage 3b from the Stage 3a model.

The training script defaults to 2 GPUs. The paper workflow below uses 4 GPUs with DeepSpeed ZeRO-2. Training, direct/No-CoT evaluation, and subset-CoT evaluation retain complete histories without truncation.

### Beauty Training

Run the shared trunk first, followed by the two independent Stage 3 branches:

```bash
bash run_train.sh --stage expand --category Beauty --gpus 4
bash run_train.sh --stage align --category Beauty --gpus 4
bash run_train.sh --stage merge --category Beauty --gpus 4
bash run_train.sh --stage rec --category Beauty --gpus 4

bash run_train.sh --stage ra --category Beauty --gpus 4
bash run_train.sh --stage sid_routing --category Beauty --gpus 4
```

Completed-stage outputs are detected and skipped on rerun. After an interrupted training process, verify that the expected final model files exist before relying on the automatic stage gate.

### Beauty Evaluation

Evaluate the Stage 2 direct model and the fixed Stage 3a epoch-2 model:

```bash
bash run_train.sh --stage eval_rec --category Beauty --gpus 4
bash run_train.sh --stage eval_ra --category Beauty --gpus 4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export COT_MAX_HISTORY_ITEMS=21
export COT_HISTORY_TRUNCATION_SIDE=tail
bash run_train.sh --stage eval_ra_full_cot --category Beauty --gpus 4
```

For Stage 3b, select the paper checkpoint before evaluating its test metrics. The interactive list prints the validation loss and epoch for each retained checkpoint. For Beauty, select the entry with `epoch=2`, then mode `3` evaluates No-CoT, subset-CoT, and full-CoT using that same checkpoint:

```bash
export ROUTING_EVAL_MODE=3
bash run_train.sh --stage eval_sid_routing_ckpt --category Beauty --gpus 4
unset ROUTING_EVAL_MODE
```

For a non-interactive run, set the exact checkpoint path explicitly:

```bash
ROUTING_CKPT=/absolute/path/to/train/results/beauty/sid_routing/checkpoint-N \
ROUTING_EVAL_MODE=3 \
bash run_train.sh --stage eval_sid_routing_ckpt --category Beauty --gpus 4
```

Replace `checkpoint-N` with the checkpoint whose recorded epoch is 2. Do not choose a checkpoint by comparing test-set metrics.

### Convenience Pipeline

The following command remains available for a resumable end-to-end convenience run:

```bash
bash run_train.sh --stage all --category Beauty --gpus 4
```

`all` trains every stage and runs full-test No-CoT plus predefined subset-CoT evaluation using the stage-specific saved models. It does not run full-test CoT and should not replace the explicit fixed-epoch Stage 3b evaluation above when reproducing the reported paper setting.

### Full-Test CoT Memory Controls

Reasoning evaluation samples 5 traces with temperature `1.5` and top-p `0.95`; each trace conditions beam-10 SID decoding with temperature `0.6`, top-p `1.0`, and at most 8 SID tokens. Full-test CoT is separate because this produces up to 50 trace-conditioned hypotheses per instance.

The following approximate P95 history lengths are memory-control settings only for full-test CoT. They are not training or general evaluation defaults.

| Dataset | Reported Stage 3b epoch | Full-CoT max history | Side |
| ------- | ------------------------: | -------------------: | ---- |
| Beauty  |                         2 |                   21 | tail |
| Sports  |                         2 |                   20 | tail |
| Toys    |                         3 |                   18 | tail |

For Sports or Toys, repeat the Beauty workflow with the corresponding category, checkpoint epoch, and history limit. For example:

```bash
CATEGORY=Sports
MAX_HISTORY_ITEMS=20
# For Toys, use: CATEGORY=Toys and MAX_HISTORY_ITEMS=18

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export COT_MAX_HISTORY_ITEMS="${MAX_HISTORY_ITEMS}"
export COT_HISTORY_TRUNCATION_SIDE=tail
bash run_train.sh --stage eval_ra_full_cot --category "${CATEGORY}" --gpus 4
ROUTING_EVAL_MODE=3 \
bash run_train.sh --stage eval_sid_routing_ckpt --category "${CATEGORY}" --gpus 4
```

The checkpoint command remains interactive unless `ROUTING_CKPT` is set. Select epoch 2 for Sports or epoch 3 for Toys before evaluation.

If `COT_MAX_HISTORY_ITEMS` is unset, full-test CoT keeps complete histories and emits an OOM warning; the code does not silently choose a dataset-specific limit. A log message that says the beam size was reduced after CUDA OOM indicates a non-canonical run and should not be used for paper comparison.

The evaluation code emits Hit@K/NDCG@K metrics and supports stratified difficulty and CoT-step statistics. Final paper-rendering and plotting utilities are intentionally excluded because they do not affect the metrics.

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

## Citation

If you use CogRec in your research, please cite:

```bibtex
@misc{liu2026cogrec,
  title         = {CogRec: Structure-Cognitive Fast-and-Slow Reasoning for Generative Recommendation},
  author        = {Xiang Liu and Jingsong Su and Shuqi Zhao and Pengbo Mo and Yiming Qiu and Huimu Wang and Mingming Li and Jiao Dai and Jizhong Han and Songlin Hu},
  year          = {2026},
  eprint        = {2607.24402},
  archivePrefix = {arXiv},
  primaryClass  = {cs.IR},
  url           = {https://arxiv.org/abs/2607.24402}
}
```

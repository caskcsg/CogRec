#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# setup_env.sh — CogRec / SID Routing Recommendation environment setup
# Target: CUDA GPU environment, DeepSpeed ZeRO-2, vLLM for Stage 3 reconstruction
# ══════════════════════════════════════════════════════════════════════
# Usage:
#   chmod +x setup_env.sh
#   bash setup_env.sh
#
# Optional overrides:
#   ENV_NAME=torch PYTHON_VERSION=3.10 bash setup_env.sh
#   PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash setup_env.sh
#   INSTALL_VLLM=0 bash setup_env.sh
#
# If you have a backup exported by backup_env.sh:
#   bash setup_env.sh --from-backup /path/to/env_backup_xxx
# ══════════════════════════════════════════════════════════════════════
set -euo pipefail

ENV_NAME="${ENV_NAME:-torch}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
INSTALL_VLLM="${INSTALL_VLLM:-1}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
BACKUP_DIR=""

if [[ "${1:-}" == "--from-backup" ]]; then
  BACKUP_DIR="${2:-}"
  if [[ -z "${BACKUP_DIR}" || ! -d "${BACKUP_DIR}" ]]; then
    echo "[ERROR] backup directory not found: ${BACKUP_DIR}" >&2
    exit 1
  fi
fi

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  CogRec Environment Setup                               ║"
echo "║  Target: CUDA + DeepSpeed ZeRO-2 + supervised CogRec    ║"
echo "╚══════════════════════════════════════════════════════════╝"

# ── 0. Basic checks ────────────────────────────────────────────────
echo ""
echo "━━━ Step 0: Basic checks ━━━"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
  echo "[OK] GPU count: ${GPU_COUNT}"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
else
  GPU_COUNT=0
  echo "[WARN] nvidia-smi not found. GPU training will not work until the driver is available."
fi

if ! command -v conda >/dev/null 2>&1; then
  if [[ -x /root/miniconda3/bin/conda ]]; then
    export PATH="/root/miniconda3/bin:$PATH"
  elif [[ -x /opt/conda/bin/conda ]]; then
    export PATH="/opt/conda/bin:$PATH"
  else
    echo "[ERROR] conda not found. Please install miniconda/anaconda first." >&2
    exit 1
  fi
fi
echo "[OK] $(conda --version)"

# ── 1. Conda env ──────────────────────────────────────────────────
echo ""
echo "━━━ Step 1: Conda environment ━━━"
eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[INFO] Environment exists: ${ENV_NAME}"
else
  echo "[INFO] Creating ${ENV_NAME} with Python ${PYTHON_VERSION}"
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi
conda activate "${ENV_NAME}"
python --version
python -m pip --version

# ── 2. pip mirror ─────────────────────────────────────────────────
echo ""
echo "━━━ Step 2: pip mirror ━━━"
python -m pip config set global.index-url "${PIP_INDEX_URL}" || true
if [[ "${PIP_INDEX_URL}" == *"tuna.tsinghua"* ]]; then
  python -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn || true
fi
python -m pip install --upgrade pip setuptools wheel

# ── 3. Backup restore path ────────────────────────────────────────
if [[ -n "${BACKUP_DIR}" ]]; then
  echo ""
  echo "━━━ Step 3: Restore from backup ━━━"
  if [[ -f "${BACKUP_DIR}/requirements.freeze.txt" ]]; then
    echo "[INFO] Installing from ${BACKUP_DIR}/requirements.freeze.txt"
    python -m pip install -r "${BACKUP_DIR}/requirements.freeze.txt"
  elif [[ -f "${BACKUP_DIR}/requirements.txt" ]]; then
    echo "[INFO] Installing from ${BACKUP_DIR}/requirements.txt"
    python -m pip install -r "${BACKUP_DIR}/requirements.txt"
  else
    echo "[ERROR] No requirements file found in backup dir." >&2
    exit 1
  fi
else
  # ── 3. PyTorch ─────────────────────────────────────────────────
  echo ""
  echo "━━━ Step 3: PyTorch CUDA 12.1 ━━━"
  TORCH_OK="$(python - <<'PY' 2>/dev/null || true
try:
    import torch
    ok = torch.cuda.is_available() and torch.version.cuda and torch.version.cuda.startswith('12')
    print('ok' if ok else 'no')
except Exception:
    print('no')
PY
)"
  if [[ "${TORCH_OK}" == "ok" ]]; then
    python - <<'PY'
import torch
print(f'[OK] PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs={torch.cuda.device_count()}')
PY
  else
    python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
      --index-url https://download.pytorch.org/whl/cu121
  fi

  # ── 4. Core dependencies ───────────────────────────────────────
  echo ""
  echo "━━━ Step 4: Core training dependencies ━━━"
  python -m pip install \
    'transformers>=4.44.0' 'peft>=0.12.0' 'datasets>=2.20.0' \
    'accelerate>=0.33.0' 'safetensors>=0.4.0' 'sentencepiece' \
    'protobuf' 'einops' 'packaging' 'ninja' 'huggingface_hub'

  echo "[INFO] Installing DeepSpeed"
  python -m pip install deepspeed

  # ── 5. Data dependencies ───────────────────────────────────────
  echo ""
  echo "━━━ Step 5: Data dependencies ━━━"
  python -m pip install \
    'sentence-transformers>=2.7.0' 'scipy>=1.12.0' 'pandas>=2.0.0' \
    'pyarrow>=14.0.0' 'numpy>=1.24.0' \
    'tqdm>=4.66.0' 'tensorboard>=2.14.0'

  # faiss-gpu availability varies by CUDA/pip index. Try conda first, fallback to faiss-cpu.
  echo "[INFO] Installing FAISS"
  conda install -y -c pytorch -c nvidia faiss-gpu || python -m pip install faiss-cpu

  # ── 6. Stage 3 reconstruction dependency ───────────────────────
  echo ""
  echo "━━━ Step 6: Stage 3 reconstruction dependency ━━━"
  if [[ "${INSTALL_VLLM}" == "1" ]]; then
    python -m pip install vllm || echo "[WARN] vLLM install failed. Stage 3 reconstruction requires vLLM."
  fi
fi

# ── 7. Runtime environment ────────────────────────────────────────
echo ""
echo "━━━ Step 7: Runtime variables ━━━"
cat > "${CONDA_PREFIX}/etc/conda/activate.d/cogrec_env.sh" <<'EOF2'
#!/usr/bin/env bash
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS="${TRANSFORMERS_NO_ADVISORY_WARNINGS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800000}"
EOF2
chmod +x "${CONDA_PREFIX}/etc/conda/activate.d/cogrec_env.sh"
# shellcheck disable=SC1090
source "${CONDA_PREFIX}/etc/conda/activate.d/cogrec_env.sh"

# ── 8. Project DeepSpeed config helper ────────────────────────────
echo ""
echo "━━━ Step 8: Project helper files ━━━"
PROJ_DIR="${PROJ_DIR:-$(pwd)}"
mkdir -p "${PROJ_DIR}/train/scripts"
if [[ "${GPU_COUNT}" != "0" ]]; then
  echo "localhost slots=${GPU_COUNT}" > "${PROJ_DIR}/train/scripts/hostfile"
  echo "[OK] hostfile: ${PROJ_DIR}/train/scripts/hostfile"
fi

if [[ ! -f "${PROJ_DIR}/train/scripts/ds_config_zero2.json" ]]; then
  cat > "${PROJ_DIR}/train/scripts/ds_config_zero2.json" <<'EOF2'
{
  "fp16": {"enabled": false},
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 200000000,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 200000000,
    "contiguous_gradients": true
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "optimizer": {
    "type": "AdamW",
    "params": {"lr": "auto", "betas": "auto", "eps": "auto", "weight_decay": "auto"}
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {"warmup_min_lr": "auto", "warmup_max_lr": "auto", "warmup_num_steps": "auto"}
  }
}
EOF2
  echo "[OK] ds_config_zero2.json created"
else
  echo "[INFO] ds_config_zero2.json already exists; not overwritten"
fi

# ── 9. Verify ─────────────────────────────────────────────────────
echo ""
echo "━━━ Step 9: Verification ━━━"
python - <<'PY'
import importlib, sys
mods = ['torch','transformers','peft','datasets','accelerate','deepspeed','numpy','pandas','pyarrow','flask']
failed = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f'OK {m}: {getattr(mod, "__version__", "unknown")}')
    except Exception as e:
        failed.append((m, e))
        print(f'FAIL {m}: {e}')
try:
    import torch
    print(f'torch.cuda.is_available={torch.cuda.is_available()}')
    print(f'torch.version.cuda={torch.version.cuda}')
    print(f'gpu_count={torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'gpu_{i}={torch.cuda.get_device_name(i)}')
except Exception as e:
    failed.append(('torch_cuda', e))
if failed:
    print('Some checks failed. Review messages above.', file=sys.stderr)
PY

echo ""
echo "[OK] Setup completed. Use: conda activate ${ENV_NAME}"
echo "[OK] W&B is disabled by default in this conda env."

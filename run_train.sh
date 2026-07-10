#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# run_train.sh — OneRec 统一训练入口
# ═══════════════════════════════════════════════════════════════════════
#
# 单卡 (--gpus 1): 原生 PyTorch
# 多卡 (--gpus 2+): DeepSpeed ZeRO-2
#
# 有效 batch 严格对齐 OneRec 原版 (8GPU):
#   S1 Align:       8 × GA × N_GPU = 64
#   S2 Rec/S3a/S3b: 2 × GA × N_GPU = 16
#
# 正式版约定:
#   - train 阶段只消费离线数据，不生产任何数据
#   - difficulty 只允许 row-aligned
#   - reconstruction 统一由 s3_reconstruct_data.py 执行
#   - eval 严格按传入 --gpus 并行
#   - trie 在评估阶段按需生成
#   - eval 时必须校验 trie metadata 与 parquet fingerprint 一致
#
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PROJECT:-}" ]]; then
    if [[ -d "${PROJECT}" ]]; then
        PROJ_ROOT="$(cd "${PROJECT}" && pwd)"
    else
        echo "ERROR: PROJECT is set but is not a directory: ${PROJECT}"
        exit 1
    fi
else
    PROJ_ROOT="${SCRIPT_PROJ_ROOT}"
fi
cd "${PROJ_ROOT}"
export PROJECT="${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}:${PYTHONPATH:-}"
export NCCL_TIMEOUT=1800000          # 30 min
export TORCH_NCCL_BLOCKING_WAIT=0
# Normalize thread env for conda shells. Invalid inherited values can abort
# libgomp on torch/pandas import; pinning to 1 also avoids BLAS/OpenMP
# oversubscription when multiple workers share the node.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAYON_NUM_THREADS=1

# Disable Weights & Biases by default for unattended server training.
# Transformers Trainer auto-enables W&B when the package is installed; without
# these guards, multi-GPU runs can fail at on_train_begin if no W&B API key is
# configured. Users may override by exporting WANDB_DISABLED=false and
# WANDB_MODE=online/offline before invoking this script.
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS="${TRANSFORMERS_NO_ADVISORY_WARNINGS:-1}"

# ── 默认参数 ─────────────────────────────────────────────────────────
STAGE="all"
CATEGORY="Beauty"
NUM_GPUS=2
DRY_RUN=false

# ── 固定配置 ─────────────────────────────────────────────────────────
BASE_MODEL_DIR="${PROJ_ROOT}/model/Qwen3-1-7B"
EXPAND_MODEL_DIR="${PROJ_ROOT}/model/Qwen3-1-7B-expand"
TRAIN_SCRIPTS="${PROJ_ROOT}/train/scripts"
DS_CONFIG="${TRAIN_SCRIPTS}/ds_config_zero2.json"

# ── 日志 ─────────────────────────────────────────────────────────────
LOG_DIR="${PROJ_ROOT}/logs"
mkdir -p "${LOG_DIR}"

log() {
    local level=$1; shift
    local ts=$(date '+%Y-%m-%d %H:%M:%S')
    local msg="[${ts}] [${level}] $*"
    echo "${msg}" | tee -a "${LOG_FILE:-/dev/stderr}"
}

# ── 参数解析 ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)    STAGE="$2";    shift 2 ;;
        --category) CATEGORY="$2"; shift 2 ;;
        --gpus)     NUM_GPUS="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=true;  shift   ;;
        -h|--help)
            echo "Usage: bash run_train.sh --stage <stage> --category <cat> --gpus <n> [--dry-run]"
            echo "Defaults: --stage all --category Beauty --gpus 2"
            echo "Stages: expand | align | merge | rec | ra | sid_routing | eval_rec | eval_ra | eval_sid_routing | eval_sid_routing_ckpt | onerec | routing | eval_ra_full_cot | eval_sid_routing_full_cot | all"
            echo "Categories: Beauty | Sports | Toys"
            exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# ── 路径推导 ─────────────────────────────────────────────────────────
CAT_LOWER=$(echo "${CATEGORY}" | tr '[:upper:]' '[:lower:]')
DATA_DIR="${PROJ_ROOT}/data/processed/${CATEGORY}"
RESULTS_DIR="${PROJ_ROOT}/train/results/${CAT_LOWER}"
MERGED_MODEL_DIR="${PROJ_ROOT}/model/merged_${CAT_LOWER}_model"

ALIGN_TRAIN="${DATA_DIR}/training_align_data_train.parquet"
ALIGN_VAL="${DATA_DIR}/training_align_data_val.parquet"
SID_TRAIN="${DATA_DIR}/training_prediction_sid_data_train.parquet"
SID_VAL="${DATA_DIR}/training_prediction_sid_data_val.parquet"
SID_TEST="${DATA_DIR}/training_prediction_sid_data_test.parquet"
RA_TRAIN="${DATA_DIR}/training_RA_train.parquet"
RA_VAL="${DATA_DIR}/training_RA_val.parquet"
RA_TEST="${DATA_DIR}/training_RA_test.parquet"
ROUTING_TRAIN="${DATA_DIR}/training_sid_routing_train.parquet"
ROUTING_VAL="${DATA_DIR}/training_sid_routing_val.parquet"
ROUTING_TEST="${DATA_DIR}/training_sid_routing_test.parquet"
RA_TEST_COT="${DATA_DIR}/training_RA_test_cot_eval.parquet"
ROUTING_TEST_COT="${DATA_DIR}/training_sid_routing_test_cot_eval.parquet"
DIFF_FULL="${DATA_DIR}/test_difficulty_labels.json"
DIFF_SUBSET="${DATA_DIR}/test_difficulty_labels_cot_eval.json"
TRIE_REC="${DATA_DIR}/exact_trie_sid_test_full.pkl"
TRIE_RA="${DATA_DIR}/exact_trie_ra_test_full.pkl"
TRIE_ROUTING="${DATA_DIR}/exact_trie_routing_test_full.pkl"

stage3_recon_dir() {
    local task=$1
    local epoch=$2
    echo "${DATA_DIR}/reconstruct/${task}/epoch_${epoch}"
}

stage3_recon_file() {
    local task=$1
    local epoch=$2
    echo "$(stage3_recon_dir "${task}" "${epoch}")/reconstructed_data.parquet"
}

stage3_recon_report() {
    local task=$1
    local epoch=$2
    echo "$(stage3_recon_dir "${task}" "${epoch}")/reconstruction_report.json"
}

run_stage3_reconstruction() {
    local task=$1
    local model_path=$2
    local data_path=$3
    local epoch=$4
    local log_sink=${5:-${LOG_FILE}}

    local recon_output
    recon_output="$(stage3_recon_file "${task}" "${epoch}")"
    local recon_report
    recon_report="$(stage3_recon_report "${task}" "${epoch}")"

    python3 -u "${TRAIN_SCRIPTS}/s3_reconstruct_data.py" \
        --task "${task}" \
        --model_path "${model_path}" \
        --data_path "${data_path}" \
        --config_name "${CATEGORY}_${task}" \
        --epoch "${epoch}" \
        --num_gpus "${NUM_GPUS}" 2>&1 | tee -a "${log_sink}" "${LOG_FILE}"

    check_files "${recon_output}" "${recon_report}"
    echo "${recon_output}"
}

LOG_FILE="${LOG_DIR}/pipeline_${CAT_LOWER}_${STAGE}_$(date '+%Y%m%d_%H%M%S').log"

# ── System Prompt 定义（训练/评估对齐） ────────────────────────────
SYS_PROMPT_RA="You are a professional recommendation expert who needs to recommend the next possible purchase for users based on their purchase history. Please predict the most likely next product that the user will purchase based on the user's historical purchase information."
 
SYS_PROMPT_ROUTING="You are a professional recommendation expert who needs to recommend the next possible purchase for users based on their purchase history. Please predict the most likely next product that the user will purchase based on the user's historical purchase information. Express your prediction as a Semantic ID by navigating the SID hierarchy."

# ── 工具函数 ─────────────────────────────────────────────────────────
check_files() {
    for f in "$@"; do
        if [[ ! -e "${f}" ]]; then
            log ERROR "文件不存在: ${f}"
            exit 1
        fi
    done
}


check_generated_training_data() {
    check_files \
        "${ALIGN_TRAIN}" "${ALIGN_VAL}" \
        "${SID_TRAIN}" "${SID_VAL}" "${SID_TEST}" \
        "${RA_TRAIN}" "${RA_VAL}" "${RA_TEST}" "${RA_TEST_COT}" \
        "${ROUTING_TRAIN}" "${ROUTING_VAL}" "${ROUTING_TEST}" "${ROUTING_TEST_COT}" \
        "${DIFF_FULL}" "${DIFF_SUBSET}"
}

check_eval_assets() {
    check_files "${TRIE_REC}" "${TRIE_RA}" "${TRIE_ROUTING}" \
        "${TRIE_REC}.meta.json" "${TRIE_RA}.meta.json" "${TRIE_ROUTING}.meta.json"
}

kill_stale_gpu() {
    local pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' || true)
    if [[ -n "$pids" ]]; then
        log WARN "清理残留 GPU 进程: $pids"
        for pid in $pids; do kill -9 "$pid" 2>/dev/null || true; done
        sleep 3
    fi
}

compute_ga() {
    local target_effective_batch=$1
    local per_device_batch=$2

    [[ "${target_effective_batch}" =~ ^[1-9][0-9]*$ ]] || {
        log ERROR "compute_ga: 非法 target_effective_batch=${target_effective_batch}"
        exit 1
    }
    [[ "${per_device_batch}" =~ ^[1-9][0-9]*$ ]] || {
        log ERROR "compute_ga: 非法 per_device_batch=${per_device_batch}"
        exit 1
    }
    [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]] || {
        log ERROR "compute_ga: 非法 NUM_GPUS=${NUM_GPUS}"
        exit 1
    }

    local denom=$((per_device_batch * NUM_GPUS))
    local ga

    # 能严格对齐时，精确对齐 OneRec 目标有效 batch
    if (( target_effective_batch % denom == 0 )); then
        ga=$((target_effective_batch / denom))
    else
        # 不能严格整除时，向上取整，避免有效 batch 低于目标
        ga=$(((target_effective_batch + denom - 1) / denom))
        log WARN \
            "无法严格对齐有效 batch: target=${target_effective_batch}, per_device=${per_device_batch}, gpus=${NUM_GPUS}; 使用 GA=${ga}, effective_batch=$((ga * denom))"
    fi

    (( ga >= 1 )) || ga=1
    echo "${ga}"
}

# GA = target_effective_batch / (per_device_batch × num_gpus)
resolve_eval_gpus() {
    local visible
    visible=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    [[ ${visible} -lt 1 ]] && visible=1

    local eval_gpus=${NUM_GPUS}
    [[ ${eval_gpus} -lt 1 ]] && eval_gpus=1
    [[ ${eval_gpus} -gt ${visible} ]] && eval_gpus=${visible}

    echo "${eval_gpus}"
}

# ═══════════════════════════════════════════════════════════════════════
# Routing Think Max Tokens — 自动解析 (env → report → fallback)
# ═══════════════════════════════════════════════════════════════════════

_resolve_routing_think_max() {
    # 优先读环境变量 (训练+评估一条龙时可用)
    if [[ -n "${ROUTING_THINK_MAX_TOKENS:-}" ]] && [[ "${ROUTING_THINK_MAX_TOKENS}" =~ ^[0-9]+$ ]]; then
        echo "${ROUTING_THINK_MAX_TOKENS}"
        return
    fi
    # 其次读预检报告 (单独跑 eval 时从文件恢复)
    local report="${DATA_DIR}/think_length_report.json"
    if [[ -f "${report}" ]]; then
        local val
        val=$(python3 -c "import json; print(json.load(open('${report}'))['recommended_think_max_tokens'])" 2>/dev/null || echo "")
        if [[ "${val}" =~ ^[0-9]+$ ]]; then
            echo "${val}"
            return
        fi
    fi
    # 兜底默认值
    echo "128"
}

# ═══════════════════════════════════════════════════════════════════════
# 门控检查工具 — 检测阶段是否已完成
# ═══════════════════════════════════════════════════════════════════════

# 检查目录是否包含完整的 LoRA adapter
_has_lora_model() {
    local dir=$1
    # 根目录有 adapter
    [[ -f "${dir}/adapter_config.json" ]] && return 0
    # checkpoint 子目录有 adapter
    local ckpt
    for ckpt in "${dir}"/checkpoint-*/; do
        [[ -f "${ckpt}adapter_config.json" ]] && return 0
    done
    return 1
}

# 检查目录是否包含完整的全参数模型 (config.json + safetensors)
_has_full_model() {
    local dir=$1
    [[ -f "${dir}/config.json" ]] || return 1
    local n_safetensors
    n_safetensors=$(find "${dir}" -maxdepth 1 -name "*.safetensors" 2>/dev/null | wc -l)
    [[ ${n_safetensors} -gt 0 ]] && return 0
    return 1
}

# ── 最佳模型保存 (修复 DeepSpeed load_best_model_at_end 崩溃) ────────
save_best_model() {
    local out_dir=$1
    log INFO "查找最佳 checkpoint..."

    local latest_ckpt=$(ls -d "${out_dir}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
    if [[ -z "${latest_ckpt}" || ! -f "${latest_ckpt}/trainer_state.json" ]]; then
        log WARN "未找到 trainer_state.json, 跳过最佳模型保存"
        return
    fi

    local best_ckpt=$(python3 -c "
import json, sys
s = json.load(open('${latest_ckpt}/trainer_state.json'))
print(s.get('best_model_checkpoint', ''))
" 2>/dev/null || true)

    if [[ -z "${best_ckpt}" || ! -d "${best_ckpt}" ]]; then
        log WARN "best_model_checkpoint 无效 (${best_ckpt}), 使用最后一个 checkpoint"
        best_ckpt="${latest_ckpt}"
    fi

    local best_metric=$(python3 -c "
import json
s = json.load(open('${latest_ckpt}/trainer_state.json'))
print(f\"{s.get('best_metric', 'N/A')}\")
" 2>/dev/null || echo "N/A")

    log INFO "最佳 checkpoint: $(basename ${best_ckpt}) (metric=${best_metric})"

    for f in \
    config.json \
    generation_config.json \
    model.safetensors \
    model-00001-of-00002.safetensors \
    model-00002-of-00002.safetensors \
    model.safetensors.index.json \
    tokenizer.json \
    tokenizer_config.json \
    special_tokens_map.json \
    vocab.json \
    merges.txt \
    added_tokens.json \
    chat_template.jinja \
    training_args.bin \
    trainer_state.json \
    adapter_config.json \
    adapter_model.safetensors
do
    if [[ -f "${best_ckpt}/${f}" ]]; then
        cp -f "${best_ckpt}/${f}" "${out_dir}/${f}"
    fi
done

    log INFO "最佳模型已保存到: ${out_dir}"
}

_find_model() {
    local dir=$1
    if [[ -f "${dir}/config.json" ]]; then
        echo "${dir}"
    else
        local ckpt=$(ls -d "${dir}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
        echo "${ckpt}"
    fi
}

# ═══════════════════════════════════════════════════════════════════════
# expand: 扩词表
# ═══════════════════════════════════════════════════════════════════════
do_expand() {
    log INFO "═══ 扩词表 ═══"
    if [[ -d "${EXPAND_MODEL_DIR}" && -f "${EXPAND_MODEL_DIR}/config.json" ]]; then
        log INFO "已存在，跳过: ${EXPAND_MODEL_DIR}"
        return
    fi
    check_files "${BASE_MODEL_DIR}"
    python3 model/expand_vocab.py \
        --base_model_dir "${BASE_MODEL_DIR}" \
        --save_dir "${EXPAND_MODEL_DIR}" 2>&1 | tee -a "${LOG_FILE}"
    log INFO "扩词表完成"
}

# ═══════════════════════════════════════════════════════════════════════
# Stage 1: Alignment  |  有效 batch = 64
# ═══════════════════════════════════════════════════════════════════════
do_align() {
    log INFO "═══ Stage 1: Alignment [${CATEGORY}] (${NUM_GPUS} GPU) ═══"

    local out_dir="${RESULTS_DIR}/align"

    if _has_lora_model "${out_dir}"; then
        log INFO "✅ Stage 1 已完成, 跳过 (adapter 存在于 ${out_dir})"
        return
    fi

    check_files "${ALIGN_TRAIN}" "${ALIGN_VAL}" "${EXPAND_MODEL_DIR}"
    kill_stale_gpu

    mkdir -p "${out_dir}"

    local PER_DEVICE_BS=4
    local TARGET_EFFECTIVE_BS=64
    local GA
    GA=$(compute_ga "${TARGET_EFFECTIVE_BS}" "${PER_DEVICE_BS}")

    log INFO "有效 batch: ${PER_DEVICE_BS} × ${GA} × ${NUM_GPUS} = $((PER_DEVICE_BS * GA * NUM_GPUS))"

    local CMD_ARGS=(
        --model_dir "${EXPAND_MODEL_DIR}"
        --train_data_path "${ALIGN_TRAIN}"
        --val_data_path "${ALIGN_VAL}"
        --per_device_train_batch_size "${PER_DEVICE_BS}"
        --gradient_accumulation_steps "${GA}"
        --num_train_epochs 15
        --gradient_checkpointing True
        --bf16 True
        --output_dir "${out_dir}"
        --logging_dir "${LOG_DIR}/tb/${CAT_LOWER}_align"
        --logging_steps 10
        --eval_strategy epoch
        --eval_on_start False
        --save_strategy epoch
        --save_total_limit 5
        --metric_for_best_model eval_loss
        --greater_is_better False
        --load_best_model_at_end False
        --optim adamw_torch
        --learning_rate 1e-4
        --warmup_ratio 0.0
        --weight_decay 0.0
        --max_grad_norm 1.0
        --adam_beta1 0.9
        --adam_beta2 0.999
        --adam_epsilon 1e-8
        --dataloader_num_workers 4
        --remove_unused_columns False
        --category "${CATEGORY}"
        --report_to none
        --seed 42
        --data_seed 42
    )

    if [[ ${NUM_GPUS} -eq 1 ]]; then
        log INFO "单卡模式: 原生 PyTorch + 单卡优化"
        CUDA_VISIBLE_DEVICES=0 python3 "${TRAIN_SCRIPTS}/s1_train_align.py" \
            "${CMD_ARGS[@]}" \
            --single_gpu_mode True \
            --per_device_eval_batch_size 8 \
            --lr_scheduler_type constant 2>&1 | tee -a "${LOG_FILE}"
    else
        log INFO "多卡模式: DeepSpeed ZeRO-2"
        deepspeed --num_gpus ${NUM_GPUS} \
            "${TRAIN_SCRIPTS}/s1_train_align.py" \
            "${CMD_ARGS[@]}" \
            --single_gpu_mode False \
            --deepspeed "${DS_CONFIG}" 2>&1 | tee -a "${LOG_FILE}"
    fi

    save_best_model "${out_dir}"
    log INFO "Stage 1 完成: ${out_dir}"
}

# ═══════════════════════════════════════════════════════════════════════
# Merge LoRA
# ═══════════════════════════════════════════════════════════════════════
do_merge() {
    log INFO "═══ Merge [${CATEGORY}] ═══"

    # ── 门控: 检查合并模型是否已存在 ──
    if _has_full_model "${MERGED_MODEL_DIR}"; then
        log INFO "✅ Merge 已完成, 跳过 (模型存在于 ${MERGED_MODEL_DIR})"
        return
    fi

    local align_dir="${RESULTS_DIR}/align"
    check_files "${align_dir}"

    python3 model/merge_model.py \
        --base_model_path "${EXPAND_MODEL_DIR}" \
        --lora_model_path "${align_dir}" \
        --output_path "${MERGED_MODEL_DIR}" 2>&1 | tee -a "${LOG_FILE}"

    log INFO "Merge 完成: ${MERGED_MODEL_DIR}"
}

# ═══════════════════════════════════════════════════════════════════════
# Stage 2: SID Recommendation  |  有效 batch = 16
# ═══════════════════════════════════════════════════════════════════════
do_rec() {
    log INFO "═══ Stage 2: SID Rec [${CATEGORY}] (${NUM_GPUS} GPU) ═══"

    local out_dir="${RESULTS_DIR}/sid_rec"

    # ── 门控: 检查是否已完成 ──
    if _has_full_model "${out_dir}"; then
        log INFO "✅ Stage 2 已完成, 跳过 (模型存在于 ${out_dir})"
        return
    fi

    check_files "${SID_TRAIN}" "${SID_VAL}" "${MERGED_MODEL_DIR}"
    kill_stale_gpu

    mkdir -p "${out_dir}"

    local GA=$(compute_ga 16 2)
    log INFO "有效 batch: 2 × ${GA} × ${NUM_GPUS} = $((2 * GA * NUM_GPUS))"

    local CMD_ARGS=(
        --model_name_or_path "${MERGED_MODEL_DIR}"
        --train_data_path "${SID_TRAIN}"
        --val_data_path "${SID_VAL}"
        --use_lora False
        --per_device_train_batch_size 2
        --gradient_accumulation_steps ${GA}
        --num_train_epochs 6
        --gradient_checkpointing True
        --bf16 True
        --output_dir "${out_dir}"
        --logging_dir "${LOG_DIR}/tb/${CAT_LOWER}_sid_rec"
        --logging_steps 10
        --eval_strategy epoch
        --eval_on_start False
        --save_strategy epoch
        --save_total_limit 2
        --metric_for_best_model eval_loss
        --greater_is_better False
        --load_best_model_at_end False
        --optim adamw_torch
        --learning_rate 1e-5
        --warmup_ratio 0.1
        --weight_decay 0.01
        --max_grad_norm 1.0
        --dataloader_num_workers 4
        --remove_unused_columns False
        --seed 42
        --data_seed 42
    )

    if [[ ${NUM_GPUS} -eq 1 ]]; then
        log INFO "单卡模式: 原生 PyTorch"
        CUDA_VISIBLE_DEVICES=0 python3 "${TRAIN_SCRIPTS}/s2_train_sid_rec.py" \
            "${CMD_ARGS[@]}" \
            --lr_scheduler_type constant_with_warmup 2>&1 | tee -a "${LOG_FILE}"
    else
        log INFO "多卡模式: DeepSpeed ZeRO-2"
        deepspeed --num_gpus ${NUM_GPUS} \
            "${TRAIN_SCRIPTS}/s2_train_sid_rec.py" \
            "${CMD_ARGS[@]}" \
            --deepspeed "${DS_CONFIG}" 2>&1 | tee -a "${LOG_FILE}"
    fi

    save_best_model "${out_dir}"
    log INFO "Stage 2 完成: ${out_dir}"
}

# ═══════════════════════════════════════════════════════════════════════
# Stage 3a: Reasoning Activation 有效 batch = 16
# ═══════════════════════════════════════════════════════════════════════
do_ra() {
    log INFO "═══ Stage 3a: RA [${CATEGORY}] (${NUM_GPUS} GPU) ═══"

    # ── 门控: 检查 RA 最终 epoch 是否已完成 ──
    local ra_dir="${RESULTS_DIR}/ra"
    local num_epochs=2
    local final_epoch_dir="${ra_dir}/epoch_${num_epochs}"

    if _has_full_model "${final_epoch_dir}"; then
        log INFO "✅ Stage 3a RA 已完成, 跳过 (模型存在于 ${final_epoch_dir})"
        return
    fi
    # 也检查 checkpoint 子目录
    local final_ckpt=$(_find_model "${final_epoch_dir}" 2>/dev/null || true)
    if [[ -n "${final_ckpt}" ]] && _has_full_model "${final_ckpt}"; then
        log INFO "✅ Stage 3a RA 已完成, 跳过 (checkpoint: ${final_ckpt})"
        return
    fi

    check_files "${RA_TRAIN}"

    local rec_dir="${RESULTS_DIR}/sid_rec"
    local last_ckpt=$(_find_model "${rec_dir}")
    if [[ -z "${last_ckpt}" ]]; then
        log ERROR "找不到 sid_rec 模型"
        exit 1
    fi

    log INFO "RA 初始模型: ${last_ckpt}"

    local current_model="${last_ckpt}"
    local current_data="${RA_TRAIN}"

    local GA=$(compute_ga 16 2)
    log INFO "有效 batch: 2 × ${GA} × ${NUM_GPUS} = $((2 * GA * NUM_GPUS))"

    export RA_VAL_DATA="${RA_VAL}"

    for (( epoch=1; epoch<=num_epochs; epoch++ )); do
        local epoch_dir="${ra_dir}/epoch_${epoch}"

        # ── 单 epoch 门控 ──
        local epoch_model=$(_find_model "${epoch_dir}" 2>/dev/null || true)
        if [[ -n "${epoch_model}" ]] && _has_full_model "${epoch_model}"; then
            log INFO "RA Epoch ${epoch} 已完成, 跳过"
            current_model="${epoch_model}"
            # 检查重构数据
            local recon_data="$(stage3_recon_file ra ${epoch})"
            if [[ ${epoch} -lt ${num_epochs} && -f "${recon_data}" ]]; then
                current_data="${recon_data}"
            fi
            continue
        fi

        mkdir -p "${epoch_dir}"
        local epoch_log="${LOG_DIR}/${CAT_LOWER}_ra_epoch_${epoch}.log"

        log INFO "RA Epoch ${epoch}/${num_epochs} | model=${current_model}"

        local CMD_ARGS=(
            --model_name_or_path "${current_model}"
            --data_path "${current_data}"
            --use_lora False
            --per_device_train_batch_size 2
            --gradient_accumulation_steps ${GA}
            --num_train_epochs 1
            --gradient_checkpointing True
            --bf16 True
            --output_dir "${epoch_dir}"
            --logging_dir "${LOG_DIR}/tb/${CAT_LOWER}_ra_epoch_${epoch}"
            --logging_steps 1
            --eval_strategy "no"
            --save_strategy "epoch"
            --save_total_limit 1
            --load_best_model_at_end False
            --optim adamw_torch
            --learning_rate 1e-5
            --warmup_ratio 0.1
            --weight_decay 0.01
            --max_grad_norm 1.0
            --dataloader_num_workers 4
            --remove_unused_columns False
            --seed 42
            --data_seed 42
        )

        if [[ ${NUM_GPUS} -eq 1 ]]; then
            kill_stale_gpu
            CUDA_VISIBLE_DEVICES=0 python3 -u "${TRAIN_SCRIPTS}/s3a_train_ra.py" \
                "${CMD_ARGS[@]}" \
                --lr_scheduler_type constant_with_warmup 2>&1 | tee -a "${epoch_log}" "${LOG_FILE}"
        else
            deepspeed --num_gpus ${NUM_GPUS} \
                "${TRAIN_SCRIPTS}/s3a_train_ra.py" \
                "${CMD_ARGS[@]}" \
                --deepspeed "${DS_CONFIG}" 2>&1 | tee -a "${epoch_log}" "${LOG_FILE}"
        fi

        save_best_model "${epoch_dir}"

        if _has_full_model "${epoch_dir}"; then
            current_model="${epoch_dir}"
        else
            local saved_ckpt=$(ls -d "${epoch_dir}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
            current_model="${saved_ckpt:-${epoch_dir}}"
        fi

        log INFO "RA Epoch ${epoch} checkpoint: ${current_model}"

        # 数据重建 (非最后一个 epoch) — 使用 vLLM
        if [[ ${epoch} -lt ${num_epochs} ]]; then
            local recon_output="$(stage3_recon_file ra ${epoch})"
            if [[ -f "${recon_output}" ]]; then
                log INFO "重构数据已存在, 跳过: ${recon_output}"
            else
                log INFO "数据重建 (vLLM, epoch ${epoch}, ${NUM_GPUS} GPU)..."
                run_stage3_reconstruction ra "${current_model}" "${RA_TRAIN}" "${epoch}" "${epoch_log}" >/dev/null
            fi
            current_data="${recon_output}"
            log INFO "重建完成: ${current_data}"
        fi
    done

    log INFO "Stage 3a RA 完成"
}


# ═══════════════════════════════════════════════════════════════════════
# Routing 离线资产校验（train 只消费离线数据）
# ═══════════════════════════════════════════════════════════════════════
validate_routing_assets() {
    check_generated_training_data
    # Trie 文件是 eval-time 按需构建的，训练阶段不检查
    check_files "${DATA_DIR}/cognitive_map.json"
}

# ═══════════════════════════════════════════════════════════════════════
# Stage 3b: SID Routing (Ours)  |  有效 batch = 16
# ═══════════════════════════════════════════════════════════════════════

do_sid_routing() {
    log INFO "═══ Stage 3b: SID Routing [${CATEGORY}] (${NUM_GPUS} GPU) ═══"

    validate_routing_assets

    check_files "${ROUTING_TRAIN}" "${ROUTING_VAL}"

    local rec_dir="${RESULTS_DIR}/sid_rec"
    local rec_model=$(_find_model "${rec_dir}")
    if [[ -z "${rec_model}" ]]; then
        log ERROR "找不到 sid_rec 模型"
        exit 1
    fi

    local out_dir="${RESULTS_DIR}/sid_routing"

    # ── 门控: 检查是否已完成 ──
    if _has_full_model "${out_dir}"; then
        log INFO "✅ Stage 3b 已完成, 跳过 (模型存在于 ${out_dir})"
        return
    fi

    mkdir -p "${out_dir}"
    export ROUTING_VAL_DATA="${ROUTING_VAL}"

    # ── Think 长度预检 ──────────────────────────────────────────────
    log INFO "预检 think token 长度..."
    local think_check_output
    think_check_output=$(python3 -u "${TRAIN_SCRIPTS}/s3b_check_think_length.py" \
        --data_path "${ROUTING_TRAIN}" \
        --model_path "${rec_model}" 2>&1)
    echo "${think_check_output}" | tee -a "${LOG_FILE}"
    # 最后一行是推荐值
    ROUTING_THINK_MAX_TOKENS=$(echo "${think_check_output}" | tail -1 | tr -d '[:space:]')
    if ! [[ "${ROUTING_THINK_MAX_TOKENS}" =~ ^[0-9]+$ ]]; then
        log WARN "预检输出异常, fallback think_max_tokens=128"
        ROUTING_THINK_MAX_TOKENS=128
    fi
    export ROUTING_THINK_MAX_TOKENS
    log INFO "Routing think_max_tokens = ${ROUTING_THINK_MAX_TOKENS}"

    # ── 训练 ────────────────────────────────────────────────────────
    kill_stale_gpu

    local GA=$(compute_ga 16 2)
    log INFO "有效 batch: 2 × ${GA} × ${NUM_GPUS} = $((2 * GA * NUM_GPUS))"

    local CMD_ARGS=(
        --model_name_or_path "${rec_model}"
        --data_path "${ROUTING_TRAIN}"
        --use_lora False
        --per_device_train_batch_size 2
        --gradient_accumulation_steps ${GA}
        --num_train_epochs 3
        --gradient_checkpointing True
        --bf16 True
        --output_dir "${out_dir}"
        --logging_dir "${LOG_DIR}/tb/${CAT_LOWER}_sid_routing"
        --logging_steps 10
        --eval_strategy epoch
        --eval_on_start False
        --save_strategy epoch
        --save_total_limit 3
        --metric_for_best_model eval_loss
        --greater_is_better False
        --load_best_model_at_end False
        --optim adamw_torch
        --learning_rate 1e-5
        --warmup_ratio 0.1
        --weight_decay 0.01
        --max_grad_norm 1.0
        --dataloader_num_workers 4
        --remove_unused_columns False
        --seed 42
        --data_seed 42
    )

    if [[ ${NUM_GPUS} -eq 1 ]]; then
        log INFO "单卡模式: 原生 PyTorch"
        CUDA_VISIBLE_DEVICES=0 python3 -u "${TRAIN_SCRIPTS}/s3b_train_sid_routing.py" \
            "${CMD_ARGS[@]}" \
            --lr_scheduler_type constant_with_warmup 2>&1 | tee -a "${LOG_FILE}"
    else
        log INFO "多卡模式: DeepSpeed ZeRO-2"
        deepspeed --num_gpus ${NUM_GPUS} \
            "${TRAIN_SCRIPTS}/s3b_train_sid_routing.py" \
            "${CMD_ARGS[@]}" \
            --deepspeed "${DS_CONFIG}" 2>&1 | tee -a "${LOG_FILE}"
    fi

    save_best_model "${out_dir}"
    log INFO "Stage 3b SID Routing 训练完成: ${out_dir}"
}

# ═══════════════════════════════════════════════════════════════════════
# 评估工具函数 — 自适应单/多卡
# ═══════════════════════════════════════════════════════════════════════

resolve_trie_file() {
    local task=$1
    case "${task}" in
        rec) echo "${TRIE_REC}" ;;
        ra) echo "${TRIE_RA}" ;;
        routing) echo "${TRIE_ROUTING}" ;;
        *) log ERROR "未知 trie task: ${task}" >&2; return 1 ;;
    esac
}

_trie_source_file() {
    local task=$1
    case "${task}" in
        rec)     echo "${SID_TEST}" ;;
        ra)      echo "${RA_TEST}" ;;
        routing) echo "${ROUTING_TEST}" ;;
        *) log ERROR "未知 trie task: ${task}" >&2; return 1 ;;
    esac
}

_ensure_trie() {
    local model_path=$1
    local task=$2
    local trie_source_file=$3
    local trie_file
    trie_file="$(resolve_trie_file "${task}")" || return 1

    if [[ ! -f "${trie_file}" ]]; then
        log INFO "构建 Trie: ${trie_file}" >&2
        CUDA_VISIBLE_DEVICES=0 python3 test/precompute_global_trie.py \
            --test_parquet_file "${trie_source_file}" \
            --model_path "${model_path}" \
            --output_file "${trie_file}" >> "${LOG_FILE}" 2>&1

        if [[ ! -f "${trie_file}" ]]; then
            log ERROR "Trie 构建失败: ${trie_file}" >&2
            return 1
        fi
        log INFO "Trie 构建完成: ${trie_file}" >&2
    else
        log INFO "Trie 已存在，复用: ${trie_file}" >&2
    fi

    echo "${trie_file}"
}

# ═══════════════════════════════════════════════════════════════════════
# 多卡评估结果聚合
# ═══════════════════════════════════════════════════════════════════════
_aggregate_eval_results() {
    local log_dir="$1"
    local num_gpus="$2"
    local summary="$3"
    local summary_json="${log_dir}/summary_results.json"

    python3 - "$log_dir" "$num_gpus" "$summary" "$summary_json" <<'PY'
import json, os, re, sys
from collections import defaultdict
from pathlib import Path

log_dir = Path(sys.argv[1])
num_gpus = int(sys.argv[2])
summary_path = Path(sys.argv[3])
summary_json_path = Path(sys.argv[4])
metrics = ["hit@1", "hit@5", "hit@10", "ndcg@5", "ndcg@10"]


def parse_progress(content):
    lines = content.splitlines()
    records = []
    i = 0
    while i < len(lines):
        line = lines[i]
        pm = re.search(r"PROGRESS REPORT\s*-\s*Step\s*(\d+)\s*/\s*(\d+)(?:\s*\(Samples:\s*(\d+)\))?", line)
        if not pm:
            i += 1
            continue
        step = int(pm.group(1))
        total_steps = int(pm.group(2))
        samples = int(pm.group(3)) if pm.group(3) else 0
        rec = {"step": step, "total_steps": total_steps, "samples": samples, "metrics": {}}
        j = i + 1
        while j < min(i + 20, len(lines)):
            sub = lines[j]
            sm = re.search(r"Processed samples:\s*(\d+)", sub)
            if sm:
                rec["samples"] = int(sm.group(1))
            mm = re.search(r"(hit@\d+|ndcg@\d+)\s*:\s*([\d.]+)", sub)
            if mm:
                rec["metrics"][mm.group(1)] = float(mm.group(2))
            if re.match(r"\s*={10,}", sub) and j > i + 1:
                break
            j += 1
        if rec["metrics"]:
            records.append(rec)
        i = j + 1
    return records


def parse_samples(content, progress):
    m = re.search(r"Total samples:\s*(\d+)", content)
    if m and int(m.group(1)) > 0:
        return int(m.group(1))
    m = re.search(r"Test data size:\s*(\d+)", content)
    if m and int(m.group(1)) > 0:
        return int(m.group(1))
    m = re.findall(r"Loaded eval samples=(\d+)", content)
    if m:
        return int(m[-1])
    samples = [p["samples"] for p in progress if p.get("samples", 0) > 0]
    return max(samples) if samples else 0


def parse_final(content):
    m = re.search(r"Final CoT Hit Rate Results.*?(hit@1:.*?ndcg@10:\s*[\d.]+)", content, flags=re.S)
    if not m:
        m = re.search(r"Final Hit Rate Results.*?(hit@1:.*?ndcg@10:\s*[\d.]+)", content, flags=re.S)
    block = m.group(1) if m else content[-1200:]
    out = {}
    for metric in metrics:
        mm = re.findall(rf"{re.escape(metric)}\s*:\s*([\d.]+)", block)
        if mm:
            out[metric] = float(mm[-1])
    return out


def detect_variant(log_dir):
    name = log_dir.name.lower()
    clean = re.sub(r"^eval_", "", name)
    clean = re.sub(r"_\d{8}_\d{6}$", "", clean)

    if "sid_routing_full_cot_ra_prompt" in name:
        return ("sid_routing_full_cot_ra_prompt", "sid_routing", "cot", "full")
    if "sid_routing_subset_cot_ra_prompt" in name:
        return ("sid_routing_subset_cot_ra_prompt", "sid_routing", "cot", "subset")
    if "sid_routing_ra_prompt" in name:
        return ("sid_routing_ra_prompt", "sid_routing", "no_cot", "full")
    if "sid_routing_full_cot" in name:
        return ("sid_routing_full_cot", "sid_routing", "cot", "full")
    if "sid_routing_subset_cot" in name or "sid_routing_cot" in name:
        return ("sid_routing_subset_cot", "sid_routing", "cot", "subset")
    if "ra_full_cot" in name:
        return ("ra_full_cot", "ra", "cot", "full")
    if "ra_subset_cot" in name or "ra_cot" in name:
        return ("ra_subset_cot", "ra", "cot", "subset")
    if "sid_routing" in name:
        return ("sid_routing", "sid_routing", "no_cot", "full")
    if "ra" in name:
        return ("ra", "ra", "no_cot", "full")
    return ("rec", "rec", "no_cot", "full")


total_metrics = defaultdict(float)
per_gpu = []
found = 0
progress_by_step = defaultdict(lambda: {"samples": [], "metrics": defaultdict(list), "total_steps": 0})
variant_name, task, mode, scope = detect_variant(log_dir)

for gpu_id in range(num_gpus):
    path = log_dir / f"gpu_{gpu_id}.log"
    if not path.exists():
        per_gpu.append({"gpu_id": gpu_id, "found": False})
        continue
    content = path.read_text(encoding="utf-8", errors="ignore")
    progress = parse_progress(content)
    final = parse_final(content)
    samples = parse_samples(content, progress)
    if final:
        found += 1
        for m, v in final.items():
            total_metrics[m] += v * samples  # 加权累加
    for rec in progress:
        cur = progress_by_step[rec["step"]]
        cur["total_steps"] = max(cur["total_steps"], rec["total_steps"])
        if rec.get("samples", 0) > 0:
            cur["samples"].append(rec["samples"])
        for m, v in rec["metrics"].items():
            cur["metrics"][m].append(v)
    per_gpu.append({
        "gpu_id": gpu_id,
        "found": bool(final),
        "samples": samples,
        "final_metrics": final,
        "progress_points": len(progress),
    })

total_samples = sum(x.get("samples", 0) for x in per_gpu if x.get("found"))
avg_metrics = {m: (total_metrics[m] / total_samples if total_samples > 0 else 0.0) for m in metrics}
progress = []
for step in sorted(progress_by_step):
    cur = progress_by_step[step]
    step_total = sum(cur["samples"]) if cur["samples"] else 0
    step_metrics = {}
    for m, vs in cur["metrics"].items():
        if vs and cur["samples"] and len(vs) == len(cur["samples"]):
            step_metrics[m] = float(sum(v * s for v, s in zip(vs, cur["samples"])) / step_total) if step_total > 0 else 0.0
        elif vs:
            step_metrics[m] = float(sum(vs) / len(vs))
    progress.append({
        "step": step,
        "total_steps": cur["total_steps"],
        "samples": step_total,
        "metrics": step_metrics,
    })

with summary_path.open("w", encoding="utf-8") as f:
    f.write("Multi-GPU Evaluation Summary\n")
    f.write("=" * 60 + "\n")
    for item in per_gpu:
        gpu_id = item["gpu_id"]
        if not item.get("found"):
            f.write(f"GPU {gpu_id}: No results found\n")
            continue
        f.write(f"GPU {gpu_id}: {item['samples']} samples\n")
        for m in metrics:
            if m in item["final_metrics"]:
                f.write(f"    {m}: {item['final_metrics'][m]:.4f}\n")
    f.write("\nFINAL WEIGHTED-AVERAGED RESULTS:\n")
    f.write("=" * 60 + "\n")
    for m in metrics:
        f.write(f"{m:>10}: {avg_metrics[m]:.4f}\n")
    f.write("=" * 60 + "\n")
    f.write(f"Total samples: {total_samples}\n")
    f.write(f"Completed GPUs: {found}/{num_gpus}\n")

payload = {
    "variant_name": variant_name,
    "task": task,
    "mode": mode,
    "scope": scope,
    "eval_log_dir": str(log_dir),
    "summary_log": str(summary_path),
    "summary_json": str(summary_json_path),
    "gpu_logs": [str(log_dir / f"gpu_{i}.log") for i in range(num_gpus) if (log_dir / f"gpu_{i}.log").exists()],
    "num_gpus": num_gpus,
    "completed_gpus": found,
    "per_gpu": per_gpu,
    "progress": progress,
    "final_metrics": avg_metrics,
    "total_samples": total_samples,
}
summary_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"wrote {summary_path}")
print(f"wrote {summary_json_path}")
PY

    if [[ -f "${summary}" ]]; then
        log INFO "聚合完成: ${summary}"
    else
        log ERROR "聚合失败: ${summary} 未生成"
    fi
}

_eval_no_cot_on_file() {
    local model_path=$1
    local test_file=$2
    local trie_file=$3
    local label=$4
    local trie_source_file=$5

    check_files "${model_path}/config.json" "${test_file}" "${trie_file}" "${trie_source_file}"

    local EVAL_GPUS
    EVAL_GPUS=$(resolve_eval_gpus)

    log INFO "评估 [${label}] 模型 (无CoT, ${EVAL_GPUS} GPU): ${model_path}"

    local eval_log_dir="${LOG_DIR}/eval_${label}_${CAT_LOWER}_$(date '+%Y%m%d_%H%M%S')"
    mkdir -p "${eval_log_dir}"

    local sys_prompt="${SYS_PROMPT_RA}"
    if [[ "${label}" == *"sid_routing"* || "${label}" == *"routing"* ]]; then
        sys_prompt="${SYS_PROMPT_ROUTING}"
    fi
 
    local COMMON_ARGS=(
        --merged_model_path "${model_path}"
        --test_parquet_file "${test_file}"
        --global_trie_file "${trie_file}"
        --trie_source_parquet_file "${trie_source_file}"
        --test_batch_size 4
        --num_beams 10
        --metrics "hit@1,hit@5,hit@10,ndcg@5,ndcg@10"
        --max_new_tokens 6
        --think_max_tokens 0
        --temperature 0.6
        --top_p 1
        --print_generations
        --system_prompt "${sys_prompt}"
    )

    if [[ ${EVAL_GPUS} -ge 2 ]]; then
        local TOTAL_SAMPLES
        TOTAL_SAMPLES=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('${test_file}')))" )
        [[ ${TOTAL_SAMPLES} -lt ${EVAL_GPUS} ]] && EVAL_GPUS=${TOTAL_SAMPLES}
        local SAMPLES_PER_GPU=$(( (TOTAL_SAMPLES + EVAL_GPUS - 1) / EVAL_GPUS ))

        log INFO "多卡评估: ${TOTAL_SAMPLES} 样本 / ${EVAL_GPUS} GPU ≈ ${SAMPLES_PER_GPU}/GPU"

        local pids=()
        for (( gpu_id=0; gpu_id<EVAL_GPUS; gpu_id++ )); do
            local offset=$((gpu_id * SAMPLES_PER_GPU))
            local gpu_log="${eval_log_dir}/gpu_${gpu_id}.log"

            CUDA_VISIBLE_DEVICES=${gpu_id} python3 -u test/test_hitrate.py \
                "${COMMON_ARGS[@]}" \
                --sample_num ${SAMPLES_PER_GPU} \
                --sample_offset ${offset} \
                --gpu_id ${gpu_id} \
                --log_file "${gpu_log}" > "${eval_log_dir}/stdout_gpu_${gpu_id}.log" 2>&1 &
            pids+=($!)
            sleep 1
        done

        for pid in "${pids[@]}"; do
            wait ${pid}
        done

        local summary_log="${eval_log_dir}/summary_results.log"
        _aggregate_eval_results "${eval_log_dir}" "${EVAL_GPUS}" "${summary_log}"
        
        local main_log="${LOG_DIR}/eval_${label}_${CAT_LOWER}.log"
        : > "${main_log}"
        for (( gpu_id=0; gpu_id<EVAL_GPUS; gpu_id++ )); do
            local gpu_log="${eval_log_dir}/gpu_${gpu_id}.log"
            [[ -f "${gpu_log}" ]] || continue
            cat "${gpu_log}" >> "${main_log}"
        done
        if [[ -f "${summary_log}" ]]; then
            cat "${summary_log}" >> "${main_log}"
        fi
        return 0
    else
        CUDA_VISIBLE_DEVICES=0 python3 -u test/test_hitrate.py \
            "${COMMON_ARGS[@]}" \
            --log_file "${LOG_DIR}/eval_${label}_${CAT_LOWER}.log" 2>&1 | tee -a "${LOG_FILE}"
    fi
}

_eval_cot_on_file() {
    local model_path=$1
    local test_file=$2
    local trie_file=$3
    local label=$4
    local trie_source_file=$5
    local think_max=${6:-128}
    local max_history_items=${7:-}
    local history_truncation_side=${8:-}
    local eval_scope=${9:-}

    if ! [[ "${max_history_items}" =~ ^[0-9]+$ ]]; then
        log ERROR "CoT max_history_items 必须是非负整数, got: ${max_history_items:-<empty>}"
        return 1
    fi
    case "${history_truncation_side}" in
        head|tail) ;;
        *)
            log ERROR "CoT history_truncation_side 必须是 head 或 tail, got: ${history_truncation_side:-<empty>}"
            return 1
            ;;
    esac
    case "${eval_scope}" in
        subset|full) ;;
        *)
            log ERROR "CoT scope 必须是 subset 或 full, got: ${eval_scope:-<empty>}"
            return 1
            ;;
    esac
    if [[ "${eval_scope}" == "subset" && "${max_history_items}" -ne 0 ]]; then
        log ERROR "Subset-CoT 不允许历史截断; max_history_items 必须为 0"
        return 1
    fi
    if [[ "${eval_scope}" == "full" && "${max_history_items}" -eq 0 ]]; then
        log WARN "Full-CoT max_history_items=0: 完整历史可能导致 OOM"
    fi

    log INFO "CoT 配置 [${label}]: scope=${eval_scope}, max_history_items=${max_history_items}, history_truncation_side=${history_truncation_side}"

    check_files "${model_path}/config.json" "${test_file}" "${trie_file}" "${trie_source_file}"

    local EVAL_GPUS
    EVAL_GPUS=$(resolve_eval_gpus)

    local TOTAL_SAMPLES
    TOTAL_SAMPLES=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('${test_file}')))")
    [[ ${TOTAL_SAMPLES} -lt ${EVAL_GPUS} ]] && EVAL_GPUS=${TOTAL_SAMPLES}

    log INFO "评估 [${label}] 模型 (CoT, ${EVAL_GPUS} GPU, ${TOTAL_SAMPLES} 样本): ${model_path}"

    local eval_log_dir="${LOG_DIR}/eval_${label}_${CAT_LOWER}_$(date '+%Y%m%d_%H%M%S')"
    mkdir -p "${eval_log_dir}"

    local sys_prompt="${SYS_PROMPT_RA}"
    if [[ "${label}" == *"sid_routing"* || "${label}" == *"routing"* ]]; then
        sys_prompt="${SYS_PROMPT_ROUTING}"
    fi
 
    local COMMON_ARGS=(
        --merged_model_path "${model_path}"
        --test_parquet_file "${test_file}"
        --global_trie_file "${trie_file}"
        --test_batch_size 4
        --num_thinking_samples 5
        --num_beams_per_sample 10
        --metrics "hit@1,hit@5,hit@10,ndcg@5,ndcg@10"
        --think_max_tokens ${think_max}
        --sid_max_tokens 8
        --think_temperature 1.5
        --think_top_p 0.95
        --sid_temperature 0.6
        --sid_top_p 1
        --print_generations
        --system_prompt "${sys_prompt}"
        --trie_source_parquet_file "${trie_source_file}"
        --max_history_items "${max_history_items}"
        --history_truncation_side "${history_truncation_side}"
        --evaluation_scope "${eval_scope}"
    )

    if [[ ${EVAL_GPUS} -ge 2 ]]; then
        local SAMPLES_PER_GPU=$(( (TOTAL_SAMPLES + EVAL_GPUS - 1) / EVAL_GPUS ))
        log INFO "多卡 CoT 评估: ${TOTAL_SAMPLES} 样本 / ${EVAL_GPUS} GPU ≈ ${SAMPLES_PER_GPU}/GPU"

        local pids=()
        for (( gpu_id=0; gpu_id<EVAL_GPUS; gpu_id++ )); do
            local offset=$((gpu_id * SAMPLES_PER_GPU))
            local gpu_log="${eval_log_dir}/gpu_${gpu_id}.log"

            CUDA_VISIBLE_DEVICES=${gpu_id} python3 -u test/test_hitrate_cot.py \
                "${COMMON_ARGS[@]}" \
                --sample_num ${SAMPLES_PER_GPU} \
                --sample_offset ${offset} \
                --gpu_id ${gpu_id} \
                --log_file "${gpu_log}" > "${eval_log_dir}/stdout_gpu_${gpu_id}.log" 2>&1 &
            pids+=($!)
            sleep 2
        done

        for pid in "${pids[@]}"; do
            wait ${pid}
        done

       local summary_log="${eval_log_dir}/summary_results.log"
        _aggregate_eval_results "${eval_log_dir}" "${EVAL_GPUS}" "${summary_log}"
        
        local main_log="${LOG_DIR}/eval_${label}_${CAT_LOWER}.log"
        : > "${main_log}"
        for (( gpu_id=0; gpu_id<EVAL_GPUS; gpu_id++ )); do
            local gpu_log="${eval_log_dir}/gpu_${gpu_id}.log"
            [[ -f "${gpu_log}" ]] || continue
            cat "${gpu_log}" >> "${main_log}"
        done
        if [[ -f "${summary_log}" ]]; then
            cat "${summary_log}" >> "${main_log}"
        fi
        return 0
    else
        CUDA_VISIBLE_DEVICES=0 python3 -u test/test_hitrate_cot.py \
            "${COMMON_ARGS[@]}" \
            --sample_num ${TOTAL_SAMPLES} \
            --sample_offset 0 \
            --gpu_id 0 \
            --log_file "${LOG_DIR}/eval_${label}_${CAT_LOWER}.log" 2>&1 | tee -a "${LOG_FILE}"
    fi
}

_eval_no_cot() {
    local model_path=$1
    local label=$2
    _eval_no_cot_on_file "${model_path}" "${SID_TEST}" "${TRIE_REC}" "${label}"
}

_resolve_ra_think_max() {
    local report="${DATA_DIR}/think_length_report_ra.json"
    if [[ -f "${report}" ]]; then
        local val
        val=$(python3 -c "import json; print(json.load(open('${report}')).get('recommended_think_max_tokens', 64))" 2>/dev/null || echo "")
        if [[ "${val}" =~ ^[0-9]+$ ]]; then
            echo "${val}"
            return
        fi
    fi
    echo 64
}


# ═══════════════════════════════════════════════════════════════════════
# 评估入口
# ═══════════════════════════════════════════════════════════════════════
do_eval_rec() {
    log INFO "═══ Eval: rec [${CATEGORY}] ═══"
    local existing_log
    existing_log=$(find "${LOG_DIR}" -name "eval_rec_${CAT_LOWER}*.log" -size +200c 2>/dev/null | head -1)
    if [[ -n "${existing_log}" ]] && grep -q "hit@1" "${existing_log}" 2>/dev/null; then
        log INFO "✅ eval_rec 已完成, 跳过 (${existing_log})"
        return
    fi

    local model=$(_find_model "${RESULTS_DIR}/sid_rec")
    [[ -n "${model}" ]] || { log ERROR "找不到 sid_rec 模型"; exit 1; }

    local trie_source_file=$(_trie_source_file rec)
    local trie_file=$(_ensure_trie "${model}" rec "${trie_source_file}")
    _eval_no_cot_on_file "${model}" "${SID_TEST}" "${trie_file}" "rec" "${SID_TEST}"
}

do_eval_ra() {
    log INFO "═══ Eval: ra [${CATEGORY}] ═══"

    local no_cot_log="${LOG_DIR}/eval_ra_${CAT_LOWER}.log"
    local subset_cot_log="${LOG_DIR}/eval_ra_subset_cot_${CAT_LOWER}.log"
    if [[ -f "${no_cot_log}" && -f "${subset_cot_log}" ]]        && grep -q "hit@1" "${no_cot_log}" 2>/dev/null        && grep -q "hit@1" "${subset_cot_log}" 2>/dev/null; then
        log INFO "✅ eval_ra 已完成, 跳过 (${no_cot_log}, ${subset_cot_log})"
        return
    fi

    local ra_dir="${RESULTS_DIR}/ra/epoch_2"
    local model=$(_find_model "${ra_dir}")
    [[ -n "${model}" ]] || { log ERROR "找不到 RA 模型"; exit 1; }

    local trie_source_file=$(_trie_source_file ra)
    local trie_file=$(_ensure_trie "${model}" ra "${trie_source_file}")

    # 1. No-CoT 评估 (全量 RA test)
    _eval_no_cot_on_file "${model}" "${RA_TEST}" "${trie_file}" "ra" "${RA_TEST}"

    # 2. CoT 评估 (subset, 默认 — 策略/参数对齐 OneRec，仅数据量缩小)
    [[ -f "${RA_TEST_COT}" ]] || { log ERROR "RA CoT 子集不存在: ${RA_TEST_COT}"; exit 1; }
    _eval_cot_on_file "${model}" "${RA_TEST_COT}" "${trie_file}" "ra_subset_cot" "${RA_TEST}" 128 0 tail subset
}

do_eval_sid_routing() {
    log INFO "═══ Eval: sid_routing [${CATEGORY}] ═══"

    local no_cot_log="${LOG_DIR}/eval_sid_routing_${CAT_LOWER}.log"
    local subset_cot_log="${LOG_DIR}/eval_sid_routing_subset_cot_${CAT_LOWER}.log"
    if [[ -f "${no_cot_log}" && -f "${subset_cot_log}" ]] \
       && grep -q "hit@1" "${no_cot_log}" 2>/dev/null \
       && grep -q "hit@1" "${subset_cot_log}" 2>/dev/null; then
        log INFO "✅ eval_sid_routing 已完成, 跳过 (${no_cot_log}, ${subset_cot_log})"
        return
    fi

    local routing_dir="${RESULTS_DIR}/sid_routing"
    local model=$(_find_model "${routing_dir}")
    [[ -n "${model}" ]] || { log ERROR "找不到 Routing 模型"; exit 1; }

    local trie_source_file=$(_trie_source_file routing)
    local trie_file=$(_ensure_trie "${model}" routing "${trie_source_file}")

    local think_max=$(_resolve_routing_think_max)

    # 1. No-CoT 评估 (全量 Routing test)
    _eval_no_cot_on_file "${model}" "${ROUTING_TEST}" "${trie_file}" "sid_routing" "${ROUTING_TEST}"

    # 2. CoT 评估 (subset, 默认)
    [[ -f "${ROUTING_TEST_COT}" ]] || { log ERROR "Routing CoT 子集不存在: ${ROUTING_TEST_COT}"; exit 1; }
    _eval_cot_on_file "${model}" "${ROUTING_TEST_COT}" "${trie_file}" "sid_routing_subset_cot" "${ROUTING_TEST}" "${think_max}" 0 tail subset
}

do_eval_ra_full_cot() {
    log INFO "═══ Eval: ra FULL CoT [${CATEGORY}] ═══"

    local existing_log="${LOG_DIR}/eval_ra_full_cot_${CAT_LOWER}.log"
    if [[ -f "${existing_log}" ]] && grep -q "hit@1" "${existing_log}" 2>/dev/null; then
        log INFO "✅ eval_ra_full_cot 已完成, 跳过 (${existing_log})"
        return
    fi

    local ra_dir="${RESULTS_DIR}/ra/epoch_2"
    local model=$(_find_model "${ra_dir}")
    [[ -n "${model}" ]] || { log ERROR "找不到 RA 模型"; exit 1; }

    local trie_source_file=$(_trie_source_file ra)
    local trie_file=$(_ensure_trie "${model}" ra "${trie_source_file}")

    _eval_cot_on_file "${model}" "${RA_TEST}" "${trie_file}" "ra_full_cot" "${RA_TEST}" 128 \
        "${COT_MAX_HISTORY_ITEMS:-0}" "${COT_HISTORY_TRUNCATION_SIDE:-tail}" full
}

do_eval_sid_routing_full_cot() {
    log INFO "═══ Eval: sid_routing FULL CoT [${CATEGORY}] ═══"

    local existing_log="${LOG_DIR}/eval_sid_routing_full_cot_${CAT_LOWER}.log"
    if [[ -f "${existing_log}" ]] && grep -q "hit@1" "${existing_log}" 2>/dev/null; then
        log INFO "✅ eval_sid_routing_full_cot 已完成, 跳过 (${existing_log})"
        return
    fi

    local routing_dir="${RESULTS_DIR}/sid_routing"
    local model=$(_find_model "${routing_dir}")
    [[ -n "${model}" ]] || { log ERROR "找不到 Routing 模型"; exit 1; }

    local trie_source_file=$(_trie_source_file routing)
    local trie_file=$(_ensure_trie "${model}" routing "${trie_source_file}")

    local think_max=$(_resolve_routing_think_max)
    _eval_cot_on_file "${model}" "${ROUTING_TEST}" "${trie_file}" "sid_routing_full_cot" "${ROUTING_TEST}" "${think_max}" \
        "${COT_MAX_HISTORY_ITEMS:-0}" "${COT_HISTORY_TRUNCATION_SIDE:-tail}" full
}

# ═══════════════════════════════════════════════════════════════════════
# 手动选择 checkpoint 评估 (No-CoT + CoT)
# ═══════════════════════════════════════════════════════════════════════

_list_routing_checkpoints() {
    local base_dir="${RESULTS_DIR}/sid_routing"
    local found=()

    # 根目录本身 (save_best_model 产物)
    if [[ -f "${base_dir}/config.json" ]]; then
        found+=("${base_dir}  [save_best_model]")
    fi

    # 所有 checkpoint 子目录
    for ckpt_dir in $(ls -d "${base_dir}"/checkpoint-* 2>/dev/null | sort -V); do
        if [[ -f "${ckpt_dir}/config.json" ]]; then
            # 尝试读取对应 eval_loss
            local eval_info=""
            local ts_file="${ckpt_dir}/trainer_state.json"
            if [[ -f "${ts_file}" ]]; then
                eval_info=$(python3 -c "
import json
s = json.load(open('${ts_file}'))
logs = s.get('log_history', [])
eval_entries = [l for l in logs if 'eval_loss' in l]
if eval_entries:
    last = eval_entries[-1]
    print(f\"eval_loss={last['eval_loss']:.4f}  step={last.get('step','?')}  epoch={last.get('epoch','?')}\")
else:
    print('no eval_loss recorded')
" 2>/dev/null || echo "")
            fi
            found+=("${ckpt_dir}  ${eval_info}")
        fi
    done

    if [[ ${#found[@]} -eq 0 ]]; then
        log ERROR "未找到任何 routing checkpoint: ${base_dir}"
        return 1
    fi

    echo ""
    echo "═══ 可用 Routing Checkpoints ═══"
    for i in "${!found[@]}"; do
        echo "  [$i]  ${found[$i]}"
    done
    echo "════════════════════════════════"
    echo ""

    # 返回纯路径列表供选择
    for entry in "${found[@]}"; do
        echo "${entry}" | awk '{print $1}'
    done
}

do_eval_sid_routing_ckpt() {
    log INFO "═══ Eval: sid_routing 手动 Checkpoint [${CATEGORY}] ═══"

    local base_dir="${RESULTS_DIR}/sid_routing"
    local model_path=""

    # ── Step 1: 选择 checkpoint ─────────────────────────────────────
    if [[ -n "${ROUTING_CKPT:-}" ]]; then
        model_path="${ROUTING_CKPT}"
        log INFO "使用环境变量指定的 checkpoint: ${model_path}"
    else
        local ckpt_paths=()
        local ckpt_labels=()

        if [[ -f "${base_dir}/config.json" ]]; then
            ckpt_paths+=("${base_dir}")
            ckpt_labels+=("[save_best_model]")
        fi

        for ckpt_dir in $(find "${base_dir}" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | sort -V); do
            if [[ -f "${ckpt_dir}/config.json" ]]; then
                local eval_info=""
                local ts_file="${ckpt_dir}/trainer_state.json"
                if [[ -f "${ts_file}" ]]; then
                    eval_info=$(python3 -c "
import json
s = json.load(open('${ts_file}'))
logs = s.get('log_history', [])
eval_entries = [l for l in logs if 'eval_loss' in l]
if eval_entries:
    last = eval_entries[-1]
    print(f\"eval_loss={last['eval_loss']:.4f}  epoch={last.get('epoch','?')}\")
else:
    print('')
" 2>/dev/null || echo "")
                fi
                ckpt_paths+=("${ckpt_dir}")
                ckpt_labels+=("${eval_info}")
            fi
        done

        if [[ ${#ckpt_paths[@]} -eq 0 ]]; then
            log ERROR "未找到任何 routing checkpoint: ${base_dir}"
            exit 1
        fi

        echo ""
        echo "═══ 可用 Checkpoints (${#ckpt_paths[@]} 个) ═══"
        for i in "${!ckpt_paths[@]}"; do
            printf "  [%d]  %s  %s\n" "$i" "${ckpt_paths[$i]}" "${ckpt_labels[$i]}"
        done
        echo "════════════════════════════════════════════════════════"
        echo ""

        local choice
        echo -n "选择 checkpoint 编号 [0-$((${#ckpt_paths[@]}-1))]: "
        read -r choice

        if ! [[ "${choice}" =~ ^[0-9]+$ ]] || [[ ${choice} -ge ${#ckpt_paths[@]} ]]; then
            log ERROR "无效选择: ${choice}"
            exit 1
        fi
        model_path="${ckpt_paths[$choice]}"
    fi

    if [[ ! -f "${model_path}/config.json" ]]; then
        log ERROR "指定路径不含有效模型: ${model_path}"
        exit 1
    fi

    # ── Step 2: 选择评估模式 ────────────────────────────────────────
    local eval_mode="${ROUTING_EVAL_MODE:-}"
    if [[ -z "${eval_mode}" ]]; then
        echo ""
        echo "═══ 评估模式 ═══"
        echo "  [1]  No-CoT + CoT Subset"
        echo "  [2]  Full CoT"
        echo "  [3]  全部 (No-CoT + CoT Subset + Full CoT)"
        echo "════════════════"
        echo ""
        echo -n "选择评估模式 [1/2/3]: "
        read -r eval_mode
    fi

    case "${eval_mode}" in
        1) local do_nocot=true  do_subset=true  do_full=false ;;
        2) local do_nocot=false do_subset=false do_full=true  ;;
        3) local do_nocot=true  do_subset=true  do_full=true  ;;
        *)
            log ERROR "无效评估模式: ${eval_mode}"
            exit 1
            ;;
    esac

    log INFO "评估模型: ${model_path}"
    log INFO "评估模式: nocot=${do_nocot} subset=${do_subset} full=${do_full}"

    local trie_source_file=$(_trie_source_file routing)
    local trie_file=$(_ensure_trie "${model_path}" routing "${trie_source_file}")

    local ckpt_tag=$(basename "${model_path}")
    [[ "${ckpt_tag}" == "sid_routing" ]] && ckpt_tag="best"

    local think_max=$(_resolve_routing_think_max)

    # ── No-CoT ──────────────────────────────────────────────────────
    if ${do_nocot}; then
        log INFO "── No-CoT 评估 [${ckpt_tag}] ──"
        _eval_no_cot_on_file "${model_path}" "${ROUTING_TEST}" "${trie_file}" \
            "sid_routing_${ckpt_tag}" "${ROUTING_TEST}"
    fi

    # ── CoT Subset ──────────────────────────────────────────────────
    if ${do_subset} && [[ -f "${ROUTING_TEST_COT}" ]]; then
        log INFO "── CoT Subset 评估 [${ckpt_tag}] ──"
        _eval_cot_on_file "${model_path}" "${ROUTING_TEST_COT}" "${trie_file}" \
            "sid_routing_subset_cot_${ckpt_tag}" "${ROUTING_TEST}" "${think_max}" 0 tail subset
    fi

    # ── Full CoT ────────────────────────────────────────────────────
    if ${do_full}; then
        log INFO "── Full CoT 评估 [${ckpt_tag}] ──"
        _eval_cot_on_file "${model_path}" "${ROUTING_TEST}" "${trie_file}" \
            "sid_routing_full_cot_${ckpt_tag}" "${ROUTING_TEST}" "${think_max}" \
            "${COT_MAX_HISTORY_ITEMS:-0}" "${COT_HISTORY_TRUNCATION_SIDE:-tail}" full
    fi

    log INFO "手动 Checkpoint 评估完成: ${ckpt_tag}"
}


# ═══════════════════════════════════════════════════════════════════════
# 主逻辑
# ═══════════════════════════════════════════════════════════════════════

log INFO "╔══════════════════════════════════════════════════════════╗"
log INFO "║  OneRec Training Pipeline  ║"
log INFO "╠══════════════════════════════════════════════════════════╣"
log INFO "║  Stage:    %-45s ║" "${STAGE}"
log INFO "║  Category: %-45s ║" "${CATEGORY}"
log INFO "║  GPUs:     %-45s ║" "${NUM_GPUS} ($([ ${NUM_GPUS} -eq 1 ] && echo '原生PyTorch' || echo 'DeepSpeed ZeRO-2'))"
log INFO "║  Data:     %-45s ║" "${DATA_DIR}"
log INFO "║  Results:  %-45s ║" "${RESULTS_DIR}"
log INFO "║  Log:      %-45s ║" "${LOG_FILE}"
log INFO "╚══════════════════════════════════════════════════════════╝"

if ${DRY_RUN}; then
    log INFO "[DRY RUN] 仅打印配置。"
    echo ""
    echo "数据检查:"
    for f in "${ALIGN_TRAIN}" "${SID_TRAIN}" "${RA_TRAIN}" "${ROUTING_TRAIN}" "${DIFF_FULL}" "${TRIE_REC}" "${TRIE_RA}" "${TRIE_ROUTING}"; do
        if [[ -f "$f" ]]; then echo "  ✓ $(basename $f)"; else echo "  ✗ $(basename $f)"; fi
    done
    echo ""
    echo "GA 计算 (${NUM_GPUS} GPU):"
    echo "  S1 Align:   8 × $(compute_ga 64 8) × ${NUM_GPUS} = $((8 * $(compute_ga 64 8) * NUM_GPUS))"
    echo "  S2 Rec:     2 × $(compute_ga 16 2) × ${NUM_GPUS} = $((2 * $(compute_ga 16 2) * NUM_GPUS))"
    echo "  S3 RA/Rout: 2 × $(compute_ga 16 2) × ${NUM_GPUS} = $((2 * $(compute_ga 16 2) * NUM_GPUS))"
    echo ""
    echo "门控检查:"
    _has_lora_model "${RESULTS_DIR}/align" && echo "  ✓ align 已完成" || echo "  ✗ align 待训练"
    _has_full_model "${MERGED_MODEL_DIR}" && echo "  ✓ merge 已完成" || echo "  ✗ merge 待执行"
    _has_full_model "${RESULTS_DIR}/sid_rec" && echo "  ✓ sid_rec 已完成" || echo "  ✗ sid_rec 待训练"
    _ra_final=$(_find_model "${RESULTS_DIR}/ra/epoch_2" 2>/dev/null || true)
    [[ -n "${_ra_final}" ]] && _has_full_model "${_ra_final}" && echo "  ✓ ra 已完成" || echo "  ✗ ra 待训练"
     _routing_final=$(_find_model "${RESULTS_DIR}/sid_routing" 2>/dev/null || true)
    [[ -n "${_routing_final}" ]] && _has_full_model "${_routing_final}" && echo "  ✓ sid_routing 已完成" || echo "  ✗ sid_routing 待训练"
    exit 0
fi

case ${STAGE} in
    expand)       do_expand ;;
    align)        do_align ;;
    merge)        do_merge ;;
    rec)          do_rec ;;
    ra)           do_ra ;;
    sid_routing)  do_sid_routing ;;
    eval_rec)     do_eval_rec ;;
    eval_ra)      do_eval_ra ;;
    eval_sid_routing) do_eval_sid_routing ;;
    eval_ra_full_cot)          do_eval_ra_full_cot ;;
    eval_sid_routing_full_cot) do_eval_sid_routing_full_cot ;;
    eval_sid_routing_ckpt) do_eval_sid_routing_ckpt ;;
    onerec)
        do_expand
        do_align
        do_merge
        do_rec
        do_eval_rec
        do_ra
        do_eval_ra
        ;;
    routing)
        do_expand
        do_align
        do_merge
        do_rec
        do_eval_rec
        do_sid_routing
        do_eval_sid_routing
        ;;
    all)
        do_expand
        do_align
        do_merge
        do_rec
        do_eval_rec
        do_ra
        do_eval_ra
        do_sid_routing
        do_eval_sid_routing
        ;;
    *)
        log ERROR "未知 stage: ${STAGE}"
        exit 1
        ;;
esac

log INFO "Pipeline [${STAGE}] 完成!"

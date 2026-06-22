#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT_DIR="${CKPT_DIR:-${ROOT_DIR}/Wan2.2-TI2V-5B}"
COGVIDEO_ROOT="${COGVIDEO_ROOT:-${ROOT_DIR}}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output/physical_ti2v_5b_49f_bs8}"
DS_CONFIG="${DS_CONFIG:-${ROOT_DIR}/finetune/deepspeed_zero3.json}"
CACHE_DIR="${CACHE_DIR:-${DATA_ROOT}/cache_wan}"
FAISS_INDEX_DIR="${FAISS_INDEX_DIR:-${DATA_ROOT}/rag/faiss_index}"
VIDEOCLIP_XL_MODEL_PATH="${VIDEOCLIP_XL_MODEL_PATH:-${ROOT_DIR}/PhysicalDB/VideoCLIP-XL/VideoCLIP-XL.bin}"

PROMPT_FILE="${DATA_ROOT}/prompts_new.txt"
if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Prompt file not found: ${PROMPT_FILE}" >&2
  exit 1
fi

DATASET_SIZE=$(wc -l < "${PROMPT_FILE}")
if [[ "${DATASET_SIZE}" -eq 0 ]]; then
  echo "Prompt file is empty: ${PROMPT_FILE}" >&2
  exit 1
fi
BATCH_SIZE=4
GRAD_ACC=2
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep . | wc -l)
if [[ "${NUM_GPUS}" -eq 0 ]]; then
  NUM_GPUS=1
fi

STEPS_PER_EPOCH=$(( (DATASET_SIZE + BATCH_SIZE*GRAD_ACC*NUM_GPUS - 1) / (BATCH_SIZE*GRAD_ACC*NUM_GPUS) ))
SAVE_STEPS=400

mkdir -p "${OUTPUT_DIR}"

# Create log directory
LOG_DIR="${OUTPUT_DIR}/log"
mkdir -p "${LOG_DIR}"

# Generate log filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_physical_${TIMESTAMP}.log"

if [[ ! -f "${DS_CONFIG}" ]]; then
  echo "DeepSpeed config not found: ${DS_CONFIG}" >&2
  exit 1
fi

# Auto-resume: prefer RESUME_FROM env, otherwise use latest checkpoint in OUTPUT_DIR.
RESUME_FROM="${RESUME_FROM:-}"
if [[ -z "${RESUME_FROM}" ]]; then
  if ls "${OUTPUT_DIR}"/checkpoint-* >/dev/null 2>&1; then
    RESUME_FROM="$(ls -d "${OUTPUT_DIR}"/checkpoint-* | sort -t '-' -k2 -n | tail -n 1)"
  fi
fi
RESUME_ARGS=()
if [[ -n "${RESUME_FROM}" ]]; then
  echo "Resuming from: ${RESUME_FROM}" | tee -a "${LOG_FILE}"
  RESUME_ARGS+=(--resume_from "${RESUME_FROM}")
fi

# Log training start information
echo "Training started at $(date)" | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "Using GPUs: ${CUDA_VISIBLE_DEVICES}" | tee -a "${LOG_FILE}"
echo "Output directory: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

# Launch training with logging
# Note: --nproc_per_node should match the number of GPUs in CUDA_VISIBLE_DEVICES
cd "${ROOT_DIR}"
torchrun --nproc_per_node "${NUM_GPUS}" finetune/train_physical_ti2v_5b.py \
  --ckpt_dir "${CKPT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --data_root "${DATA_ROOT}" \
  --caption_column "prompts_new.txt" \
  --video_column "videos_new.txt" \
  --train_resolution "49x480x704" \
  --cache_dir "${CACHE_DIR}" \
  --batch_size 4 \
  --gradient_accumulation_steps 2 \
  --train_epochs 20 \
  --learning_rate 1e-6 \
  --weight_decay 0.01 \
  --mixed_precision bf16 \
  --t5_cpu \
  --num_workers 1 \
  --cogvideo_root "${COGVIDEO_ROOT}" \
  --log_steps 10 \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 2 \
  --gradient_checkpointing \
  --train_mode full \
  --deepspeed \
  --deepspeed_config "${DS_CONFIG}" \
  --faiss_index_dir "${FAISS_INDEX_DIR}" \
  --videoclip_xl_model_path "${VIDEOCLIP_XL_MODEL_PATH}" \
  "${RESUME_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"

# Log training completion time
echo "========================================" | tee -a "${LOG_FILE}"
echo "Training completed at $(date)" | tee -a "${LOG_FILE}"

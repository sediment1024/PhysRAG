#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT_DIR="${CKPT_DIR:-${ROOT_DIR}/Wan2.2-TI2V-5B}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}"
CACHE_DIR="${CACHE_DIR:-${DATA_ROOT}/cache_wan}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/output/cache_logs}"
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep . | wc -l)

mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/build_cache_wan_49f_704_${TIMESTAMP}.log"

echo "Cache build started at $(date)" | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "Using GPUs: ${CUDA_VISIBLE_DEVICES}" | tee -a "${LOG_FILE}"
echo "Cache dir: ${CACHE_DIR}" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

cd "${ROOT_DIR}"
torchrun --nproc_per_node "${NUM_GPUS}" finetune/build_cache_wan.py \
  --ckpt_dir "${CKPT_DIR}" \
  --data_root "${DATA_ROOT}" \
  --caption_column "prompts_new.txt" \
  --video_column "videos_new.txt" \
  --train_resolution "49x480x704" \
  --cache_dir "${CACHE_DIR}" \
  --log_every 50 2>&1 | tee -a "${LOG_FILE}"

echo "========================================" | tee -a "${LOG_FILE}"
echo "Cache build completed at $(date)" | tee -a "${LOG_FILE}"

#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi
export TOKENIZERS_PARALLELISM=false

CKPT_DIR="${CKPT_DIR:?Set CKPT_DIR to the Wan2.2-TI2V-5B directory}"
PHYSICAL_CKPT="${PHYSICAL_CKPT:-${ROOT_DIR}/checkpoints/merged_model.pt}"
PHYGENBENCH_ROOT="${PHYGENBENCH_ROOT:?Set PHYGENBENCH_ROOT to a PhyGenBench checkout}"
COGVIDEO_ROOT="${COGVIDEO_ROOT:-${ROOT_DIR}}"
MODEL_NAME="${MODEL_NAME:-phyrag-wan22-ti2v-5b}"
FAISS_INDEX_DIR="${FAISS_INDEX_DIR:-${ROOT_DIR}/data/rag/faiss_index}"
VIDEOCLIP_XL_MODEL_PATH="${VIDEOCLIP_XL_MODEL_PATH:-${ROOT_DIR}/PhysicalDB/VideoCLIP-XL/VideoCLIP-XL.bin}"

python "${ROOT_DIR}/finetune/infer_phygenbench_wan.py" \
  --ckpt_dir "${CKPT_DIR}" \
  --physical_ckpt "${PHYSICAL_CKPT}" \
  --phygenbench_root "${PHYGENBENCH_ROOT}" \
  --cogvideo_root "${COGVIDEO_ROOT}" \
  --modelname "${MODEL_NAME}" \
  --size "704*480" \
  --frame_num 49 \
  --faiss_index_dir "${FAISS_INDEX_DIR}" \
  --videoclip_xl_model_path "${VIDEOCLIP_XL_MODEL_PATH}" \
  --t5_cpu \
  --skip_existing

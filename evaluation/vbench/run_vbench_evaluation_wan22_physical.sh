#!/usr/bin/env bash
# Inference script (not training). Runs on Linux server. Keep LF line endings (no CRLF).
# VBench inference: generate videos with Wan2.2 Physical TI2V for 1st-gen VBench evaluation.
#
# Usage: CUDA_VISIBLE_DEVICES=<gpu> bash run_vbench_evaluation_wan22_physical.sh <repo_root> <modelname> [cuda_device] [checkpoint_step] --categories ...
# Example: bash run_vbench_evaluation_wan22_physical.sh /path/to/Wan2.2 physical_ti2v_5b_49f_bs8 0 4000 --categories color
#
# Physical fine-tuned weights: <repo_root>/output/<modelname>/checkpoint-<step>/merged_model.pt
# Output videos: <repo_root>/benchmark/VBench/results/<modelname>_ck<step>

set -euo pipefail

VBENCH_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repo root: from script path (vbench -> evaluation -> DiT-Mem -> VBench -> Benchmark -> repo, 5 levels up)
DEFAULT_REPO_ROOT="$(cd "$VBENCH_SCRIPT_DIR/../../../../../" && pwd)"

REPO_ROOT="${1:-$DEFAULT_REPO_ROOT}"
MODELNAME="${2:-}"
CUDA_DEVICE="${3:-0}"
# Checkpoint step: 4th positional arg (from 4gpu_new) wins, then env CHECKPOINT_STEP, else 4000.
if [[ "${4:-}" =~ ^[0-9]+$ ]]; then
  CHECKPOINT_STEP="$4"
elif [[ -n "${CHECKPOINT_STEP:-}" ]] && [[ "${CHECKPOINT_STEP}" =~ ^[0-9]+$ ]]; then
  :
else
  CHECKPOINT_STEP="${CHECKPOINT_STEP:-4000}"
fi
# Parse categories: either $5=--categories $6+=names, or $4=--categories $5+=names
if [[ "${4:-}" == "--categories" ]] && [[ -n "${5:-}" ]]; then
  CATEGORIES="${*:5}"
elif [[ "${5:-}" == "--categories" ]] && [[ -n "${6:-}" ]]; then
  CATEGORIES="${*:6}"
fi

if [[ -z "$MODELNAME" ]]; then
  echo "Usage: $0 <repo_root> <modelname> [cuda_device] [checkpoint_step] [--categories cat ...]" >&2
  echo "  repo_root       : Wan2.2 repo root" >&2
  echo "  modelname      : e.g. physical_ti2v_5b_49f_bs8" >&2
  echo "  cuda_device    : GPU id (default: 0)" >&2
  echo "  checkpoint_step: e.g. 860, 4000, 6000 (default: 4000)" >&2
  echo "Example: bash $0 /path/to/PhyRAG physical_ti2v_5b_49f 0 860 --categories color" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
# PhysicalDB lives under CogVideo-main; required for PhysRAG (RAG on by default).
COGVIDEO_ROOT="${COGVIDEO_ROOT:-${REPO_ROOT}}"
if [[ -d "$COGVIDEO_ROOT" ]]; then
  export PYTHONPATH="${COGVIDEO_ROOT}:${PYTHONPATH}"
fi
export PATH="${CONDA_PREFIX:-}/bin:$PATH"

# CHECKPOINT_STEP set above from $4 or env
# Dir containing merged_model.pt (physical fine-tuned weights from training run)
CHECKPOINT_DIR="${REPO_ROOT}/output/${MODELNAME}/checkpoint-${CHECKPOINT_STEP}"
# Base Wan2.2-TI2V-5B model dir for inference; override with env BASE_MODEL_DIR if needed
BASE_MODEL_DIR="${BASE_MODEL_DIR:-${REPO_ROOT}/Wan2.2-TI2V-5B}"
# Output resolution (e.g. 1280*704); override with env SIZE. Used for path label (1280x704).
SIZE="${SIZE:-1280*704}"
RESOLUTION_LABEL="${SIZE//\*/x}"
# Where to write generated videos; include resolution to distinguish different resolutions
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/benchmark/VBench/results/${MODELNAME}_ckpt${CHECKPOINT_STEP}_${RESOLUTION_LABEL}}"

VBENCH_PROMPT_DIR="${VBENCH_SCRIPT_DIR}/prompts"
NUM_INFERENCE_STEPS=40
FPS=16
FRAME_NUM=49
# PhysRAG: RAG is on by default. Set DISABLE_RAG=1 to run without CogVideo/PhysicalDB (physical weights only).
DISABLE_RAG="${DISABLE_RAG:-0}"
# Optional: override RAG paths (defaults in Python match run_wan22_ti2v_5b_batch.py)
FAISS_INDEX_DIR="${FAISS_INDEX_DIR:-}"
VIDEOCLIP_XL_MODEL_PATH="${VIDEOCLIP_XL_MODEL_PATH:-}"
# CATEGORIES set above when --categories passed; else from env
CATEGORIES="${CATEGORIES:-}"

echo "════════════════════════════════════════════════════════════════════════════════"
echo "VBench inference (Wan2.2 Physical TI2V)"
echo "════════════════════════════════════════════════════════════════════════════════"
echo "REPO_ROOT:       $REPO_ROOT"
echo "MODELNAME:       $MODELNAME"
echo "CHECKPOINT_DIR:  $CHECKPOINT_DIR"
echo "CHECKPOINT_STEP: $CHECKPOINT_STEP"
echo "OUTPUT_DIR:      $OUTPUT_DIR"
echo "BASE_MODEL_DIR:  $BASE_MODEL_DIR"
echo "CUDA_DEVICE:     $CUDA_DEVICE"
echo "SIZE:            $SIZE"
echo "RAG:             $([ "$DISABLE_RAG" = "1" ] && echo "disabled" || echo "enabled (PhysRAG)")"
echo "════════════════════════════════════════════════════════════════════════════════"

if [[ "$DISABLE_RAG" != "1" ]] && [[ ! -d "$COGVIDEO_ROOT" ]]; then
  echo "Error: RAG is enabled but COGVIDEO_ROOT not found: $COGVIDEO_ROOT (set COGVIDEO_ROOT or DISABLE_RAG=1)" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT_DIR/merged_model.pt" ]]; then
  echo "Error: merged_model.pt not found at $CHECKPOINT_DIR/merged_model.pt" >&2
  exit 1
fi

if [[ ! -d "$BASE_MODEL_DIR" ]]; then
  echo "Error: Base model dir not found: $BASE_MODEL_DIR (set BASE_MODEL_DIR if needed)" >&2
  exit 1
fi

if [[ ! -d "$VBENCH_PROMPT_DIR" ]]; then
  echo "Error: VBench prompt dir not found: $VBENCH_PROMPT_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Inference with RAG by default (same as run_wan22_ti2v_5b_batch.py). Use --disable_rag in Python to disable.
CMD="CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python ${VBENCH_SCRIPT_DIR}/generate_vbench_videos_wan22_physical.py \
  --ckpt_dir \"$BASE_MODEL_DIR\" \
  --checkpoint_dir \"$CHECKPOINT_DIR\" \
  --vbench_prompt_dir \"$VBENCH_PROMPT_DIR\" \
  --output_dir \"$OUTPUT_DIR\" \
  --num_inference_steps $NUM_INFERENCE_STEPS \
  --fps $FPS \
  --frame_num $FRAME_NUM \
  --size \"$SIZE\" \
  --cuda_device $CUDA_DEVICE"

if [[ "$DISABLE_RAG" == "1" ]]; then
  CMD="$CMD --disable_rag"
fi
if [[ -n "$FAISS_INDEX_DIR" ]]; then
  CMD="$CMD --faiss_index_dir \"$FAISS_INDEX_DIR\""
fi
if [[ -n "$VIDEOCLIP_XL_MODEL_PATH" ]]; then
  CMD="$CMD --videoclip_xl_model_path \"$VIDEOCLIP_XL_MODEL_PATH\""
fi
if [[ -n "$CATEGORIES" ]]; then
  CMD="$CMD --categories $CATEGORIES"
fi

eval $CMD
EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
  echo "VBench videos written to: $OUTPUT_DIR"
else
  echo "VBench generation failed (exit $EXIT_CODE)." >&2
fi
exit $EXIT_CODE

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:?Usage: $0 <phygenbench_root> <modelname> [skip_existing:1|0]}"
MODELNAME="${2:-}"
SKIP_EXISTING="${3:-1}"

if [[ -z "$MODELNAME" ]]; then
  echo "Usage: $0 <repo_root> <modelname> [skip_existing:1|0]" >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
LLAVA_MODEL_PATH="${LLAVA_MODEL_PATH:?Set LLAVA_MODEL_PATH}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:?Set CLIP_MODEL_PATH}"
INTERNVIDEO_MODEL_PATH="${INTERNVIDEO_MODEL_PATH:?Set INTERNVIDEO_MODEL_PATH}"

run_if_missing() {
  local output_path="$1"
  shift
  if [[ "$SKIP_EXISTING" == "1" && -f "$output_path" ]]; then
    echo "Skip (exists): $output_path"
    return 0
  fi
  "$@"
}

SINGLE_OUT="$REPO_ROOT/PhyGenEval/single/prompt_replace_augment_single_question_${MODELNAME}_res.json"
MULTI_CLIP_OUT="$REPO_ROOT/PhyGenEval/multi/prompt_replace_augment_multi_question1_${MODELNAME}_res1_imageclip.json"
MULTI_LLAVA_OUT="$REPO_ROOT/PhyGenEval/multi/prompt_replace_augment_multi_question_${MODELNAME}_res_llava.json"
VIDEO_INTERN_OUT="$REPO_ROOT/PhyGenEval/video/prompt_replace_augment_video_question_${MODELNAME}_res_intern.json"
OVERALL_OUT="$REPO_ROOT/result/${MODELNAME}.json"

run_if_missing "$SINGLE_OUT" \
  python "$REPO_ROOT/PhyGenEval/single/vqascore.py" \
    --repo-root "$REPO_ROOT" \
    --modelname "$MODELNAME" \
    --skip-missing

run_if_missing "$MULTI_CLIP_OUT" \
  python "$REPO_ROOT/PhyGenEval/multi/multiimage_clip.py" \
    --repo-root "$REPO_ROOT" \
    --modelname "$MODELNAME" \
    --skip-missing

run_if_missing "$MULTI_LLAVA_OUT" \
  python "$REPO_ROOT/PhyGenEval/multi/LLaVA-NeXT-interleave_inference/llava/eval/model_vqa_multi.py" \
    --repo-root "$REPO_ROOT" \
    --modelname "$MODELNAME" \
    --model-path "${LLAVA_MODEL_PATH}" \
    --clip-path "${CLIP_MODEL_PATH}" \
    --skip-missing

run_if_missing "$VIDEO_INTERN_OUT" \
  python "$REPO_ROOT/PhyGenEval/video/MTScore/InternVideo_physical.py" \
    --repo-root "$REPO_ROOT" \
    --modelname "$MODELNAME" \
    --model_pth "${INTERNVIDEO_MODEL_PATH}" \
    --skip-missing

run_if_missing "$OVERALL_OUT" \
  python "$REPO_ROOT/PhyGenEval/overall.py" \
    --repo-root "$REPO_ROOT" \
    --modelname "$MODELNAME"

python "$REPO_ROOT/scripts/compute_phygenbench_metrics.py" \
  --input "$OVERALL_OUT" \
  --modelname "$MODELNAME" \
  --score-key "${MODELNAME}_average" \
  --normalize

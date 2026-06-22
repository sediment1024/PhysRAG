#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1) Select which zip IDs to download (edit this list).
ZIP_IDS=(8 9 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99)

# 2) Paths / config.
OUT_ROOT="${OUT_ROOT:?Set OUT_ROOT to the WISA working directory}"
MODEL_REPO="${MODEL_REPO:-qihoo360/WISA-80K}"
MODEL_REV="${MODEL_REV:-dddbd5683581c2ebf0b463e2b1c3342b2094bfb3}"
FILTER_SCRIPT="${FILTER_SCRIPT:-${SCRIPT_DIR}/filter_all_categories_qwen3vl.py}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TOP_K_CAPTION="0.1"   # keep 10% after caption-only
TOP_K_FRAME="0.9"     # keep 90% after caption+first-frame
FIRST_FRAME_SIZE=256

# 2.5) Label mapping for per-phenomenon counts.
LABELS=(
  "collision"
  "combustion"
  "deformation"
  "elastic motion"
  "explosion"
  "gas motion"
  "interference and diffraction"
  "liquefaction"
  "liquid motion"
  "melting"
  "reflection"
  "refraction"
  "rigid body motion"
  "scattering"
  "solidification"
  "unnatural light source"
  "vaporization"
)
LABEL_DIRS=(
  "collision"
  "combustion"
  "deformation"
  "elastic_motion"
  "explosion"
  "gas_motion"
  "interference_and_diffraction"
  "liquefaction"
  "liquid_motion"
  "melting"
  "reflection"
  "refraction"
  "rigid_body_motion"
  "scattering"
  "solidification"
  "unnatural_light_source"
  "vaporization"
)

# 3) Cleanup toggles.
CLEANUP_ZIPS=true
CLEANUP_RAW=true
PRUNE_AFTER_FRAME=true
KEEP_ONLY_FINAL=true
UNZIP_FLAGS="-n"  # -n = skip existing files

# 4) HF cache (official HF, no mirror).
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HOME}/hub"

HF_CMD=""
if command -v hf >/dev/null 2>&1; then
  HF_CMD="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CMD="huggingface-cli"
else
  echo "Error: hf or huggingface-cli not found in PATH." >&2
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "Error: unzip is required but not found in PATH." >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"

echo "=== Config ==="
echo "ZIP_IDS: ${ZIP_IDS[*]}"
echo "OUT_ROOT: ${OUT_ROOT}"
echo "DEVICE: ${DEVICE}  BATCH_SIZE: ${BATCH_SIZE}"
echo "TOP_K_CAPTION: ${TOP_K_CAPTION}  TOP_K_FRAME: ${TOP_K_FRAME}"
echo "FIRST_FRAME_SIZE: ${FIRST_FRAME_SIZE}"
echo "==============="

# Download + unzip each zip.
for zip_id in "${ZIP_IDS[@]}"; do
  target_dir="${OUT_ROOT}/${zip_id}"
  if [[ -d "${target_dir}" && -f "${target_dir}/.extracted_by_script" ]]; then
    mp4_count=$(find "${target_dir}" -type f -name "*.mp4" | wc -l | tr -d ' ')
    if [[ "${mp4_count}" != "0" ]]; then
      echo "[zip ${zip_id}] already extracted (${mp4_count} mp4), skip download/unzip."
      continue
    fi
  fi

  "${HF_CMD}" download "${MODEL_REPO}" \
    --repo-type dataset \
    --revision "${MODEL_REV}" \
    --include "data/videos/${zip_id}.zip" \
    --local-dir "${OUT_ROOT}"

  zip_path="${OUT_ROOT}/data/videos/${zip_id}.zip"
  if [[ ! -f "${zip_path}" ]]; then
    zip_path="${OUT_ROOT}/${zip_id}.zip"
  fi
  if [[ ! -f "${zip_path}" ]]; then
    echo "Error: ${zip_id}.zip not found after download." >&2
    exit 1
  fi

  mkdir -p "${target_dir}"
  unzip -q ${UNZIP_FLAGS} "${zip_path}" -d "${target_dir}"
  touch "${target_dir}/.extracted_by_script"
  zip_count=$(find "${target_dir}" -type f -name "*.mp4" | wc -l | tr -d ' ')
  echo "[zip ${zip_id}] extracted mp4: ${zip_count}"

  if [[ "${CLEANUP_ZIPS}" == "true" ]]; then
    rm -f "${zip_path}"
  fi

done

subdirs_csv=$(IFS=,; echo "${ZIP_IDS[*]}")

zip_ids_env=$(IFS=' '; echo "${ZIP_IDS[*]}")
labels_env=$(IFS='||'; echo "${LABELS[*]}")

echo "=== Raw label counts (selected zips) ==="
ZIP_IDS="${zip_ids_env}" LABELS="${labels_env}" OUT_ROOT="${OUT_ROOT}" python - <<'PY'
import json
import os

zip_ids = os.environ.get("ZIP_IDS", "").split()
out_root = os.environ.get("OUT_ROOT", "")
labels = os.environ.get("LABELS", "").split("||")
json_path = os.path.join(out_root, "wisa-80k.json")

names = set()
for zid in zip_ids:
    base = os.path.join(out_root, zid)
    if not os.path.isdir(base):
        continue
    for root, _, files in os.walk(base):
        for f in files:
            if f.endswith(".mp4"):
                names.add(f)

print(f"raw_total={len(names)}")
if not os.path.isfile(json_path):
    print("wisa-80k.json not found, skip label counts.")
    raise SystemExit(0)

with open(json_path, "r") as f:
    data = json.load(f)

counts = {l: 0 for l in labels if l}
other = 0
for item in data:
    if item.get("video_name") in names:
        label = item.get("label")
        if label in counts:
            counts[label] += 1
        else:
            other += 1

for label in labels:
    if label:
        print(f"{label}: {counts.get(label, 0)}")
if other:
    print(f"other_labels: {other}")
PY

echo "=== Stage A: caption-only (top ${TOP_K_CAPTION}) ==="
# Stage A: caption-only top 10% (writes *_qwen3vl_caption_top10).
python "${FILTER_SCRIPT}" \
  --use_caption \
  --recursive \
  --video_root "${OUT_ROOT}" \
  --video_subdirs "${subdirs_csv}" \
  --output_root "${OUT_ROOT}" \
  --top_k_percent "${TOP_K_CAPTION}" \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}"

echo "=== Stage A results (caption_top10) ==="
for i in "${!LABELS[@]}"; do
  label="${LABELS[$i]}"
  dir="${LABEL_DIRS[$i]}"
  label_dir="${OUT_ROOT}/${dir}_qwen3vl_caption_top10"
  if [[ -d "${label_dir}" ]]; then
    count=$(find "${label_dir}" -type f -name "*.mp4" | wc -l | tr -d ' ')
    echo "[${label}] caption_top10 mp4: ${count}"
  else
    echo "[${label}] caption_top10 mp4: 0 (missing)"
  fi
done

if [[ "${CLEANUP_RAW}" == "true" ]]; then
  for zip_id in "${ZIP_IDS[@]}"; do
    target_dir="${OUT_ROOT}/${zip_id}"
    if [[ -f "${target_dir}/.extracted_by_script" ]]; then
      rm -rf "${target_dir}"
    fi
  done
fi

# Stage B: caption + first-frame top 90% (writes *_qwen3vl_caption_frame_top10).
echo "=== Stage B: caption+first-frame (top ${TOP_K_FRAME}) ==="
for i in "${!LABELS[@]}"; do
  label="${LABELS[$i]}"
  dir="${LABEL_DIRS[$i]}"
  label_dir="${OUT_ROOT}/${dir}_qwen3vl_caption_top10"
  if [[ ! -d "${label_dir}" ]]; then
    echo "Skip missing: ${label_dir}" >&2
    continue
  fi

  python "${FILTER_SCRIPT}" \
    --use_caption \
    --caption_with_first_frame \
    --first_frame_size "${FIRST_FRAME_SIZE}" \
    --label "${label}" \
    --top_k_percent "${TOP_K_FRAME}" \
    --batch_size "${BATCH_SIZE}" \
    --video_root "${label_dir}" \
    --output_root "${label_dir}" \
    --device "${DEVICE}"

  if [[ "${PRUNE_AFTER_FRAME}" == "true" ]]; then
    frame_dir="${label_dir}/${dir}_qwen3vl_caption_frame_top10"
    if [[ -d "${frame_dir}" ]]; then
      find "${label_dir}" -type f -name "*.mp4" ! -path "${frame_dir}/*" -delete
    fi
  fi

  if [[ "${KEEP_ONLY_FINAL}" == "true" ]]; then
    frame_dir="${label_dir}/${dir}_qwen3vl_caption_frame_top10"
    if [[ -d "${frame_dir}" ]]; then
      keep_name="$(basename "${frame_dir}")"
      find "${label_dir}" -mindepth 1 -maxdepth 1 ! -name "${keep_name}" -exec rm -rf {} +
    fi
  fi

done

echo "=== Stage B results (caption_frame_top10) ==="
for i in "${!LABELS[@]}"; do
  label="${LABELS[$i]}"
  dir="${LABEL_DIRS[$i]}"
  frame_dir="${OUT_ROOT}/${dir}_qwen3vl_caption_top10/${dir}_qwen3vl_caption_frame_top10"
  if [[ -d "${frame_dir}" ]]; then
    count=$(find "${frame_dir}" -type f -name "*.mp4" | wc -l | tr -d ' ')
    echo "[${label}] caption_frame_top10 mp4: ${count}"
  else
    echo "[${label}] caption_frame_top10 mp4: 0 (missing)"
  fi
done

echo "All done. Final videos are under: ${OUT_ROOT}/*_qwen3vl_caption_top10/*_qwen3vl_caption_frame_top10"

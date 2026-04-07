#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
GEO_ROOT=$(cd -- "${REPO_ROOT}/.." && pwd)

NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-10}
NUM_PARALLEL_CLIENTS=${NUM_PARALLEL_CLIENTS:-4}
BASE_PORT=${BASE_PORT:-25000}
CAMERA_VARIATION_CONFIG=${CAMERA_VARIATION_CONFIG:-${GEO_ROOT}/LIBERO-Camera/configs/camera_variation/libero_object_left_variation.json}
CAMERA_VARIATION_COUNT=${CAMERA_VARIATION_COUNT:-11}

BASELINE_CKPT_ROOT=${BASELINE_CKPT_ROOT:-${REPO_ROOT}/checkpoints/pi0_libero_pytorch_full_finetune/pi0_libero_object_camtrain_o00_02_03_05_06_09_10_baseline_60k}
SIMPLE_CKPT_ROOT=${SIMPLE_CKPT_ROOT:-${REPO_ROOT}/checkpoints/pi0_libero_cam_pytorch_full_finetune/pi0_libero_object_camtrain_o00_02_03_05_06_09_10_simple_60k}
STANDARD_CKPT_ROOT=${STANDARD_CKPT_ROOT:-${REPO_ROOT}/checkpoints/pi0_libero_cam_pytorch_full_finetune_standard/pi0_libero_object_camtrain_o00_02_03_05_06_09_10_standard_60k}

BASELINE_LOG_ROOT=${BASELINE_LOG_ROOT:-${REPO_ROOT}/log/libero_object_multicam_eval/camtrain60k_ckpt_sweep/baseline}
SIMPLE_LOG_ROOT=${SIMPLE_LOG_ROOT:-${REPO_ROOT}/log/libero_object_multicam_eval/camtrain60k_ckpt_sweep/simple}
STANDARD_LOG_ROOT=${STANDARD_LOG_ROOT:-${REPO_ROOT}/log/libero_object_multicam_eval/camtrain60k_ckpt_sweep/standard}
MANIFEST_DIR=${MANIFEST_DIR:-${REPO_ROOT}/log/libero_object_multicam_eval/camtrain60k_ckpt_sweep}

mkdir -p "${BASELINE_LOG_ROOT}" "${SIMPLE_LOG_ROOT}" "${STANDARD_LOG_ROOT}" "${MANIFEST_DIR}"

collect_steps() {
  local ckpt_root="$1"
  local -a steps=()
  local path
  for path in "${ckpt_root}"/*; do
    [[ -d "${path}" ]] || continue
    local step
    step=$(basename -- "${path}")
    [[ "${step}" =~ ^[0-9]+$ ]] || continue
    if (( step % 5000 == 0 )); then
      steps+=("${step}")
    fi
  done

  if ((${#steps[@]} == 0)); then
    return 1
  fi

  printf '%s\n' "${steps[@]}" | sort -n
}

mapfile -t baseline_steps < <(collect_steps "${BASELINE_CKPT_ROOT}")
mapfile -t simple_steps < <(collect_steps "${SIMPLE_CKPT_ROOT}")
mapfile -t standard_steps < <(collect_steps "${STANDARD_CKPT_ROOT}")

manifest="${MANIFEST_DIR}/submit_manifest_$(date +%Y%m%d_%H%M%S).tsv"
printf 'model\tstep\tjob_id\tlog_root\n' > "${manifest}"

port_offset=0

submit_one() {
  local model_label="$1"
  local config_name="$2"
  local ckpt_root="$3"
  local ckpt_asset_id="$4"
  local serve_asset_id="$5"
  local log_root="$6"
  local step="$7"
  local port_base="$8"

  local job_id
  job_id=$(
    sbatch --parsable \
      --job-name="camtrain_${model_label}_${step}" \
      --output="${log_root}/slurm-%j.out" \
      --error="${log_root}/slurm-%j.err" \
      --export=ALL,MODEL_NAME="${model_label}",CONFIG_NAME="${config_name}",CHECKPOINT_ROOT="${ckpt_root}",CHECKPOINT_STEP="${step}",CHECKPOINT_ASSET_ID="${ckpt_asset_id}",SERVE_ASSET_ID="${serve_asset_id}",NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}",NUM_PARALLEL_CLIENTS="${NUM_PARALLEL_CLIENTS}",PORT_BASE="${port_base}",CAMERA_VARIATION_CONFIG="${CAMERA_VARIATION_CONFIG}",CAMERA_VARIATION_COUNT="${CAMERA_VARIATION_COUNT}",LOG_ROOT="${log_root}" \
      "${SCRIPT_DIR}/infer_libero_object_multicam_parallel.sbatch"
  )

  printf '%s\t%s\t%s\t%s\n' "${model_label}" "${step}" "${job_id}" "${log_root}" >> "${manifest}"
  printf '%s step %s -> job %s\n' "${model_label}" "${step}" "${job_id}"
}

for step in "${baseline_steps[@]}"; do
  submit_one \
    "camtrain60k_baseline" \
    "pi0_libero_pytorch_full_finetune" \
    "${BASELINE_CKPT_ROOT}" \
    "glbreeze/libero_object_cam_train_o00_02_03_05_06_09_10" \
    "glbreeze/libero" \
    "${BASELINE_LOG_ROOT}" \
    "${step}" \
    "$((BASE_PORT + port_offset))"
  port_offset=$((port_offset + 1))
done

for step in "${simple_steps[@]}"; do
  submit_one \
    "camtrain60k_simple" \
    "pi0_libero_cam_pytorch_full_finetune" \
    "${SIMPLE_CKPT_ROOT}" \
    "glbreeze/libero_object_cam_train_o00_02_03_05_06_09_10" \
    "glbreeze/libero_cam" \
    "${SIMPLE_LOG_ROOT}" \
    "${step}" \
    "$((BASE_PORT + port_offset))"
  port_offset=$((port_offset + 1))
done

for step in "${standard_steps[@]}"; do
  submit_one \
    "camtrain60k_standard" \
    "pi0_libero_cam_pytorch_full_finetune_standard" \
    "${STANDARD_CKPT_ROOT}" \
    "glbreeze/libero_object_cam_train_o00_02_03_05_06_09_10" \
    "glbreeze/libero_cam" \
    "${STANDARD_LOG_ROOT}" \
    "${step}" \
    "$((BASE_PORT + port_offset))"
  port_offset=$((port_offset + 1))
done

echo "Manifest: ${manifest}"

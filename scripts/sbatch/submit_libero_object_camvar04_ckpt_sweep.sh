#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
GEO_ROOT=$(cd -- "${REPO_ROOT}/.." && pwd)

CAMERA04_CONFIG=${CAMERA04_CONFIG:-${GEO_ROOT}/LIBERO-Camera/configs/camera_variation/libero_object_left_camvar_04_only.json}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-50}
NUM_PARALLEL_CLIENTS=${NUM_PARALLEL_CLIENTS:-4}
BASE_PORT=${BASE_PORT:-21000}

PI0_CKPT_ROOT=${PI0_CKPT_ROOT:-${REPO_ROOT}/checkpoints/pi0_libero_pytorch_full_finetune/pi0_libero_object_cam_pi0_baseline}
POSE_CKPT_ROOT=${POSE_CKPT_ROOT:-${REPO_ROOT}/checkpoints/pi0_libero_cam_pytorch_full_finetune/pi0_libero_object_cam_posefusion}

PI0_LOG_ROOT=${PI0_LOG_ROOT:-${REPO_ROOT}/log/libero_object_multicam_eval/camvar04_ckpt_sweep/cam_pi0_baseline}
POSE_LOG_ROOT=${POSE_LOG_ROOT:-${REPO_ROOT}/log/libero_object_multicam_eval/camvar04_ckpt_sweep/posefusion_multi}
MANIFEST_DIR=${MANIFEST_DIR:-${REPO_ROOT}/log/libero_object_multicam_eval/camvar04_ckpt_sweep}

mkdir -p "${PI0_LOG_ROOT}" "${POSE_LOG_ROOT}" "${MANIFEST_DIR}"

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

mapfile -t pi0_steps < <(collect_steps "${PI0_CKPT_ROOT}")
mapfile -t pose_steps < <(collect_steps "${POSE_CKPT_ROOT}")

manifest="${MANIFEST_DIR}/submit_manifest_$(date +%Y%m%d_%H%M%S).tsv"
printf 'model\tstep\tjob_id\tlog_root\n' > "${manifest}"

port_offset=0

submit_one() {
  local model_label="$1"
  local config_name="$2"
  local ckpt_root="$3"
  local ckpt_step="$4"
  local ckpt_asset_id="$5"
  local serve_asset_id="$6"
  local log_root="$7"
  local port_base="$8"

  mkdir -p "${log_root}"

  local job_id
  job_id=$(
    sbatch --parsable \
      --job-name="cam4_${model_label}_${ckpt_step}" \
      --output="${log_root}/slurm-%j.out" \
      --error="${log_root}/slurm-%j.err" \
      --export=ALL,MODEL_NAME="${model_label}",CONFIG_NAME="${config_name}",CHECKPOINT_ROOT="${ckpt_root}",CHECKPOINT_STEP="${ckpt_step}",CHECKPOINT_ASSET_ID="${ckpt_asset_id}",SERVE_ASSET_ID="${serve_asset_id}",CAMERA_VARIATION_CONFIG="${CAMERA04_CONFIG}",CAMERA_VARIATION_COUNT=1,CAMERA_VARIATION_INCLUDE_ORIGINAL=false,NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}",NUM_PARALLEL_CLIENTS="${NUM_PARALLEL_CLIENTS}",PORT_BASE="${port_base}",LOG_ROOT="${log_root}" \
      "${SCRIPT_DIR}/infer_libero_object_multicam_parallel.sbatch"
  )

  printf '%s\t%s\t%s\t%s\n' "${model_label}" "${ckpt_step}" "${job_id}" "${log_root}" >> "${manifest}"
  printf '%s step %s -> job %s\n' "${model_label}" "${ckpt_step}" "${job_id}"
}

for step in "${pi0_steps[@]}"; do
  submit_one \
    "cam_pi0_baseline_cam4" \
    "pi0_libero_pytorch_full_finetune" \
    "${PI0_CKPT_ROOT}" \
    "${step}" \
    "glbreeze/libero_object_cam" \
    "glbreeze/libero" \
    "${PI0_LOG_ROOT}" \
    "$((BASE_PORT + port_offset))"
  port_offset=$((port_offset + 1))
done

for step in "${pose_steps[@]}"; do
  submit_one \
    "posefusion_multi_cam4" \
    "pi0_libero_cam_pytorch_full_finetune" \
    "${POSE_CKPT_ROOT}" \
    "${step}" \
    "glbreeze/libero_object_cam" \
    "glbreeze/libero_cam" \
    "${POSE_LOG_ROOT}" \
    "$((BASE_PORT + port_offset))"
  port_offset=$((port_offset + 1))
done

echo "Manifest: ${manifest}"

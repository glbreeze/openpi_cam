#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
GEO_ROOT=$(cd -- "${REPO_ROOT}/.." && pwd)

NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-10}
NUM_PARALLEL_CLIENTS=${NUM_PARALLEL_CLIENTS:-4}
BASE_PORT=${BASE_PORT:-26400}
CAMERA_VARIATION_CONFIG=${CAMERA_VARIATION_CONFIG:-${GEO_ROOT}/LIBERO-Camera/configs/camera_variation/libero_object_left_variation.json}
CAMERA_VARIATION_COUNT=${CAMERA_VARIATION_COUNT:-11}

MODEL_LABEL=${MODEL_LABEL:-camtrain30k_agentcam_relpose_nocross_twostage_v1}
CONFIG_NAME=${CONFIG_NAME:-pi0_libero_object_cam_pytorch_relative_pose_agentcam_action_nocross_finetune}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/pi0_libero_object_cam_pytorch_relative_pose_agentcam_action_finetune/pi0_libero_object_camtrain_o00_02_03_05_06_09_10_agentcam_action_relpose_nocross_30k_twostage_v1}
CHECKPOINT_ASSET_ID=${CHECKPOINT_ASSET_ID:-glbreeze/libero_object_cam_train_o00_02_03_05_06_09_10_agentcam_action}
SERVE_ASSET_ID=${SERVE_ASSET_ID:-glbreeze/libero_agentcam_action_evalalias}
LOG_ROOT=${LOG_ROOT:-${REPO_ROOT}/log/libero_object_multicam_eval/camtrain30k_agentcam_relpose_nocross_twostage_v1}

mkdir -p "${LOG_ROOT}" "${LOG_ROOT}/summary"

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
    echo "No numeric 5k checkpoints found under ${ckpt_root}" >&2
    return 1
  fi

  printf '%s\n' "${steps[@]}" | sort -n
}

has_result() {
  local step="$1"
  compgen -G "${LOG_ROOT}/summary/aggregate_${step}_*.json" > /dev/null
}

mapfile -t steps < <(collect_steps "${CHECKPOINT_ROOT}")

manifest="${LOG_ROOT}/submit_manifest_$(date +%Y%m%d_%H%M%S).tsv"
printf 'model\tstep\tjob_id\tlog_root\n' > "${manifest}"

port_offset=0
for step in "${steps[@]}"; do
  if has_result "${step}"; then
    printf 'skip step %s: existing aggregate found\n' "${step}"
    continue
  fi

  job_id=$(
    sbatch --parsable \
      --job-name="agnox_${step}" \
      --output="${LOG_ROOT}/slurm-%j.out" \
      --error="${LOG_ROOT}/slurm-%j.err" \
      --export=ALL,MODEL_NAME="${MODEL_LABEL}",CONFIG_NAME="${CONFIG_NAME}",CHECKPOINT_ROOT="${CHECKPOINT_ROOT}",CHECKPOINT_STEP="${step}",CHECKPOINT_ASSET_ID="${CHECKPOINT_ASSET_ID}",SERVE_ASSET_ID="${SERVE_ASSET_ID}",NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}",NUM_PARALLEL_CLIENTS="${NUM_PARALLEL_CLIENTS}",PORT_BASE="$((BASE_PORT + port_offset))",CAMERA_VARIATION_CONFIG="${CAMERA_VARIATION_CONFIG}",CAMERA_VARIATION_COUNT="${CAMERA_VARIATION_COUNT}",LOG_ROOT="${LOG_ROOT}" \
      "${SCRIPT_DIR}/infer_libero_object_multicam_parallel.sbatch"
  )

  printf '%s\t%s\t%s\t%s\n' "${MODEL_LABEL}" "${step}" "${job_id}" "${LOG_ROOT}" >> "${manifest}"
  printf '%s step %s -> job %s\n' "${MODEL_LABEL}" "${step}" "${job_id}"
  port_offset=$((port_offset + 1))
done

echo "Manifest: ${manifest}"

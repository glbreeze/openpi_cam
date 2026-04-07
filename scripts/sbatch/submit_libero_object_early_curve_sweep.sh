#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-10}
BASE_PORT=${BASE_PORT:-24000}

BASELINE_CKPT_ROOT=${BASELINE_CKPT_ROOT:-${REPO_ROOT}/checkpoints/pi0_libero_pytorch_full_finetune/pi0_libero_object_4690327}
SIMPLE_CKPT_ROOT=${SIMPLE_CKPT_ROOT:-${REPO_ROOT}/checkpoints/pi0_libero_cam_pytorch_full_finetune/pi0_libero_object_cam_4711127}

BASELINE_LOG_ROOT=${BASELINE_LOG_ROOT:-${REPO_ROOT}/log/libero_object_eval/early_curve/baseline}
SIMPLE_LOG_ROOT=${SIMPLE_LOG_ROOT:-${REPO_ROOT}/log/libero_object_eval/early_curve/simple}
BASELINE_SUMMARY_ROOT=${BASELINE_SUMMARY_ROOT:-${REPO_ROOT}/log/libero_object_eval/summaries/early_curve_baseline}
SIMPLE_SUMMARY_ROOT=${SIMPLE_SUMMARY_ROOT:-${REPO_ROOT}/log/libero_object_eval/summaries/early_curve_simple}
MANIFEST_DIR=${MANIFEST_DIR:-${REPO_ROOT}/log/libero_object_eval/early_curve}

mkdir -p "${BASELINE_LOG_ROOT}" "${SIMPLE_LOG_ROOT}" "${BASELINE_SUMMARY_ROOT}" "${SIMPLE_SUMMARY_ROOT}" "${MANIFEST_DIR}"

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

manifest="${MANIFEST_DIR}/submit_manifest_$(date +%Y%m%d_%H%M%S).tsv"
printf 'model\tstep\tjob_id\tlog_root\tsummary_root\n' > "${manifest}"

port_offset=0

submit_one() {
  local model_label="$1"
  local sbatch_script="$2"
  local ckpt_root="$3"
  local log_root="$4"
  local summary_root="$5"
  local ckpt_asset_id="$6"
  local serve_asset_id="$7"
  local config_name="$8"
  local step="$9"
  local port_base="${10}"

  local job_id
  job_id=$(
    sbatch --parsable \
      --job-name="objcurve_${model_label}_${step}" \
      --output="${log_root}/slurm-%j.out" \
      --error="${log_root}/slurm-%j.err" \
      --export=ALL,MODEL_NAME="${model_label}",CONFIG_NAME="${config_name}",CHECKPOINT_ROOT="${ckpt_root}",CHECKPOINT_STEP="${step}",CHECKPOINT_ASSET_ID="${ckpt_asset_id}",SERVE_ASSET_ID="${serve_asset_id}",NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}",PORT_BASE="${port_base}",SUMMARY_ROOT="${summary_root}" \
      "${sbatch_script}"
  )

  printf '%s\t%s\t%s\t%s\t%s\n' "${model_label}" "${step}" "${job_id}" "${log_root}" "${summary_root}" >> "${manifest}"
  printf '%s step %s -> job %s\n' "${model_label}" "${step}" "${job_id}"
}

for step in "${baseline_steps[@]}"; do
  submit_one \
    "early_baseline" \
    "${SCRIPT_DIR}/eval_libero_object_ckpts_no_cam.sbatch" \
    "${BASELINE_CKPT_ROOT}" \
    "${BASELINE_LOG_ROOT}" \
    "${BASELINE_SUMMARY_ROOT}" \
    "glbreeze/libero_object" \
    "glbreeze/libero" \
    "pi0_libero_pytorch_full_finetune" \
    "${step}" \
    "$((BASE_PORT + port_offset))"
  port_offset=$((port_offset + 1))
done

for step in "${simple_steps[@]}"; do
  submit_one \
    "early_simple" \
    "${SCRIPT_DIR}/eval_libero_object_ckpts_cam.sbatch" \
    "${SIMPLE_CKPT_ROOT}" \
    "${SIMPLE_LOG_ROOT}" \
    "${SIMPLE_SUMMARY_ROOT}" \
    "glbreeze/libero_object_cam" \
    "glbreeze/libero_cam" \
    "pi0_libero_cam_pytorch_full_finetune" \
    "${step}" \
    "$((BASE_PORT + port_offset))"
  port_offset=$((port_offset + 1))
done

echo "Manifest: ${manifest}"

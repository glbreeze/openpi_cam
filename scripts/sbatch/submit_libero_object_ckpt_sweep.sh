#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

MODEL_NAME=${MODEL_NAME:-pi0_cam_v3_fr_s2_fgfg_2gpu_b16_eval50_h100_sweep}
CONFIG_NAME=${CONFIG_NAME:-pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage2_fgfg}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage2_fgfg/pi0_libero_object_cam_v3_prope_ray_view_distill_fullres_twostage_v1_stage2_fgfg_2gpu_b16}
CHECKPOINT_ASSET_ID=${CHECKPOINT_ASSET_ID:-glbreeze/libero_object_cam_v3}
SERVE_ASSET_ID=${SERVE_ASSET_ID:-glbreeze/libero_object_cam_v3}

SUITE_NAME=${SUITE_NAME:-libero_object}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-50}
NUM_PARALLEL_CLIENTS=${NUM_PARALLEL_CLIENTS:-10}
TASK_ID_START=${TASK_ID_START:-0}
TASK_ID_END=${TASK_ID_END:--1}

PARTITION=${PARTITION:-h100_tandon}
ACCOUNT=${ACCOUNT:-torch_pr_637_tandon_advanced}
GRES=${GRES:-gpu:1}
CPUS_PER_TASK=${CPUS_PER_TASK:-24}
MEMORY=${MEMORY:-160G}
TIME_LIMIT=${TIME_LIMIT:-02:00:00}

BASE_PORT=${BASE_PORT:-28000}
JOB_NAME_PREFIX=${JOB_NAME_PREFIX:-fgfg_ckpt}
STEP_INTERVAL=${STEP_INTERVAL:-5000}
LOG_ROOT=${LOG_ROOT:-${REPO_ROOT}/log/libero_object_eval/ckpt_sweep/${MODEL_NAME}}

mkdir -p "${LOG_ROOT}"

collect_steps() {
  local ckpt_root="$1"
  local -a steps=()
  local path
  for path in "${ckpt_root}"/*; do
    [[ -d "${path}" ]] || continue
    local step
    step=$(basename -- "${path}")
    [[ "${step}" =~ ^[0-9]+$ ]] || continue
    if (( step % STEP_INTERVAL == 0 )); then
      steps+=("${step}")
    fi
  done

  if ((${#steps[@]} == 0)); then
    return 1
  fi

  printf '%s\n' "${steps[@]}" | sort -n
}

if [[ ! -d "${CHECKPOINT_ROOT}" ]]; then
  echo "Missing checkpoint root: ${CHECKPOINT_ROOT}" >&2
  exit 1
fi

mapfile -t steps < <(collect_steps "${CHECKPOINT_ROOT}")

manifest="${LOG_ROOT}/submit_manifest_$(date +%Y%m%d_%H%M%S).tsv"
printf 'model\tstep\tjob_id\tlog_root\n' > "${manifest}"

port_offset=0

submit_one() {
  local step="$1"
  local port="$2"
  local job_id

  job_id=$(
    sbatch --parsable \
      --job-name="${JOB_NAME_PREFIX}_${step}" \
      --partition="${PARTITION}" \
      --account="${ACCOUNT}" \
      --gres="${GRES}" \
      --cpus-per-task="${CPUS_PER_TASK}" \
      --mem="${MEMORY}" \
      --time="${TIME_LIMIT}" \
      --output="${LOG_ROOT}/slurm-%j.out" \
      --error="${LOG_ROOT}/slurm-%j.err" \
      --export=ALL,REPO_ROOT="${REPO_ROOT}",MODEL_NAME="${MODEL_NAME}",CONFIG_NAME="${CONFIG_NAME}",CHECKPOINT_ROOT="${CHECKPOINT_ROOT}",CHECKPOINT_STEP="${step}",CHECKPOINT_ASSET_ID="${CHECKPOINT_ASSET_ID}",SERVE_ASSET_ID="${SERVE_ASSET_ID}",SUITE_NAME="${SUITE_NAME}",NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}",NUM_PARALLEL_CLIENTS="${NUM_PARALLEL_CLIENTS}",TASK_ID_START="${TASK_ID_START}",TASK_ID_END="${TASK_ID_END}",SERVER_BATCH_MAX_SIZE="${NUM_PARALLEL_CLIENTS}",PORT="${port}",LOG_ROOT="${LOG_ROOT}" \
      "${SCRIPT_DIR}/infer_libero_object_parallel.sbatch"
  )

  printf '%s\t%s\t%s\t%s\n' "${MODEL_NAME}" "${step}" "${job_id}" "${LOG_ROOT}" >> "${manifest}"
  printf '%s step %s -> job %s\n' "${MODEL_NAME}" "${step}" "${job_id}"
}

for step in "${steps[@]}"; do
  submit_one "${step}" "$((BASE_PORT + port_offset))"
  port_offset=$((port_offset + 1))
done

echo "Manifest: ${manifest}"

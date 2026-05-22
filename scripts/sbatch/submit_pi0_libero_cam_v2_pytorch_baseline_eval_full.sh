#!/bin/bash

set -euo pipefail

resolve_repo_root() {
  local candidate
  local script_root=""

  if script_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd); then
    :
  else
    script_root=""
  fi

  for candidate in "${REPO_ROOT:-}" "${OPENPI_CAM_ROOT:-}" "${PWD:-}" "${script_root}"; do
    [[ -n "${candidate}" ]] || continue
    while [[ "${candidate}" != "/" ]]; do
      if [[ -f "${candidate}/scripts/env/activate_env.sh" ]]; then
        printf '%s\n' "${candidate}"
        return 0
      fi
      candidate=$(dirname -- "${candidate}")
    done
  done

  return 1
}

collect_steps() {
  local ckpt_root="$1"
  local min_step="$2"
  local max_step="$3"
  local stride="$4"
  local path

  for path in "${ckpt_root}"/*; do
    [[ -d "${path}" ]] || continue
    local step
    step=$(basename -- "${path}")
    [[ "${step}" =~ ^[0-9]+$ ]] || continue
    if (( step < min_step )); then
      continue
    fi
    if (( max_step >= 0 && step > max_step )); then
      continue
    fi
    if (( step % stride == 0 )); then
      printf '%s\n' "${step}"
    fi
  done | sort -n
}

REPO_ROOT=${REPO_ROOT:-$(resolve_repo_root)} || {
  echo "Unable to locate openpi_cam repo root. Set REPO_ROOT or run from inside the repo." >&2
  exit 1
}

TARGET_SCRIPT="${REPO_ROOT}/scripts/sbatch/eval_pi0_libero_cam_v2_pytorch_baseline_full.sbatch"
EXP_NAME=${EXP_NAME:-pi0_libero_cam_v2_pytorch_baseline_v1}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/pi0_libero_cam_v2_pytorch_baseline/${EXP_NAME}}
SUITE_NAME=${SUITE_NAME:-libero_90}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-50}
TASK_ID_START=${TASK_ID_START:-0}
TASK_ID_END=${TASK_ID_END:--1}
STEP_STRIDE=${STEP_STRIDE:-5000}
MIN_STEP=${MIN_STEP:-0}
MAX_STEP=${MAX_STEP:--1}
BASE_PORT=${BASE_PORT:-19000}
LOG_ROOT=${LOG_ROOT:-${REPO_ROOT}/log/libero_eval/${EXP_NAME}/${SUITE_NAME}/full}
MANIFEST_DIR=${MANIFEST_DIR:-${LOG_ROOT}}

SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-}
SBATCH_PARTITION=${SBATCH_PARTITION:-}
SBATCH_GRES=${SBATCH_GRES:-}
SBATCH_TIME=${SBATCH_TIME:-}
SBATCH_CPUS=${SBATCH_CPUS:-}
SBATCH_MEM=${SBATCH_MEM:-}

mkdir -p "${LOG_ROOT}" "${MANIFEST_DIR}"

mapfile -t steps < <(collect_steps "${CHECKPOINT_ROOT}" "${MIN_STEP}" "${MAX_STEP}" "${STEP_STRIDE}")
if ((${#steps[@]} == 0)); then
  echo "No checkpoint steps found under ${CHECKPOINT_ROOT}" >&2
  exit 1
fi

manifest="${MANIFEST_DIR}/submit_manifest_$(date +%Y%m%d_%H%M%S).tsv"
printf 'step\tjob_id\tlog_root\n' > "${manifest}"

echo "Submitting full baseline LIBERO eval sweep"
echo "repo root: ${REPO_ROOT}"
echo "target script: ${TARGET_SCRIPT}"
echo "config: pi0_libero_cam_v2_pytorch_baseline"
echo "exp name: ${EXP_NAME}"
echo "checkpoint root: ${CHECKPOINT_ROOT}"
echo "steps: ${steps[*]}"
echo "suite: ${SUITE_NAME}"
echo "num_trials_per_task: ${NUM_TRIALS_PER_TASK}"
echo "task range: [${TASK_ID_START}, ${TASK_ID_END})"
echo "log root: ${LOG_ROOT}"
echo "manifest: ${manifest}"
echo "default account: ${SBATCH_ACCOUNT:-torch_pr_926_general}"
echo "default partition: ${SBATCH_PARTITION:-l40s_public}"
echo "default gres: ${SBATCH_GRES:-gpu:l40s:1}"

for idx in "${!steps[@]}"; do
  step="${steps[$idx]}"
  port=$((BASE_PORT + idx))
  step_log_root="${LOG_ROOT}/step_${step}"
  mkdir -p "${step_log_root}"

  SBATCH_ARGS=()
  [[ -n "${SBATCH_ACCOUNT}" ]] && SBATCH_ARGS+=(--account="${SBATCH_ACCOUNT}")
  [[ -n "${SBATCH_PARTITION}" ]] && SBATCH_ARGS+=(--partition="${SBATCH_PARTITION}")
  [[ -n "${SBATCH_GRES}" ]] && SBATCH_ARGS+=(--gres="${SBATCH_GRES}")
  [[ -n "${SBATCH_TIME}" ]] && SBATCH_ARGS+=(--time="${SBATCH_TIME}")
  [[ -n "${SBATCH_CPUS}" ]] && SBATCH_ARGS+=(--cpus-per-task="${SBATCH_CPUS}")
  [[ -n "${SBATCH_MEM}" ]] && SBATCH_ARGS+=(--mem="${SBATCH_MEM}")
  SBATCH_ARGS+=(--job-name="pi0_lv2_eval_${step}")
  SBATCH_ARGS+=(--output="${step_log_root}/slurm-%j.out" --error="${step_log_root}/slurm-%j.err")

  job_id=$(
    sbatch --parsable "${SBATCH_ARGS[@]}" \
      --export=ALL,CONFIG_NAME=pi0_libero_cam_v2_pytorch_baseline,EXP_NAME="${EXP_NAME}",CHECKPOINT_ROOT="${CHECKPOINT_ROOT}",CHECKPOINT_STEP="${step}",SUITE_NAME="${SUITE_NAME}",NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}",TASK_ID_START="${TASK_ID_START}",TASK_ID_END="${TASK_ID_END}",SERVER_PORT="${port}",LOG_ROOT="${step_log_root}",MODEL_NAME="${EXP_NAME}_ckpt${step}" \
      "${TARGET_SCRIPT}"
  )

  printf '%s\t%s\t%s\n' "${step}" "${job_id}" "${step_log_root}" >> "${manifest}"
  printf 'step %s -> job %s\n' "${step}" "${job_id}"
done

echo "Manifest: ${manifest}"

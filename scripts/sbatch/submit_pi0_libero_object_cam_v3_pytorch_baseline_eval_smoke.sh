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

REPO_ROOT=${REPO_ROOT:-$(resolve_repo_root)} || {
  echo "Unable to locate openpi_cam repo root. Set REPO_ROOT or run from inside the repo." >&2
  exit 1
}

TARGET_SCRIPT="${REPO_ROOT}/scripts/sbatch/eval_pi0_libero_object_cam_v3_pytorch_baseline_smoke.sbatch"
EXP_NAME=${EXP_NAME:-pi0_libero_object_cam_v3_pytorch_baseline_v1}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/pi0_libero_object_pytorch_baseline/${EXP_NAME}}
CHECKPOINT_STEP=${CHECKPOINT_STEP:-latest}
SUITE_NAME=${SUITE_NAME:-libero_object}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-1}
TASK_ID_START=${TASK_ID_START:-0}
TASK_ID_END=${TASK_ID_END:-1}
SERVER_PORT=${SERVER_PORT:-18000}
LOG_ROOT=${LOG_ROOT:-${REPO_ROOT}/log/libero_eval/${EXP_NAME}/smoke}

if [[ -z "${TASK_ID_END:-}" || "${TASK_ID_END}" == "-1" ]]; then
  TASK_ID_END=1
fi

SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-}
SBATCH_PARTITION=${SBATCH_PARTITION:-}
SBATCH_GRES=${SBATCH_GRES:-}
SBATCH_TIME=${SBATCH_TIME:-}
SBATCH_CPUS=${SBATCH_CPUS:-}
SBATCH_MEM=${SBATCH_MEM:-}

mkdir -p "${LOG_ROOT}"

SBATCH_ARGS=()
[[ -n "${SBATCH_ACCOUNT}" ]] && SBATCH_ARGS+=(--account="${SBATCH_ACCOUNT}")
[[ -n "${SBATCH_PARTITION}" ]] && SBATCH_ARGS+=(--partition="${SBATCH_PARTITION}")
[[ -n "${SBATCH_GRES}" ]] && SBATCH_ARGS+=(--gres="${SBATCH_GRES}")
[[ -n "${SBATCH_TIME}" ]] && SBATCH_ARGS+=(--time="${SBATCH_TIME}")
[[ -n "${SBATCH_CPUS}" ]] && SBATCH_ARGS+=(--cpus-per-task="${SBATCH_CPUS}")
[[ -n "${SBATCH_MEM}" ]] && SBATCH_ARGS+=(--mem="${SBATCH_MEM}")
SBATCH_ARGS+=(--output="${LOG_ROOT}/slurm-%j.out" --error="${LOG_ROOT}/slurm-%j.err")

echo "Submitting baseline LIBERO smoke eval"
echo "repo root: ${REPO_ROOT}"
echo "target script: ${TARGET_SCRIPT}"
echo "config: pi0_libero_object_pytorch_baseline"
echo "exp name: ${EXP_NAME}"
echo "checkpoint root: ${CHECKPOINT_ROOT}"
echo "checkpoint step: ${CHECKPOINT_STEP}"
echo "suite: ${SUITE_NAME}"
echo "num_trials_per_task: ${NUM_TRIALS_PER_TASK}"
echo "task range: [${TASK_ID_START}, ${TASK_ID_END})"
echo "log root: ${LOG_ROOT}"
echo "default account: ${SBATCH_ACCOUNT:-torch_pr_926_general}"
echo "default partition: ${SBATCH_PARTITION:-l40s_public}"
echo "default gres: ${SBATCH_GRES:-gpu:l40s:1}"

sbatch "${SBATCH_ARGS[@]}" \
  --export=ALL,CONFIG_NAME=pi0_libero_object_pytorch_baseline,EXP_NAME="${EXP_NAME}",CHECKPOINT_ROOT="${CHECKPOINT_ROOT}",CHECKPOINT_STEP="${CHECKPOINT_STEP}",SUITE_NAME="${SUITE_NAME}",NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}",TASK_ID_START="${TASK_ID_START}",TASK_ID_END="${TASK_ID_END}",SERVER_PORT="${SERVER_PORT}",LOG_ROOT="${LOG_ROOT}" \
  "${TARGET_SCRIPT}"

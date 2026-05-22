#!/bin/bash
#SBATCH --job-name=pi0_obj_eval_smoke
#SBATCH --output=/scratch/%u/openpi_cam/log/libero_eval/slurm-%j.out
#SBATCH --error=/scratch/%u/openpi_cam/log/libero_eval/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_926_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --export=ALL

set -euo pipefail

resolve_repo_root() {
  local candidate
  local script_root=""

  if script_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd); then
    :
  else
    script_root=""
  fi

  for candidate in "${REPO_ROOT:-}" "${OPENPI_CAM_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "${PWD:-}" "${script_root}"; do
    [[ -n "${candidate}" ]] || continue
    while [[ "${candidate}" != "/" ]]; do
      if [[ -f "${candidate}/scripts/sbatch/run_libero_checkpoint_eval.sh" ]]; then
        printf '%s\n' "${candidate}"
        return 0
      fi
      candidate=$(dirname -- "${candidate}")
    done
  done

  return 1
}

REPO_ROOT=${REPO_ROOT:-$(resolve_repo_root)} || {
  echo "Unable to locate openpi_cam repo root. Set REPO_ROOT or submit from inside the repo." >&2
  exit 1
}

export CONFIG_NAME=${CONFIG_NAME:-pi0_libero_object_pytorch_baseline}
export EXP_NAME=${EXP_NAME:-pi0_libero_object_cam_v3_pytorch_baseline_v1}
export CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/pi0_libero_object_pytorch_baseline/${EXP_NAME}}
export CHECKPOINT_STEP=${CHECKPOINT_STEP:-latest}
export SUITE_NAME=${SUITE_NAME:-libero_object}
export NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-1}
export TASK_ID_START=${TASK_ID_START:-0}
export TASK_ID_END=${TASK_ID_END:-1}
export SERVER_PORT=${SERVER_PORT:-18000}
export LOG_ROOT=${LOG_ROOT:-${REPO_ROOT}/log/libero_eval/${EXP_NAME}/smoke}
export MODEL_NAME=${MODEL_NAME:-${EXP_NAME}_smoke}

if [[ -z "${TASK_ID_END:-}" || "${TASK_ID_END}" == "-1" ]]; then
  export TASK_ID_END=1
fi

bash "${REPO_ROOT}/scripts/sbatch/run_libero_checkpoint_eval.sh"

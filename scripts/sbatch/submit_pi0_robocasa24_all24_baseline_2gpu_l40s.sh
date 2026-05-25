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
  echo "Unable to locate repo root. Set REPO_ROOT or run from inside the repo." >&2
  exit 1
}

TARGET_SCRIPT="${REPO_ROOT}/scripts/sbatch/train_pi0_robocasa24_all24_baseline_2gpu_l40s.sbatch"

SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-torch_pr_926_general}
SBATCH_PARTITION=${SBATCH_PARTITION:-l40s_public}
SBATCH_GRES=${SBATCH_GRES:-gpu:l40s:2}
SBATCH_TIME=${SBATCH_TIME:-24:00:00}
SBATCH_CPUS=${SBATCH_CPUS:-32}
SBATCH_MEM=${SBATCH_MEM:-256G}

SBATCH_ARGS=()
[[ -n "${SBATCH_ACCOUNT}" ]] && SBATCH_ARGS+=(--account="${SBATCH_ACCOUNT}")
[[ -n "${SBATCH_PARTITION}" ]] && SBATCH_ARGS+=(--partition="${SBATCH_PARTITION}")
[[ -n "${SBATCH_GRES}" ]] && SBATCH_ARGS+=(--gres="${SBATCH_GRES}")
[[ -n "${SBATCH_TIME}" ]] && SBATCH_ARGS+=(--time="${SBATCH_TIME}")
[[ -n "${SBATCH_CPUS}" ]] && SBATCH_ARGS+=(--cpus-per-task="${SBATCH_CPUS}")
[[ -n "${SBATCH_MEM}" ]] && SBATCH_ARGS+=(--mem="${SBATCH_MEM}")

echo "Submitting RoboCasa24 Pi0 baseline training (2 GPU)"
echo "repo root: ${REPO_ROOT}"
echo "target script: ${TARGET_SCRIPT}"
echo "config: ${CONFIG_NAME:-pi0_robocasa24_all24_baseline}"
echo "dataset repo id: ${DATASET_REPO_ID:-robocasa24/all24_human_camaware}"
echo "norm asset id: ${NORM_ASSET_ID:-robocasa24/all24_human_camaware}"
echo "norm stats root: ${NORM_STATS_ROOT:-/scratch/${USER}/pi0_robocasa24}"
echo "base model dir: ${BASE_MODEL_DIR:-/scratch/${USER}/pi0_base}"
echo "checkpoint base dir: ${CHECKPOINT_BASE_DIR:-/scratch/${USER}/tmp/openpi_cam/checkpoints}"
echo "exp name: ${EXP_NAME:-pi0_robocasa24_all24_baseline_2gpu_v1}"
echo "resume: ${RESUME:-false}"
echo "wandb enabled: ${WANDB_ENABLED:-false}"
echo "default account: ${SBATCH_ACCOUNT}"
echo "default partition: ${SBATCH_PARTITION}"
echo "default gres: ${SBATCH_GRES}"
echo "default cpus-per-task: ${SBATCH_CPUS}"
echo "default mem: ${SBATCH_MEM}"
echo "default time: ${SBATCH_TIME}"
echo "default batch size: ${BATCH_SIZE:-32}"
echo "default num workers: ${NUM_WORKERS:-4}"
echo "default num train steps: ${NUM_TRAIN_STEPS:-30000}"

sbatch "${SBATCH_ARGS[@]}" "${TARGET_SCRIPT}"

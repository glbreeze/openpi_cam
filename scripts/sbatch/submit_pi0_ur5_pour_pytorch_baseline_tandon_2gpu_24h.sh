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

TARGET_SCRIPT="${REPO_ROOT}/scripts/sbatch/train_pi0_ur5_pour_pytorch_baseline_tandon_2gpu_24h.sbatch"

SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-torch_pr_69_tandon_advanced}
SBATCH_PARTITION=${SBATCH_PARTITION:-a100_tandon,h100_tandon,h200_tandon}
SBATCH_GRES=${SBATCH_GRES:-gpu:2}
SBATCH_TIME=${SBATCH_TIME:-24:00:00}
SBATCH_CPUS=${SBATCH_CPUS:-24}
SBATCH_MEM=${SBATCH_MEM:-192G}
SBATCH_TEST_ONLY=${SBATCH_TEST_ONLY:-false}

SBATCH_ARGS=()
if [[ "${SBATCH_TEST_ONLY}" == "true" ]]; then
  SBATCH_ARGS+=(--test-only)
fi
[[ -n "${SBATCH_ACCOUNT}" ]] && SBATCH_ARGS+=(--account="${SBATCH_ACCOUNT}")
[[ -n "${SBATCH_PARTITION}" ]] && SBATCH_ARGS+=(--partition="${SBATCH_PARTITION}")
[[ -n "${SBATCH_GRES}" ]] && SBATCH_ARGS+=(--gres="${SBATCH_GRES}")
[[ -n "${SBATCH_TIME}" ]] && SBATCH_ARGS+=(--time="${SBATCH_TIME}")
[[ -n "${SBATCH_CPUS}" ]] && SBATCH_ARGS+=(--cpus-per-task="${SBATCH_CPUS}")
[[ -n "${SBATCH_MEM}" ]] && SBATCH_ARGS+=(--mem="${SBATCH_MEM}")

echo "Submitting Pi0 UR5 pour baseline training"
echo "repo root: ${REPO_ROOT}"
echo "target script: ${TARGET_SCRIPT}"
echo "test only: ${SBATCH_TEST_ONLY}"
echo "config: ${CONFIG_NAME:-pi0_ur5_real_robot_pytorch_baseline}"
echo "dataset dir: ${DATASET_DIR:-/scratch/${USER}/real_robot_data/ur5_place_and_pour_nuts_camera_shifts}"
echo "dataset repo id: ${DATASET_REPO_ID:-ur5_place_and_pour_nuts_camera_shifts}"
echo "array cache dir: ${DATASET_DIR:-/scratch/${USER}/real_robot_data/ur5_place_and_pour_nuts_camera_shifts}_array_cache"
echo "norm stats root: ${REAL_ROBOT_NORM_ROOT:-/scratch/${USER}/pi0_ur5_real_robot}"
echo "base model dir: ${OPENPI_PI0_BASE_DIR:-/scratch/${USER}/pi0_base}"
echo "checkpoint base dir: ${CHECKPOINT_BASE_DIR:-/scratch/${USER}/tmp/openpi_cam/checkpoints}"
echo "exp name: ${EXP_NAME:-pi0_ur5_pour_pytorch_baseline_tandon_2gpu_b16}"
echo "default account: ${SBATCH_ACCOUNT}"
echo "default partition: ${SBATCH_PARTITION}"
echo "default gres: ${SBATCH_GRES}"
echo "default cpus-per-task: ${SBATCH_CPUS}"
echo "default mem: ${SBATCH_MEM}"
echo "default time: ${SBATCH_TIME}"
echo "default batch size: ${BATCH_SIZE:-16}"
echo "default num workers per GPU/rank: ${NUM_WORKERS:-6}"
echo "default num train steps: ${NUM_TRAIN_STEPS:-30000}"
echo "default save interval: ${SAVE_INTERVAL:-5000}"
echo "default keep period: ${KEEP_PERIOD:-5000}"

sbatch "${SBATCH_ARGS[@]}" "${TARGET_SCRIPT}"

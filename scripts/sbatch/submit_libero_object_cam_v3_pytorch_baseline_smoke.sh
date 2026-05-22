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

TARGET_SCRIPT="${REPO_ROOT}/scripts/sbatch/train_pi0_libero_object_cam_v3_pytorch_baseline_smoke.sbatch"
SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-}
SBATCH_PARTITION=${SBATCH_PARTITION:-}
SBATCH_GRES=${SBATCH_GRES:-}
SBATCH_TIME=${SBATCH_TIME:-}
SBATCH_CPUS=${SBATCH_CPUS:-}
SBATCH_MEM=${SBATCH_MEM:-}

SBATCH_ARGS=()
[[ -n "${SBATCH_ACCOUNT}" ]] && SBATCH_ARGS+=(--account="${SBATCH_ACCOUNT}")
[[ -n "${SBATCH_PARTITION}" ]] && SBATCH_ARGS+=(--partition="${SBATCH_PARTITION}")
[[ -n "${SBATCH_GRES}" ]] && SBATCH_ARGS+=(--gres="${SBATCH_GRES}")
[[ -n "${SBATCH_TIME}" ]] && SBATCH_ARGS+=(--time="${SBATCH_TIME}")
[[ -n "${SBATCH_CPUS}" ]] && SBATCH_ARGS+=(--cpus-per-task="${SBATCH_CPUS}")
[[ -n "${SBATCH_MEM}" ]] && SBATCH_ARGS+=(--mem="${SBATCH_MEM}")

echo "Submitting PyTorch baseline smoke test"
echo "repo root: ${REPO_ROOT}"
echo "target script: ${TARGET_SCRIPT}"
echo "dataset repo id: ${DATASET_REPO_ID:-glbreeze/libero_object_cam_v3}"
echo "dataset dir: ${DATASET_DIR:-/scratch/yz11445/cam-aware-data/libero_object_cam_v3}"
echo "lerobot home: ${HF_LEROBOT_HOME:-/scratch/yz11445/.huggingface/lerobot}"
echo "config: pi0_libero_object_pytorch_baseline"
echo "exp name: ${EXP_NAME:-pi0_libero_object_cam_v3_pytorch_baseline_smoke}"
echo "num_gpus: ${NUM_GPUS:-1}"
echo "batch_size: ${BATCH_SIZE:-2}"
echo "num_workers: ${NUM_WORKERS:-2}"
echo "num_train_steps: ${NUM_TRAIN_STEPS:-20}"
echo "save_interval: ${SAVE_INTERVAL:-10}"
echo "keep_period: ${KEEP_PERIOD:-10}"
echo "wandb project: ${WANDB_PROJECT:-openpi_cam}"
echo "wandb entity: ${WANDB_ENTITY:-NYU-robotics}"
echo "wandb dir: ${WANDB_DIR:-/scratch/yz11445/wandb}"
echo "wandb config dir: ${WANDB_CONFIG_DIR:-/scratch/yz11445/.config/wandb}"
echo "wandb cache dir: ${WANDB_CACHE_DIR:-/scratch/yz11445/.cache/wandb}"
echo "default account: ${SBATCH_ACCOUNT:-torch_pr_69_tandon_advanced}"
echo "default partition: ${SBATCH_PARTITION:-h100_tandon,h200_tandon}"
echo "default gres: ${SBATCH_GRES:-gpu:1}"

sbatch "${SBATCH_ARGS[@]}" "${TARGET_SCRIPT}"

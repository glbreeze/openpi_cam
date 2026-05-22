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

RESUME=${RESUME:-false}
SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-}
SBATCH_PARTITION=${SBATCH_PARTITION:-}
SBATCH_GRES=${SBATCH_GRES:-}
SBATCH_TIME=${SBATCH_TIME:-}
SBATCH_CPUS=${SBATCH_CPUS:-}
SBATCH_MEM=${SBATCH_MEM:-}

if [[ "${RESUME}" == "true" ]]; then
  TARGET_SCRIPT="${REPO_ROOT}/scripts/sbatch/train_pi0_libero_cam_v2_pytorch_baseline_2gpu_resume.sbatch"
else
  TARGET_SCRIPT="${REPO_ROOT}/scripts/sbatch/train_pi0_libero_cam_v2_pytorch_baseline_2gpu.sbatch"
fi

SBATCH_ARGS=()
[[ -n "${SBATCH_ACCOUNT}" ]] && SBATCH_ARGS+=(--account="${SBATCH_ACCOUNT}")
[[ -n "${SBATCH_PARTITION}" ]] && SBATCH_ARGS+=(--partition="${SBATCH_PARTITION}")
[[ -n "${SBATCH_GRES}" ]] && SBATCH_ARGS+=(--gres="${SBATCH_GRES}")
[[ -n "${SBATCH_TIME}" ]] && SBATCH_ARGS+=(--time="${SBATCH_TIME}")
[[ -n "${SBATCH_CPUS}" ]] && SBATCH_ARGS+=(--cpus-per-task="${SBATCH_CPUS}")
[[ -n "${SBATCH_MEM}" ]] && SBATCH_ARGS+=(--mem="${SBATCH_MEM}")

echo "Submitting PyTorch baseline 2-GPU training"
echo "repo root: ${REPO_ROOT}"
echo "target script: ${TARGET_SCRIPT}"
echo "resume: ${RESUME}"
echo "dataset repo id: ${DATASET_REPO_ID:-glbreeze/libero_cam_v2}"
echo "dataset dir: ${DATASET_DIR:-/scratch/yp2841/cache/lerobot/glbreeze/libero_cam_v2}"
echo "lerobot home: ${HF_LEROBOT_HOME:-/scratch/yz11445/.huggingface/lerobot}"
echo "config: pi0_libero_cam_v2_pytorch_baseline"
echo "exp name: ${EXP_NAME:-pi0_libero_cam_v2_pytorch_baseline_2gpu_v1}"
echo "wandb project: ${WANDB_PROJECT:-openpi_cam}"
echo "wandb entity: ${WANDB_ENTITY:-NYU-robotics}"
echo "wandb dir: ${WANDB_DIR:-/scratch/yz11445/wandb}"
echo "wandb config dir: ${WANDB_CONFIG_DIR:-/scratch/yz11445/.config/wandb}"
echo "wandb cache dir: ${WANDB_CACHE_DIR:-/scratch/yz11445/.cache/wandb}"
echo "default account: ${SBATCH_ACCOUNT:-torch_pr_69_tandon_advanced}"
echo "default partition: ${SBATCH_PARTITION:-h100_tandon,h200_tandon}"
echo "default gres: ${SBATCH_GRES:-gpu:2}"
echo "default time: ${SBATCH_TIME:-16:00:00}"

sbatch "${SBATCH_ARGS[@]}" "${TARGET_SCRIPT}"

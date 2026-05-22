#!/bin/bash
#SBATCH --job-name=pi0_robocasa24_base
#SBATCH --output=/scratch/%u/openpi_cam/log/pi0_robocasa24_ft/slurm-%j.out
#SBATCH --error=/scratch/%u/openpi_cam/log/pi0_robocasa24_ft/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=h100_tandon,h200_tandon,a100_tandon
#SBATCH --gres=gpu:4
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
      if [[ -f "${candidate}/scripts/env/activate_env.sh" ]]; then
        printf '%s\n' "${candidate}"
        return 0
      fi
      candidate=$(dirname -- "${candidate}")
    done
  done

  return 1
}

if [[ -z "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(resolve_repo_root || true)"
fi
if [[ -z "${REPO_ROOT}" && -d "/scratch/${USER}/openpi_cam" ]]; then
  REPO_ROOT="/scratch/${USER}/openpi_cam"
fi
if [[ -z "${REPO_ROOT}" ]]; then
  echo "Unable to locate repo root." >&2
  exit 1
fi

export USE_CAM=false
export CONFIG_NAME=pi0_robocasa24_all24_pytorch_baseline
export TRAIN_RECIPE=single_stage
export RESUME="${RESUME:-false}"

export WANDB_ENABLED="${WANDB_ENABLED:-true}"
export WANDB_ENTITY="${WANDB_ENTITY:-NYU-robotics}"
export WANDB_PROJECT="${WANDB_PROJECT:-openpi_cam_robocasa}"
export WANDB_DIR="${WANDB_DIR:-/scratch/${USER}/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/scratch/${USER}/.config/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/scratch/${USER}/.cache/wandb}"

export GEO_ROOT="${GEO_ROOT:-/scratch/${USER}/geometry-vla}"
export OPENPI_GEO_ROOT="${OPENPI_GEO_ROOT:-${GEO_ROOT}}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${GEO_ROOT}}"
export OPENPI_PI0_BASE_DIR="${OPENPI_PI0_BASE_DIR:-${GEO_ROOT}/pi0_base}"
export OPENPI_PI0_ROBOCASA_NORM_DIR="${OPENPI_PI0_ROBOCASA_NORM_DIR:-${GEO_ROOT}/pi0_robocasa24}"
export OPENPI_PI0_LIBERO_NORM_DIR="${OPENPI_PI0_LIBERO_NORM_DIR:-${OPENPI_PI0_ROBOCASA_NORM_DIR}}"

export DATASET_REPO_ID="${DATASET_REPO_ID:-robocasa24/all24_human_camaware}"
export DATASET_DIR="${DATASET_DIR:-${HF_LEROBOT_HOME}/${DATASET_REPO_ID}}"
export OPENPI_ROBOCASA_DATASET_DIR="${OPENPI_ROBOCASA_DATASET_DIR:-${DATASET_DIR}}"
export NORM_ASSET_ID="${NORM_ASSET_ID:-robocasa24/all24_human_camaware}"
export EXP_NAME="${EXP_NAME:-pi0_robocasa24_all24_pytorch_baseline_v1}"
export CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-/scratch/${USER}/openpi_cam/checkpoints}"

export NUM_GPUS="${NUM_GPUS:-4}"
export NUM_WORKERS="${NUM_WORKERS:-16}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
export KEEP_PERIOD="${KEEP_PERIOD:-5000}"

mkdir -p "/scratch/${USER}/openpi_cam/log/pi0_robocasa24_ft" "${CHECKPOINT_BASE_DIR}"

exec bash "${REPO_ROOT}/scripts/sbatch/train_pi0_libero_object_ft.sbatch"

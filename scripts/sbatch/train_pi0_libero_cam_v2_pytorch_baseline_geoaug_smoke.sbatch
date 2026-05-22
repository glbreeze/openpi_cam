#!/bin/bash
#SBATCH --job-name=pi0_libero_v2_gsmk
#SBATCH --output=/scratch/%u/openpi_cam/log/pi0_libero_cam_v2_ft/slurm-%j.out
#SBATCH --error=/scratch/%u/openpi_cam/log/pi0_libero_cam_v2_ft/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=h100_tandon,h200_tandon
#SBATCH --gres=gpu:1
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

export USE_CAM=true
export CONFIG_NAME=pi0_libero_cam_v2_pytorch_baseline_geoaug
export TRAIN_RECIPE=single_stage
export WANDB_ENABLED="${WANDB_ENABLED:-true}"
export WANDB_ENTITY="${WANDB_ENTITY:-NYU-robotics}"
export WANDB_PROJECT="${WANDB_PROJECT:-openpi_cam}"
export WANDB_DIR="${WANDB_DIR:-/scratch/yz11445/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/scratch/yz11445/.config/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/scratch/yz11445/.cache/wandb}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/scratch/yz11445/.huggingface/lerobot}"
export DATASET_DIR="${DATASET_DIR:-/scratch/yp2841/cache/lerobot/glbreeze/libero_cam_v2}"
export NUM_GPUS="${NUM_GPUS:-1}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-20}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
export KEEP_PERIOD="${KEEP_PERIOD:-10}"

export DATASET_REPO_ID="${DATASET_REPO_ID:-glbreeze/libero_cam_v2}"
export NORM_ASSET_ID="${NORM_ASSET_ID:-glbreeze/libero_cam_v2}"
export EXP_NAME="${EXP_NAME:-pi0_libero_cam_v2_pytorch_baseline_geoaug_smoke}"

exec bash "${REPO_ROOT}/scripts/sbatch/train_pi0_libero_object_ft.sbatch"

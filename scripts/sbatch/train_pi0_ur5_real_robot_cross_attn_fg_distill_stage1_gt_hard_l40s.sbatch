#!/bin/bash
#SBATCH --job-name=pi0_ur5_fg_gt_s1
#SBATCH --output=/scratch/%u/openpi_cam/log/pi0_ur5_real_robot_ft/slurm-%j.out
#SBATCH --error=/scratch/%u/openpi_cam/log/pi0_ur5_real_robot_ft/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_69_general
#SBATCH --partition=l40s_public
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
export CONFIG_NAME="${CONFIG_NAME:-pi0_ur5_real_robot_pytorch_cross_attn_fg_distill_stage1_hard}"
export TRAIN_RECIPE=single_stage
export RESUME="${RESUME:-false}"

export WANDB_ENABLED="${WANDB_ENABLED:-true}"
export WANDB_ENTITY="${WANDB_ENTITY:-NYU-robotics}"
export WANDB_PROJECT="${WANDB_PROJECT:-openpi_cam_real_robot}"
export WANDB_DIR="${WANDB_DIR:-/scratch/${USER}/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/scratch/${USER}/.config/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/scratch/${USER}/.cache/wandb}"

export OPENPI_GEO_ROOT="${OPENPI_GEO_ROOT:-/scratch/${USER}}"
export OPENPI_PI0_BASE_DIR="${OPENPI_PI0_BASE_DIR:-${OPENPI_GEO_ROOT}/pi0_base}"
export OPENPI_PI0_LIBERO_NORM_DIR="${OPENPI_PI0_LIBERO_NORM_DIR:-${OPENPI_GEO_ROOT}/pi0_ur5_real_robot}"
export OPENPI_PI3X_TARGETS_224_BASE_DIR="${OPENPI_PI3X_TARGETS_224_BASE_DIR:-/scratch/${USER}/openpi_cache/pi3x_targets_224}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${OPENPI_GEO_ROOT}/.huggingface/lerobot}"

export DATASET_DIR="${DATASET_DIR:-/scratch/${USER}/real_robot_data/ur5_lab_test_tube_camera_shifts}"
export DATASET_REPO_ID="${DATASET_REPO_ID:-ur5_lab_test_tube_camera_shifts}"
export NORM_ASSET_ID="${NORM_ASSET_ID:-ur5_lab_test_tube_camera_shifts}"
export EXP_NAME="${EXP_NAME:-pi0_ur5_real_robot_cross_attn_fg_distill_gt_hard_stage1}"
export CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-/scratch/${USER}/tmp/openpi_cam/checkpoints}"

export NUM_GPUS="${NUM_GPUS:-4}"
export NUM_WORKERS="${NUM_WORKERS:-16}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-5000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
export KEEP_PERIOD="${KEEP_PERIOD:-5000}"

mkdir -p "/scratch/${USER}/tmp/openpi_cam/log/pi0_ur5_real_robot_ft" "${CHECKPOINT_BASE_DIR}"

exec bash "${REPO_ROOT}/scripts/sbatch/train_pi0_libero_object_ft.sbatch"

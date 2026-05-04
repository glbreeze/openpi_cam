#!/bin/bash
#SBATCH --job-name=pi0_cam_v3_fr_s2
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_libero_object_ft/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_libero_object_ft/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_637_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
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
if [[ -z "${REPO_ROOT}" && -d "/scratch/yp2841/geometry-vla/openpi_cam" ]]; then
  REPO_ROOT="/scratch/yp2841/geometry-vla/openpi_cam"
fi
if [[ -z "${REPO_ROOT}" ]]; then
  echo "Unable to locate repo root." >&2
  exit 1
fi

CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:-${REPO_ROOT}/checkpoints}
STAGE1_EXP_NAME=${STAGE1_EXP_NAME:-pi0_libero_object_cam_v3_prope_ray_view_distill_fullres_twostage_v1_stage1}
STAGE1_WEIGHT_PATH=${STAGE1_WEIGHT_PATH:-${CHECKPOINT_BASE_DIR}/pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage1/${STAGE1_EXP_NAME}/5000}

export USE_CAM=true
export CONFIG_NAME=pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage2
export TRAIN_RECIPE=single_stage
export WANDB_ENABLED=true
export WANDB_PROJECT=pi0_libero_cam
export WANDB_ENTITY=NYU-robotics
export WANDB_API_KEY_OVERRIDE="${WANDB_API_KEY_OVERRIDE:?Set WANDB_API_KEY_OVERRIDE}"
export HF_LEROBOT_HOME=/scratch/yp2841/.huggingface/lerobot
export NUM_GPUS=4
export NUM_WORKERS=16
export BATCH_SIZE=32
export NUM_TRAIN_STEPS=30000
export SAVE_INTERVAL=1000
export KEEP_PERIOD=5000

export DATASET_REPO_ID=glbreeze/libero_object_cam_v3
export NORM_ASSET_ID=glbreeze/libero_object_cam_v3
export EXP_NAME=pi0_libero_object_cam_v3_prope_ray_view_distill_fullres_twostage_v1_stage2
export PYTORCH_WEIGHT_PATH_OVERRIDE="${STAGE1_WEIGHT_PATH}"

exec bash "${REPO_ROOT}/scripts/sbatch/train_pi0_libero_object_ft.sbatch"

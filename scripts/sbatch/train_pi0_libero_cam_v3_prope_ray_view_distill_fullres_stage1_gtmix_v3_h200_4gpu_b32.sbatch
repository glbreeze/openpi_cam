#!/bin/bash
#SBATCH --job-name=pi0_s1_gtmix_v3_h200
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_libero_object_ft/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_libero_object_ft/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --account=torch_pr_637_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:h200:4
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

export USE_CAM=${USE_CAM:-true}
export CONFIG_NAME=${CONFIG_NAME:-pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage1_gtmix}
export TRAIN_RECIPE=${TRAIN_RECIPE:-single_stage}
export WANDB_ENABLED=${WANDB_ENABLED:-true}
export WANDB_PROJECT=${WANDB_PROJECT:-pi0_libero_cam}
export WANDB_ENTITY=${WANDB_ENTITY:-NYU-robotics}
export WANDB_API_KEY_OVERRIDE=${WANDB_API_KEY_OVERRIDE:-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}
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-/scratch/yp2841/.huggingface/lerobot}
export NUM_GPUS=${NUM_GPUS:-4}
export NUM_WORKERS=${NUM_WORKERS:-16}
export BATCH_SIZE=${BATCH_SIZE:-32}
export NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-5000}
export SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
export KEEP_PERIOD=${KEEP_PERIOD:-5000}

export DATASET_REPO_ID=${DATASET_REPO_ID:-glbreeze/libero_object_cam_v3}
export NORM_ASSET_ID=${NORM_ASSET_ID:-glbreeze/libero_object_cam_v3}
export PI3X_TARGETS_ROOT_OVERRIDE=${PI3X_TARGETS_ROOT_OVERRIDE:-/scratch/yp2841/geometry-vla/.cache/openpi/pi3x_targets_224/libero_object_cam_v3}
export GT_POINT_TARGETS_ROOT_OVERRIDE=${GT_POINT_TARGETS_ROOT_OVERRIDE:-/scratch/yp2841/geometry-vla/.cache/openpi/gt_point_targets_224/libero_object_cam_v3_aligned}
export EXP_NAME=${EXP_NAME:-pi0_libero_object_cam_v3_prope_ray_view_distill_fullres_stage1_gtmix_v3_h200_4gpu_b32_5k}

exec bash "${REPO_ROOT}/scripts/sbatch/train_pi0_libero_object_ft.sbatch"

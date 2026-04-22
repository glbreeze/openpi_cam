#!/bin/bash
#SBATCH --job-name=pi0_obj_simple_twostage
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_libero_object_ft/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_libero_object_ft/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=16:00:00
#SBATCH --account=torch_pr_676_tandon_advanced
#SBATCH --partition=h100_tandon,h200_tandon
#SBATCH --gres=gpu:4

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

export USE_CAM=true
export CONFIG_NAME=pi0_libero_cam_pytorch_full_finetune
export CROSS_VIEW_TYPE=simple
export TRAIN_RECIPE=simple_two_stage
export MODEL_POSE_ENC_TYPE=absolute_pose
export WANDB_ENABLED=true
export WANDB_PROJECT=openpi_cam_libero_object
export NUM_GPUS=4
export NUM_WORKERS=16
export BATCH_SIZE=32

export EXP_NAME=pi0_libero_object_camtrain_o00_02_03_05_06_09_10_simple_60k_twostage_v1
export DATASET_REPO_ID=glbreeze/libero_object_cam_train_o00_02_03_05_06_09_10
export NORM_ASSET_ID=glbreeze/libero_object_cam_train_o00_02_03_05_06_09_10

export SIMPLE_STAGE1_STEPS=5000
export SIMPLE_STAGE2_STEPS=55000
export SIMPLE_STAGE1_WARMUP_STEPS=500
export SIMPLE_STAGE1_PEAK_LR=1e-4
export SIMPLE_STAGE1_DECAY_STEPS=5000
export SIMPLE_STAGE1_END_LR=1e-5

export SIMPLE_STAGE2_WARMUP_STEPS=1000
export SIMPLE_STAGE2_PEAK_LR=2.5e-5
export SIMPLE_STAGE2_DECAY_STEPS=55000
export SIMPLE_STAGE2_END_LR=2.5e-6
export SIMPLE_NEW_MODULE_PREFIXES=paligemma_with_expert.cam_pose_encoder,paligemma_with_expert.view_embedding,paligemma_with_expert.cross_view_fusion,paligemma_with_expert.ray_embed
export SIMPLE_STAGE2_NEW_MODULE_LR_MULTIPLIER=4.0

exec bash "${REPO_ROOT}/scripts/sbatch/train_pi0_libero_object_ft.sbatch"

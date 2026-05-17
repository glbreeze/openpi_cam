#!/bin/bash
#SBATCH --job-name=pi0_cam_v2_s2_h100
#SBATCH --output=/scratch/lg154/Research/openpi/logs/pi0_libero_cam_v2/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/openpi/logs/pi0_libero_cam_v2/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:2
#SBATCH --export=ALL

set -euo pipefail

export REPO_ROOT=/scratch/lg154/Research/openpi

# --- env consumed by train_pi0_libero.sbatch ---
export HF_LEROBOT_HOME=/scratch/lg154/cache/lerobot
export CONFIG_NAME=pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage2

export DATASET_REPO_ID=glbreeze/libero_cam_v2
export NORM_ASSET_ID=glbreeze/libero_cam_v2
export PI3X_TARGETS_ROOT_OVERRIDE=/scratch/lg154/.cache/openpi/pi3x_targets_224/libero_cam_v2

# Stage 1 must have finished; point at its final-step checkpoint. The
# `<config>/<exp_name>/<step>` layout is the default for `train_pi0_libero.sbatch`
# (CHECKPOINT_BASE_DIR=${REPO_ROOT}/checkpoints).
STAGE1_CONFIG=pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage1
STAGE1_EXP_NAME=pi0_libero_cam_v2_prope_ray_view_distill_fullres_stage1_h100_2gpu_b16_lg
STAGE1_FINAL_STEP=5000
export PYTORCH_WEIGHT_PATH_OVERRIDE="${REPO_ROOT}/checkpoints/${STAGE1_CONFIG}/${STAGE1_EXP_NAME}/${STAGE1_FINAL_STEP}"

export NUM_GPUS=2
export NUM_WORKERS=16
export BATCH_SIZE=16
export NUM_TRAIN_STEPS=30000
export SAVE_INTERVAL=1000
export KEEP_PERIOD=5000

# --- wandb ---
export WANDB_ENABLED=true
export WANDB_PROJECT=pi0_libero_lg
export WANDB_ENTITY=NYU-robotics
export WANDB_API_KEY_OVERRIDE="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"

export EXP_NAME=pi0_libero_cam_v2_prope_ray_view_distill_fullres_stage2_h100_2gpu_b16_lg

# Pre-flight: surface a clear error if Stage 1's checkpoint isn't there yet
# (otherwise the inner runner's missing-weights check would fire deep in setup).
if [[ ! -f "${PYTORCH_WEIGHT_PATH_OVERRIDE}/model.safetensors" ]]; then
  echo "Stage 1 checkpoint not found at: ${PYTORCH_WEIGHT_PATH_OVERRIDE}" >&2
  echo "Run stage 1 first (scripts/sbatch_lg/train_pi0_libero_cam_v2_prope_ray_view_distill_fullres_stage1.sbatch)" >&2
  echo "or update STAGE1_EXP_NAME / STAGE1_FINAL_STEP in this wrapper if the names differ." >&2
  exit 1
fi

exec bash "${REPO_ROOT}/scripts/sbatch_lg/train_pi0_libero.sbatch"

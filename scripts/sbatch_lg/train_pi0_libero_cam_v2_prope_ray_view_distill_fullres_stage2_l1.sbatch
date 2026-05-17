#!/bin/bash
#SBATCH --job-name=pi0_cam_v2_s2l1_h100
#SBATCH --output=/scratch/lg154/Research/openpi/logs/pi0_libero_cam_v2/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/openpi/logs/pi0_libero_cam_v2/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --gres=gpu:2
#SBATCH --export=ALL

set -euo pipefail

export REPO_ROOT=/scratch/lg154/Research/openpi

# Stage 2 paired with the L1-aux Stage 1 run (`..._stage1_l1`). Same Stage 2
# recipe as `..._stage2.sbatch`, with two differences:
#   1. PYTORCH_WEIGHT_PATH_OVERRIDE points at the L1 Stage 1 final checkpoint.
#   2. EXTRA_TRAIN_ARGS forces the aux loss to L1 (consistent with Stage 1).

# --- env consumed by train_pi0_libero.sbatch ---
export HF_LEROBOT_HOME=/scratch/lg154/cache/lerobot
export CONFIG_NAME=pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage2

export DATASET_REPO_ID=glbreeze/libero_cam_v2
export NORM_ASSET_ID=glbreeze/libero_cam_v2
export PI3X_TARGETS_ROOT_OVERRIDE=/scratch/lg154/.cache/openpi/pi3x_targets_224/libero_cam_v2

# Stage 1 (L1) must have finished; point at its final-step checkpoint.
STAGE1_CONFIG=pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage1
STAGE1_EXP_NAME=pi0_libero_cam_v2_prope_ray_view_distill_fullres_stage1_l1
STAGE1_FINAL_STEP=5000
export PYTORCH_WEIGHT_PATH_OVERRIDE="${REPO_ROOT}/checkpoints/${STAGE1_CONFIG}/${STAGE1_EXP_NAME}/${STAGE1_FINAL_STEP}"

# Keep the aux loss in L1 mode for stage 2 (matches stage 1's training signal).
export EXTRA_TRAIN_ARGS="--model.aux_point_head.legacy_conf_loss_order 1"

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

export EXP_NAME=pi0_libero_cam_v2_prope_ray_view_distill_fullres_stage2_l1

# Pre-flight: surface a clear error if Stage 1's checkpoint isn't there yet.
if [[ ! -f "${PYTORCH_WEIGHT_PATH_OVERRIDE}/model.safetensors" ]]; then
  echo "Stage 1 (L1) checkpoint not found at: ${PYTORCH_WEIGHT_PATH_OVERRIDE}" >&2
  echo "Run stage 1 first (scripts/sbatch_lg/train_pi0_libero_cam_v2_prope_ray_view_distill_fullres_stage1_l1.sbatch)" >&2
  echo "or update STAGE1_EXP_NAME / STAGE1_FINAL_STEP in this wrapper if the names differ." >&2
  exit 1
fi

exec bash "${REPO_ROOT}/scripts/sbatch_lg/train_pi0_libero.sbatch"

#!/bin/bash
#SBATCH --job-name=pi0_cam_v2_s1l1_h100
#SBATCH --output=/scratch/lg154/Research/openpi/logs/pi0_libero_cam_v2/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/openpi/logs/pi0_libero_cam_v2/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:2
#SBATCH --export=ALL

set -euo pipefail

export REPO_ROOT=/scratch/lg154/Research/openpi

# Same Stage 1 recipe as `..._stage1.sbatch`, but the aux point-distillation loss
# uses L1 instead of MSE on xy/logz. The model config is identical -- the
# `legacy_conf_loss_order=1` override is passed via EXTRA_TRAIN_ARGS so we don't
# need to maintain a duplicate TrainConfig.

# --- env consumed by train_pi0_libero.sbatch ---
export HF_LEROBOT_HOME=/scratch/lg154/cache/lerobot
export CONFIG_NAME=pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage1

export DATASET_REPO_ID=glbreeze/libero_cam_v2
export NORM_ASSET_ID=glbreeze/libero_cam_v2
export PI3X_TARGETS_ROOT_OVERRIDE=/scratch/lg154/.cache/openpi/pi3x_targets_224/libero_cam_v2
export PYTORCH_WEIGHT_PATH_OVERRIDE=/scratch/lg154/Research/openpi/pi0_base_ckpt/pi0_base_ckpt_torch

# L1 aux loss instead of MSE (1 = L1, 2 = MSE; default in the dataclass is 2).
export EXTRA_TRAIN_ARGS="--model.aux_point_head.legacy_conf_loss_order 1"

export NUM_GPUS=2
export NUM_WORKERS=16
export BATCH_SIZE=16
export NUM_TRAIN_STEPS=5000
export SAVE_INTERVAL=1000
export KEEP_PERIOD=1000

# --- wandb (same project/entity as the MSE stage 1) ---
export WANDB_ENABLED=true
export WANDB_PROJECT=pi0_libero_lg
export WANDB_ENTITY=NYU-robotics
export WANDB_API_KEY_OVERRIDE="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"

export EXP_NAME=pi0_libero_cam_v2_prope_ray_view_distill_fullres_stage1_l1

exec bash "${REPO_ROOT}/scripts/sbatch_lg/train_pi0_libero.sbatch"

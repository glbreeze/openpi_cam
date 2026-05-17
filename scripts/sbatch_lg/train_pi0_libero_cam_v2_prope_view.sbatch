#!/bin/bash
#SBATCH --job-name=pi0_cam_v2_prope_view
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
# Ablation A5 (drop ray_embed): same setup as pi0_libero_cam_v2_prope_ray_view
# but ray_enc_type=False. PRoPE + standard cross-view fusion stay on; no aux
# distillation. Single-stage 30k from pi0_base, matching the geom-only budget.
export HF_LEROBOT_HOME=/scratch/lg154/cache/lerobot
export CONFIG_NAME=pi0_libero_cam_pytorch_prope_view

export DATASET_REPO_ID=glbreeze/libero_cam_v2
export NORM_ASSET_ID=glbreeze/libero_cam_v2
export PYTORCH_WEIGHT_PATH_OVERRIDE=/scratch/lg154/Research/openpi/pi0_base_ckpt/pi0_base_ckpt_torch
# NOTE: PI3X_TARGETS_ROOT_OVERRIDE intentionally unset -> aux distillation is off.

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

export EXP_NAME=pi0_libero_cam_v2_prope_view

exec bash "${REPO_ROOT}/scripts/sbatch_lg/train_pi0_libero.sbatch"

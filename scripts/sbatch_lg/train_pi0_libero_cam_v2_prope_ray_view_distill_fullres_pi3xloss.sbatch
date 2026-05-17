#!/bin/bash
#SBATCH --job-name=pi0_cam_v2_distill_pi3xloss
#SBATCH --output=/scratch/lg154/Research/openpi/logs/pi0_libero_cam_v2/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/openpi/logs/pi0_libero_cam_v2/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:2
#SBATCH --export=ALL

set -euo pipefail

export REPO_ROOT=/scratch/lg154/Research/openpi

# --- env consumed by train_pi0_libero.sbatch ---
# Ablation A4 (loss-formulation): single-stage 30k full-res Pi3X distillation
# using the SCALE-ALIGNED pi3x_local_pointmap loss. The existing
# pi0_libero_cam_pytorch_prope_ray_view_distill_fullres config already defaults
# to loss_type="pi3x_local_pointmap" (see AuxPointHeadConfig in
# src/openpi/models/point_head_config.py:24-27), so no per-flag override is
# needed -- this is the unmodified single-stage recipe.
#
# Counterfactual to the user's existing legacy_conf_mse stage1+stage2 runs
# (stage2 default = MSE, stage2_l1 = L1 inside the legacy path). Isolates the
# value of the scale-aligned local-pointmap loss vs. the legacy gated loss.
#
# Same architecture as cam_v2_prope_ray_view (R + P + X all on) + aux head with
# loss_weight=0.05, output_resolution=224. Init from pi0_base (no stage1
# warmup). Compute budget matches the geom-only and stage2 runs at 30k.
export HF_LEROBOT_HOME=/scratch/lg154/cache/lerobot
export CONFIG_NAME=pi0_libero_cam_pytorch_prope_ray_view_distill_fullres

export DATASET_REPO_ID=glbreeze/libero_cam_v2
export NORM_ASSET_ID=glbreeze/libero_cam_v2
export PYTORCH_WEIGHT_PATH_OVERRIDE=/scratch/lg154/Research/openpi/pi0_base_ckpt/pi0_base_ckpt_torch
export PI3X_TARGETS_ROOT_OVERRIDE=/scratch/lg154/.cache/openpi/pi3x_targets_224/libero_cam_v2

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

export EXP_NAME=pi0_libero_cam_v2_prope_ray_view_distill_fullres_pi3xloss

exec bash "${REPO_ROOT}/scripts/sbatch_lg/train_pi0_libero.sbatch"

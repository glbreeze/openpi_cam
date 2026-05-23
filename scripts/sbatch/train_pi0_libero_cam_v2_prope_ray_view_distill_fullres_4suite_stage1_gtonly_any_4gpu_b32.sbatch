#!/bin/bash
#SBATCH --job-name=pi0_4s_gt_s1
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_libero_object_ft/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_libero_object_ft/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon
#SBATCH --gres=gpu:4
#SBATCH --export=ALL

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/scratch/yp2841/geometry-vla/openpi_cam}

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/scratch/yp2841/cache/lerobot}"
export USE_CAM=true
export CONFIG_NAME=pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_4suite_stage1_gtonly
export TRAIN_RECIPE=single_stage
export WANDB_ENABLED=true
export WANDB_PROJECT=pi0_libero_cam
export WANDB_ENTITY=NYU-robotics
export NUM_GPUS=4
export NUM_WORKERS=16
export BATCH_SIZE=32
export NUM_TRAIN_STEPS=5000
export SAVE_INTERVAL=1000
export KEEP_PERIOD=5000

export DATASET_REPO_ID=glbreeze/libero_cam_v2
export NORM_ASSET_ID=glbreeze/libero_cam_v2
export GT_POINT_TARGETS_ROOT_OVERRIDE=/scratch/yp2841/geometry-vla/.cache/openpi/gt_point_targets_224/libero_cam_v2_aligned
export EXP_NAME=pi0_libero_cam_v2_prope_ray_view_distill_fullres_4suite_stage1_gtonly_769_any_4gpu_b32_5k

exec bash "${REPO_ROOT}/scripts/sbatch/train_pi0_libero_object_ft.sbatch"

#!/bin/bash
#SBATCH --job-name=pi0_robotwin_s1
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_robotwin_s1/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_robotwin_s1/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_637_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:h200:4
#SBATCH --export=ALL
#
# Stage 1 (5,000 steps, freeze backbone) — PI3X-ONLY distillation, single-block
# fg cross-view fusion (mirrors `pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage1`).
#
# This is the vanilla recipe (no simulator-GT mixed dual loss). For the
# fgfg gtdual variant, use train_pi0_robotwin_s1_gtdual_h200.sbatch instead.
#
# Submit per-task with:
#   sbatch --export=ALL,TASK=handover_block scripts/sbatch/train_pi0_robotwin_s1_h200.sbatch
#
# Available TASK values:
#   beat_block_hammer  handover_block  stack_blocks_three
#   place_container_plate  place_dual_shoes  lift_pot

set -euo pipefail
TASK=${TASK:-handover_block}

REPO_ROOT=${REPO_ROOT:-/scratch/yp2841/geometry-vla/openpi_cam}
GEO_ROOT=${GEO_ROOT:-/scratch/yp2841/geometry-vla}
TASK_CONFIG=${TASK_CONFIG:-demo_clean_camaware}
REPO_ID="robotwin/${TASK}_${TASK_CONFIG}_50"
SLUG="robotwin_${TASK}_${TASK_CONFIG}_50"

mkdir -p ${GEO_ROOT}/openpi_cam/log/pi0_robotwin_s1
cd ${REPO_ROOT}
source ${REPO_ROOT}/scripts/env/activate_env.sh

export HF_LEROBOT_HOME=${GEO_ROOT}
export OPENPI_PI0_BASE_DIR=${GEO_ROOT}/pi0_base
export OPENPI_CACHE_DIR=${GEO_ROOT}/.cache/openpi
export OPENPI_DATA_HOME=${OPENPI_CACHE_DIR}
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=${WANDB_PROJECT:-pi0_robotwin_cam}

echo "===== ROBOTWIN STAGE 1 (Pi3X-only) ====="
echo "task         : ${TASK}"
echo "config       : pi0_robotwin_cam_prope_ray_view_distill_fullres_stage1"
echo "repo_id      : ${REPO_ID}"
echo "pi3x_root    : ${OPENPI_CACHE_DIR}/pi3x_targets_224/${SLUG}"
echo "n gpus       : ${SLURM_GPUS_ON_NODE:-4}"
hostname; python -V; date

uv run torchrun --standalone --nnodes=1 --nproc_per_node=${SLURM_GPUS_ON_NODE:-4} \
    scripts/train_pytorch.py \
    pi0_robotwin_cam_prope_ray_view_distill_fullres_stage1 \
    --exp_name=stage1_${TASK} \
    --batch_size=${BATCH_SIZE:-32} \
    --num_workers=${NUM_WORKERS:-16} \
    --num_train_steps=${NUM_TRAIN_STEPS:-5000} \
    --save_interval=${SAVE_INTERVAL:-1000} \
    --keep_period=${KEEP_PERIOD:-5000} \
    --data.repo_id ${REPO_ID} \
    --data.assets.assets_dir ${GEO_ROOT}/pi0_libero \
    --data.assets.asset_id ${REPO_ID} \
    --data.pi3x_targets_root ${OPENPI_CACHE_DIR}/pi3x_targets_224/${SLUG}

#!/bin/bash
#SBATCH --job-name=pi0_robotwin_s2_gtdual
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_robotwin_s2/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_robotwin_s2/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=18:00:00
#SBATCH --account=torch_pr_637_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:h200:4
#SBATCH --export=ALL
#
# Stage 2 (30,000 steps, unfreeze, warm-start from Stage 1 ckpt) for the
# cam-aware Pi3X mixed-GT distillation recipe on RoboTwin2.0.
#
# Submit after stage 1 succeeds:
#   S1=$(sbatch --parsable --export=ALL,TASK=handover_block scripts/sbatch/train_pi0_robotwin_s1_gtdual_h200.sbatch)
#   sbatch --dependency=afterok:${S1} --export=ALL,TASK=handover_block scripts/sbatch/train_pi0_robotwin_s2_gtdual_h200.sbatch

set -euo pipefail
TASK=${TASK:-handover_block}

REPO_ROOT=${REPO_ROOT:-/scratch/yp2841/geometry-vla/openpi_cam}
GEO_ROOT=${GEO_ROOT:-/scratch/yp2841/geometry-vla}
TASK_CONFIG=${TASK_CONFIG:-demo_clean_camaware}
REPO_ID="robotwin/${TASK}_${TASK_CONFIG}_50"
SLUG="robotwin_${TASK}_${TASK_CONFIG}_50"

S1_CKPT=${S1_CKPT:-${REPO_ROOT}/checkpoints/pi0_robotwin_cam_prope_ray_view_distill_fullres_stage1_gtdual/stage1_${TASK}/5000}

if [[ ! -d "${S1_CKPT}" ]]; then
    echo "ERROR: Stage 1 checkpoint not found at ${S1_CKPT}" >&2
    echo "Override with --export=ALL,TASK=...,S1_CKPT=/path/to/ckpt" >&2
    exit 1
fi

mkdir -p ${GEO_ROOT}/openpi_cam/log/pi0_robotwin_s2
cd ${REPO_ROOT}
source ${REPO_ROOT}/scripts/env/activate_env.sh

export HF_LEROBOT_HOME=${GEO_ROOT}
export OPENPI_PI0_BASE_DIR=${GEO_ROOT}/pi0_base
export OPENPI_CACHE_DIR=${GEO_ROOT}/.cache/openpi
export OPENPI_DATA_HOME=${OPENPI_CACHE_DIR}
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=${WANDB_PROJECT:-pi0_robotwin_cam}

echo "===== ROBOTWIN STAGE 2 GTDUAL ====="
echo "task         : ${TASK}"
echo "config       : pi0_robotwin_cam_prope_ray_view_distill_fullres_stage2_gtdual"
echo "repo_id      : ${REPO_ID}"
echo "S1_CKPT      : ${S1_CKPT}"
echo "n gpus       : ${SLURM_GPUS_ON_NODE:-4}"
hostname; python -V; date

EXTRA_ARGS=()
if [[ "${RESUME:-false}" == "true" ]]; then
    EXTRA_ARGS+=(--resume)
fi

uv run torchrun --standalone --nnodes=1 --nproc_per_node=${SLURM_GPUS_ON_NODE:-4} \
    scripts/train_pytorch.py \
    pi0_robotwin_cam_prope_ray_view_distill_fullres_stage2_gtdual \
    --exp_name=stage2_${TASK} \
    --batch_size=${BATCH_SIZE:-32} \
    --num_workers=${NUM_WORKERS:-16} \
    --num_train_steps=${NUM_TRAIN_STEPS:-30000} \
    --save_interval=${SAVE_INTERVAL:-2000} \
    --keep_period=${KEEP_PERIOD:-10000} \
    --pytorch_weight_path=${S1_CKPT} \
    --data.repo_id ${REPO_ID} \
    --data.assets.assets_dir ${GEO_ROOT}/pi0_libero \
    --data.assets.asset_id ${REPO_ID} \
    --data.pi3x_targets_root  ${OPENPI_CACHE_DIR}/pi3x_targets_224/${SLUG} \
    --data.gt_point_targets_root ${OPENPI_CACHE_DIR}/gt_point_targets_224/${SLUG} \
    "${EXTRA_ARGS[@]}"

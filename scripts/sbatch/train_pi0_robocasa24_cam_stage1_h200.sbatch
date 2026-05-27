#!/bin/bash
#SBATCH --job-name=pi0_rc24_s1
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_robocasa24_s1/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_robocasa24_s1/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_676_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --gres=gpu:h100:1
#SBATCH --export=ALL
#
# RoboCasa24 all-task cam-aware Pi0 stage 1.
# Freezes the base model and trains cross-view fusion, ray embedding, and the
# auxiliary point head with the LIBERO 4suite full-res Pi3X distillation recipe.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/scratch/yp2841/geometry-vla/openpi_cam}
GEO_ROOT=${GEO_ROOT:-/scratch/yp2841/geometry-vla}
EXP_NAME=${EXP_NAME:-stage1_all24_libero4suite_pi3x224}

mkdir -p "${REPO_ROOT}/log/pi0_robocasa24_s1"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/env/activate_env.sh"

export OPENPI_GEO_ROOT=${GEO_ROOT}
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-${GEO_ROOT}}
export OPENPI_PI0_BASE_DIR=${OPENPI_PI0_BASE_DIR:-${GEO_ROOT}/pi0_base}
export OPENPI_CACHE_DIR=${OPENPI_CACHE_DIR:-${GEO_ROOT}/.cache/openpi}
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-${OPENPI_CACHE_DIR}}

export ROBOCASA24_LEROBOT_ROOT=${ROBOCASA24_LEROBOT_ROOT:-/scratch/yz11445/robocasa24/all24_human_camaware}
export ROBOCASA24_ASSETS_DIR=${ROBOCASA24_ASSETS_DIR:-/scratch/yz11445/pi0_robocasa24}
export ROBOCASA24_BUNDLE=${ROBOCASA24_BUNDLE:-/scratch/yz11445/robocasa-human50}
export ROBOCASA24_PI3X_TARGETS_ROOT=${ROBOCASA24_PI3X_TARGETS_ROOT:-/scratch/yz11445/pi3x_targets_224/robocasa24_all24_human_camaware}

if [[ ! -d "${ROBOCASA24_PI3X_TARGETS_ROOT}" ]]; then
    echo "ERROR: full-res Pi3X target cache not found at ${ROBOCASA24_PI3X_TARGETS_ROOT}" >&2
    echo "This LIBERO-4suite-matched config expects 224x224 Pi3X targets." >&2
    exit 1
fi

export TMPDIR=${TMPDIR:-/scratch/yp2841/tmp}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/scratch/yp2841/.cache}
export HF_HOME=${HF_HOME:-/scratch/yp2841/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export TORCH_HOME=${TORCH_HOME:-/scratch/yp2841/.cache/torch}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/scratch/yp2841/.cache/pip}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/scratch/yp2841/tmp/matplotlib}
mkdir -p "${TMPDIR}" "${HF_DATASETS_CACHE}" "${TORCH_HOME}" "${PIP_CACHE_DIR}" "${MPLCONFIGDIR}"
unset TRANSFORMERS_CACHE

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export OPENPI_MICRO_BATCH_SIZE=${OPENPI_MICRO_BATCH_SIZE:-64}
export OPENPI_DATALOADER_PREFETCH_FACTOR=${OPENPI_DATALOADER_PREFETCH_FACTOR:-4}
export WANDB_MODE=${WANDB_MODE:-disabled}
export WANDB_DISABLED=${WANDB_DISABLED:-true}
export WANDB_PROJECT=${WANDB_PROJECT:-pi0_robocasa24_cam}
export WANDB_ENTITY=${WANDB_ENTITY:-NYU-robotics}
export WANDB_DIR=${WANDB_DIR:-${REPO_ROOT}/log/wandb}
mkdir -p "${WANDB_DIR}"

echo "===== ROBOCASA24 CAM-AWARE STAGE 1 ====="
echo "config        : pi0_robocasa24_all24_cam_prope_ray_view_distill_fullres_stage1"
echo "exp_name      : ${EXP_NAME}"
echo "lerobot_root  : ${ROBOCASA24_LEROBOT_ROOT}"
echo "assets_dir    : ${ROBOCASA24_ASSETS_DIR}"
echo "pi3x_root     : ${ROBOCASA24_PI3X_TARGETS_ROOT}"
echo "gt_root       : <disabled>"
echo "n gpus        : ${SLURM_GPUS_ON_NODE:-4}"
echo "micro batch   : ${OPENPI_MICRO_BATCH_SIZE}"
echo "num workers   : ${NUM_WORKERS:-16}"
echo "prefetch      : ${OPENPI_DATALOADER_PREFETCH_FACTOR}"
hostname
python -V
date

EXTRA_ARGS=()
if [[ "${RESUME:-false}" == "true" ]]; then
    EXTRA_ARGS+=(--resume)
fi

uv run torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-4}} \
    scripts/train_pytorch.py \
    pi0_robocasa24_all24_cam_prope_ray_view_distill_fullres_stage1 \
    --exp_name=${EXP_NAME} \
    --batch_size=${BATCH_SIZE:-64} \
    --num_workers=${NUM_WORKERS:-16} \
    --num_train_steps=${NUM_TRAIN_STEPS:-5000} \
    --save_interval=${SAVE_INTERVAL:-200} \
    --keep_period=${KEEP_PERIOD:-5000} \
    "${EXTRA_ARGS[@]}"

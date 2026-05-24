#!/bin/bash
#SBATCH --job-name=pi0_rc24_s2
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_robocasa24_s2/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_robocasa24_s2/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --account=torch_pr_637_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:h200:4
#SBATCH --export=ALL
#
# RoboCasa24 all-task cam-aware Pi0 stage 2.
# Warm-starts from the LIBERO-4suite-matched stage-1 checkpoint and trains the
# full model with full-res Pi3X distillation.
#
# Typical chained submit:
#   S1=$(sbatch --parsable scripts/sbatch/train_pi0_robocasa24_cam_stage1_h200.sbatch)
#   sbatch --dependency=afterok:${S1} scripts/sbatch/train_pi0_robocasa24_cam_stage2_h200.sbatch

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/scratch/yp2841/geometry-vla/openpi_cam}
GEO_ROOT=${GEO_ROOT:-/scratch/yp2841/geometry-vla}
S1_EXP_NAME=${S1_EXP_NAME:-stage1_all24_libero4suite_pi3x224}
EXP_NAME=${EXP_NAME:-stage2_all24_libero4suite_pi3x224}
S1_CKPT=${S1_CKPT:-${REPO_ROOT}/checkpoints/pi0_robocasa24_all24_cam_prope_ray_view_distill_fullres_stage1/${S1_EXP_NAME}/5000}

if [[ ! -d "${S1_CKPT}" ]]; then
    echo "ERROR: Stage 1 checkpoint not found at ${S1_CKPT}" >&2
    echo "Override with --export=ALL,S1_CKPT=/path/to/ckpt if needed." >&2
    exit 1
fi

mkdir -p "${REPO_ROOT}/log/pi0_robocasa24_s2"
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
export ROBOCASA24_PI3X_TARGETS_ROOT=${ROBOCASA24_PI3X_TARGETS_ROOT:-${ROBOCASA24_BUNDLE}/pi3x_targets_224}

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
export WANDB_PROJECT=${WANDB_PROJECT:-pi0_robocasa24_cam}
export WANDB_ENTITY=${WANDB_ENTITY:-NYU-robotics}
export WANDB_DIR=${WANDB_DIR:-${REPO_ROOT}/log/wandb}
mkdir -p "${WANDB_DIR}"

echo "===== ROBOCASA24 CAM-AWARE STAGE 2 ====="
echo "config        : pi0_robocasa24_all24_cam_prope_ray_view_distill_fullres_stage2"
echo "exp_name      : ${EXP_NAME}"
echo "S1_CKPT       : ${S1_CKPT}"
echo "lerobot_root  : ${ROBOCASA24_LEROBOT_ROOT}"
echo "assets_dir    : ${ROBOCASA24_ASSETS_DIR}"
echo "pi3x_root     : ${ROBOCASA24_PI3X_TARGETS_ROOT}"
echo "gt_root       : <disabled>"
echo "n gpus        : ${SLURM_GPUS_ON_NODE:-4}"
hostname
python -V
date

EXTRA_ARGS=()
if [[ "${RESUME:-false}" == "true" ]]; then
    EXTRA_ARGS+=(--resume)
fi

uv run torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-4}} \
    scripts/train_pytorch.py \
    pi0_robocasa24_all24_cam_prope_ray_view_distill_fullres_stage2 \
    --exp_name=${EXP_NAME} \
    --batch_size=${BATCH_SIZE:-32} \
    --num_workers=${NUM_WORKERS:-16} \
    --num_train_steps=${NUM_TRAIN_STEPS:-30000} \
    --save_interval=${SAVE_INTERVAL:-2000} \
    --keep_period=${KEEP_PERIOD:-10000} \
    --pytorch_weight_path=${S1_CKPT} \
    "${EXTRA_ARGS[@]}"

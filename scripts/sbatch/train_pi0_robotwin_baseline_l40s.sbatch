#!/bin/bash
#SBATCH --job-name=pi0_robotwin_baseline
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_robotwin_baseline/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_robotwin_baseline/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=18:00:00
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:4
#SBATCH --export=ALL
#
# RoboTwin cam-aware baseline for A/B against the geometry distillation recipe.
# Uses the same task/data/base init/action schedule as Stage 2, but no PRoPE,
# no ray encoder, no cross-view fusion, no aux point head, and no extrinsics.

set -euo pipefail
TASK=${TASK:-handover_block}

REPO_ROOT=${REPO_ROOT:-/scratch/yp2841/geometry-vla/openpi_cam}
GEO_ROOT=${GEO_ROOT:-/scratch/yp2841/geometry-vla}
TASK_CONFIG=${TASK_CONFIG:-demo_clean_camaware}
REPO_ID="robotwin/${TASK}_${TASK_CONFIG}_50"
# Keep the default away from older baseline_${TASK} checkpoints, which were
# trained before the RoboTwin adapt_to_pi=False fix.
EXP_NAME=${EXP_NAME:-baseline_${TASK}_adaptfix}

mkdir -p ${GEO_ROOT}/openpi_cam/log/pi0_robotwin_baseline
cd ${REPO_ROOT}
source ${REPO_ROOT}/scripts/env/activate_env.sh

export HF_LEROBOT_HOME=${GEO_ROOT}
export OPENPI_PI0_BASE_DIR=${GEO_ROOT}/pi0_base
export OPENPI_CACHE_DIR=${GEO_ROOT}/.cache/openpi
export OPENPI_DATA_HOME=${OPENPI_CACHE_DIR}
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=${WANDB_PROJECT:-pi0_robotwin_cam}
export WANDB_ENTITY=${WANDB_ENTITY:-NYU-robotics}
export WANDB_API_KEY_OVERRIDE=${WANDB_API_KEY_OVERRIDE:-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}
export WANDB_API_KEY="${WANDB_API_KEY_OVERRIDE}"
export WANDB_DIR=${WANDB_DIR:-${GEO_ROOT}/openpi_cam/log/wandb}
mkdir -p "${WANDB_DIR}"

echo "===== ROBOTWIN BASELINE ====="
echo "task         : ${TASK}"
echo "config       : pi0_robotwin_cam_baseline"
echo "repo_id      : ${REPO_ID}"
echo "exp_name     : ${EXP_NAME}"
echo "adapt_to_pi  : false (LeRobotRobotwinCamDataConfig default)"
echo "n gpus       : ${SLURM_GPUS_ON_NODE:-4}"
hostname; python -V; date

EXTRA_ARGS=()
if [[ "${RESUME:-false}" == "true" ]]; then
    EXTRA_ARGS+=(--resume)
fi

uv run torchrun --standalone --nnodes=1 --nproc_per_node=${SLURM_GPUS_ON_NODE:-4} \
    scripts/train_pytorch.py \
    pi0_robotwin_cam_baseline \
    --exp_name=${EXP_NAME} \
    --batch_size=${BATCH_SIZE:-32} \
    --num_workers=${NUM_WORKERS:-4} \
    --num_train_steps=${NUM_TRAIN_STEPS:-30000} \
    --save_interval=${SAVE_INTERVAL:-2000} \
    --keep_period=${KEEP_PERIOD:-10000} \
    --data.repo_id ${REPO_ID} \
    --data.assets.assets_dir ${GEO_ROOT}/pi0_libero \
    --data.assets.asset_id ${REPO_ID} \
    "${EXTRA_ARGS[@]}"

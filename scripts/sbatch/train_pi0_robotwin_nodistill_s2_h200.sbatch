#!/bin/bash
#SBATCH --job-name=pi0_rw_nodist_s2
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_robotwin_nodistill_s2/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_robotwin_nodistill_s2/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=18:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:h200:4
#SBATCH --export=ALL

set -euo pipefail
TASK=${TASK:-handover_block}

REPO_ROOT=${REPO_ROOT:-/scratch/yp2841/geometry-vla/openpi_cam}
GEO_ROOT=${GEO_ROOT:-/scratch/yp2841/geometry-vla}
TASK_CONFIG=${TASK_CONFIG:-demo_clean_camaware}
REPO_ID="robotwin/${TASK}_${TASK_CONFIG}_50"
EXP_NAME=${EXP_NAME:-stage2_${TASK}_nodistill_fg_b64_20260515}
S1_EXP_NAME=${S1_EXP_NAME:-stage1_${TASK}_nodistill_fg_20260515}
S1_CKPT=${S1_CKPT:-${REPO_ROOT}/checkpoints/pi0_robotwin_cam_prope_ray_view_nodistill_stage1/${S1_EXP_NAME}/5000}

if [[ ! -d "${S1_CKPT}" ]]; then
    echo "ERROR: Stage 1 checkpoint not found at ${S1_CKPT}" >&2
    echo "Override with --export=ALL,TASK=...,S1_CKPT=/path/to/ckpt" >&2
    exit 1
fi

mkdir -p ${GEO_ROOT}/openpi_cam/log/pi0_robotwin_nodistill_s2
cd ${REPO_ROOT}
source ${REPO_ROOT}/scripts/env/activate_env.sh

export HF_LEROBOT_HOME=${GEO_ROOT}
export OPENPI_PI0_BASE_DIR=${GEO_ROOT}/pi0_base
export OPENPI_CACHE_DIR=${GEO_ROOT}/.cache/openpi
export OPENPI_DATA_HOME=${OPENPI_CACHE_DIR}
export MPLCONFIGDIR=${GEO_ROOT}/tmp/matplotlib
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=${WANDB_PROJECT:-pi0_robotwin_cam}
export WANDB_ENTITY=${WANDB_ENTITY:-NYU-robotics}
export WANDB_API_KEY_OVERRIDE=${WANDB_API_KEY_OVERRIDE:-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}
export WANDB_API_KEY="${WANDB_API_KEY_OVERRIDE}"
export WANDB_DIR=${WANDB_DIR:-${GEO_ROOT}/openpi_cam/log/wandb}
mkdir -p "${WANDB_DIR}" "${MPLCONFIGDIR}"

echo "===== ROBOTWIN STAGE 2 (no distillation loss) ====="
echo "task         : ${TASK}"
echo "config       : pi0_robotwin_cam_prope_ray_view_nodistill_stage2"
echo "repo_id      : ${REPO_ID}"
echo "exp_name     : ${EXP_NAME}"
echo "S1_CKPT      : ${S1_CKPT}"
echo "distillation : disabled (no pi3x_targets_root)"
echo "n gpus       : ${SLURM_GPUS_ON_NODE:-4}"
hostname; python -V; date

EXTRA_ARGS=()
if [[ "${RESUME:-false}" == "true" ]]; then
    EXTRA_ARGS+=(--resume)
fi

uv run torchrun --standalone --nnodes=1 --nproc_per_node=${SLURM_GPUS_ON_NODE:-4} \
    scripts/train_pytorch.py \
    pi0_robotwin_cam_prope_ray_view_nodistill_stage2 \
    --exp_name=${EXP_NAME} \
    --batch_size=${BATCH_SIZE:-64} \
    --num_workers=${NUM_WORKERS:-16} \
    --num_train_steps=${NUM_TRAIN_STEPS:-30000} \
    --save_interval=${SAVE_INTERVAL:-2000} \
    --keep_period=${KEEP_PERIOD:-10000} \
    --pytorch_weight_path=${S1_CKPT} \
    --data.repo_id ${REPO_ID} \
    --data.assets.assets_dir ${GEO_ROOT}/pi0_libero \
    --data.assets.asset_id ${REPO_ID} \
    "${EXTRA_ARGS[@]}"

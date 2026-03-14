#!/bin/bash
#SBATCH --job-name=pi0_libero_fullft
#SBATCH --output=/scratch/yp2841/TFP/geo/openpi_cam/log/pi0_libero_fullft/slurm-%j.out
#SBATCH --error=/scratch/yp2841/TFP/geo/openpi_cam/log/pi0_libero_fullft/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_637_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:h200:4
#SBATCH --export=ALL

set -euo pipefail

REPO_ROOT=/scratch/yp2841/TFP/geo/openpi_cam
GEO_ROOT=/scratch/yp2841/TFP/geo

CONFIG_NAME=${CONFIG_NAME:-pi0_libero_pytorch_full_finetune}
EXP_NAME=${EXP_NAME:-pi0_libero_fullft_${SLURM_JOB_ID:-manual}}
WANDB_ENABLED=${WANDB_ENABLED:-true}
WANDB_PROJECT=${WANDB_PROJECT:-openpi_cam}
NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-4}}
NUM_WORKERS=${NUM_WORKERS:-16}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-30000}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:-${REPO_ROOT}/checkpoints}
CACHE_ROOT=${CACHE_ROOT:-${GEO_ROOT}/.cache/huggingface}
OPENPI_CACHE_ROOT=${OPENPI_CACHE_ROOT:-${GEO_ROOT}/.cache/openpi}

JOB_TMP_BASE="${SLURM_TMPDIR:-/tmp/$USER}"
JOB_TMP_ROOT="${JOB_TMP_BASE}/openpi_cam_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOB_TMP_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export DATASETS_TMPDIR="${JOB_TMP_ROOT}/datasets_tmp"
export HOME="${JOB_TMP_ROOT}/home"
export XDG_CACHE_HOME="${JOB_TMP_ROOT}/xdg_cache"
export XDG_CONFIG_HOME="${JOB_TMP_ROOT}/xdg_config"
export XDG_STATE_HOME="${JOB_TMP_ROOT}/xdg_state"

mkdir -p "${TMPDIR}" "${DATASETS_TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}"
chmod 700 "${TMPDIR}" "${DATASETS_TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" || true

cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/env/activate_env.sh"

export HF_HOME="${CACHE_ROOT}"
export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
export HF_HUB_CACHE="${CACHE_ROOT}/hub"
export OPENPI_CACHE_DIR="${OPENPI_CACHE_ROOT}"
export OPENPI_DATA_HOME="${OPENPI_CACHE_ROOT}"
unset TRANSFORMERS_CACHE

mkdir -p \
  "${HF_HOME}" \
  "${HF_DATASETS_CACHE}" \
  "${HF_HUB_CACHE}" \
  "${OPENPI_CACHE_DIR}" \
  "${CHECKPOINT_BASE_DIR}"

if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "NUM_GPUS must be >= 1, got ${NUM_GPUS}" >&2
  exit 1
fi

THREADS_PER_PROC=$(( SLURM_CPUS_PER_TASK / NUM_GPUS ))
if [[ "${THREADS_PER_PROC}" -lt 1 ]]; then
  THREADS_PER_PROC=1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${THREADS_PER_PROC}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ "${WANDB_ENABLED}" == "true" ]]; then
  unset WANDB_DIR WANDB_CACHE_DIR WANDB_CONFIG_DIR WANDB_DATA_DIR
  export WANDB_API_KEY="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"
  export WANDB_DIR="${JOB_TMP_ROOT}/wandb_root"
  export WANDB_CACHE_DIR="${JOB_TMP_ROOT}/wandb_cache"
  export WANDB_CONFIG_DIR="${JOB_TMP_ROOT}/wandb_config"
  export WANDB_DATA_DIR="${JOB_TMP_ROOT}/wandb_data"
  mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${WANDB_DATA_DIR}"
else
  export WANDB_MODE=disabled
fi

DATASET_DIR="${HF_LEROBOT_HOME}/glbreeze/libero"
BASE_MODEL_DIR="${OPENPI_PI0_BASE_DIR}"
NORM_STATS_DIR="${OPENPI_PI0_LIBERO_NORM_DIR}/glbreeze/libero"

if [[ ! -d "${DATASET_DIR}/data" || ! -f "${DATASET_DIR}/meta/info.json" ]]; then
  echo "Missing local LeRobot dataset at ${DATASET_DIR}" >&2
  exit 1
fi

if [[ ! -f "${BASE_MODEL_DIR}/model.safetensors" || ! -f "${BASE_MODEL_DIR}/config.json" ]]; then
  echo "Missing local PyTorch base model at ${BASE_MODEL_DIR}" >&2
  exit 1
fi

if [[ ! -f "${NORM_STATS_DIR}/norm_stats.json" ]]; then
  echo "Missing local norm stats at ${NORM_STATS_DIR}/norm_stats.json" >&2
  exit 1
fi

WAND_ARG="--no-wandb-enabled"
if [[ "${WANDB_ENABLED}" == "true" ]]; then
  WAND_ARG="--wandb-enabled"
fi

echo "===== ENV CHECK ====="
echo "repo root: ${REPO_ROOT}"
echo "config: ${CONFIG_NAME}"
echo "exp: ${EXP_NAME}"
echo "dataset: ${DATASET_DIR}"
echo "base model: ${BASE_MODEL_DIR}"
echo "norm stats: ${NORM_STATS_DIR}"
echo "checkpoints: ${CHECKPOINT_BASE_DIR}"
echo "wandb root: ${WANDB_DIR:-disabled}"
which python
python -V
hostname
nvidia-smi -L || true

srun --export=ALL --ntasks=1 --ntasks-per-node=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --kill-on-bad-exit=1 \
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" \
    scripts/train_pytorch.py "${CONFIG_NAME}" \
    --exp-name="${EXP_NAME}" \
    --project-name="${WANDB_PROJECT}" \
    --checkpoint-base-dir="${CHECKPOINT_BASE_DIR}" \
    --batch-size="${BATCH_SIZE}" \
    --num-workers="${NUM_WORKERS}" \
    --num-train-steps="${NUM_TRAIN_STEPS}" \
    --save-interval="${SAVE_INTERVAL}" \
    --overwrite \
    "${WAND_ARG}"

#!/bin/bash
#SBATCH --job-name=pi0_rc24_ta4
#SBATCH --output=/scratch/%u/openpi_cam/log/pi0_robocasa24_baseline/slurm-%j.out
#SBATCH --error=/scratch/%u/openpi_cam/log/pi0_robocasa24_baseline/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=20:00:00
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --gres=gpu:4
#SBATCH --export=ALL

set -euo pipefail

resolve_repo_root() {
  local candidate
  local script_root=""

  if script_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd); then
    :
  else
    script_root=""
  fi

  for candidate in "${REPO_ROOT:-}" "${OPENPI_CAM_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "${PWD:-}" "${script_root}"; do
    [[ -n "${candidate}" ]] || continue
    while [[ "${candidate}" != "/" ]]; do
      if [[ -f "${candidate}/scripts/env/activate_env.sh" ]]; then
        printf '%s\n' "${candidate}"
        return 0
      fi
      candidate=$(dirname -- "${candidate}")
    done
  done

  return 1
}

REPO_ROOT=${REPO_ROOT:-$(resolve_repo_root)} || {
  echo "Unable to locate repo root. Set REPO_ROOT or submit from inside the repo." >&2
  exit 1
}
GEO_ROOT=${GEO_ROOT:-$(cd -- "${REPO_ROOT}/.." && pwd)}

CONFIG_NAME=${CONFIG_NAME:-pi0_robocasa24_all24_baseline}
DATASET_REPO_ID=${DATASET_REPO_ID:-robocasa24/all24_human_camaware}
NORM_ASSET_ID=${NORM_ASSET_ID:-robocasa24/all24_human_camaware}
NORM_STATS_ROOT=${NORM_STATS_ROOT:-${GEO_ROOT}/pi0_robocasa24}
BASE_MODEL_DIR=${BASE_MODEL_DIR:-${GEO_ROOT}/pi0_base}
CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:-/scratch/${USER}/tmp/openpi_cam/checkpoints}
EXP_NAME=${EXP_NAME:-pi0_robocasa24_all24_baseline_tandon_4gpu_v1}

WANDB_ENABLED=${WANDB_ENABLED:-false}
WANDB_PROJECT=${WANDB_PROJECT:-openpi_cam_robocasa24}
WANDB_ENTITY=${WANDB_ENTITY:-NYU-robotics}
WANDB_DIR=${WANDB_DIR:-${GEO_ROOT}/openpi_cam/log/wandb}
WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR:-/scratch/${USER}/.config/wandb}
WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/scratch/${USER}/.cache/wandb}

BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-8}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-30000}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
KEEP_PERIOD=${KEEP_PERIOD:-5000}
RESUME=${RESUME:-false}

JOB_TMP_BASE="${SLURM_TMPDIR:-/tmp/$USER}"
JOB_TMP_ROOT="${JOB_TMP_BASE}/openpi_train_rc24_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOB_TMP_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export DATASETS_TMPDIR="${JOB_TMP_ROOT}/datasets_tmp"
export HOME="${JOB_TMP_ROOT}/home"
export XDG_CACHE_HOME="${JOB_TMP_ROOT}/xdg_cache"
export XDG_CONFIG_HOME="${JOB_TMP_ROOT}/xdg_config"
export XDG_STATE_HOME="${JOB_TMP_ROOT}/xdg_state"
export PYTHONUNBUFFERED=1

mkdir -p \
  "${TMPDIR}" \
  "${DATASETS_TMPDIR}" \
  "${HOME}" \
  "${XDG_CACHE_HOME}" \
  "${XDG_CONFIG_HOME}" \
  "${XDG_STATE_HOME}" \
  "${REPO_ROOT}/log/pi0_robocasa24_baseline" \
  "${CHECKPOINT_BASE_DIR}" \
  "${WANDB_DIR}" \
  "${WANDB_CONFIG_DIR}" \
  "${WANDB_CACHE_DIR}"
chmod 700 "${TMPDIR}" "${DATASETS_TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" || true

cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/env/activate_env.sh"

export OPENPI_GEO_ROOT="${GEO_ROOT}"
export OPENPI_PI0_BASE_DIR="${BASE_MODEL_DIR}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${GEO_ROOT}}"
export HF_HOME="${HF_HOME:-${GEO_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export OPENPI_CACHE_DIR="${OPENPI_CACHE_DIR:-${GEO_ROOT}/.cache/openpi}"
export OPENPI_DATA_HOME="${OPENPI_CACHE_DIR}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
unset TRANSFORMERS_CACHE

mkdir -p "${HF_LEROBOT_HOME}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${OPENPI_CACHE_DIR}"

DATASET_DIR="${HF_LEROBOT_HOME}/${DATASET_REPO_ID}"
NORM_STATS_DIR="${NORM_STATS_ROOT}/${NORM_ASSET_ID}"

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

wandb_flag="--no-wandb-enabled"
if [[ "${WANDB_ENABLED}" == "true" ]]; then
  wandb_flag="--wandb-enabled"
fi

resume_or_overwrite_arg="--overwrite"
if [[ "${RESUME}" == "true" ]]; then
  resume_or_overwrite_arg="--resume"
fi

echo "===== RoboCasa24 Pi0 baseline (Tandon Advanced 4 GPU) ====="
echo "job id: ${SLURM_JOB_ID:-manual}"
echo "host: $(hostname)"
echo "repo root: ${REPO_ROOT}"
echo "geo root: ${GEO_ROOT}"
echo "config: ${CONFIG_NAME}"
echo "dataset repo id: ${DATASET_REPO_ID}"
echo "dataset dir: ${DATASET_DIR}"
echo "norm asset id: ${NORM_ASSET_ID}"
echo "norm stats dir: ${NORM_STATS_DIR}"
echo "base model dir: ${BASE_MODEL_DIR}"
echo "checkpoint base dir: ${CHECKPOINT_BASE_DIR}"
echo "exp name: ${EXP_NAME}"
echo "batch size: ${BATCH_SIZE}"
echo "num workers: ${NUM_WORKERS}"
echo "num train steps: ${NUM_TRAIN_STEPS}"
echo "save interval: ${SAVE_INTERVAL}"
echo "keep period: ${KEEP_PERIOD}"
echo "resume: ${RESUME}"
echo "wandb enabled: ${WANDB_ENABLED}"
date

uv run torchrun --standalone --nnodes=1 --nproc_per_node="${SLURM_GPUS_ON_NODE:-4}" \
  scripts/train_pytorch.py \
  "${CONFIG_NAME}" \
  --exp-name="${EXP_NAME}" \
  --project-name="${WANDB_PROJECT}" \
  --checkpoint-base-dir="${CHECKPOINT_BASE_DIR}" \
  --batch-size="${BATCH_SIZE}" \
  --num-workers="${NUM_WORKERS}" \
  --num-train-steps="${NUM_TRAIN_STEPS}" \
  --save-interval="${SAVE_INTERVAL}" \
  --keep-period="${KEEP_PERIOD}" \
  "${resume_or_overwrite_arg}" \
  --pytorch_weight_path "${BASE_MODEL_DIR}" \
  --data.repo_id "${DATASET_REPO_ID}" \
  --data.assets.assets_dir "${NORM_STATS_ROOT}" \
  --data.assets.asset_id "${NORM_ASSET_ID}" \
  "${wandb_flag}"

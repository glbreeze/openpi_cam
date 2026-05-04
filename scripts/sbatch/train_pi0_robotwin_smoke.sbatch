#!/bin/bash
#SBATCH --job-name=pi0_robotwin_smoke
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_robotwin_smoke/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_robotwin_smoke/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --account=torch_pr_637_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --gres=gpu:1
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

if [[ -z "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(resolve_repo_root || true)"
fi
if [[ -z "${REPO_ROOT}" && -d "/scratch/yp2841/geometry-vla/openpi_cam" ]]; then
  REPO_ROOT="/scratch/yp2841/geometry-vla/openpi_cam"
fi
if [[ -z "${REPO_ROOT}" ]]; then
  echo "Unable to locate repo root." >&2
  exit 1
fi
GEO_ROOT=${GEO_ROOT:-$(cd -- "${REPO_ROOT}/.." && pwd)}

JOB_TMP_BASE="${SLURM_TMPDIR:-/tmp/$USER}"
JOB_TMP_ROOT="${JOB_TMP_BASE}/robotwin_train_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOB_TMP_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export DATASETS_TMPDIR="${JOB_TMP_ROOT}/datasets_tmp"
export HOME="${JOB_TMP_ROOT}/home"
export XDG_CACHE_HOME="${JOB_TMP_ROOT}/xdg_cache"
export XDG_CONFIG_HOME="${JOB_TMP_ROOT}/xdg_config"
export XDG_STATE_HOME="${JOB_TMP_ROOT}/xdg_state"

mkdir -p \
  "${TMPDIR}" \
  "${DATASETS_TMPDIR}" \
  "${HOME}" \
  "${XDG_CACHE_HOME}" \
  "${XDG_CONFIG_HOME}" \
  "${XDG_STATE_HOME}" \
  /scratch/yp2841/geometry-vla/openpi_cam/log/pi0_robotwin_smoke
chmod 700 "${TMPDIR}" "${DATASETS_TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" || true

cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/env/activate_env.sh"

export HF_LEROBOT_HOME="${GEO_ROOT}"
export HF_HOME="${GEO_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export OPENPI_CACHE_DIR="${GEO_ROOT}/.cache/openpi"
export OPENPI_DATA_HOME="${OPENPI_CACHE_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
unset TRANSFORMERS_CACHE

CONFIG_NAME=${CONFIG_NAME:-pi0_robotwin_smoke}
EXP_NAME=${EXP_NAME:-robotwin_smoke}
DATASET_REPO_ID=${DATASET_REPO_ID:-lerobot/robotwin_unified}
NORM_ASSET_ID=${NORM_ASSET_ID:-lerobot/robotwin_unified}
CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:-${REPO_ROOT}/checkpoints}
NUM_WORKERS=${NUM_WORKERS:-4}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-20}
SAVE_INTERVAL=${SAVE_INTERVAL:-100}

DATASET_DIR="${HF_LEROBOT_HOME}/${DATASET_REPO_ID}"
BASE_MODEL_DIR="${OPENPI_PI0_BASE_DIR}"
NORM_STATS_DIR="${GEO_ROOT}/pi0_libero/${NORM_ASSET_ID}"

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${OPENPI_CACHE_DIR}" "${CHECKPOINT_BASE_DIR}"

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

echo "===== ROBOTWIN SMOKE TRAIN ====="
echo "repo root: ${REPO_ROOT}"
echo "geo root: ${GEO_ROOT}"
echo "config name: ${CONFIG_NAME}"
echo "exp name: ${EXP_NAME}"
echo "dataset dir: ${DATASET_DIR}"
echo "norm stats dir: ${NORM_STATS_DIR}"
echo "checkpoint base dir: ${CHECKPOINT_BASE_DIR}"
hostname
python -V

python scripts/train.py \
  "${CONFIG_NAME}" \
  --exp-name="${EXP_NAME}" \
  --checkpoint-base-dir="${CHECKPOINT_BASE_DIR}" \
  --batch-size="${BATCH_SIZE}" \
  --num-workers="${NUM_WORKERS}" \
  --num-train-steps="${NUM_TRAIN_STEPS}" \
  --save-interval="${SAVE_INTERVAL}" \
  --overwrite \
  --no-wandb-enabled \
  --data.repo_id "${DATASET_REPO_ID}" \
  --data.assets.assets_dir "${GEO_ROOT}/pi0_libero" \
  --data.assets.asset_id "${NORM_ASSET_ID}"

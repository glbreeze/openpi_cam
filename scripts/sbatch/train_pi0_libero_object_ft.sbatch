#!/bin/bash
#SBATCH --job-name=pi0_libero_object_ft
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_libero_object_ft/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_libero_object_ft/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=16:00:00
#SBATCH --account=torch_pr_637_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:h200:4
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

  for candidate in "${REPO_ROOT:-}" "${OPENPI_CAM_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "${script_root}"; do
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
  echo "Unable to locate repo root. Set REPO_ROOT or submit from inside the repo." >&2
  exit 1
fi
GEO_ROOT=${GEO_ROOT:-$(cd -- "${REPO_ROOT}/.." && pwd)}

USE_CAM=${USE_CAM:-false}
WANDB_ENABLED=${WANDB_ENABLED:-true}
WANDB_PROJECT=${WANDB_PROJECT:-openpi_cam_libero_object}
NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-4}}
NUM_WORKERS=${NUM_WORKERS:-16}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-30000}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:-${REPO_ROOT}/checkpoints}
CACHE_ROOT=${CACHE_ROOT:-${GEO_ROOT}/.cache/huggingface}
OPENPI_CACHE_ROOT=${OPENPI_CACHE_ROOT:-${GEO_ROOT}/.cache/openpi}

if [[ "${USE_CAM}" == "true" ]]; then
  CONFIG_NAME=${CONFIG_NAME:-pi0_libero_cam_pytorch_full_finetune}
  DATASET_REPO_ID=${DATASET_REPO_ID:-glbreeze/libero_object_cam}
  NORM_ASSET_ID=${NORM_ASSET_ID:-glbreeze/libero_object_cam}
  EXP_NAME=${EXP_NAME:-pi0_libero_object_cam_${SLURM_JOB_ID:-manual}}
else
  CONFIG_NAME=${CONFIG_NAME:-pi0_libero_pytorch_full_finetune}
  DATASET_REPO_ID=${DATASET_REPO_ID:-glbreeze/libero_object}
  NORM_ASSET_ID=${NORM_ASSET_ID:-glbreeze/libero_object}
  EXP_NAME=${EXP_NAME:-pi0_libero_object_${SLURM_JOB_ID:-manual}}
fi

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

DATASET_DIR="${HF_LEROBOT_HOME}/${DATASET_REPO_ID}"
BASE_MODEL_DIR="${OPENPI_PI0_BASE_DIR}"
NORM_STATS_ROOT="${OPENPI_PI0_LIBERO_NORM_DIR}"
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

WAND_ARG="--no-wandb-enabled"
if [[ "${WANDB_ENABLED}" == "true" ]]; then
  WAND_ARG="--wandb-enabled"
fi

TRAIN_ARGS=(
  "${CONFIG_NAME}"
  --exp-name="${EXP_NAME}"
  --project-name="${WANDB_PROJECT}"
  --checkpoint-base-dir="${CHECKPOINT_BASE_DIR}"
  --batch-size="${BATCH_SIZE}"
  --num-workers="${NUM_WORKERS}"
  --num-train-steps="${NUM_TRAIN_STEPS}"
  --save-interval="${SAVE_INTERVAL}"
  --overwrite
  --pytorch_weight_path "${BASE_MODEL_DIR}"
  --data.repo_id "${DATASET_REPO_ID}"
  --data.assets.assets_dir "${NORM_STATS_ROOT}"
  --data.assets.asset_id "${NORM_ASSET_ID}"
  "${WAND_ARG}"
)

if [[ "${USE_CAM}" == "true" ]]; then
  TRAIN_ARGS+=(
    --model.pose_enc_type absolute_pose
    --model.cross_view_fusion
  )
fi

echo "===== TRAIN ENV CHECK ====="
echo "repo root: ${REPO_ROOT}"
echo "use_cam: ${USE_CAM}"
echo "config: ${CONFIG_NAME}"
echo "exp: ${EXP_NAME}"
echo "dataset: ${DATASET_DIR}"
echo "base model: ${BASE_MODEL_DIR}"
echo "norm stats: ${NORM_STATS_DIR}"
echo "checkpoints: ${CHECKPOINT_BASE_DIR}"
echo "wandb root: ${WANDB_DIR:-disabled}"
echo "num_gpus: ${NUM_GPUS}"
echo "batch_size: ${BATCH_SIZE}"
echo "num_train_steps: ${NUM_TRAIN_STEPS}"
which python
python -V
hostname
nvidia-smi -L || true

srun --export=ALL --ntasks=1 --ntasks-per-node=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --kill-on-bad-exit=1 \
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" \
    scripts/train_pytorch.py "${TRAIN_ARGS[@]}"

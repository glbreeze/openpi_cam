#!/bin/bash
#SBATCH --job-name=pi0_pour_base_2g
#SBATCH --output=/scratch/%u/openpi_cam/log/pi0_ur5_pour_baseline/slurm-%j.out
#SBATCH --error=/scratch/%u/openpi_cam/log/pi0_ur5_pour_baseline/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --gres=gpu:2
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
if [[ -z "${REPO_ROOT}" && -d "/scratch/${USER}/openpi_cam" ]]; then
  REPO_ROOT="/scratch/${USER}/openpi_cam"
fi
if [[ -z "${REPO_ROOT}" ]]; then
  echo "Unable to locate repo root." >&2
  exit 1
fi

export USE_CAM=false
export CONFIG_NAME="${CONFIG_NAME:-pi0_ur5_real_robot_pytorch_baseline}"
export TRAIN_RECIPE=single_stage
export RESUME="${RESUME:-false}"

export WANDB_ENABLED="${WANDB_ENABLED:-true}"
export WANDB_ENTITY="${WANDB_ENTITY:-NYU-robotics}"
export WANDB_PROJECT="${WANDB_PROJECT:-openpi_cam_real_robot}"
export WANDB_DIR="${WANDB_DIR:-/scratch/${USER}/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/scratch/${USER}/.config/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/scratch/${USER}/.cache/wandb}"

export OPENPI_GEO_ROOT="${OPENPI_GEO_ROOT:-/scratch/${USER}}"
export OPENPI_PI0_BASE_DIR="${OPENPI_PI0_BASE_DIR:-${OPENPI_GEO_ROOT}/pi0_base}"
export REAL_ROBOT_NORM_ROOT="${REAL_ROBOT_NORM_ROOT:-${OPENPI_GEO_ROOT}/pi0_ur5_real_robot}"
export OPENPI_PI0_LIBERO_NORM_DIR="${REAL_ROBOT_NORM_ROOT}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${OPENPI_GEO_ROOT}/.huggingface/lerobot}"

export DATASET_DIR="${DATASET_DIR:-/scratch/${USER}/real_robot_data/ur5_place_and_pour_nuts_camera_shifts}"
export DATASET_REPO_ID="${DATASET_REPO_ID:-ur5_place_and_pour_nuts_camera_shifts}"
export NORM_ASSET_ID="${NORM_ASSET_ID:-ur5_place_and_pour_nuts_camera_shifts}"
export EXP_NAME="${EXP_NAME:-pi0_ur5_pour_pytorch_baseline_tandon_2gpu_b16}"
export CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-/scratch/${USER}/tmp/openpi_cam/checkpoints}"

export NUM_GPUS="${NUM_GPUS:-2}"
export NUM_WORKERS="${NUM_WORKERS:-6}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
export KEEP_PERIOD="${KEEP_PERIOD:-5000}"
export OPENPI_ENABLE_GPU_FILLER="${OPENPI_ENABLE_GPU_FILLER:-true}"
export OPENPI_GPU_FILLER_LOG_DIR="${OPENPI_GPU_FILLER_LOG_DIR:-/scratch/${USER}/openpi_cam/log/pi0_ur5_pour_baseline}"

mkdir -p "/scratch/${USER}/openpi_cam/log/pi0_ur5_pour_baseline" "${CHECKPOINT_BASE_DIR}"

NORM_STATS_DIR="${REAL_ROBOT_NORM_ROOT}/${NORM_ASSET_ID}"
ARRAY_CACHE_ROOT="${DATASET_DIR}_array_cache"

if [[ ! -d "${DATASET_DIR}/data" || ! -f "${DATASET_DIR}/meta/info.json" || ! -f "${DATASET_DIR}/meta/episodes.jsonl" ]]; then
  echo "Missing local LeRobot pour dataset at ${DATASET_DIR}." >&2
  exit 1
fi
if [[ ! -d "${ARRAY_CACHE_ROOT}/videos/chunk-000/observation.images.context_left_rgb" ||
      ! -d "${ARRAY_CACHE_ROOT}/videos/chunk-000/observation.images.wrist_right_rgb" ]]; then
  echo "Missing local video array cache for context_left_rgb and wrist_right_rgb under ${ARRAY_CACHE_ROOT}." >&2
  exit 1
fi
if [[ ! -f "${OPENPI_PI0_BASE_DIR}/model.safetensors" || ! -f "${OPENPI_PI0_BASE_DIR}/config.json" ]]; then
  echo "Missing Pi0 base checkpoint under ${OPENPI_PI0_BASE_DIR}." >&2
  exit 1
fi
if [[ ! -f "${NORM_STATS_DIR}/norm_stats.json" ]]; then
  echo "Missing Pi0 pour norm stats at ${NORM_STATS_DIR}/norm_stats.json." >&2
  exit 1
fi

echo "===== PI0 UR5 POUR BASELINE (TANDON A100/H100/H200 2 GPU) ====="
echo "repo root: ${REPO_ROOT}"
echo "config: ${CONFIG_NAME}"
echo "dataset: ${DATASET_DIR}"
echo "array cache: ${ARRAY_CACHE_ROOT}"
echo "norm stats: ${NORM_STATS_DIR}"
echo "base model: ${OPENPI_PI0_BASE_DIR}"
echo "checkpoint base: ${CHECKPOINT_BASE_DIR}"
echo "exp name: ${EXP_NAME}"
echo "num_gpus: ${NUM_GPUS}"
echo "num_workers per rank/gpu: ${NUM_WORKERS}"
echo "batch_size: ${BATCH_SIZE}"
echo "num_train_steps: ${NUM_TRAIN_STEPS}"
echo "save_interval: ${SAVE_INTERVAL}"
echo "keep_period: ${KEEP_PERIOD}"
echo "geometry augmentation: disabled in config"
echo "new modules: pose/ray/view/cross-view/aux point head disabled in config"

exec bash "${REPO_ROOT}/scripts/sbatch/train_pi0_libero_object_ft.sbatch"

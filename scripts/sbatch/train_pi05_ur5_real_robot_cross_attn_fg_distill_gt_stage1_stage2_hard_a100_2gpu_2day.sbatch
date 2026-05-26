#!/bin/bash
#SBATCH --job-name=pi05_ur5_fg_gt_s1s2_a100_2g
#SBATCH --output=/scratch/%u/openpi_cam/log/pi0_ur5_real_robot_ft/slurm-%j.out
#SBATCH --error=/scratch/%u/openpi_cam/log/pi0_ur5_real_robot_ft/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=a100_tandon
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
export TRAIN_RECIPE=single_stage

export WANDB_ENABLED="${WANDB_ENABLED:-true}"
export WANDB_ENTITY="${WANDB_ENTITY:-NYU-robotics}"
export WANDB_PROJECT="${WANDB_PROJECT:-openpi_cam_real_robot}"
export WANDB_DIR="${WANDB_DIR:-/scratch/${USER}/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/scratch/${USER}/.config/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/scratch/${USER}/.cache/wandb}"

export OPENPI_GEO_ROOT="${OPENPI_GEO_ROOT:-/scratch/${USER}}"
export OPENPI_PI0_BASE_DIR="${OPENPI_PI0_BASE_DIR:-${OPENPI_GEO_ROOT}/pi05_base}"
export REAL_ROBOT_NORM_ROOT="${REAL_ROBOT_NORM_ROOT:-${OPENPI_GEO_ROOT}/pi05_ur5_real_robot}"
export OPENPI_PI0_LIBERO_NORM_DIR="${REAL_ROBOT_NORM_ROOT}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${OPENPI_GEO_ROOT}/.huggingface/lerobot}"

export DATASET_DIR="${DATASET_DIR:-/scratch/${USER}/real_robot_data/ur5_lab_test_tube_camera_shifts}"
export DATASET_REPO_ID="${DATASET_REPO_ID:-ur5_lab_test_tube_camera_shifts}"
export NORM_ASSET_ID="${NORM_ASSET_ID:-ur5_lab_test_tube_camera_shifts}"
export PI3X_TARGETS_ROOT_OVERRIDE="${PI3X_TARGETS_ROOT_OVERRIDE:-}"
export GT_POINT_TARGETS_ROOT_OVERRIDE="${GT_POINT_TARGETS_ROOT_OVERRIDE:-${OPENPI_GEO_ROOT}/gt_point_targets_grid224/ur5_lab_test_tube_camera_shifts}"
export CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-/scratch/${USER}/tmp/openpi_cam/checkpoints}"

export NUM_GPUS="${NUM_GPUS:-2}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export OPENPI_ENABLE_GPU_FILLER="${OPENPI_ENABLE_GPU_FILLER:-true}"
export OPENPI_GPU_FILLER_LOG_DIR="${OPENPI_GPU_FILLER_LOG_DIR:-/scratch/${USER}/openpi_cam/log/pi0_ur5_real_robot_ft}"

STAGE1_CONFIG_NAME="${STAGE1_CONFIG_NAME:-pi05_ur5_real_robot_pytorch_cross_attn_fg_distill_gt_stage1_hard}"
STAGE2_CONFIG_NAME="${STAGE2_CONFIG_NAME:-pi05_ur5_real_robot_pytorch_cross_attn_fg_distill_gt_stage2_hard}"
STAGE1_EXP_NAME="${STAGE1_EXP_NAME:-pi05_ur5_real_robot_cross_attn_fg_distill_gt_hard_stage1_a100_2gpu_b32}"
STAGE2_EXP_NAME="${STAGE2_EXP_NAME:-pi05_ur5_real_robot_cross_attn_fg_distill_gt_hard_stage2_a100_2gpu_b32}"

STAGE1_NUM_TRAIN_STEPS="${STAGE1_NUM_TRAIN_STEPS:-5000}"
STAGE1_CHECKPOINT_STEP="${STAGE1_CHECKPOINT_STEP:-${STAGE1_NUM_TRAIN_STEPS}}"
STAGE1_SAVE_INTERVAL="${STAGE1_SAVE_INTERVAL:-${STAGE1_NUM_TRAIN_STEPS}}"
STAGE1_KEEP_PERIOD="${STAGE1_KEEP_PERIOD:-${STAGE1_NUM_TRAIN_STEPS}}"
STAGE1_RESUME="${STAGE1_RESUME:-false}"

STAGE2_NUM_TRAIN_STEPS="${STAGE2_NUM_TRAIN_STEPS:-30000}"
STAGE2_SAVE_INTERVAL="${STAGE2_SAVE_INTERVAL:-5000}"
STAGE2_KEEP_PERIOD="${STAGE2_KEEP_PERIOD:-5000}"
STAGE2_RESUME="${STAGE2_RESUME:-false}"

mkdir -p "/scratch/${USER}/openpi_cam/log/pi0_ur5_real_robot_ft" "${CHECKPOINT_BASE_DIR}"

ARRAY_CACHE_ROOT="${DATASET_DIR}_array_cache"
if [[ ! -d "${ARRAY_CACHE_ROOT}/videos/chunk-000/observation.images.context_left_rgb" ||
      ! -d "${ARRAY_CACHE_ROOT}/videos/chunk-000/observation.images.wrist_right_rgb" ]]; then
  echo "Missing local video array cache for context_left_rgb and wrist_right_rgb under ${ARRAY_CACHE_ROOT}." >&2
  exit 1
fi

if [[ -n "${PI3X_TARGETS_ROOT_OVERRIDE}" ]]; then
  echo "PI3X_TARGETS_ROOT_OVERRIDE must be empty for this GT-only run, got: ${PI3X_TARGETS_ROOT_OVERRIDE}" >&2
  exit 1
fi

if [[ ! -d "${GT_POINT_TARGETS_ROOT_OVERRIDE}/base" ||
      ! -d "${GT_POINT_TARGETS_ROOT_OVERRIDE}/left_wrist" ]]; then
  echo "Missing GT target subdirs under ${GT_POINT_TARGETS_ROOT_OVERRIDE}; expected base/ and left_wrist/." >&2
  exit 1
fi

shopt -s nullglob
base_npzs=("${GT_POINT_TARGETS_ROOT_OVERRIDE}"/base/episode_*.npz)
left_wrist_npzs=("${GT_POINT_TARGETS_ROOT_OVERRIDE}"/left_wrist/episode_*.npz)
context_left_cache=("${ARRAY_CACHE_ROOT}"/videos/chunk-000/observation.images.context_left_rgb/episode_*.npy)
wrist_right_cache=("${ARRAY_CACHE_ROOT}"/videos/chunk-000/observation.images.wrist_right_rgb/episode_*.npy)
shopt -u nullglob
episode_count=$(grep -cve '^[[:space:]]*$' "${DATASET_DIR}/meta/episodes.jsonl")
if [[ "${#base_npzs[@]}" -eq 0 || "${#left_wrist_npzs[@]}" -eq 0 ]]; then
  echo "GT target subdirs exist but episode npz files are missing under ${GT_POINT_TARGETS_ROOT_OVERRIDE}." >&2
  exit 1
fi
if [[ "${#context_left_cache[@]}" -eq 0 || "${#wrist_right_cache[@]}" -eq 0 ]]; then
  echo "Array cache subdirs exist but episode npy files are missing under ${ARRAY_CACHE_ROOT}." >&2
  exit 1
fi
if [[ "${#base_npzs[@]}" -ne "${episode_count}" || "${#left_wrist_npzs[@]}" -ne "${episode_count}" ]]; then
  echo "GT cache file count mismatch: episodes=${episode_count}, base=${#base_npzs[@]}, left_wrist=${#left_wrist_npzs[@]}." >&2
  exit 1
fi
if [[ "${#context_left_cache[@]}" -ne "${episode_count}" || "${#wrist_right_cache[@]}" -ne "${episode_count}" ]]; then
  echo "Array cache file count mismatch: episodes=${episode_count}, context_left=${#context_left_cache[@]}, wrist_right=${#wrist_right_cache[@]}." >&2
  exit 1
fi

run_common_launcher() {
  bash "${REPO_ROOT}/scripts/sbatch/train_pi0_libero_object_ft.sbatch"
}

echo "===== COMBINED PI0.5 UR5 GT HARD STAGE1+STAGE2 (A100 2 GPU) ====="
echo "repo root: ${REPO_ROOT}"
echo "dataset: ${DATASET_DIR}"
echo "array cache: ${ARRAY_CACHE_ROOT}"
echo "norm stats root: ${REAL_ROBOT_NORM_ROOT}"
echo "gt target root: ${GT_POINT_TARGETS_ROOT_OVERRIDE}"
echo "checkpoint base: ${CHECKPOINT_BASE_DIR}"
echo "num_gpus: ${NUM_GPUS}"
echo "num_workers: ${NUM_WORKERS}"
echo "batch_size: ${BATCH_SIZE}"
echo "stage1 config: ${STAGE1_CONFIG_NAME}"
echo "stage1 exp: ${STAGE1_EXP_NAME}"
echo "stage2 config: ${STAGE2_CONFIG_NAME}"
echo "stage2 exp: ${STAGE2_EXP_NAME}"

export CONFIG_NAME="${STAGE1_CONFIG_NAME}"
export EXP_NAME="${STAGE1_EXP_NAME}"
export RESUME="${STAGE1_RESUME}"
export NUM_TRAIN_STEPS="${STAGE1_NUM_TRAIN_STEPS}"
export SAVE_INTERVAL="${STAGE1_SAVE_INTERVAL}"
export KEEP_PERIOD="${STAGE1_KEEP_PERIOD}"
export PYTORCH_WEIGHT_PATH_OVERRIDE="${OPENPI_PI0_BASE_DIR}"

echo "===== RUNNING STAGE 1 ====="
run_common_launcher

STAGE1_CKPT_DIR="${CHECKPOINT_BASE_DIR}/${STAGE1_CONFIG_NAME}/${STAGE1_EXP_NAME}/${STAGE1_CHECKPOINT_STEP}"
if [[ ! -f "${STAGE1_CKPT_DIR}/model.safetensors" ]]; then
  echo "Expected Stage 1 checkpoint at ${STAGE1_CKPT_DIR}/model.safetensors" >&2
  exit 1
fi

export CONFIG_NAME="${STAGE2_CONFIG_NAME}"
export EXP_NAME="${STAGE2_EXP_NAME}"
export RESUME="${STAGE2_RESUME}"
export NUM_TRAIN_STEPS="${STAGE2_NUM_TRAIN_STEPS}"
export SAVE_INTERVAL="${STAGE2_SAVE_INTERVAL}"
export KEEP_PERIOD="${STAGE2_KEEP_PERIOD}"
export PYTORCH_WEIGHT_PATH_OVERRIDE="${STAGE1_CKPT_DIR}"

echo "===== RUNNING STAGE 2 ====="
echo "stage2 weight init: ${PYTORCH_WEIGHT_PATH_OVERRIDE}"
run_common_launcher

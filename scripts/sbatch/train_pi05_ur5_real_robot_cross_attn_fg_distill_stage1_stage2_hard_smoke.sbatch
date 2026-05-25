#!/bin/bash
#SBATCH --job-name=pi05_ur5_fg_s1s2_sm
#SBATCH --output=/scratch/%u/openpi_cam/log/pi0_ur5_real_robot_ft/slurm-%j.out
#SBATCH --error=/scratch/%u/openpi_cam/log/pi0_ur5_real_robot_ft/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=h100_tandon,h200_tandon,a100_tandon
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
if [[ -z "${REPO_ROOT}" && -d "/scratch/${USER}/openpi_cam" ]]; then
  REPO_ROOT="/scratch/${USER}/openpi_cam"
fi
if [[ -z "${REPO_ROOT}" ]]; then
  echo "Unable to locate repo root." >&2
  exit 1
fi

export USE_CAM=false
export TRAIN_RECIPE=single_stage

export WANDB_ENABLED="${WANDB_ENABLED:-false}"
export WANDB_ENTITY="${WANDB_ENTITY:-NYU-robotics}"
export WANDB_PROJECT="${WANDB_PROJECT:-openpi_cam_real_robot}"

export OPENPI_GEO_ROOT="${OPENPI_GEO_ROOT:-/scratch/${USER}}"
export OPENPI_PI0_BASE_DIR="${OPENPI_PI0_BASE_DIR:-${OPENPI_GEO_ROOT}/pi05_base}"
export REAL_ROBOT_NORM_ROOT="${REAL_ROBOT_NORM_ROOT:-${OPENPI_GEO_ROOT}/pi05_ur5_real_robot}"
export REAL_ROBOT_PI3X_TARGETS_224_BASE_DIR="${REAL_ROBOT_PI3X_TARGETS_224_BASE_DIR:-${OPENPI_GEO_ROOT}/pi3x_targets_224}"
export OPENPI_PI0_LIBERO_NORM_DIR="${REAL_ROBOT_NORM_ROOT}"
export OPENPI_PI3X_TARGETS_224_BASE_DIR="${REAL_ROBOT_PI3X_TARGETS_224_BASE_DIR}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${OPENPI_GEO_ROOT}/.huggingface/lerobot}"

export DATASET_DIR="${DATASET_DIR:-/scratch/${USER}/real_robot_data/ur5_lab_test_tube_camera_shifts}"
export DATASET_REPO_ID="${DATASET_REPO_ID:-ur5_lab_test_tube_camera_shifts}"
export NORM_ASSET_ID="${NORM_ASSET_ID:-ur5_lab_test_tube_camera_shifts}"
export PI3X_TARGETS_ROOT_OVERRIDE="${PI3X_TARGETS_ROOT_OVERRIDE:-${REAL_ROBOT_PI3X_TARGETS_224_BASE_DIR}/ur5_lab_test_tube_camera_shifts}"
export GT_POINT_TARGETS_ROOT_OVERRIDE="${GT_POINT_TARGETS_ROOT_OVERRIDE:-}"
export CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-/scratch/${USER}/tmp/openpi_cam/checkpoints}"

export NUM_GPUS="${NUM_GPUS:-1}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export OPENPI_ENABLE_GPU_FILLER="${OPENPI_ENABLE_GPU_FILLER:-false}"

STAGE1_CONFIG_NAME="${STAGE1_CONFIG_NAME:-pi05_ur5_real_robot_pytorch_cross_attn_fg_distill_stage1_hard}"
STAGE2_CONFIG_NAME="${STAGE2_CONFIG_NAME:-pi05_ur5_real_robot_pytorch_cross_attn_fg_distill_stage2_hard}"
STAGE1_EXP_NAME="${STAGE1_EXP_NAME:-pi05_ur5_real_robot_cross_attn_fg_distill_hard_stage1_targetfix_smoke}"
STAGE2_EXP_NAME="${STAGE2_EXP_NAME:-pi05_ur5_real_robot_cross_attn_fg_distill_hard_stage2_targetfix_smoke}"

STAGE1_NUM_TRAIN_STEPS="${STAGE1_NUM_TRAIN_STEPS:-2}"
STAGE1_CHECKPOINT_STEP="${STAGE1_CHECKPOINT_STEP:-${STAGE1_NUM_TRAIN_STEPS}}"
STAGE1_SAVE_INTERVAL="${STAGE1_SAVE_INTERVAL:-${STAGE1_NUM_TRAIN_STEPS}}"
STAGE1_KEEP_PERIOD="${STAGE1_KEEP_PERIOD:-${STAGE1_NUM_TRAIN_STEPS}}"
STAGE1_RESUME="${STAGE1_RESUME:-false}"

STAGE2_NUM_TRAIN_STEPS="${STAGE2_NUM_TRAIN_STEPS:-2}"
STAGE2_SAVE_INTERVAL="${STAGE2_SAVE_INTERVAL:-${STAGE2_NUM_TRAIN_STEPS}}"
STAGE2_KEEP_PERIOD="${STAGE2_KEEP_PERIOD:-${STAGE2_NUM_TRAIN_STEPS}}"
STAGE2_RESUME="${STAGE2_RESUME:-false}"

SMOKE_LOG_DIR="${SMOKE_LOG_DIR:-/scratch/${USER}/openpi_cam/log/pi0_ur5_real_robot_ft/smoke_aux_checks}"
mkdir -p "/scratch/${USER}/tmp/openpi_cam/log/pi0_ur5_real_robot_ft" "${CHECKPOINT_BASE_DIR}" "${SMOKE_LOG_DIR}"

if [[ ! -d "${PI3X_TARGETS_ROOT_OVERRIDE}/agent" || ! -d "${PI3X_TARGETS_ROOT_OVERRIDE}/wrist" ]]; then
  echo "Missing Pi3X target subdirs under ${PI3X_TARGETS_ROOT_OVERRIDE}; expected agent/ and wrist/." >&2
  exit 1
fi

shopt -s nullglob
agent_npzs=("${PI3X_TARGETS_ROOT_OVERRIDE}"/agent/episode_*.npz)
wrist_npzs=("${PI3X_TARGETS_ROOT_OVERRIDE}"/wrist/episode_*.npz)
shopt -u nullglob
if [[ "${#agent_npzs[@]}" -eq 0 || "${#wrist_npzs[@]}" -eq 0 ]]; then
  echo "Pi3X target subdirs exist but episode npz files are missing under ${PI3X_TARGETS_ROOT_OVERRIDE}." >&2
  exit 1
fi

cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/env/activate_env.sh"

run_target_passthrough_check() {
  python - <<'PY'
import numpy as np

from openpi.models import model as _model
from openpi.policies.real_robot_policy import RealRobotUR5Inputs

data = {
    "observation/base_image": np.zeros((8, 8, 3), dtype=np.uint8),
    "observation/wrist_image": np.zeros((8, 8, 3), dtype=np.uint8),
    "observation/state": np.zeros(7, dtype=np.float32),
    "actions": np.zeros((10, 7), dtype=np.float32),
    "prompt": "smoke",
    "pi3x_target_xy": np.ones((2, 224, 224, 2), dtype=np.float16),
    "pi3x_target_logz": np.ones((2, 224, 224, 1), dtype=np.float16),
    "pi3x_target_conf": np.ones((2, 224, 224, 1), dtype=np.float16),
}
out = RealRobotUR5Inputs(_model.ModelType.PI05)(data)
for key in ("pi3x_target_xy", "pi3x_target_logz", "pi3x_target_conf"):
    if key not in out:
        raise SystemExit(f"{key} was dropped by RealRobotUR5Inputs")
print("target_passthrough_check_ok")
PY
}

run_common_launcher_to_log() {
  local log_file=$1
  bash "${REPO_ROOT}/scripts/sbatch/train_pi0_libero_object_ft.sbatch" 2>&1 | tee "${log_file}"
}

validate_aux_log() {
  local label=$1
  local log_file=$2
  python - "${label}" "${log_file}" <<'PY'
import re
import sys

label, log_file = sys.argv[1], sys.argv[2]
text = open(log_file, encoding="utf-8", errors="replace").read()
matches = re.findall(
    r"loss_breakdown .*?aux_loss=([0-9.eE+-]+).*?aux_pi3x_frac=([0-9.eE+-]+)",
    text,
)
if not matches:
    raise SystemExit(f"{label}: no loss_breakdown with aux_loss and aux_pi3x_frac found in {log_file}")
active = [(float(aux), float(frac)) for aux, frac in matches if float(aux) > 0.0 and float(frac) > 0.5]
if not active:
    raise SystemExit(
        f"{label}: aux supervision inactive in {log_file}; observed pairs: {matches[:5]}"
    )
print(f"{label}: aux supervision active; first active aux_loss={active[0][0]:.6g}, aux_pi3x_frac={active[0][1]:.3f}")
PY
}

echo "===== PI0.5 UR5 PI3X HARD STAGE1+STAGE2 SMOKE ====="
echo "repo root: ${REPO_ROOT}"
echo "dataset: ${DATASET_DIR}"
echo "pi3x target root: ${PI3X_TARGETS_ROOT_OVERRIDE}"
echo "num_gpus: ${NUM_GPUS}"
echo "num_workers per GPU/rank: ${NUM_WORKERS}"
echo "batch_size: ${BATCH_SIZE}"

run_target_passthrough_check

STAGE1_LOG="${SMOKE_LOG_DIR}/stage1-${SLURM_JOB_ID:-manual}.log"
STAGE2_LOG="${SMOKE_LOG_DIR}/stage2-${SLURM_JOB_ID:-manual}.log"

export CONFIG_NAME="${STAGE1_CONFIG_NAME}"
export EXP_NAME="${STAGE1_EXP_NAME}"
export RESUME="${STAGE1_RESUME}"
export NUM_TRAIN_STEPS="${STAGE1_NUM_TRAIN_STEPS}"
export SAVE_INTERVAL="${STAGE1_SAVE_INTERVAL}"
export KEEP_PERIOD="${STAGE1_KEEP_PERIOD}"
export PYTORCH_WEIGHT_PATH_OVERRIDE="${OPENPI_PI0_BASE_DIR}"

echo "===== SMOKE STAGE 1 ====="
run_common_launcher_to_log "${STAGE1_LOG}"
validate_aux_log "stage1" "${STAGE1_LOG}"

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

echo "===== SMOKE STAGE 2 ====="
run_common_launcher_to_log "${STAGE2_LOG}"
validate_aux_log "stage2" "${STAGE2_LOG}"

echo "smoke_ok stage1_log=${STAGE1_LOG} stage2_log=${STAGE2_LOG}"

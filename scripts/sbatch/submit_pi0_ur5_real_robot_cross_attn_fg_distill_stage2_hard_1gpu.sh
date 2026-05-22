#!/bin/bash

set -euo pipefail

resolve_repo_root() {
  local candidate
  local script_root=""

  if script_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd); then
    :
  else
    script_root=""
  fi

  for candidate in "${REPO_ROOT:-}" "${OPENPI_CAM_ROOT:-}" "${PWD:-}" "${script_root}"; do
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
  echo "Unable to locate openpi_cam repo root. Set REPO_ROOT or run from inside the repo." >&2
  exit 1
}

TARGET_SCRIPT="${REPO_ROOT}/scripts/sbatch/train_pi0_ur5_real_robot_cross_attn_fg_distill_stage2_hard_1gpu.sbatch"

SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-}
SBATCH_PARTITION=${SBATCH_PARTITION:-}
SBATCH_GRES=${SBATCH_GRES:-}
SBATCH_TIME=${SBATCH_TIME:-}
SBATCH_CPUS=${SBATCH_CPUS:-}
SBATCH_MEM=${SBATCH_MEM:-}

SBATCH_ARGS=()
[[ -n "${SBATCH_ACCOUNT}" ]] && SBATCH_ARGS+=(--account="${SBATCH_ACCOUNT}")
[[ -n "${SBATCH_PARTITION}" ]] && SBATCH_ARGS+=(--partition="${SBATCH_PARTITION}")
[[ -n "${SBATCH_GRES}" ]] && SBATCH_ARGS+=(--gres="${SBATCH_GRES}")
[[ -n "${SBATCH_TIME}" ]] && SBATCH_ARGS+=(--time="${SBATCH_TIME}")
[[ -n "${SBATCH_CPUS}" ]] && SBATCH_ARGS+=(--cpus-per-task="${SBATCH_CPUS}")
[[ -n "${SBATCH_MEM}" ]] && SBATCH_ARGS+=(--mem="${SBATCH_MEM}")

echo "Submitting UR5 real-robot fg + hard Stage 2 (1 GPU)"
echo "repo root: ${REPO_ROOT}"
echo "target script: ${TARGET_SCRIPT}"
echo "config: ${CONFIG_NAME:-pi0_ur5_real_robot_pytorch_cross_attn_fg_distill_stage2_hard}"
echo "dataset dir: ${DATASET_DIR:-/scratch/${USER}/real_robot_data/ur5_lab_test_tube_camera_shifts}"
echo "pi3x target root: ${OPENPI_PI3X_TARGETS_224_BASE_DIR:-/scratch/${USER}/pi3x_targets_224}/ur5_lab_test_tube_camera_shifts"
echo "stage1 checkpoint root: ${STAGE1_CHECKPOINT_ROOT:-/scratch/${USER}/tmp/openpi_cam/checkpoints/pi0_ur5_real_robot_pytorch_cross_attn_fg_distill_stage1_hard/pi0_ur5_real_robot_cross_attn_fg_distill_hard_stage1}"
echo "stage1 checkpoint step: ${STAGE1_CHECKPOINT_STEP:-5000}"
echo "checkpoint base dir: ${CHECKPOINT_BASE_DIR:-/scratch/${USER}/tmp/openpi_cam/checkpoints}"
echo "exp name: ${EXP_NAME:-pi0_ur5_real_robot_cross_attn_fg_distill_hard_stage2_1gpu}"
echo "default account: ${SBATCH_ACCOUNT:-torch_pr_69_tandon_advanced}"
echo "default partition: ${SBATCH_PARTITION:-h100_tandon,h200_tandon,a100_tandon}"
echo "default gres: ${SBATCH_GRES:-gpu:1}"
echo "default num train steps: ${NUM_TRAIN_STEPS:-30000}"
echo "default save interval: ${SAVE_INTERVAL:-1000}"

sbatch "${SBATCH_ARGS[@]}" "${TARGET_SCRIPT}"

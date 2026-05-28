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

TARGET_SCRIPT="${REPO_ROOT}/scripts/sbatch/train_pi05_ur5_real_robot_cross_attn_fg_distill_gt_stage1_stage2_hard_a100_2gpu_2day.sbatch"

SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-torch_pr_69_tandon_advanced}
SBATCH_PARTITION=${SBATCH_PARTITION:-a100_tandon,h100_tandon,h200_tandon}
SBATCH_GRES=${SBATCH_GRES:-gpu:2}
SBATCH_TIME=${SBATCH_TIME:-48:00:00}
SBATCH_CPUS=${SBATCH_CPUS:-32}
SBATCH_MEM=${SBATCH_MEM:-256G}

SBATCH_ARGS=()
[[ -n "${SBATCH_ACCOUNT}" ]] && SBATCH_ARGS+=(--account="${SBATCH_ACCOUNT}")
[[ -n "${SBATCH_PARTITION}" ]] && SBATCH_ARGS+=(--partition="${SBATCH_PARTITION}")
[[ -n "${SBATCH_GRES}" ]] && SBATCH_ARGS+=(--gres="${SBATCH_GRES}")
[[ -n "${SBATCH_TIME}" ]] && SBATCH_ARGS+=(--time="${SBATCH_TIME}")
[[ -n "${SBATCH_CPUS}" ]] && SBATCH_ARGS+=(--cpus-per-task="${SBATCH_CPUS}")
[[ -n "${SBATCH_MEM}" ]] && SBATCH_ARGS+=(--mem="${SBATCH_MEM}")

echo "Submitting pi0.5 UR5 real-robot fg + hard GT combined Stage 1 + Stage 2 (Tandon A100/H100/H200, 2 GPU, 2 day)"
echo "repo root: ${REPO_ROOT}"
echo "target script: ${TARGET_SCRIPT}"
echo "dataset dir: ${DATASET_DIR:-/scratch/${USER}/real_robot_data/ur5_lab_test_tube_camera_shifts}"
echo "array cache dir: ${DATASET_DIR:-/scratch/${USER}/real_robot_data/ur5_lab_test_tube_camera_shifts}_array_cache"
echo "norm stats root: ${REAL_ROBOT_NORM_ROOT:-/scratch/${USER}/pi05_ur5_real_robot}"
echo "gt target root: ${GT_POINT_TARGETS_ROOT_OVERRIDE:-/scratch/${USER}/gt_point_targets_grid224/ur5_lab_test_tube_camera_shifts}"
echo "checkpoint base dir: ${CHECKPOINT_BASE_DIR:-/scratch/${USER}/tmp/openpi_cam/checkpoints}"
echo "stage1 config: ${STAGE1_CONFIG_NAME:-pi05_ur5_real_robot_pytorch_cross_attn_fg_distill_gt_stage1_hard}"
echo "stage1 exp: ${STAGE1_EXP_NAME:-pi05_ur5_real_robot_cross_attn_fg_distill_gt_hard_stage1_tandon_2gpu_b16}"
echo "stage1 steps: ${STAGE1_NUM_TRAIN_STEPS:-5000}"
echo "stage2 config: ${STAGE2_CONFIG_NAME:-pi05_ur5_real_robot_pytorch_cross_attn_fg_distill_gt_stage2_hard}"
echo "stage2 exp: ${STAGE2_EXP_NAME:-pi05_ur5_real_robot_cross_attn_fg_distill_gt_hard_stage2_tandon_2gpu_b16}"
echo "stage2 steps: ${STAGE2_NUM_TRAIN_STEPS:-30000}"
echo "default account: ${SBATCH_ACCOUNT}"
echo "default partition: ${SBATCH_PARTITION}"
echo "default gres: ${SBATCH_GRES}"
echo "default time: ${SBATCH_TIME}"
echo "default cpus: ${SBATCH_CPUS}"
echo "default workers per GPU/rank: ${NUM_WORKERS:-8}"
echo "default batch size: ${BATCH_SIZE:-16}"

sbatch "${SBATCH_ARGS[@]}" "${TARGET_SCRIPT}"

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

TARGET_SCRIPT="${REPO_ROOT}/scripts/sbatch/train_pi0_ur5_pour_cross_attn_fg_distill_gt_stage1_stage2_hard_smoke.sbatch"
DEFAULT_DATASET_DIR="/scratch/${USER}/real_robot_data/ur5_place_and_pour_nuts_camera_shifts"
DEFAULT_GT_ROOT="/scratch/${USER}/gt_point_targets_grid224/ur5_place_and_pour_nuts_camera_shifts"

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

echo "Submitting Pi0 UR5 pour fg + hard GT zero-init Stage 1 + Stage 2 smoke"
echo "repo root: ${REPO_ROOT}"
echo "target script: ${TARGET_SCRIPT}"
echo "dataset dir: ${DATASET_DIR:-${DEFAULT_DATASET_DIR}}"
echo "array cache dir: ${DATASET_DIR:-${DEFAULT_DATASET_DIR}}_array_cache"
echo "gt target root: ${GT_POINT_TARGETS_ROOT_OVERRIDE:-${DEFAULT_GT_ROOT}}"
echo "pi3x target root: ${PI3X_TARGETS_ROOT_OVERRIDE:-<none>}"
echo "stage1 config: ${STAGE1_CONFIG_NAME:-pi0_ur5_pour_pytorch_cross_attn_fg_distill_gt_stage1_hard}"
echo "stage1 exp: ${STAGE1_EXP_NAME:-pi0_ur5_pour_cross_attn_fg_zeroinit_gt_hard_stage1_smoke}"
echo "stage2 config: ${STAGE2_CONFIG_NAME:-pi0_ur5_pour_pytorch_cross_attn_fg_distill_gt_stage2_hard}"
echo "stage2 exp: ${STAGE2_EXP_NAME:-pi0_ur5_pour_cross_attn_fg_zeroinit_gt_hard_stage2_smoke}"
echo "default partition: ${SBATCH_PARTITION:-a100_tandon,h100_tandon,h200_tandon}"
echo "default gres: ${SBATCH_GRES:-gpu:1}"
echo "default time: ${SBATCH_TIME:-00:45:00}"
echo "default workers per GPU/rank: ${NUM_WORKERS:-2}"
echo "default batch size: ${BATCH_SIZE:-2}"
echo "checks: pour paths, cached context_left/wrist_right video arrays, GT base/left_wrist targets, nonzero GT aux loss, stage1 parameter deltas for cross_view_fusion/ray_embed/aux_point_head, stage2 checkpoint handoff"
echo "preflight only: ${SMOKE_PREFLIGHT_ONLY:-false}"

sbatch "${SBATCH_ARGS[@]}" "${TARGET_SCRIPT}"

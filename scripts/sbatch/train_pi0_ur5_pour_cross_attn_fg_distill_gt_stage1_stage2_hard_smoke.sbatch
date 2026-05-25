#!/bin/bash
#SBATCH --job-name=pi0_pour_gt_smoke
#SBATCH --output=/scratch/%u/openpi_cam/log/pi0_ur5_pour_real_robot_ft/slurm-%j.out
#SBATCH --error=/scratch/%u/openpi_cam/log/pi0_ur5_pour_real_robot_ft/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --account=torch_pr_69_tandon_advanced
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
if [[ -z "${REPO_ROOT}" && -d "/scratch/${USER}/openpi_cam" ]]; then
  REPO_ROOT="/scratch/${USER}/openpi_cam"
fi
if [[ -z "${REPO_ROOT}" ]]; then
  echo "Unable to locate repo root." >&2
  exit 1
fi

export OPENPI_GEO_ROOT="${OPENPI_GEO_ROOT:-/scratch/${USER}}"
export OPENPI_PI0_BASE_DIR="${OPENPI_PI0_BASE_DIR:-${OPENPI_GEO_ROOT}/pi0_base}"
export REAL_ROBOT_NORM_ROOT="${REAL_ROBOT_NORM_ROOT:-${OPENPI_GEO_ROOT}/pi0_ur5_real_robot}"
export OPENPI_PI0_LIBERO_NORM_DIR="${REAL_ROBOT_NORM_ROOT}"

STAGE1_CONFIG_NAME="${STAGE1_CONFIG_NAME:-pi0_ur5_pour_pytorch_cross_attn_fg_distill_gt_stage1_hard}"
STAGE2_CONFIG_NAME="${STAGE2_CONFIG_NAME:-pi0_ur5_pour_pytorch_cross_attn_fg_distill_gt_stage2_hard}"
STAGE1_EXP_NAME="${STAGE1_EXP_NAME:-pi0_ur5_pour_cross_attn_fg_zeroinit_gt_hard_stage1_smoke}"
STAGE2_EXP_NAME="${STAGE2_EXP_NAME:-pi0_ur5_pour_cross_attn_fg_zeroinit_gt_hard_stage2_smoke}"
SMOKE_LOG_DIR="${SMOKE_LOG_DIR:-/scratch/${USER}/openpi_cam/log/pi0_ur5_pour_real_robot_ft/smoke_gt_aux_checks}"
SMOKE_DISPLAY_NAME="PI0"
SMOKE_EXPECT_PI05=false
SMOKE_MODEL_TYPE=PI0

export STAGE1_CONFIG_NAME STAGE2_CONFIG_NAME STAGE1_EXP_NAME STAGE2_EXP_NAME
export SMOKE_LOG_DIR SMOKE_DISPLAY_NAME SMOKE_EXPECT_PI05 SMOKE_MODEL_TYPE

source "${REPO_ROOT}/scripts/sbatch/ur5_pour_fg_gt_smoke_body.sh"

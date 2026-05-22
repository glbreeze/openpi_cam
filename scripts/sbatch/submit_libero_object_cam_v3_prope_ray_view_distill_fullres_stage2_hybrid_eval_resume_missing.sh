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

export TARGET_SCRIPT="${REPO_ROOT}/scripts/sbatch/eval_pi0_libero_object_cam_v3_pytorch_baseline_full.sbatch"
export CONFIG_NAME="${CONFIG_NAME:-pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage2_hybrid}"
export EXP_NAME="${EXP_NAME:-pi0_libero_object_cam_v3_prope_ray_view_distill_fullres_twostage_decouple_fg_hybrid_stage2_2gpu}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage2_hybrid/${EXP_NAME}}"
export SUITE_NAME="${SUITE_NAME:-libero_object}"
export STEP_STRIDE="${STEP_STRIDE:-5000}"
export LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/libero_eval/${EXP_NAME}/${SUITE_NAME}}"
export NUM_PARALLEL_CLIENTS="${NUM_PARALLEL_CLIENTS:-4}"

exec "${REPO_ROOT}/scripts/sbatch/submit_pi0_libero_object_cam_v3_eval_resume_missing.sh"

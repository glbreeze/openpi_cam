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

TARGET_SCRIPT="${REPO_ROOT}/scripts/sbatch/train_pi0_libero_cam_v3_prope_ray_view_distill_fullres_stage2_fgfg_hard_2gpu.sbatch"
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

echo "Submitting fgfg + hard Stage 2 (2 GPU)"
echo "repo root: ${REPO_ROOT}"
echo "target script: ${TARGET_SCRIPT}"
echo "config: pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage2_fgfg_hard"
echo "exp name: ${EXP_NAME:-pi0_libero_object_cam_v3_prope_ray_view_distill_fullres_twostage_decouple_fgfg_hard_stage2_2gpu}"
echo "stage1 checkpoint root: ${STAGE1_CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage1_fgfg_pi3xray_hard/pi0_libero_object_cam_v3_prope_ray_view_distill_fullres_twostage_decouple_fgfg_hard_stage1_l40s}"
echo "stage1 checkpoint step: ${STAGE1_CHECKPOINT_STEP:-5000}"
echo "dataset repo id: ${DATASET_REPO_ID:-glbreeze/libero_object_cam_v3}"
echo "lerobot home: ${HF_LEROBOT_HOME:-/scratch/${USER}/.huggingface/lerobot}"
echo "default account: ${SBATCH_ACCOUNT:-torch_pr_69_tandon_advanced}"
echo "default partition: ${SBATCH_PARTITION:-h100_tandon,h200_tandon,a100_tandon}"
echo "default gres: ${SBATCH_GRES:-gpu:2}"

sbatch "${SBATCH_ARGS[@]}" "${TARGET_SCRIPT}"

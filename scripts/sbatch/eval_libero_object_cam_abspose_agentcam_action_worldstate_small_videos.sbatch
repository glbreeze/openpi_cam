#!/bin/bash
# Small camera-frame-action eval focused on saving rollout videos for debugging.
# Default scale:
#   - 1 task (task 0)
#   - 2 trials per task
#   - original + 2 camera variations
# This should produce 6 videos if all rollouts run.

#SBATCH --job-name=absact_vid_eval
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/libero_object_multicam_eval/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/libero_object_multicam_eval/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_637_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1

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
if [[ -z "${REPO_ROOT}" && -d "/scratch/yp2841/geometry-vla/openpi_cam" ]]; then
  REPO_ROOT="/scratch/yp2841/geometry-vla/openpi_cam"
fi
if [[ -z "${REPO_ROOT}" ]]; then
  echo "Unable to locate repo root." >&2
  exit 1
fi

GEO_ROOT=${GEO_ROOT:-$(cd -- "${REPO_ROOT}/.." && pwd)}

export MODEL_NAME=${MODEL_NAME:-camtrain30k_abspose_agentcam_action_worldstate_nocross_small_videos}
export CONFIG_NAME=${CONFIG_NAME:-pi0_libero_object_cam_pytorch_absolute_pose_agentcam_action_worldstate_nocross_finetune}
export CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/pi0_libero_object_cam_pytorch_absolute_pose_agentcam_action_worldstate_nocross_finetune/pi0_libero_object_camtrain_o00_02_03_05_06_09_10_abspose_agentcam_action_worldstate_nocross_30k_twostage_v1}
export CHECKPOINT_STEP=${CHECKPOINT_STEP:-30000}
export CHECKPOINT_ASSET_ID=${CHECKPOINT_ASSET_ID:-glbreeze/libero_object_cam_train_o00_02_03_05_06_09_10_abspose_agentcam_action_worldstate}
export SERVE_ASSET_ID=${SERVE_ASSET_ID:-glbreeze/libero_abspose_agentcam_action_worldstate_evalalias}

export SUITE_NAME=${SUITE_NAME:-libero_object}
export TASK_ID_START=${TASK_ID_START:-0}
export TASK_ID_END=1
export NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-2}
export NUM_PARALLEL_CLIENTS=${NUM_PARALLEL_CLIENTS:-1}

export CAMERA_VARIATION_CONFIG=${CAMERA_VARIATION_CONFIG:-${GEO_ROOT}/LIBERO-Camera/configs/camera_variation/libero_object_left_variation.json}
export CAMERA_VARIATION_COUNT=${CAMERA_VARIATION_COUNT:-11}
export CAMERA_VARIATION_INCLUDE_ORIGINAL=${CAMERA_VARIATION_INCLUDE_ORIGINAL:-true}
export CAMERA_VARIATION_TARGET_CAMERA=${CAMERA_VARIATION_TARGET_CAMERA:-agentview}
export SAVE_VIDEOS=${SAVE_VIDEOS:-true}

export PORT_BASE=${PORT_BASE:-26620}
export LOG_ROOT=${LOG_ROOT:-${REPO_ROOT}/log/libero_object_multicam_eval/camtrain30k_abspose_agentcam_action_worldstate_nocross_small_videos}
export VIDEO_OUT_ROOT=${VIDEO_OUT_ROOT:-${LOG_ROOT}/videos}

exec bash "${REPO_ROOT}/scripts/sbatch/infer_libero_object_multicam_parallel.sbatch"

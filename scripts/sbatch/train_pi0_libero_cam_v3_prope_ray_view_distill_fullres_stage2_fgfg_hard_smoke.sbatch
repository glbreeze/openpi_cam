#!/bin/bash
#SBATCH --job-name=pi0_cam_v3_s2_fgh_sm
#SBATCH --output=/scratch/%u/openpi_cam/log/pi0_libero_object_ft/slurm-%j.out
#SBATCH --error=/scratch/%u/openpi_cam/log/pi0_libero_object_ft/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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

export USE_CAM=true
export CONFIG_NAME=pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage2_fgfg_hard
export TRAIN_RECIPE=single_stage
export WANDB_ENABLED="${WANDB_ENABLED:-true}"
export WANDB_PROJECT="${WANDB_PROJECT:-pi0_libero_cam}"
export WANDB_ENTITY="${WANDB_ENTITY:-NYU-robotics}"
export WANDB_API_KEY_OVERRIDE="${WANDB_API_KEY_OVERRIDE:-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}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/scratch/${USER}/.huggingface/lerobot}"
export NUM_GPUS="${NUM_GPUS:-1}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-20}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
export KEEP_PERIOD="${KEEP_PERIOD:-10}"

export DATASET_REPO_ID="${DATASET_REPO_ID:-glbreeze/libero_object_cam_v3}"
export NORM_ASSET_ID="${NORM_ASSET_ID:-glbreeze/libero_object_cam_v3}"
export EXP_NAME="${EXP_NAME:-pi0_libero_object_cam_v3_prope_ray_view_distill_fullres_twostage_decouple_fgfg_hard_stage2_smoke}"
export STAGE1_CHECKPOINT_ROOT="${STAGE1_CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage1_fgfg_pi3xray_hard/pi0_libero_object_cam_v3_prope_ray_view_distill_fullres_twostage_decouple_fgfg_hard_stage1_l40s}"
export STAGE1_CHECKPOINT_STEP="${STAGE1_CHECKPOINT_STEP:-5000}"
export PYTORCH_WEIGHT_PATH_OVERRIDE="${STAGE1_CHECKPOINT_ROOT}/${STAGE1_CHECKPOINT_STEP}"

exec bash "${REPO_ROOT}/scripts/sbatch/train_pi0_libero_object_ft.sbatch"

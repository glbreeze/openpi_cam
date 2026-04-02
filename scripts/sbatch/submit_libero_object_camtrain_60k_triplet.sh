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
TRAIN_SCRIPT="${REPO_ROOT}/scripts/sbatch/train_pi0_libero_object_ft.sbatch"

DATASET_REPO_ID=${DATASET_REPO_ID:-glbreeze/libero_object_cam_train_o00_02_03_05_06_09_10}
NORM_ASSET_ID=${NORM_ASSET_ID:-${DATASET_REPO_ID}}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-60000}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
KEEP_PERIOD=${KEEP_PERIOD:-5000}
WANDB_PROJECT=${WANDB_PROJECT:-openpi_cam_libero_object}
COMMON_EXP_PREFIX=${COMMON_EXP_PREFIX:-pi0_libero_object_camtrain_o00_02_03_05_06_09_10}

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

echo "Submitting 60k training triplet"
echo "repo root: ${REPO_ROOT}"
echo "dataset repo id: ${DATASET_REPO_ID}"
echo "norm asset id: ${NORM_ASSET_ID}"
echo "num train steps: ${NUM_TRAIN_STEPS}"
echo "save interval: ${SAVE_INTERVAL}"
echo "keep period: ${KEEP_PERIOD}"
echo "wandb project: ${WANDB_PROJECT}"
echo "common exp prefix: ${COMMON_EXP_PREFIX}"

baseline_job=$(
  sbatch --parsable "${SBATCH_ARGS[@]}" \
    --job-name=pi0_obj60k_baseline \
    --export=ALL,USE_CAM=false,CONFIG_NAME=pi0_libero_pytorch_full_finetune,EXP_NAME="${COMMON_EXP_PREFIX}_baseline_60k",DATASET_REPO_ID="${DATASET_REPO_ID}",NORM_ASSET_ID="${NORM_ASSET_ID}",NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS}",SAVE_INTERVAL="${SAVE_INTERVAL}",KEEP_PERIOD="${KEEP_PERIOD}",WANDB_PROJECT="${WANDB_PROJECT}" \
    "${TRAIN_SCRIPT}"
)

simple_job=$(
  sbatch --parsable "${SBATCH_ARGS[@]}" \
    --job-name=pi0_obj60k_simple \
    --export=ALL,USE_CAM=true,CONFIG_NAME=pi0_libero_cam_pytorch_full_finetune,EXP_NAME="${COMMON_EXP_PREFIX}_simple_60k",DATASET_REPO_ID="${DATASET_REPO_ID}",NORM_ASSET_ID="${NORM_ASSET_ID}",NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS}",SAVE_INTERVAL="${SAVE_INTERVAL}",KEEP_PERIOD="${KEEP_PERIOD}",WANDB_PROJECT="${WANDB_PROJECT}",CROSS_VIEW_TYPE=simple \
    "${TRAIN_SCRIPT}"
)

standard_job=$(
  sbatch --parsable "${SBATCH_ARGS[@]}" \
    --job-name=pi0_obj60k_standard \
    --export=ALL,USE_CAM=true,CONFIG_NAME=pi0_libero_cam_pytorch_full_finetune_standard,EXP_NAME="${COMMON_EXP_PREFIX}_standard_60k",DATASET_REPO_ID="${DATASET_REPO_ID}",NORM_ASSET_ID="${NORM_ASSET_ID}",NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS}",SAVE_INTERVAL="${SAVE_INTERVAL}",KEEP_PERIOD="${KEEP_PERIOD}",WANDB_PROJECT="${WANDB_PROJECT}",CROSS_VIEW_TYPE=standard \
    "${TRAIN_SCRIPT}"
)

printf 'baseline=%s\nsimple=%s\nstandard=%s\n' "${baseline_job}" "${simple_job}" "${standard_job}"

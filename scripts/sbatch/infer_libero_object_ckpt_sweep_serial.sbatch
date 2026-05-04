#!/bin/bash
#SBATCH --job-name=libero_obj_ckpt_sweep
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/libero_object_eval/ckpt_sweep/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/libero_object_eval/ckpt_sweep/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --account=torch_pr_637_tandon_advanced
#SBATCH --partition=h100_tandon
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

  for candidate in "${REPO_ROOT:-}" "${OPENPI_CAM_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "${PWD:-}" "${script_root}" "/scratch/yp2841/geometry-vla/openpi_cam"; do
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

if [[ -n "${REPO_ROOT:-}" ]] && [[ ! -f "${REPO_ROOT}/scripts/env/activate_env.sh" ]]; then
  REPO_ROOT=""
fi
if [[ -z "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(resolve_repo_root || true)"
fi
if [[ -z "${REPO_ROOT}" && -d "/scratch/yp2841/geometry-vla/openpi_cam" ]]; then
  REPO_ROOT="/scratch/yp2841/geometry-vla/openpi_cam"
fi
if [[ -z "${REPO_ROOT}" ]]; then
  echo "Unable to locate repo root. Set REPO_ROOT or submit from inside the repo." >&2
  exit 1
fi
INFER_SCRIPT="${REPO_ROOT}/scripts/sbatch/infer_libero_object_parallel.sbatch"

MODEL_NAME=${MODEL_NAME:-pi0_cam_v3_fr_s2_fgfg_2gpu_b16_eval50_h100_sweep_serial}
CONFIG_NAME=${CONFIG_NAME:-pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage2_fgfg}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage2_fgfg/pi0_libero_object_cam_v3_prope_ray_view_distill_fullres_twostage_v1_stage2_fgfg_2gpu_b16}
CHECKPOINT_ASSET_ID=${CHECKPOINT_ASSET_ID:-glbreeze/libero_object_cam_v3}
SERVE_ASSET_ID=${SERVE_ASSET_ID:-glbreeze/libero_object_cam_v3}

SUITE_NAME=${SUITE_NAME:-libero_object}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-50}
NUM_PARALLEL_CLIENTS=${NUM_PARALLEL_CLIENTS:-10}
TASK_ID_START=${TASK_ID_START:-0}
TASK_ID_END=${TASK_ID_END:--1}

STEP_INTERVAL=${STEP_INTERVAL:-5000}
BASE_PORT=${BASE_PORT:-28500}
LOG_ROOT=${LOG_ROOT:-${REPO_ROOT}/log/libero_object_eval/ckpt_sweep/${MODEL_NAME}}

mkdir -p "${LOG_ROOT}"

collect_steps() {
  local ckpt_root="$1"
  local -a steps=()
  local path
  for path in "${ckpt_root}"/*; do
    [[ -d "${path}" ]] || continue
    local step
    step=$(basename -- "${path}")
    [[ "${step}" =~ ^[0-9]+$ ]] || continue
    if (( step % STEP_INTERVAL == 0 )); then
      steps+=("${step}")
    fi
  done

  if ((${#steps[@]} == 0)); then
    return 1
  fi

  printf '%s\n' "${steps[@]}" | sort -n
}

if [[ ! -d "${CHECKPOINT_ROOT}" ]]; then
  echo "Missing checkpoint root: ${CHECKPOINT_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${INFER_SCRIPT}" ]]; then
  echo "Missing infer script: ${INFER_SCRIPT}" >&2
  exit 1
fi

mapfile -t steps < <(collect_steps "${CHECKPOINT_ROOT}")

manifest="${LOG_ROOT}/serial_manifest_${SLURM_JOB_ID:-manual}.tsv"
printf 'step\tport\tjob_id\tstatus\n' > "${manifest}"

port_offset=0
for step in "${steps[@]}"; do
  port=$((BASE_PORT + port_offset))
  printf 'Running checkpoint step %s on port %s\n' "${step}" "${port}"
  printf '%s\t%s\t%s\tstarted\n' "${step}" "${port}" "${SLURM_JOB_ID:-manual}" >> "${manifest}"

  REPO_ROOT="${REPO_ROOT}" \
  MODEL_NAME="${MODEL_NAME}" \
  CONFIG_NAME="${CONFIG_NAME}" \
  CHECKPOINT_ROOT="${CHECKPOINT_ROOT}" \
  CHECKPOINT_STEP="${step}" \
  CHECKPOINT_ASSET_ID="${CHECKPOINT_ASSET_ID}" \
  SERVE_ASSET_ID="${SERVE_ASSET_ID}" \
  SUITE_NAME="${SUITE_NAME}" \
  NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}" \
  NUM_PARALLEL_CLIENTS="${NUM_PARALLEL_CLIENTS}" \
  TASK_ID_START="${TASK_ID_START}" \
  TASK_ID_END="${TASK_ID_END}" \
  SERVER_BATCH_MAX_SIZE="${NUM_PARALLEL_CLIENTS}" \
  PORT="${port}" \
  LOG_ROOT="${LOG_ROOT}" \
  bash "${INFER_SCRIPT}"

  printf '%s\t%s\t%s\tcompleted\n' "${step}" "${port}" "${SLURM_JOB_ID:-manual}" >> "${manifest}"
  port_offset=$((port_offset + 1))
done

echo "Serial sweep complete."
echo "Manifest: ${manifest}"

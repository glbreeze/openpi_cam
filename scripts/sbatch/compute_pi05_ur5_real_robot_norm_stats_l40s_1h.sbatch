#!/bin/bash
#SBATCH --job-name=pi05_ur5_norm
#SBATCH --output=/scratch/%u/openpi_cam/log/pi05_ur5_real_robot_norm/slurm-%j.out
#SBATCH --error=/scratch/%u/openpi_cam/log/pi05_ur5_real_robot_norm/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --account=torch_pr_69_general
#SBATCH --partition=l40s_public
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

source "${REPO_ROOT}/scripts/env/activate_env.sh"

export OPENPI_GEO_ROOT="${OPENPI_GEO_ROOT:-/scratch/${USER}}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${OPENPI_GEO_ROOT}/.huggingface/lerobot}"
export DATASET_DIR="${DATASET_DIR:-/scratch/${USER}/real_robot_data/ur5_lab_test_tube_camera_shifts}"
export DATASET_REPO_ID="${DATASET_REPO_ID:-ur5_lab_test_tube_camera_shifts}"
export OPENPI_PI3X_TARGETS_224_BASE_DIR="${OPENPI_PI3X_TARGETS_224_BASE_DIR:-/scratch/${USER}/openpi_cache/pi3x_targets_224}"
export CONFIG_NAME="${CONFIG_NAME:-pi05_ur5_real_robot_pytorch_baseline}"
export MAX_FRAMES="${MAX_FRAMES:-}"

mkdir -p "/scratch/${USER}/openpi_cam/log/pi05_ur5_real_robot_norm"

echo "repo root: ${REPO_ROOT}"
echo "config name: ${CONFIG_NAME}"
echo "dataset dir: ${DATASET_DIR}"
echo "dataset repo id: ${DATASET_REPO_ID}"
echo "openpi geo root: ${OPENPI_GEO_ROOT}"
echo "pi3x targets 224 base dir: ${OPENPI_PI3X_TARGETS_224_BASE_DIR}"
if [[ -n "${MAX_FRAMES}" ]]; then
  echo "max frames: ${MAX_FRAMES}"
fi

CMD=(
  uv run python "${REPO_ROOT}/scripts/compute_norm_stats.py"
  --config-name "${CONFIG_NAME}"
)

if [[ -n "${MAX_FRAMES}" ]]; then
  CMD+=(--max-frames "${MAX_FRAMES}")
fi

printf 'command:'
printf ' %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"

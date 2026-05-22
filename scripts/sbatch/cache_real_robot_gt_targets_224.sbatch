#!/bin/bash
#SBATCH --job-name=ur5_gt_cache_224
#SBATCH --output=/scratch/%u/openpi_cam/log/ur5_gt_cache/slurm-%j.out
#SBATCH --error=/scratch/%u/openpi_cam/log/ur5_gt_cache/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
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
export DATASET_DIR="${DATASET_DIR:-/scratch/${USER}/real_robot_data/ur5_lab_test_tube_camera_shifts}"
export GT_OUTPUT_ROOT="${GT_OUTPUT_ROOT:-/scratch/${USER}/openpi_cache/pi3x_targets_224/ur5_lab_test_tube_camera_shifts}"
export OPENPI_PI3X_TARGETS_224_BASE_DIR="${OPENPI_PI3X_TARGETS_224_BASE_DIR:-/scratch/${USER}/openpi_cache/pi3x_targets_224}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export DEVICE="${DEVICE:-cuda}"
export EPISODE_RANGE="${EPISODE_RANGE:-}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

mkdir -p \
  "/scratch/${USER}/openpi_cam/log/ur5_gt_cache" \
  "$(dirname "${GT_OUTPUT_ROOT}")" \
  "/scratch/${USER}/tmp"

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/env/activate_env.sh"

echo "repo root: ${REPO_ROOT}"
echo "dataset dir: ${DATASET_DIR}"
echo "gt output root: ${GT_OUTPUT_ROOT}"
echo "OPENPI_PI3X_TARGETS_224_BASE_DIR=${OPENPI_PI3X_TARGETS_224_BASE_DIR}"
echo "batch size: ${BATCH_SIZE}"
echo "device: ${DEVICE}"
[[ -n "${EPISODE_RANGE}" ]] && echo "episode range: ${EPISODE_RANGE}"

cmd=(
  python
  "${REPO_ROOT}/scripts/cache_real_robot_gt_targets.py"
  --dataset-root "${DATASET_DIR}"
  --output-root "${GT_OUTPUT_ROOT}"
  --device "${DEVICE}"
  --batch-size "${BATCH_SIZE}"
  --log-level "${LOG_LEVEL}"
)

if [[ -n "${EPISODE_RANGE}" ]]; then
  cmd+=(--episode-range "${EPISODE_RANGE}")
fi

printf 'running command:'
printf ' %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"

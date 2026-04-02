#!/bin/bash
#SBATCH --job-name=validate_libero_object_hdf5
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/libero_object_validate/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/libero_object_validate/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --account=torch_pr_637_general
#SBATCH --partition=cpu_short
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

  for candidate in "${REPO_ROOT:-}" "${OPENPI_CAM_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "${script_root}"; do
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
  echo "Unable to locate repo root. Set REPO_ROOT or submit from inside the repo." >&2
  exit 1
}
GEO_ROOT=${GEO_ROOT:-$(cd -- "${REPO_ROOT}/.." && pwd)}

RAW_HDF5_ROOT=${RAW_HDF5_ROOT:-${GEO_ROOT}/libero_object_camera_left}
INCLUDE_CAMERA_LABELS=${INCLUDE_CAMERA_LABELS:-}
EXCLUDE_CAMERA_LABELS=${EXCLUDE_CAMERA_LABELS:-}
VALIDATE_NUM_WORKERS=${VALIDATE_NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}
DELETE_BAD=${DELETE_BAD:-false}

cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/env/activate_env.sh"

mkdir -p "${REPO_ROOT}/log/libero_object_validate"

if [[ ! -d "${RAW_HDF5_ROOT}" ]]; then
  echo "Missing raw HDF5 directory at ${RAW_HDF5_ROOT}" >&2
  exit 1
fi

echo "===== LIBERO HDF5 VALIDATION ENV CHECK ====="
echo "repo root: ${REPO_ROOT}"
echo "raw hdf5 root: ${RAW_HDF5_ROOT}"
echo "include camera labels: ${INCLUDE_CAMERA_LABELS:-<none>}"
echo "exclude camera labels: ${EXCLUDE_CAMERA_LABELS:-<none>}"
echo "validate workers: ${VALIDATE_NUM_WORKERS}"
echo "delete bad: ${DELETE_BAD}"
which python
python -V
hostname

VALIDATE_ARGS=(
  --dataset-root "${RAW_HDF5_ROOT}"
  --num-workers "${VALIDATE_NUM_WORKERS}"
)

if [[ -n "${INCLUDE_CAMERA_LABELS}" ]]; then
  VALIDATE_ARGS+=(--include-camera-labels "${INCLUDE_CAMERA_LABELS}")
fi
if [[ -n "${EXCLUDE_CAMERA_LABELS}" ]]; then
  VALIDATE_ARGS+=(--exclude-camera-labels "${EXCLUDE_CAMERA_LABELS}")
fi
if [[ "${DELETE_BAD,,}" =~ ^(1|true|yes|y|on)$ ]]; then
  VALIDATE_ARGS+=(--delete-bad)
fi

python examples/libero/validate_libero_hdf5_integrity.py "${VALIDATE_ARGS[@]}"

echo "Validation completed successfully."

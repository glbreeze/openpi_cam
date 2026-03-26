#!/bin/bash
#SBATCH --job-name=convert_libero_object_hdf5
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/libero_object_convert/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/libero_object_convert/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_637_general
#SBATCH --partition=cs
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
ORIGINAL_REPO_ID=${ORIGINAL_REPO_ID:-glbreeze/libero_object}
CAM_REPO_ID=${CAM_REPO_ID:-glbreeze/libero_object_cam}
ORIGINAL_MAX_FILES=${ORIGINAL_MAX_FILES:-0}
CAM_MAX_FILES=${CAM_MAX_FILES:-0}
MAX_EPISODES_PER_FILE=${MAX_EPISODES_PER_FILE:-0}
CONVERT_FPS=${CONVERT_FPS:-10}
CONVERT_IMAGE_SIZE=${CONVERT_IMAGE_SIZE:-256}
CACHE_ROOT=${CACHE_ROOT:-${GEO_ROOT}/.cache/huggingface}
OPENPI_CACHE_ROOT=${OPENPI_CACHE_ROOT:-${GEO_ROOT}/.cache/openpi}

JOB_TMP_BASE="${SLURM_TMPDIR:-/tmp/$USER}"
JOB_TMP_ROOT="${JOB_TMP_BASE}/openpi_cam_convert_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOB_TMP_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export DATASETS_TMPDIR="${JOB_TMP_ROOT}/datasets_tmp"
export HOME="${JOB_TMP_ROOT}/home"
export XDG_CACHE_HOME="${JOB_TMP_ROOT}/xdg_cache"
export XDG_CONFIG_HOME="${JOB_TMP_ROOT}/xdg_config"
export XDG_STATE_HOME="${JOB_TMP_ROOT}/xdg_state"

mkdir -p "${TMPDIR}" "${DATASETS_TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}"
chmod 700 "${TMPDIR}" "${DATASETS_TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" || true

cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/env/activate_env.sh"

export HF_HOME="${CACHE_ROOT}"
export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
export HF_HUB_CACHE="${CACHE_ROOT}/hub"
export OPENPI_CACHE_DIR="${OPENPI_CACHE_ROOT}"
export OPENPI_DATA_HOME="${OPENPI_CACHE_ROOT}"
unset TRANSFORMERS_CACHE

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${OPENPI_CACHE_DIR}"

if [[ ! -d "${RAW_HDF5_ROOT}" ]]; then
  echo "Missing raw HDF5 directory at ${RAW_HDF5_ROOT}" >&2
  exit 1
fi

echo "===== HDF5 CONVERSION ENV CHECK ====="
echo "repo root: ${REPO_ROOT}"
echo "raw hdf5 root: ${RAW_HDF5_ROOT}"
echo "original repo id: ${ORIGINAL_REPO_ID}"
echo "cam repo id: ${CAM_REPO_ID}"
echo "original max files: ${ORIGINAL_MAX_FILES}"
echo "cam max files: ${CAM_MAX_FILES}"
echo "max episodes per file: ${MAX_EPISODES_PER_FILE}"
echo "hf lerobot home: ${HF_LEROBOT_HOME}"
echo "norm stats root: ${OPENPI_PI0_LIBERO_NORM_DIR}"
which python
python -V
hostname

python examples/libero/convert_libero_hdf5_to_lerobot.py \
  --dataset-root "${RAW_HDF5_ROOT}" \
  --mode original_only \
  --repo-id "${ORIGINAL_REPO_ID}" \
  --max-files "${ORIGINAL_MAX_FILES}" \
  --max-episodes-per-file "${MAX_EPISODES_PER_FILE}" \
  --fps "${CONVERT_FPS}" \
  --image-size "${CONVERT_IMAGE_SIZE}"

python examples/libero/convert_libero_hdf5_to_lerobot.py \
  --dataset-root "${RAW_HDF5_ROOT}" \
  --mode all_views \
  --repo-id "${CAM_REPO_ID}" \
  --max-files "${CAM_MAX_FILES}" \
  --max-episodes-per-file "${MAX_EPISODES_PER_FILE}" \
  --fps "${CONVERT_FPS}" \
  --image-size "${CONVERT_IMAGE_SIZE}"

python scripts/compute_lerobot_norm_stats.py \
  --repo-id "${ORIGINAL_REPO_ID}" \
  --output-root "${OPENPI_PI0_LIBERO_NORM_DIR}"

python scripts/compute_lerobot_norm_stats.py \
  --repo-id "${CAM_REPO_ID}" \
  --output-root "${OPENPI_PI0_LIBERO_NORM_DIR}"

echo "HDF5 conversion and norm-stats generation completed successfully."

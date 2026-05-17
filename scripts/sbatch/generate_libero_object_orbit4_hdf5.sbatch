#!/bin/bash
#SBATCH --job-name=libero_obj_orbit4_gen
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/libero_object_orbit4_gen/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/libero_object_orbit4_gen/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
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
  echo "Unable to locate openpi_cam repo root." >&2
  exit 1
}
GEO_ROOT=${GEO_ROOT:-$(cd -- "${REPO_ROOT}/.." && pwd)}
LIBERO_CAMERA_ROOT=${LIBERO_CAMERA_ROOT:-${GEO_ROOT}/LIBERO-Camera}

PYTHON_BIN=${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}
DATASET_DOWNLOAD_ROOT=${DATASET_DOWNLOAD_ROOT:-${GEO_ROOT}/libero_raw_datasets}
DATASET_ROOT=${DATASET_ROOT:-${DATASET_DOWNLOAD_ROOT}/libero_object}
CAMERA_VARIATION_CONFIG=${CAMERA_VARIATION_CONFIG:-${LIBERO_CAMERA_ROOT}/configs/camera_variation/train_orbit_yaw_4view_pm25_pm8.json}
PREVIEW_OUTPUT_DIR=${PREVIEW_OUTPUT_DIR:-${GEO_ROOT}/libero_object_preview_orbit_yaw_4view_pm25_pm8}
DATASET_CAMERA_ROOT=${DATASET_CAMERA_ROOT:-${GEO_ROOT}/libero_object_camera_orbit_yaw_4view_pm25_pm8}
FAIL_LOG=${FAIL_LOG:-${DATASET_CAMERA_ROOT}/pipeline_failures.txt}
PREVIEW_FAIL_LOG=${PREVIEW_FAIL_LOG:-${DATASET_CAMERA_ROOT}/preview_failures.txt}
MAX_FILES=${MAX_FILES:-0}
SKIP_PREVIEW=${SKIP_PREVIEW:-true}
NUM_WORKERS=${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}

JOB_TMP_BASE="${SLURM_TMPDIR:-/tmp/$USER}"
JOB_TMP_ROOT="${JOB_TMP_BASE}/libero_object_orbit4_gen_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOB_TMP_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export HOME="${JOB_TMP_ROOT}/home"
export XDG_CACHE_HOME="${JOB_TMP_ROOT}/xdg_cache"
export XDG_CONFIG_HOME="${JOB_TMP_ROOT}/xdg_config"
export XDG_STATE_HOME="${JOB_TMP_ROOT}/xdg_state"
export LIBERO_CONFIG_PATH="${JOB_TMP_ROOT}/libero_config"

mkdir -p \
  "${TMPDIR}" \
  "${HOME}" \
  "${XDG_CACHE_HOME}" \
  "${XDG_CONFIG_HOME}" \
  "${XDG_STATE_HOME}" \
  "${LIBERO_CONFIG_PATH}" \
  "${REPO_ROOT}/log/libero_object_orbit4_gen" \
  "${DATASET_DOWNLOAD_ROOT}" \
  "${PREVIEW_OUTPUT_DIR}" \
  "${DATASET_CAMERA_ROOT}"
chmod 700 "${TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" || true

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing executable python at ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${CAMERA_VARIATION_CONFIG}" ]]; then
  echo "Missing camera variation config: ${CAMERA_VARIATION_CONFIG}" >&2
  exit 1
fi

cd "${LIBERO_CAMERA_ROOT}"

echo "===== LIBERO OBJECT ORBIT-4 CAMERA GEN ====="
echo "openpi_cam repo root: ${REPO_ROOT}"
echo "libero-camera repo root: ${LIBERO_CAMERA_ROOT}"
echo "python: ${PYTHON_BIN}"
echo "dataset root: ${DATASET_ROOT}"
echo "camera variation config: ${CAMERA_VARIATION_CONFIG}"
echo "preview output: ${PREVIEW_OUTPUT_DIR}"
echo "camera output: ${DATASET_CAMERA_ROOT}"
echo "max files: ${MAX_FILES}"
echo "skip preview: ${SKIP_PREVIEW}"
echo "num workers: ${NUM_WORKERS}"
"${PYTHON_BIN}" -V
hostname

if ! find "${DATASET_ROOT}" -maxdepth 1 -name '*.hdf5' -print -quit | grep -q .; then
  echo "No raw libero_object HDF5 found under ${DATASET_ROOT}; downloading from Hugging Face..."
  "${PYTHON_BIN}" benchmark_scripts/download_libero_datasets.py \
    --download-dir "${DATASET_DOWNLOAD_ROOT}" \
    --datasets libero_object \
    --use-huggingface
fi

PIPELINE_ARGS=(
  scripts/batch_camera_variation_pipeline_libero_object_left.py
  --dataset-root "${DATASET_ROOT}"
  --preview-output-dir "${PREVIEW_OUTPUT_DIR}"
  --dataset-camera-root "${DATASET_CAMERA_ROOT}"
  --camera-variation-config "${CAMERA_VARIATION_CONFIG}"
  --resume
  --continue-on-error
  --fail-log "${FAIL_LOG}"
  --preview-fail-log "${PREVIEW_FAIL_LOG}"
  --num-workers "${NUM_WORKERS}"
  --no-include-original-hdf5
)

if [[ "${MAX_FILES}" -gt 0 ]]; then
  PIPELINE_ARGS+=(--max-files "${MAX_FILES}")
fi

if [[ "${SKIP_PREVIEW}" == "true" ]]; then
  PIPELINE_ARGS+=(--skip-preview)
fi

"${PYTHON_BIN}" "${PIPELINE_ARGS[@]}"

echo "===== GENERATION SUMMARY ====="
echo "raw input count:"
find "${DATASET_ROOT}" -maxdepth 1 -name '*.hdf5' | wc -l
echo "generated output count:"
find "${DATASET_CAMERA_ROOT}" -name '*.hdf5' | wc -l

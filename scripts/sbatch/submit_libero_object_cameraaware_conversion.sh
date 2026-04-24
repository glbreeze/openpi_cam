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
  echo "Unable to locate repo root. Set REPO_ROOT or run from inside the repo." >&2
  exit 1
}
GEO_ROOT=${GEO_ROOT:-$(cd -- "${REPO_ROOT}/.." && pwd)}

RAW_HDF5_ROOT=${RAW_HDF5_ROOT:-${GEO_ROOT}/libero_object_cameraaware/libero_object}
OUTPUT_REPO_ID=${OUTPUT_REPO_ID:-glbreeze/libero_object_cam}
RUN_TAG=${RUN_TAG:-cameraaware_224_$(date +%Y%m%d_%H%M%S)}
TMP_REPO_PREFIX=${TMP_REPO_PREFIX:-glbreeze/libero_object_cam_tmp_shards_${RUN_TAG}}
SHARDS_ROOT=${SHARDS_ROOT:-${GEO_ROOT}/${TMP_REPO_PREFIX}}
FILE_COUNT=${FILE_COUNT:-$(find "${RAW_HDF5_ROOT}" -maxdepth 1 -name '*.hdf5' | wc -l)}
CONVERT_IMAGE_SIZE=${CONVERT_IMAGE_SIZE:-224}
CONVERT_FPS=${CONVERT_FPS:-10}
IMAGE_WRITER_THREADS=${IMAGE_WRITER_THREADS:-4}
IMAGE_WRITER_PROCESSES=${IMAGE_WRITER_PROCESSES:-2}
STAGE_FILES_TO_LOCAL=${STAGE_FILES_TO_LOCAL:-true}

if [[ ! -d "${RAW_HDF5_ROOT}" ]]; then
  echo "Missing raw HDF5 root at ${RAW_HDF5_ROOT}" >&2
  exit 1
fi

if [[ "${FILE_COUNT}" -lt 1 ]]; then
  echo "No HDF5 files found under ${RAW_HDF5_ROOT}" >&2
  exit 1
fi

mkdir -p \
  "${REPO_ROOT}/log/libero_object_convert_shard" \
  "${REPO_ROOT}/log/libero_object_merge"

echo "Submitting LIBERO object camera-aware conversion"
echo "repo root: ${REPO_ROOT}"
echo "raw hdf5 root: ${RAW_HDF5_ROOT}"
echo "output repo id: ${OUTPUT_REPO_ID}"
echo "tmp repo prefix: ${TMP_REPO_PREFIX}"
echo "shards root: ${SHARDS_ROOT}"
echo "file count: ${FILE_COUNT}"
echo "run tag: ${RUN_TAG}"

array_job=$(
  sbatch --parsable \
    --array="0-$(( FILE_COUNT - 1 ))" \
    --export=ALL,RAW_HDF5_ROOT="${RAW_HDF5_ROOT}",MODE=original_only,NUM_SHARDS="${FILE_COUNT}",REPO_ID_PREFIX="${TMP_REPO_PREFIX}",CONVERT_IMAGE_SIZE="${CONVERT_IMAGE_SIZE}",CONVERT_FPS="${CONVERT_FPS}",IMAGE_WRITER_THREADS="${IMAGE_WRITER_THREADS}",IMAGE_WRITER_PROCESSES="${IMAGE_WRITER_PROCESSES}",STAGE_FILES_TO_LOCAL="${STAGE_FILES_TO_LOCAL}" \
    "${REPO_ROOT}/scripts/sbatch/convert_libero_object_hdf5_shard.sbatch"
)

merge_job=$(
  sbatch --parsable \
    --dependency="afterok:${array_job}" \
    --export=ALL,SHARDS_ROOT="${SHARDS_ROOT}",OUTPUT_REPO_ID="${OUTPUT_REPO_ID}",COMPUTE_NORM_STATS=true \
    "${REPO_ROOT}/scripts/sbatch/merge_libero_object_lerobot_shards.sbatch"
)

printf 'convert_array=%s\nmerge_and_norm=%s\n' "${array_job}" "${merge_job}"

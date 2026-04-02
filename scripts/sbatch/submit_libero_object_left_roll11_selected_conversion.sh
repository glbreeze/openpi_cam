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
GEO_ROOT=${GEO_ROOT:-$(cd -- "${REPO_ROOT}/.." && pwd)}
LIBERO_CAMERA_ROOT=${LIBERO_CAMERA_ROOT:-${GEO_ROOT}/LIBERO-Camera}

RAW_DATASET_ROOT=${RAW_DATASET_ROOT:-${GEO_ROOT}/libero_raw_datasets/libero_object}
PREVIEW_OUTPUT_DIR=${PREVIEW_OUTPUT_DIR:-${GEO_ROOT}/libero_object_preview_left_roll11}
RAW_HDF5_ROOT=${RAW_HDF5_ROOT:-${GEO_ROOT}/libero_object_camera_left_roll11}
FAIL_LOG=${FAIL_LOG:-${RAW_HDF5_ROOT}/pipeline_failures.txt}
PREVIEW_FAIL_LOG=${PREVIEW_FAIL_LOG:-${RAW_HDF5_ROOT}/preview_failures.txt}

OUTPUT_REPO_ID=${OUTPUT_REPO_ID:-glbreeze/libero_object_cam_train_o00_02_03_05_06_09_10}
TMP_REPO_PREFIX=${TMP_REPO_PREFIX:-glbreeze/libero_object_cam_train_o00_02_03_05_06_09_10_tmp_shards}
SHARDS_ROOT=${SHARDS_ROOT:-${GEO_ROOT}/${TMP_REPO_PREFIX}}

INCLUDE_CAMERA_LABELS=${INCLUDE_CAMERA_LABELS:-original+camvar_00+camvar_02+camvar_03+camvar_05+camvar_06+camvar_09+camvar_10}
NUM_GEN_WORKERS=${NUM_GEN_WORKERS:-8}
NUM_CONVERT_SHARDS=${NUM_CONVERT_SHARDS:-8}
SKIP_PREVIEW=${SKIP_PREVIEW:-true}
MAX_FILES=${MAX_FILES:-0}
CONVERT_FPS=${CONVERT_FPS:-10}
CONVERT_IMAGE_SIZE=${CONVERT_IMAGE_SIZE:-256}
STAGE_FILES_TO_LOCAL=${STAGE_FILES_TO_LOCAL:-true}
IMAGE_WRITER_THREADS=${IMAGE_WRITER_THREADS:-4}
IMAGE_WRITER_PROCESSES=${IMAGE_WRITER_PROCESSES:-2}
GEN_PARTITION=${GEN_PARTITION:-cpu_short}
VALIDATE_PARTITION=${VALIDATE_PARTITION:-cpu_short}
CONVERT_PARTITION=${CONVERT_PARTITION:-cs}
MERGE_PARTITION=${MERGE_PARTITION:-cpu_short}
GEN_ACCOUNT=${GEN_ACCOUNT:-torch_pr_637_general}
VALIDATE_ACCOUNT=${VALIDATE_ACCOUNT:-${GEN_ACCOUNT}}
CONVERT_ACCOUNT=${CONVERT_ACCOUNT:-torch_pr_637_general}
MERGE_ACCOUNT=${MERGE_ACCOUNT:-torch_pr_637_general}
GEN_CPUS=${GEN_CPUS:-8}
GEN_MEM=${GEN_MEM:-96G}
GEN_TIME=${GEN_TIME:-04:00:00}
VALIDATE_CPUS=${VALIDATE_CPUS:-4}
VALIDATE_MEM=${VALIDATE_MEM:-16G}
VALIDATE_TIME=${VALIDATE_TIME:-00:30:00}
VALIDATE_NUM_WORKERS=${VALIDATE_NUM_WORKERS:-${VALIDATE_CPUS}}
CONVERT_CPUS=${CONVERT_CPUS:-4}
CONVERT_MEM=${CONVERT_MEM:-32G}
CONVERT_TIME=${CONVERT_TIME:-08:00:00}

mkdir -p \
  "${REPO_ROOT}/log/libero_object_convert_shard" \
  "${REPO_ROOT}/log/libero_object_validate" \
  "${REPO_ROOT}/log/libero_object_merge" \
  "${LIBERO_CAMERA_ROOT}/log/libero_object_left_cam"

echo "Submitting LIBERO object left-roll11 selected-view conversion pipeline"
echo "openpi_cam repo root: ${REPO_ROOT}"
echo "libero-camera repo root: ${LIBERO_CAMERA_ROOT}"
echo "raw dataset root: ${RAW_DATASET_ROOT}"
echo "generated hdf5 root: ${RAW_HDF5_ROOT}"
echo "preview output dir: ${PREVIEW_OUTPUT_DIR}"
echo "include camera labels: ${INCLUDE_CAMERA_LABELS}"
echo "num generation workers: ${NUM_GEN_WORKERS}"
echo "num convert shards: ${NUM_CONVERT_SHARDS}"
echo "output repo id: ${OUTPUT_REPO_ID}"
echo "tmp shard prefix: ${TMP_REPO_PREFIX}"
echo "image writer threads: ${IMAGE_WRITER_THREADS}"
echo "image writer processes: ${IMAGE_WRITER_PROCESSES}"
echo "generation partition: ${GEN_PARTITION}"
echo "validate partition: ${VALIDATE_PARTITION}"
echo "convert partition: ${CONVERT_PARTITION}"
echo "merge partition: ${MERGE_PARTITION}"
echo "generation account: ${GEN_ACCOUNT}"
echo "validate account: ${VALIDATE_ACCOUNT}"
echo "convert account: ${CONVERT_ACCOUNT}"
echo "merge account: ${MERGE_ACCOUNT}"
echo "generation cpus: ${GEN_CPUS}"
echo "generation mem: ${GEN_MEM}"
echo "generation time: ${GEN_TIME}"
echo "validate cpus: ${VALIDATE_CPUS}"
echo "validate mem: ${VALIDATE_MEM}"
echo "validate time: ${VALIDATE_TIME}"
echo "convert cpus: ${CONVERT_CPUS}"
echo "convert mem: ${CONVERT_MEM}"
echo "convert time: ${CONVERT_TIME}"

generate_job=$(
  sbatch --parsable \
    --job-name=libero_obj_left_roll11_gen \
    --account="${GEN_ACCOUNT}" \
    --partition="${GEN_PARTITION}" \
    --cpus-per-task="${GEN_CPUS}" \
    --mem="${GEN_MEM}" \
    --time="${GEN_TIME}" \
    --export=ALL,REPO_ROOT="${LIBERO_CAMERA_ROOT}",DATASET_ROOT="${RAW_DATASET_ROOT}",PREVIEW_OUTPUT_DIR="${PREVIEW_OUTPUT_DIR}",DATASET_CAMERA_ROOT="${RAW_HDF5_ROOT}",FAIL_LOG="${FAIL_LOG}",PREVIEW_FAIL_LOG="${PREVIEW_FAIL_LOG}",NUM_WORKERS="${NUM_GEN_WORKERS}",SKIP_PREVIEW="${SKIP_PREVIEW}",MAX_FILES="${MAX_FILES}" \
    "${LIBERO_CAMERA_ROOT}/scripts/sbatch/generate_libero_object_left_camera_hdf5.sbatch"
)

validate_job=$(
  sbatch --parsable \
    --dependency="afterok:${generate_job}" \
    --job-name=validate_left_roll11_sel \
    --account="${VALIDATE_ACCOUNT}" \
    --partition="${VALIDATE_PARTITION}" \
    --cpus-per-task="${VALIDATE_CPUS}" \
    --mem="${VALIDATE_MEM}" \
    --time="${VALIDATE_TIME}" \
    --export=ALL,RAW_HDF5_ROOT="${RAW_HDF5_ROOT}",INCLUDE_CAMERA_LABELS="${INCLUDE_CAMERA_LABELS}",VALIDATE_NUM_WORKERS="${VALIDATE_NUM_WORKERS}" \
    "${REPO_ROOT}/scripts/sbatch/validate_libero_object_hdf5_selected_views.sbatch"
)

convert_job=$(
  sbatch --parsable \
    --dependency="afterok:${validate_job}" \
    --array="0-$(( NUM_CONVERT_SHARDS - 1 ))" \
    --job-name=convert_left_roll11_sel \
    --account="${CONVERT_ACCOUNT}" \
    --partition="${CONVERT_PARTITION}" \
    --cpus-per-task="${CONVERT_CPUS}" \
    --mem="${CONVERT_MEM}" \
    --time="${CONVERT_TIME}" \
    --export=ALL,RAW_HDF5_ROOT="${RAW_HDF5_ROOT}",MODE=all_views,NUM_SHARDS="${NUM_CONVERT_SHARDS}",REPO_ID_PREFIX="${TMP_REPO_PREFIX}",INCLUDE_CAMERA_LABELS="${INCLUDE_CAMERA_LABELS}",CONVERT_FPS="${CONVERT_FPS}",CONVERT_IMAGE_SIZE="${CONVERT_IMAGE_SIZE}",IMAGE_WRITER_THREADS="${IMAGE_WRITER_THREADS}",IMAGE_WRITER_PROCESSES="${IMAGE_WRITER_PROCESSES}",STAGE_FILES_TO_LOCAL="${STAGE_FILES_TO_LOCAL}" \
    "${REPO_ROOT}/scripts/sbatch/convert_libero_object_hdf5_shard.sbatch"
)

merge_job=$(
  sbatch --parsable \
    --dependency="afterok:${convert_job}" \
    --job-name=merge_left_roll11_sel \
    --account="${MERGE_ACCOUNT}" \
    --partition="${MERGE_PARTITION}" \
    --export=ALL,SHARDS_ROOT="${SHARDS_ROOT}",OUTPUT_REPO_ID="${OUTPUT_REPO_ID}",COMPUTE_NORM_STATS=true \
    "${REPO_ROOT}/scripts/sbatch/merge_libero_object_lerobot_shards.sbatch"
)

printf 'generate=%s\nvalidate=%s\nconvert_array=%s\nmerge=%s\n' "${generate_job}" "${validate_job}" "${convert_job}" "${merge_job}"

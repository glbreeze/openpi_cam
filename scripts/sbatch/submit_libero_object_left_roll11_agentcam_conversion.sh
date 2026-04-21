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

RAW_HDF5_ROOT=${RAW_HDF5_ROOT:-${GEO_ROOT}/libero_object_camera_left_roll11}
OUTPUT_REPO_ID=${OUTPUT_REPO_ID:-glbreeze/libero_object_cam_train_o00_02_03_05_06_09_10_agentcam_action}
TMP_REPO_PREFIX=${TMP_REPO_PREFIX:-glbreeze/libero_object_cam_train_o00_02_03_05_06_09_10_agentcam_action_tmp_shards}
SHARDS_ROOT=${SHARDS_ROOT:-${GEO_ROOT}/${TMP_REPO_PREFIX}}

INCLUDE_CAMERA_LABELS=${INCLUDE_CAMERA_LABELS:-original+camvar_00+camvar_02+camvar_03+camvar_05+camvar_06+camvar_09+camvar_10}
NUM_CONVERT_SHARDS=${NUM_CONVERT_SHARDS:-8}
CONVERT_FPS=${CONVERT_FPS:-10}
CONVERT_IMAGE_SIZE=${CONVERT_IMAGE_SIZE:-256}
STAGE_FILES_TO_LOCAL=${STAGE_FILES_TO_LOCAL:-true}
IMAGE_WRITER_THREADS=${IMAGE_WRITER_THREADS:-4}
IMAGE_WRITER_PROCESSES=${IMAGE_WRITER_PROCESSES:-2}

CONVERT_ACCOUNT=${CONVERT_ACCOUNT:-torch_pr_637_general}
MERGE_ACCOUNT=${MERGE_ACCOUNT:-torch_pr_637_general}
CONVERT_PARTITION=${CONVERT_PARTITION:-cs}
MERGE_PARTITION=${MERGE_PARTITION:-cpu_short}
CONVERT_CPUS=${CONVERT_CPUS:-4}
CONVERT_MEM=${CONVERT_MEM:-32G}
CONVERT_TIME=${CONVERT_TIME:-08:00:00}

mkdir -p \
  "${REPO_ROOT}/log/libero_object_convert_shard" \
  "${REPO_ROOT}/log/libero_object_merge"

if [[ ! -d "${RAW_HDF5_ROOT}" ]]; then
  echo "Missing raw HDF5 root: ${RAW_HDF5_ROOT}" >&2
  exit 1
fi

echo "Submitting LIBERO object selected-view agent-camera action conversion"
echo "openpi_cam repo root: ${REPO_ROOT}"
echo "generated hdf5 root: ${RAW_HDF5_ROOT}"
echo "include camera labels: ${INCLUDE_CAMERA_LABELS}"
echo "num convert shards: ${NUM_CONVERT_SHARDS}"
echo "output repo id: ${OUTPUT_REPO_ID}"
echo "tmp shard prefix: ${TMP_REPO_PREFIX}"
echo "action frame: agent_camera"

convert_job=$(
  sbatch --parsable \
    --array="0-$(( NUM_CONVERT_SHARDS - 1 ))" \
    --job-name=convert_agentcam_o11 \
    --account="${CONVERT_ACCOUNT}" \
    --partition="${CONVERT_PARTITION}" \
    --cpus-per-task="${CONVERT_CPUS}" \
    --mem="${CONVERT_MEM}" \
    --time="${CONVERT_TIME}" \
    --export=ALL,RAW_HDF5_ROOT="${RAW_HDF5_ROOT}",MODE=all_views,NUM_SHARDS="${NUM_CONVERT_SHARDS}",REPO_ID_PREFIX="${TMP_REPO_PREFIX}",INCLUDE_CAMERA_LABELS="${INCLUDE_CAMERA_LABELS}",CONVERT_FPS="${CONVERT_FPS}",CONVERT_IMAGE_SIZE="${CONVERT_IMAGE_SIZE}",IMAGE_WRITER_THREADS="${IMAGE_WRITER_THREADS}",IMAGE_WRITER_PROCESSES="${IMAGE_WRITER_PROCESSES}",STAGE_FILES_TO_LOCAL="${STAGE_FILES_TO_LOCAL}",ACTION_FRAME=agent_camera \
    "${REPO_ROOT}/scripts/sbatch/convert_libero_object_hdf5_shard.sbatch"
)

merge_job=$(
  sbatch --parsable \
    --dependency="afterok:${convert_job}" \
    --job-name=merge_agentcam_o11 \
    --account="${MERGE_ACCOUNT}" \
    --partition="${MERGE_PARTITION}" \
    --export=ALL,SHARDS_ROOT="${SHARDS_ROOT}",OUTPUT_REPO_ID="${OUTPUT_REPO_ID}",COMPUTE_NORM_STATS=true \
    "${REPO_ROOT}/scripts/sbatch/merge_libero_object_lerobot_shards.sbatch"
)

printf 'convert_array=%s\nmerge=%s\n' "${convert_job}" "${merge_job}"

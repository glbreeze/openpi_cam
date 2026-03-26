#!/bin/bash

set -euo pipefail

resolve_repo_root() {
  local candidate
  local script_root=""

  if script_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd); then
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

RAW_HDF5_ROOT=${RAW_HDF5_ROOT:-${GEO_ROOT}/libero_object_camera_left}
ORIGINAL_REPO_ID=${ORIGINAL_REPO_ID:-glbreeze/libero_object}
CAM_REPO_ID=${CAM_REPO_ID:-glbreeze/libero_object_cam}
TMP_REPO_PREFIX=${TMP_REPO_PREFIX:-glbreeze/libero_object_tmp_shards}
ORIGINAL_SHARDS_ROOT=${ORIGINAL_SHARDS_ROOT:-${GEO_ROOT}/${TMP_REPO_PREFIX}/original}
CAM_SHARDS_ROOT=${CAM_SHARDS_ROOT:-${GEO_ROOT}/${TMP_REPO_PREFIX}/cam}
ORIGINAL_SHARD_PREFIX=${ORIGINAL_SHARD_PREFIX:-${TMP_REPO_PREFIX}/original}
CAM_SHARD_PREFIX=${CAM_SHARD_PREFIX:-${TMP_REPO_PREFIX}/cam}

ORIGINAL_FILE_COUNT=${ORIGINAL_FILE_COUNT:-$(find "${RAW_HDF5_ROOT}" -maxdepth 1 -name '*.hdf5' ! -name '*_camvar_*' | wc -l)}
CAM_FILE_COUNT=${CAM_FILE_COUNT:-$(find "${RAW_HDF5_ROOT}" -maxdepth 1 -name '*.hdf5' | wc -l)}

mkdir -p \
  "${REPO_ROOT}/log/libero_object_convert_shard" \
  "${REPO_ROOT}/log/libero_object_merge" \
  "${REPO_ROOT}/log/pi0_libero_object_ft"

echo "Submitting parallel LIBERO object pipeline"
echo "repo root: ${REPO_ROOT}"
echo "raw hdf5 root: ${RAW_HDF5_ROOT}"
echo "original file count: ${ORIGINAL_FILE_COUNT}"
echo "cam file count: ${CAM_FILE_COUNT}"

orig_array_job=$(
  sbatch --parsable \
    --array="0-$(( ORIGINAL_FILE_COUNT - 1 ))" \
    --export=ALL,RAW_HDF5_ROOT="${RAW_HDF5_ROOT}",MODE=original_only,NUM_SHARDS="${ORIGINAL_FILE_COUNT}",REPO_ID_PREFIX="${ORIGINAL_SHARD_PREFIX}" \
    "${REPO_ROOT}/scripts/sbatch/convert_libero_object_hdf5_shard.sbatch"
)

cam_array_job=$(
  sbatch --parsable \
    --array="0-$(( CAM_FILE_COUNT - 1 ))" \
    --export=ALL,RAW_HDF5_ROOT="${RAW_HDF5_ROOT}",MODE=all_views,NUM_SHARDS="${CAM_FILE_COUNT}",REPO_ID_PREFIX="${CAM_SHARD_PREFIX}" \
    "${REPO_ROOT}/scripts/sbatch/convert_libero_object_hdf5_shard.sbatch"
)

orig_merge_job=$(
  sbatch --parsable \
    --dependency="afterok:${orig_array_job}" \
    --export=ALL,SHARDS_ROOT="${ORIGINAL_SHARDS_ROOT}",OUTPUT_REPO_ID="${ORIGINAL_REPO_ID}",COMPUTE_NORM_STATS=true \
    "${REPO_ROOT}/scripts/sbatch/merge_libero_object_lerobot_shards.sbatch"
)

cam_merge_job=$(
  sbatch --parsable \
    --dependency="afterok:${cam_array_job}" \
    --export=ALL,SHARDS_ROOT="${CAM_SHARDS_ROOT}",OUTPUT_REPO_ID="${CAM_REPO_ID}",COMPUTE_NORM_STATS=true \
    "${REPO_ROOT}/scripts/sbatch/merge_libero_object_lerobot_shards.sbatch"
)

baseline_train_job=$(
  sbatch --parsable \
    --dependency="afterok:${orig_merge_job}" \
    --job-name=pi0_libero_object_baseline_full \
    --export=ALL,USE_CAM=false,DATASET_REPO_ID="${ORIGINAL_REPO_ID}",NORM_ASSET_ID="${ORIGINAL_REPO_ID}",EXP_NAME=pi0_libero_object_baseline_full,WANDB_PROJECT=openpi_cam_libero_object \
    "${REPO_ROOT}/scripts/sbatch/train_pi0_libero_object_ft.sbatch"
)

cam_train_job=$(
  sbatch --parsable \
    --dependency="afterok:${cam_merge_job}" \
    --job-name=pi0_libero_object_cam_full \
    --export=ALL,USE_CAM=true,DATASET_REPO_ID="${CAM_REPO_ID}",NORM_ASSET_ID="${CAM_REPO_ID}",EXP_NAME=pi0_libero_object_cam_full,WANDB_PROJECT=openpi_cam_libero_object \
    "${REPO_ROOT}/scripts/sbatch/train_pi0_libero_object_ft.sbatch"
)

printf 'orig_array=%s\ncam_array=%s\norig_merge=%s\ncam_merge=%s\nbaseline_train=%s\ncam_train=%s\n' \
  "${orig_array_job}" "${cam_array_job}" "${orig_merge_job}" "${cam_merge_job}" "${baseline_train_job}" "${cam_train_job}"

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

SRC_ROOT=${SRC_ROOT:-/scratch/yz11445/robocasa-human50/raw_human_im}
CACHE_DIR=${CACHE_DIR:-/scratch/yz11445/robocasa-human50/stage1_cam_matrix_cache}
REPO_ID=${REPO_ID:-robocasa24/all24_human_camaware}
SOURCE_TYPE=${SOURCE_TYPE:-human_im}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-torch_pr_926_general}
SLURM_PARTITION=${SLURM_PARTITION:-l40s_public}
SLURM_TIME=${SLURM_TIME:-}
RESUME=${RESUME:-0}
SBATCH_SCRIPT="${REPO_ROOT}/scripts/sbatch/convert_robocasa24_all24_human_camaware_l40s_30m.sbatch"

if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
  echo "Missing sbatch script at ${SBATCH_SCRIPT}" >&2
  exit 1
fi
if [[ ! -d "${SRC_ROOT}" ]]; then
  echo "Missing SRC_ROOT at ${SRC_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${CACHE_DIR}" ]]; then
  echo "Missing CACHE_DIR at ${CACHE_DIR}" >&2
  exit 1
fi

mkdir -p "${REPO_ROOT}/log/robocasa24_convert"

echo "Submitting RoboCasa24 all24 human cam-aware conversion"
echo "repo root: ${REPO_ROOT}"
echo "geo root: ${GEO_ROOT}"
echo "src root: ${SRC_ROOT}"
echo "cache dir: ${CACHE_DIR}"
echo "repo id: ${REPO_ID}"
echo "source type: ${SOURCE_TYPE}"
echo "slurm account: ${SLURM_ACCOUNT}"
echo "slurm partition: ${SLURM_PARTITION}"
echo "slurm time: ${SLURM_TIME:-<sbatch default>}"
echo "resume: ${RESUME}"
echo "sbatch script: ${SBATCH_SCRIPT}"

sbatch_time_args=()
if [[ -n "${SLURM_TIME}" ]]; then
  sbatch_time_args+=(--time="${SLURM_TIME}")
fi

job_id=$(
  sbatch --parsable \
    --account="${SLURM_ACCOUNT}" \
    --partition="${SLURM_PARTITION}" \
    "${sbatch_time_args[@]}" \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",GEO_ROOT="${GEO_ROOT}",SRC_ROOT="${SRC_ROOT}",CACHE_DIR="${CACHE_DIR}",REPO_ID="${REPO_ID}",SOURCE_TYPE="${SOURCE_TYPE}",RESUME="${RESUME}" \
    "${SBATCH_SCRIPT}"
)

printf 'convert_job=%s\n' "${job_id}"

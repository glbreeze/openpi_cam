#!/bin/bash
#SBATCH --job-name=merge_libero_object_shards
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/libero_object_merge/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/libero_object_merge/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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
  echo "Unable to locate repo root. Set REPO_ROOT or submit from inside the repo." >&2
  exit 1
}
GEO_ROOT=${GEO_ROOT:-$(cd -- "${REPO_ROOT}/.." && pwd)}

SHARDS_ROOT=${SHARDS_ROOT:-${GEO_ROOT}/glbreeze/libero_object_tmp_shards}
OUTPUT_REPO_ID=${OUTPUT_REPO_ID:-glbreeze/libero_object}
COMPUTE_NORM_STATS=${COMPUTE_NORM_STATS:-true}
CACHE_ROOT=${CACHE_ROOT:-${GEO_ROOT}/.cache/huggingface}
OPENPI_CACHE_ROOT=${OPENPI_CACHE_ROOT:-${GEO_ROOT}/.cache/openpi}

JOB_TMP_BASE="${SLURM_TMPDIR:-/tmp/$USER}"
JOB_TMP_ROOT="${JOB_TMP_BASE}/openpi_cam_merge_${SLURM_JOB_ID:-$$}"
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

if [[ ! -d "${SHARDS_ROOT}" ]]; then
  echo "Missing shard root at ${SHARDS_ROOT}" >&2
  exit 1
fi

echo "===== LEROBOT SHARD MERGE ENV CHECK ====="
echo "repo root: ${REPO_ROOT}"
echo "shards root: ${SHARDS_ROOT}"
echo "output repo id: ${OUTPUT_REPO_ID}"
echo "hf lerobot home: ${HF_LEROBOT_HOME}"
echo "norm stats root: ${OPENPI_PI0_LIBERO_NORM_DIR}"
which python
python -V
hostname

python examples/libero/merge_lerobot_shards.py \
  --shards-root "${SHARDS_ROOT}" \
  --repo-id "${OUTPUT_REPO_ID}"

if [[ "${COMPUTE_NORM_STATS}" == "true" ]]; then
  python scripts/compute_lerobot_norm_stats.py \
    --repo-id "${OUTPUT_REPO_ID}" \
    --output-root "${OPENPI_PI0_LIBERO_NORM_DIR}"
fi

echo "Shard merge completed successfully."

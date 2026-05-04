#!/bin/bash
#SBATCH --job-name=norm_libcam_v2
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/compute_norm_stats_libero_cam_v2/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/compute_norm_stats_libero_cam_v2/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=torch_pr_676_general
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

JOB_TMP_BASE="${SLURM_TMPDIR:-/tmp/$USER}"
JOB_TMP_ROOT="${JOB_TMP_BASE}/openpi_norm_stats_${SLURM_JOB_ID:-$$}"
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

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/scratch/yp2841/cache/lerobot}"

REPO_ID=${REPO_ID:-glbreeze/libero_cam_v2}
OUTPUT_ROOT=${OUTPUT_ROOT:-${GEO_ROOT}/pi0_libero}
DATASET_DIR="${HF_LEROBOT_HOME}/${REPO_ID}"
OUTPUT_DIR="${OUTPUT_ROOT}/${REPO_ID}"

mkdir -p "$(dirname -- "${OUTPUT_DIR}")" /scratch/yp2841/geometry-vla/openpi_cam/log/compute_norm_stats_libero_cam_v2

if [[ ! -d "${DATASET_DIR}/data" || ! -f "${DATASET_DIR}/meta/info.json" ]]; then
  echo "Missing local LeRobot dataset at ${DATASET_DIR}" >&2
  exit 1
fi

echo "===== NORM STATS ENV ====="
echo "repo root: ${REPO_ROOT}"
echo "hf lerobot home: ${HF_LEROBOT_HOME}"
echo "repo id: ${REPO_ID}"
echo "dataset dir: ${DATASET_DIR}"
echo "output dir: ${OUTPUT_DIR}"
hostname
python -V

python scripts/compute_lerobot_norm_stats.py --repo-id "${REPO_ID}" --output-root "${OUTPUT_ROOT}"

if [[ ! -f "${OUTPUT_DIR}/norm_stats.json" ]]; then
  echo "Expected norm stats missing at ${OUTPUT_DIR}/norm_stats.json" >&2
  exit 1
fi

echo "Norm stats ready at ${OUTPUT_DIR}/norm_stats.json"

#!/bin/bash
#SBATCH --job-name=norm_robotwin
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/compute_norm_stats_robotwin/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/compute_norm_stats_robotwin/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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

JOB_TMP_BASE="${SLURM_TMPDIR:-/tmp/$USER}"
JOB_TMP_ROOT="${JOB_TMP_BASE}/robotwin_norm_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOB_TMP_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export DATASETS_TMPDIR="${JOB_TMP_ROOT}/datasets_tmp"
export HOME="${JOB_TMP_ROOT}/home"
export XDG_CACHE_HOME="${JOB_TMP_ROOT}/xdg_cache"
export XDG_CONFIG_HOME="${JOB_TMP_ROOT}/xdg_config"
export XDG_STATE_HOME="${JOB_TMP_ROOT}/xdg_state"

mkdir -p \
  "${TMPDIR}" \
  "${DATASETS_TMPDIR}" \
  "${HOME}" \
  "${XDG_CACHE_HOME}" \
  "${XDG_CONFIG_HOME}" \
  "${XDG_STATE_HOME}" \
  /scratch/yp2841/geometry-vla/openpi_cam/log/compute_norm_stats_robotwin
chmod 700 "${TMPDIR}" "${DATASETS_TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" || true

cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/env/activate_env.sh"

export HF_LEROBOT_HOME="${GEO_ROOT}"
export HF_HOME="${GEO_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export OPENPI_CACHE_DIR="${GEO_ROOT}/.cache/openpi"
export OPENPI_DATA_HOME="${OPENPI_CACHE_DIR}"
unset TRANSFORMERS_CACHE

CONFIG_NAME=${CONFIG_NAME:-pi0_robotwin_smoke}
REPO_ID=${REPO_ID:-lerobot/robotwin_unified}
MAX_FRAMES=${MAX_FRAMES:-50000}
BATCH_SIZE=${BATCH_SIZE:-32}
SOURCE_OUTPUT_DIR="${GEO_ROOT}/pi0_libero/${CONFIG_NAME}/${REPO_ID}"
FINAL_OUTPUT_DIR="${GEO_ROOT}/pi0_libero/${REPO_ID}"
DATASET_DIR="${HF_LEROBOT_HOME}/${REPO_ID}"

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${OPENPI_CACHE_DIR}"

if [[ ! -d "${DATASET_DIR}/data" || ! -f "${DATASET_DIR}/meta/info.json" ]]; then
  echo "Missing local LeRobot dataset at ${DATASET_DIR}" >&2
  exit 1
fi

echo "===== ROBOTWIN NORM STATS ====="
echo "repo root: ${REPO_ROOT}"
echo "geo root: ${GEO_ROOT}"
echo "config name: ${CONFIG_NAME}"
echo "max frames: ${MAX_FRAMES}"
echo "batch size override: ${BATCH_SIZE}"
echo "hf lerobot home: ${HF_LEROBOT_HOME}"
echo "dataset dir: ${DATASET_DIR}"
echo "source output dir: ${SOURCE_OUTPUT_DIR}"
echo "final output dir: ${FINAL_OUTPUT_DIR}"
hostname
python -V

python scripts/compute_norm_stats.py \
  --config-name="${CONFIG_NAME}" \
  --repo-id="${REPO_ID}" \
  --max-frames="${MAX_FRAMES}" \
  --batch-size="${BATCH_SIZE}"

if [[ ! -f "${SOURCE_OUTPUT_DIR}/norm_stats.json" ]]; then
  echo "Expected norm stats missing at ${SOURCE_OUTPUT_DIR}/norm_stats.json" >&2
  exit 1
fi

mkdir -p "${FINAL_OUTPUT_DIR}"
cp "${SOURCE_OUTPUT_DIR}/norm_stats.json" "${FINAL_OUTPUT_DIR}/norm_stats.json"

echo "Norm stats ready at ${FINAL_OUTPUT_DIR}/norm_stats.json"

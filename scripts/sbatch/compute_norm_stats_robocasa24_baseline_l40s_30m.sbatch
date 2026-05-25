#!/bin/bash
#SBATCH --job-name=norm_rc24_base
#SBATCH --output=/scratch/%u/openpi_cam/log/compute_norm_stats_robocasa24/slurm-%j.out
#SBATCH --error=/scratch/%u/openpi_cam/log/compute_norm_stats_robocasa24/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --account=torch_pr_926_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
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
CONFIG_NAME=${CONFIG_NAME:-pi0_robocasa24_all24_baseline}
OPENPI_PYTHON=${OPENPI_PYTHON:-${REPO_ROOT}/.venv/bin/python}

JOB_TMP_BASE="${SLURM_TMPDIR:-/tmp/$USER}"
JOB_TMP_ROOT="${JOB_TMP_BASE}/openpi_norm_rc24_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOB_TMP_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export DATASETS_TMPDIR="${JOB_TMP_ROOT}/datasets_tmp"
export HOME="${JOB_TMP_ROOT}/home"
export XDG_CACHE_HOME="${JOB_TMP_ROOT}/xdg_cache"
export XDG_CONFIG_HOME="${JOB_TMP_ROOT}/xdg_config"
export XDG_STATE_HOME="${JOB_TMP_ROOT}/xdg_state"
export PYTHONUNBUFFERED=1

mkdir -p \
  "${TMPDIR}" \
  "${DATASETS_TMPDIR}" \
  "${HOME}" \
  "${XDG_CACHE_HOME}" \
  "${XDG_CONFIG_HOME}" \
  "${XDG_STATE_HOME}" \
  "${REPO_ROOT}/log/compute_norm_stats_robocasa24"
chmod 700 "${TMPDIR}" "${DATASETS_TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" || true

cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/env/activate_env.sh"

export OPENPI_GEO_ROOT="${GEO_ROOT}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${GEO_ROOT}}"
export HF_HOME="${HF_HOME:-${GEO_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export OPENPI_CACHE_DIR="${OPENPI_CACHE_DIR:-${GEO_ROOT}/.cache/openpi}"
export OPENPI_DATA_HOME="${OPENPI_CACHE_DIR}"
unset TRANSFORMERS_CACHE

mkdir -p "${HF_LEROBOT_HOME}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${OPENPI_CACHE_DIR}" "${GEO_ROOT}/pi0_robocasa24"

DATASET_DIR="${HF_LEROBOT_HOME}/robocasa24/all24_human_camaware"
OUTPUT_DIR="${GEO_ROOT}/pi0_robocasa24/robocasa24/all24_human_camaware"
RESUME_STATE_PATH="${RESUME_STATE_PATH:-${OUTPUT_DIR}/norm_stats.resume.pkl}"
SAVE_EVERY_BATCHES="${SAVE_EVERY_BATCHES:-100}"

if [[ ! -x "${OPENPI_PYTHON}" ]]; then
  echo "Missing OPENPI_PYTHON: ${OPENPI_PYTHON}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_DIR}/data" || ! -f "${DATASET_DIR}/meta/info.json" ]]; then
  echo "Missing local LeRobot dataset at ${DATASET_DIR}" >&2
  exit 1
fi

echo "===== RoboCasa24 baseline norm stats ====="
echo "job id: ${SLURM_JOB_ID:-manual}"
echo "host: $(hostname)"
echo "repo root: ${REPO_ROOT}"
echo "geo root: ${GEO_ROOT}"
echo "hf lerobot home: ${HF_LEROBOT_HOME}"
echo "dataset dir: ${DATASET_DIR}"
echo "output dir: ${OUTPUT_DIR}"
echo "resume state: ${RESUME_STATE_PATH}"
echo "save every batches: ${SAVE_EVERY_BATCHES}"
echo "config: ${CONFIG_NAME}"
date

"${OPENPI_PYTHON}" scripts/compute_norm_stats.py \
  --config-name="${CONFIG_NAME}" \
  --resume \
  --resume-state-path="${RESUME_STATE_PATH}" \
  --save-every-batches="${SAVE_EVERY_BATCHES}"

if [[ ! -f "${OUTPUT_DIR}/norm_stats.json" ]]; then
  echo "Expected norm stats missing at ${OUTPUT_DIR}/norm_stats.json" >&2
  exit 1
fi

echo "Norm stats ready at ${OUTPUT_DIR}/norm_stats.json"
date

#!/bin/bash
#SBATCH --job-name=rc24_convert
#SBATCH --output=/scratch/%u/openpi_cam/log/robocasa24_convert/slurm-%j.out
#SBATCH --error=/scratch/%u/openpi_cam/log/robocasa24_convert/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --account=torch_pr_637_general
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

SRC_ROOT=${SRC_ROOT:-/scratch/yz11445/robocasa-human50/raw_human_im}
CACHE_DIR=${CACHE_DIR:-/scratch/yz11445/robocasa-human50/stage1_cam_matrix_cache}
REPO_ID=${REPO_ID:-robocasa24/all24_human_camaware}
SOURCE_TYPE=${SOURCE_TYPE:-human_im}
OPENPI_PYTHON=${OPENPI_PYTHON:-${REPO_ROOT}/.venv/bin/python}
RESUME=${RESUME:-0}

JOB_TMP_BASE="${SLURM_TMPDIR:-/tmp/$USER}"
JOB_TMP_ROOT="${JOB_TMP_BASE}/robocasa24_convert_${SLURM_JOB_ID:-$$}"
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
  "${REPO_ROOT}/log/robocasa24_convert"
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

mkdir -p "${HF_LEROBOT_HOME}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${OPENPI_CACHE_DIR}"

if [[ ! -x "${OPENPI_PYTHON}" ]]; then
  echo "Missing OPENPI_PYTHON: ${OPENPI_PYTHON}" >&2
  exit 1
fi
if [[ ! -d "${SRC_ROOT}" ]]; then
  echo "Missing SRC_ROOT: ${SRC_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${CACHE_DIR}" ]]; then
  echo "Missing CACHE_DIR: ${CACHE_DIR}" >&2
  exit 1
fi

echo "===== RoboCasa24 LeRobot conversion ====="
echo "job id: ${SLURM_JOB_ID:-manual}"
echo "host: $(hostname)"
echo "repo root: ${REPO_ROOT}"
echo "geo root: ${GEO_ROOT}"
echo "hf lerobot home: ${HF_LEROBOT_HOME}"
echo "src root: ${SRC_ROOT}"
echo "cache dir: ${CACHE_DIR}"
echo "repo id: ${REPO_ID}"
echo "source type: ${SOURCE_TYPE}"
echo "resume: ${RESUME}"
echo "python: ${OPENPI_PYTHON}"
date

resume_args=()
if [[ "${RESUME}" == "1" || "${RESUME,,}" == "true" ]]; then
  resume_args=(--resume)
fi

"${OPENPI_PYTHON}" scripts/convert_robocasa24_to_camaware_lerobot.py \
  --src-root "${SRC_ROOT}" \
  --cache-dir "${CACHE_DIR}" \
  --repo-id "${REPO_ID}" \
  --source-type "${SOURCE_TYPE}" \
  "${resume_args[@]}"

OUTPUT_DIR="${HF_LEROBOT_HOME}/${REPO_ID}"
if [[ ! -d "${OUTPUT_DIR}" ]]; then
  echo "Expected output dir missing at ${OUTPUT_DIR}" >&2
  exit 1
fi

echo "Conversion complete: ${OUTPUT_DIR}"
date

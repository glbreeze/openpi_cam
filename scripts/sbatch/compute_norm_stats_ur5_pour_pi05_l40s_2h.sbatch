#!/bin/bash
#SBATCH --job-name=norm_ur5_pour_pi05
#SBATCH --output=/scratch/%u/openpi_cam/log/compute_norm_stats_ur5_pour/slurm-pi05-%j.out
#SBATCH --error=/scratch/%u/openpi_cam/log/compute_norm_stats_ur5_pour/slurm-pi05-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --account=torch_pr_69_general
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

  for candidate in "${REPO_ROOT:-}" "${OPENPI_CAM_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "${PWD:-}" "${script_root}"; do
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

if [[ -z "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(resolve_repo_root || true)"
fi
if [[ -z "${REPO_ROOT}" && -d "/scratch/${USER}/openpi_cam" ]]; then
  REPO_ROOT="/scratch/${USER}/openpi_cam"
fi
if [[ -z "${REPO_ROOT}" ]]; then
  echo "Unable to locate repo root." >&2
  exit 1
fi

export OPENPI_GEO_ROOT="${OPENPI_GEO_ROOT:-/scratch/${USER}}"
export DATASET_REPO_ID="${DATASET_REPO_ID:-ur5_place_and_pour_nuts_camera_shifts}"
export NORM_ASSET_ID="${NORM_ASSET_ID:-ur5_place_and_pour_nuts_camera_shifts}"
export DATASET_DIR="${DATASET_DIR:-${OPENPI_GEO_ROOT}/real_robot_data/${DATASET_REPO_ID}}"
export CONFIG_NAME="${CONFIG_NAME:-pi05_ur5_real_robot_pytorch_cross_attn_fg_distill_stage1_hard}"
export OUTPUT_DIR="${OUTPUT_DIR:-${OPENPI_GEO_ROOT}/pi05_ur5_real_robot/${NORM_ASSET_ID}}"
export OPENPI_PYTHON="${OPENPI_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
export SAVE_EVERY_BATCHES="${SAVE_EVERY_BATCHES:-100}"
export RESUME_STATE_PATH="${RESUME_STATE_PATH:-${OUTPUT_DIR}/norm_stats.resume.pkl}"
export PYTHONUNBUFFERED=1

JOB_TMP_BASE="${SLURM_TMPDIR:-/tmp/${USER}}"
JOB_TMP_ROOT="${JOB_TMP_BASE}/openpi_norm_ur5_pour_pi05_${SLURM_JOB_ID:-$$}"
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
  "${OUTPUT_DIR}" \
  "${REPO_ROOT}/log/compute_norm_stats_ur5_pour"
chmod 700 "${TMPDIR}" "${DATASETS_TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" || true

if [[ ! -x "${OPENPI_PYTHON}" ]]; then
  echo "Missing OPENPI_PYTHON: ${OPENPI_PYTHON}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_DIR}/data" || ! -f "${DATASET_DIR}/meta/info.json" ]]; then
  echo "Missing LeRobot dataset at ${DATASET_DIR}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_DIR}_array_cache/videos/chunk-000/observation.images.context_left_rgb" ||
      ! -d "${DATASET_DIR}_array_cache/videos/chunk-000/observation.images.wrist_right_rgb" ]]; then
  echo "Missing pour video array cache under ${DATASET_DIR}_array_cache" >&2
  exit 1
fi

cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/env/activate_env.sh"

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${OPENPI_GEO_ROOT}/real_robot_data}"
export HF_HOME="${HF_HOME:-${OPENPI_GEO_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export OPENPI_CACHE_DIR="${OPENPI_CACHE_DIR:-${OPENPI_GEO_ROOT}/.cache/openpi}"
export OPENPI_DATA_HOME="${OPENPI_CACHE_DIR}"
unset TRANSFORMERS_CACHE

echo "===== UR5 pour pi0.5 norm stats ====="
echo "job id: ${SLURM_JOB_ID:-manual}"
echo "host: $(hostname)"
echo "repo root: ${REPO_ROOT}"
echo "openpi geo root: ${OPENPI_GEO_ROOT}"
echo "config: ${CONFIG_NAME}"
echo "dataset repo id: ${DATASET_REPO_ID}"
echo "dataset dir: ${DATASET_DIR}"
echo "array cache: ${DATASET_DIR}_array_cache"
echo "output dir: ${OUTPUT_DIR}"
echo "resume state: ${RESUME_STATE_PATH}"
date

"${OPENPI_PYTHON}" scripts/compute_norm_stats.py \
  --config-name="${CONFIG_NAME}" \
  --repo-id="${DATASET_REPO_ID}" \
  --asset-id="${NORM_ASSET_ID}" \
  --resume \
  --resume-state-path="${RESUME_STATE_PATH}" \
  --save-every-batches="${SAVE_EVERY_BATCHES}"

if [[ ! -f "${OUTPUT_DIR}/norm_stats.json" ]]; then
  echo "Expected norm stats missing at ${OUTPUT_DIR}/norm_stats.json" >&2
  exit 1
fi

echo "Norm stats ready at ${OUTPUT_DIR}/norm_stats.json"
date

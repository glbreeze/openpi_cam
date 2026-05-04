#!/bin/bash
#SBATCH --job-name=robotwin_conv
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/convert_robotwin_official/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/convert_robotwin_official/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
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
  echo "Unable to locate repo root." >&2
  exit 1
}
GEO_ROOT=${GEO_ROOT:-$(cd -- "${REPO_ROOT}/.." && pwd)}

JOB_TMP_BASE="${SLURM_TMPDIR:-/tmp/$USER}"
JOB_TMP_ROOT="${JOB_TMP_BASE}/robotwin_convert_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOB_TMP_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export HOME="${JOB_TMP_ROOT}/home"
export XDG_CACHE_HOME="${JOB_TMP_ROOT}/xdg_cache"
export XDG_CONFIG_HOME="${JOB_TMP_ROOT}/xdg_config"
export XDG_STATE_HOME="${JOB_TMP_ROOT}/xdg_state"

mkdir -p "${TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" /scratch/yp2841/geometry-vla/openpi_cam/log/convert_robotwin_official
chmod 700 "${TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" || true

cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/env/activate_env.sh"

export HF_LEROBOT_HOME="${GEO_ROOT}"
export HF_HOME="${GEO_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"

TASK_NAME=${TASK_NAME:-beat_block_hammer}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
EXPERT_DATA_NUM=${EXPERT_DATA_NUM:-50}
RAW_ROOT=${RAW_ROOT:-${GEO_ROOT}/robotwin_raw}
PROCESSED_ROOT=${PROCESSED_ROOT:-${REPO_ROOT}/tmp/robotwin_processed}
TRAINING_ROOT=${TRAINING_ROOT:-${REPO_ROOT}/tmp/robotwin_training_data}
REPO_ID=${REPO_ID:-robotwin/${TASK_NAME}_${TASK_CONFIG}_${EXPERT_DATA_NUM}}

PROCESSED_DIR="${PROCESSED_ROOT}/${TASK_NAME}-${TASK_CONFIG}-${EXPERT_DATA_NUM}"
TRAINING_DIR="${TRAINING_ROOT}/${TASK_CONFIG}/${TASK_NAME}-${TASK_CONFIG}-${EXPERT_DATA_NUM}"
OUTPUT_DIR="${HF_LEROBOT_HOME}/${REPO_ID}"

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${PROCESSED_ROOT}" "${TRAINING_ROOT}" "$(dirname -- "${OUTPUT_DIR}")"

echo "===== ROBOTWIN OFFICIAL CONVERSION ====="
echo "task name: ${TASK_NAME}"
echo "task config: ${TASK_CONFIG}"
echo "expert data num: ${EXPERT_DATA_NUM}"
echo "raw root: ${RAW_ROOT}"
echo "processed dir: ${PROCESSED_DIR}"
echo "training dir: ${TRAINING_DIR}"
echo "repo id: ${REPO_ID}"
echo "output dir: ${OUTPUT_DIR}"
hostname
python -V

python scripts/robotwin_official_process_data.py \
  "${TASK_NAME}" \
  "${TASK_CONFIG}" \
  "${EXPERT_DATA_NUM}" \
  --raw-root "${RAW_ROOT}" \
  --processed-root "${PROCESSED_ROOT}"

mkdir -p "$(dirname -- "${TRAINING_DIR}")"
rm -rf "${TRAINING_DIR}"
cp -r "${PROCESSED_DIR}" "${TRAINING_DIR}"

python scripts/convert_aloha_data_to_lerobot_robotwin_official.py \
  --raw-dir "${TRAINING_DIR}" \
  --repo-id "${REPO_ID}"

if [[ ! -d "${OUTPUT_DIR}/data" || ! -f "${OUTPUT_DIR}/meta/info.json" ]]; then
  echo "Official conversion did not produce a valid LeRobot dataset at ${OUTPUT_DIR}" >&2
  exit 1
fi

echo "Converted RoboTwin dataset ready at ${OUTPUT_DIR}"

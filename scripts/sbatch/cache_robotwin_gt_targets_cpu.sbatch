#!/bin/bash
#SBATCH --job-name=rtw_gt_cache
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/cache_robotwin_gt/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/cache_robotwin_gt/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --account=torch_pr_637_general
#SBATCH --partition=cs
#SBATCH --export=ALL

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/scratch/yp2841/geometry-vla/openpi_cam}
GEO_ROOT=${GEO_ROOT:-/scratch/yp2841/geometry-vla}
JOB_TMP_BASE="${SLURM_TMPDIR:-/tmp/$USER}"
JOB_TMP_ROOT="${JOB_TMP_BASE}/robotwin_gt_cache_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOB_TMP_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export HOME="${JOB_TMP_ROOT}/home"
export XDG_CACHE_HOME="${JOB_TMP_ROOT}/xdg_cache"
export XDG_CONFIG_HOME="${JOB_TMP_ROOT}/xdg_config"
export XDG_STATE_HOME="${JOB_TMP_ROOT}/xdg_state"

mkdir -p "${TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" /scratch/yp2841/geometry-vla/openpi_cam/log/cache_robotwin_gt
chmod 700 "${TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" || true

cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/env/activate_env.sh"

export HF_LEROBOT_HOME="${GEO_ROOT}"
export HF_HOME="${GEO_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export OPENPI_CACHE_DIR="${GEO_ROOT}/.cache/openpi"
export OPENPI_DATA_HOME="${OPENPI_CACHE_DIR}"
unset TRANSFORMERS_CACHE

TASK_NAME=${TASK_NAME:-handover_block}
TASK_CONFIG=${TASK_CONFIG:-demo_clean_camaware}
EXPERT_DATA_NUM=${EXPERT_DATA_NUM:-50}
ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-${GEO_ROOT}/RoboTwin}
RAW_DIR=${RAW_DIR:-${ROBOTWIN_ROOT}/data/${TASK_NAME}/${TASK_CONFIG}}
REPO_ID=${REPO_ID:-robotwin/${TASK_NAME}_${TASK_CONFIG}_${EXPERT_DATA_NUM}}
SLUG=${REPO_ID//\//_}
GT_ROOT=${GT_ROOT:-${OPENPI_CACHE_DIR}/gt_point_targets_224/${SLUG}}
TARGET_RESOLUTION=${TARGET_RESOLUTION:-224}
OUTPUT_RESOLUTION=${OUTPUT_RESOLUTION:-224}
MAX_EPISODES=${MAX_EPISODES:-0}

if [[ ! -d "${RAW_DIR}/data" ]]; then
  echo "Missing raw RoboTwin HDF5 directory at ${RAW_DIR}/data" >&2
  exit 1
fi

echo "===== ROBOTWIN GT POINT TARGET CACHE ====="
echo "task name: ${TASK_NAME}"
echo "task config: ${TASK_CONFIG}"
echo "raw dir: ${RAW_DIR}"
echo "repo id: ${REPO_ID}"
echo "output root: ${GT_ROOT}"
hostname
python -V

args=(
  --raw-dir "${RAW_DIR}"
  --output-root "${GT_ROOT}"
  --target-resolution "${TARGET_RESOLUTION}"
  --output-resolution "${OUTPUT_RESOLUTION}"
  --resume
)
if [[ "${MAX_EPISODES}" != "0" ]]; then
  args+=(--max-episodes "${MAX_EPISODES}")
fi

PYTHONPATH=src python scripts/cache_robotwin_gt_point_targets.py "${args[@]}"

for cam in cam_high cam_left_wrist cam_right_wrist; do
  if ! find "${GT_ROOT}/${cam}" -maxdepth 1 -name 'episode_*.npz' -print -quit | grep -q .; then
    echo "GT cache missing files for ${cam} under ${GT_ROOT}/${cam}" >&2
    exit 1
  fi
done

du -sh "${GT_ROOT}"
echo "GT point target cache ready at ${GT_ROOT}"

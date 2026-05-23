#!/bin/bash
#SBATCH --job-name=robotwin_cam_conv
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/convert_robotwin_cam/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/convert_robotwin_cam/slurm-%j.err
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
ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-${GEO_ROOT}/RoboTwin}

JOB_TMP_BASE="${SLURM_TMPDIR:-/tmp/$USER}"
JOB_TMP_ROOT="${JOB_TMP_BASE}/robotwin_cam_convert_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOB_TMP_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export HOME="${JOB_TMP_ROOT}/home"
export XDG_CACHE_HOME="${GEO_ROOT}/.cache/xdg"
export XDG_CONFIG_HOME="${JOB_TMP_ROOT}/xdg_config"
export XDG_STATE_HOME="${JOB_TMP_ROOT}/xdg_state"
export HF_HOME="${GEO_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_LEROBOT_HOME="${GEO_ROOT}"

mkdir -p \
  "${TMPDIR}" \
  "${HOME}" \
  "${XDG_CACHE_HOME}" \
  "${XDG_CONFIG_HOME}" \
  "${XDG_STATE_HOME}" \
  "${HF_HOME}" \
  "${HF_DATASETS_CACHE}" \
  "${HF_HUB_CACHE}" \
  /scratch/yp2841/geometry-vla/openpi_cam/log/convert_robotwin_cam
chmod 700 "${TMPDIR}" "${HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" || true

cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/env/activate_env.sh"

TASK_NAME=${TASK_NAME:-open_laptop}
TASK_CONFIG=${TASK_CONFIG:-demo_clean_camaware}
EXPERT_DATA_NUM=${EXPERT_DATA_NUM:-50}
RAW_DIR=${RAW_DIR:-${ROBOTWIN_ROOT}/data/${TASK_NAME}/${TASK_CONFIG}}
REPO_ID=${REPO_ID:-robotwin/${TASK_NAME}_${TASK_CONFIG}_${EXPERT_DATA_NUM}}
IMAGE_SIZE=${IMAGE_SIZE:-224}
export REPO_ID

echo "===== ROBOTWIN CAM-AWARE CONVERSION ====="
echo "task name       : ${TASK_NAME}"
echo "task config     : ${TASK_CONFIG}"
echo "expert data num : ${EXPERT_DATA_NUM}"
echo "raw dir         : ${RAW_DIR}"
echo "repo id         : ${REPO_ID}"
echo "output dir      : ${HF_LEROBOT_HOME}/${REPO_ID}"
echo "image size      : ${IMAGE_SIZE}"
hostname
python -V

python scripts/convert_robotwin_cam_to_lerobot.py \
  --raw-dir "${RAW_DIR}" \
  --repo-id "${REPO_ID}" \
  --image-size "${IMAGE_SIZE}" \
  --max-episodes "${EXPERT_DATA_NUM}"

python - <<'PY'
import os
from pathlib import Path

import pyarrow.parquet as pq

repo = Path(os.environ["HF_LEROBOT_HOME"]) / os.environ["REPO_ID"]
episode = repo / "data" / "chunk-000" / "episode_000000.parquet"
table = pq.read_table(episode)
required = []
for cam in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
    required.extend([f"observation.{cam}_extrinsic", f"observation.{cam}_intrinsic"])
missing = [col for col in required if col not in table.column_names]
if missing:
    raise SystemExit(f"Converted dataset is missing camera calibration columns: {missing}")
print("camera calibration columns verified:", ", ".join(required))
PY

echo "Converted cam-aware RoboTwin dataset ready at ${HF_LEROBOT_HOME}/${REPO_ID}"

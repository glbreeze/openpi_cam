#!/bin/bash
#SBATCH --job-name=robotwin_collect
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/collect_robotwin_raw/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/collect_robotwin_raw/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --account=torch_pr_637_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --export=ALL

set -euo pipefail

CONDA_SH=${CONDA_SH:-/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh}
ROBOTWIN_CONDA_ENV=${ROBOTWIN_CONDA_ENV:-robotwin}
ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-/scratch/yp2841/geometry-vla/RoboTwin}
TASK_NAME=${TASK_NAME:-beat_block_hammer}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
GPU_ID=${GPU_ID:-0}

mkdir -p /scratch/yp2841/geometry-vla/openpi_cam/log/collect_robotwin_raw

if [[ ! -d "${ROBOTWIN_ROOT}" ]]; then
  echo "Missing RoboTwin repo at ${ROBOTWIN_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Missing conda.sh at ${CONDA_SH}" >&2
  exit 1
fi

source "${CONDA_SH}"
conda activate "${ROBOTWIN_CONDA_ENV}"

cd "${ROBOTWIN_ROOT}"

echo "===== ROBOTWIN RAW COLLECTION ====="
echo "robotwin root: ${ROBOTWIN_ROOT}"
echo "task name: ${TASK_NAME}"
echo "task config: ${TASK_CONFIG}"
echo "gpu id: ${GPU_ID}"
hostname
python -V

bash collect_data.sh "${TASK_NAME}" "${TASK_CONFIG}" "${GPU_ID}"

RAW_DIR="${ROBOTWIN_ROOT}/data/${TASK_NAME}/${TASK_CONFIG}"
if [[ ! -d "${RAW_DIR}/data" || ! -d "${RAW_DIR}/instructions" ]]; then
  echo "Expected raw collection output missing under ${RAW_DIR}" >&2
  exit 1
fi

echo "Collected raw RoboTwin data at ${RAW_DIR}"

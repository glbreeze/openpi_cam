#!/bin/bash
#SBATCH --job-name=merge_robotwin_subset
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/merge_robotwin_subset/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/merge_robotwin_subset/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --account=torch_pr_637_general
#SBATCH --partition=cs
#SBATCH --export=ALL

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/scratch/yp2841/geometry-vla/openpi_cam}
GEO_ROOT=${GEO_ROOT:-/scratch/yp2841/geometry-vla}
OPENPI_CACHE_DIR=${OPENPI_CACHE_DIR:-${GEO_ROOT}/.cache/openpi}

OUT_REPO_ID=${OUT_REPO_ID:-robotwin/open_laptop_stack_bowls_two_demo_clean_camaware_100}
OUT_SLUG=${OUT_REPO_ID//\//_}

INPUT_REPO_1=${INPUT_REPO_1:-${GEO_ROOT}/robotwin/open_laptop_demo_clean_camaware_50}
INPUT_REPO_2=${INPUT_REPO_2:-${GEO_ROOT}/robotwin/stack_bowls_two_demo_clean_camaware_50}
INPUT_GT_1=${INPUT_GT_1:-${OPENPI_CACHE_DIR}/gt_point_targets_224/robotwin_open_laptop_demo_clean_camaware_50}
INPUT_GT_2=${INPUT_GT_2:-${OPENPI_CACHE_DIR}/gt_point_targets_224/robotwin_stack_bowls_two_demo_clean_camaware_50}
OUTPUT_REPO=${OUTPUT_REPO:-${GEO_ROOT}/${OUT_REPO_ID}}
OUTPUT_GT=${OUTPUT_GT:-${OPENPI_CACHE_DIR}/gt_point_targets_224/${OUT_SLUG}}

JOB_TMP_BASE="${SLURM_TMPDIR:-/tmp/$USER}"
JOB_TMP_ROOT="${JOB_TMP_BASE}/merge_robotwin_subset_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOB_TMP_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export HOME="${JOB_TMP_ROOT}/home"
export XDG_CACHE_HOME="${JOB_TMP_ROOT}/xdg_cache"
export XDG_CONFIG_HOME="${JOB_TMP_ROOT}/xdg_config"
export XDG_STATE_HOME="${JOB_TMP_ROOT}/xdg_state"

mkdir -p "${TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" \
  /scratch/yp2841/geometry-vla/openpi_cam/log/merge_robotwin_subset
chmod 700 "${TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" || true

cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/env/activate_env.sh"

echo "===== MERGE ROBOTWIN SUBSET ====="
echo "input repo 1: ${INPUT_REPO_1}"
echo "input repo 2: ${INPUT_REPO_2}"
echo "output repo : ${OUTPUT_REPO}"
echo "input gt 1  : ${INPUT_GT_1}"
echo "input gt 2  : ${INPUT_GT_2}"
echo "output gt   : ${OUTPUT_GT}"
hostname
python -V
date

python scripts/merge_robotwin_lerobot_subset.py \
  --input-repo "${INPUT_REPO_1}" \
  --input-repo "${INPUT_REPO_2}" \
  --output-repo "${OUTPUT_REPO}" \
  --input-gt "${INPUT_GT_1}" \
  --input-gt "${INPUT_GT_2}" \
  --output-gt "${OUTPUT_GT}"

test -f "${OUTPUT_REPO}/meta/info.json"
test "$(find "${OUTPUT_REPO}/data" -name 'episode_*.parquet' | wc -l)" = "100"
test "$(find "${OUTPUT_GT}" -name 'episode_*.npz' | wc -l)" = "300"

du -sh "${OUTPUT_REPO}" "${OUTPUT_GT}"
echo "Merged subset ready."

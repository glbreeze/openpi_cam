#!/bin/bash
#SBATCH --job-name=robotwin_cuda
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/robotwin_curobo/slurm-cuda-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/robotwin_curobo/slurm-cuda-%j.err
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

ROBOTWIN_ENV=${ROBOTWIN_ENV:-/scratch/yp2841/tmp/robotwin2}
JOB_TMP_ROOT=${JOB_TMP_ROOT:-/scratch/yp2841/tmp/robotwin_cuda_${SLURM_JOB_ID:-$$}}
export TMPDIR=${JOB_TMP_ROOT}/tmp
export HOME=${JOB_TMP_ROOT}/home
export XDG_CACHE_HOME=${JOB_TMP_ROOT}/xdg
export CONDA_PKGS_DIRS=${JOB_TMP_ROOT}/conda_pkgs

mkdir -p \
  "${TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${CONDA_PKGS_DIRS}" \
  /scratch/yp2841/geometry-vla/openpi_cam/log/robotwin_curobo

module load anaconda3/2025.06

conda install -p "${ROBOTWIN_ENV}" -y --override-channels -c "nvidia/label/cuda-12.1.0" cuda-toolkit

"${ROBOTWIN_ENV}/bin/nvcc" --version

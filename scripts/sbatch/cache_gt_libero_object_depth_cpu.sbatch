#!/bin/bash
#SBATCH --job-name=gt_cache_libero_object_cpu
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/libero_gt_cache/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/libero_gt_cache/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=torch_pr_637_general
#SBATCH --partition=cs
#SBATCH --export=ALL

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/scratch/yp2841/geometry-vla/openpi_cam}
export MUJOCO_GL=${MUJOCO_GL:-osmesa}

exec bash "${REPO_ROOT}/scripts/sbatch/cache_gt_libero_object_depth_l40s.sbatch"

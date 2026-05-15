#!/bin/bash
#SBATCH --job-name=robotwin_action_mse
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/robotwin_action_mse/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/robotwin_action_mse/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --export=ALL

set -euo pipefail

GEO_ROOT=${GEO_ROOT:-/scratch/yp2841/geometry-vla}
REPO_ROOT=${REPO_ROOT:-${GEO_ROOT}/openpi_cam}
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/env/activate_env.sh"

export TMPDIR=${TMPDIR:-/scratch/yp2841/tmp}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/scratch/yp2841/.cache}
export HF_HOME=${HF_HOME:-/scratch/yp2841/.cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-/scratch/yp2841/.cache/torch}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/scratch/yp2841/.cache/pip}
export OPENPI_DISABLE_TORCH_COMPILE=1
export TOKENIZERS_PARALLELISM=false
export HF_LEROBOT_HOME=${GEO_ROOT}
export OPENPI_GEO_ROOT=${GEO_ROOT}

mkdir -p "${REPO_ROOT}/log/robotwin_action_mse" "${TMPDIR}" "${XDG_CACHE_HOME}" "${HF_HOME}" "${TORCH_HOME}" "${PIP_CACHE_DIR}"

BASELINE_CKPT=${BASELINE_CKPT:-${REPO_ROOT}/checkpoints/pi0_robotwin_cam_baseline/baseline_handover_block_natfix_prio_20260511/20000}
CAM_CKPT=${CAM_CKPT:-${REPO_ROOT}/checkpoints/pi0_robotwin_cam_prope_ray_view_distill_fullres_stage2/stage2_handover_block_camaware_natfix_20260512/20000}
OUT=${OUT:-${REPO_ROOT}/log/robotwin_action_mse/sample_chunk_mse_${SLURM_JOB_ID:-manual}.json}

uv run python scripts/compare_robotwin_sample_chunk_mse.py \
  --baseline-checkpoint "${BASELINE_CKPT}" \
  --cam-checkpoint "${CAM_CKPT}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --num-batches "${NUM_BATCHES:-16}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --num-steps "${NUM_STEPS:-10}" \
  --seed "${SEED:-0}" \
  --out "${OUT}"

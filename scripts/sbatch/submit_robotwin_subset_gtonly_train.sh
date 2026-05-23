#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/scratch/yp2841/geometry-vla/openpi_cam}
GEO_ROOT=${GEO_ROOT:-/scratch/yp2841/geometry-vla}
REPO_ID=${REPO_ID:-robotwin/open_laptop_stack_bowls_two_demo_clean_camaware_100}
STAGE1_EXP_NAME=${STAGE1_EXP_NAME:-stage1_open_laptop_stack_bowls_two_gtonly_20260520}
STAGE2_EXP_NAME=${STAGE2_EXP_NAME:-stage2_open_laptop_stack_bowls_two_gtonly_20260520}

MERGE_SCRIPT=${MERGE_SCRIPT:-${REPO_ROOT}/scripts/sbatch/merge_robotwin_subset_cpu.sbatch}
NORM_SCRIPT=${NORM_SCRIPT:-${REPO_ROOT}/scripts/sbatch/compute_norm_stats_robotwin.sbatch}
S1_SCRIPT=${S1_SCRIPT:-${REPO_ROOT}/scripts/sbatch/train_pi0_robotwin_s1_gtonly_any_4gpu.sbatch}
S2_SCRIPT=${S2_SCRIPT:-${REPO_ROOT}/scripts/sbatch/train_pi0_robotwin_s2_gtonly_any_4gpu.sbatch}

CPU_ACCOUNT=${CPU_ACCOUNT:-torch_pr_637_general}
CPU_PARTITION=${CPU_PARTITION:-cs}
GPU_ACCOUNT=${GPU_ACCOUNT:-torch_pr_637_tandon_advanced}
GPU_PARTITION=${GPU_PARTITION:-a100_tandon,h100_tandon,h200_tandon}

merge_job=$(sbatch --parsable -A "${CPU_ACCOUNT}" -p "${CPU_PARTITION}" \
  --export=ALL,REPO_ROOT="${REPO_ROOT}",GEO_ROOT="${GEO_ROOT}",OUT_REPO_ID="${REPO_ID}" \
  "${MERGE_SCRIPT}")
echo "merge job: ${merge_job}"

norm_job=$(sbatch --parsable --dependency="afterok:${merge_job}" -A "${CPU_ACCOUNT}" -p "${CPU_PARTITION}" \
  --export=ALL,REPO_ROOT="${REPO_ROOT}",GEO_ROOT="${GEO_ROOT}",CONFIG_NAME=pi0_robotwin_cam_baseline,REPO_ID="${REPO_ID}",MAX_FRAMES=50000,BATCH_SIZE=32 \
  "${NORM_SCRIPT}")
echo "norm job: ${norm_job}"

s1_job=$(sbatch --parsable --dependency="afterok:${norm_job}" -A "${GPU_ACCOUNT}" -p "${GPU_PARTITION}" --gres=gpu:4 \
  --export=ALL,REPO_ROOT="${REPO_ROOT}",GEO_ROOT="${GEO_ROOT}",REPO_ID="${REPO_ID}",EXP_NAME="${STAGE1_EXP_NAME}",BATCH_SIZE=32 \
  "${S1_SCRIPT}")
echo "stage1 job: ${s1_job}"

s2_job=$(sbatch --parsable --dependency="afterok:${s1_job}" -A "${GPU_ACCOUNT}" -p "${GPU_PARTITION}" --gres=gpu:4 \
  --export=ALL,REPO_ROOT="${REPO_ROOT}",GEO_ROOT="${GEO_ROOT}",REPO_ID="${REPO_ID}",S1_EXP_NAME="${STAGE1_EXP_NAME}",EXP_NAME="${STAGE2_EXP_NAME}",BATCH_SIZE=64 \
  "${S2_SCRIPT}")
echo "stage2 job: ${s2_job}"

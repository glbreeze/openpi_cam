#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/scratch/yp2841/geometry-vla/openpi_cam}
GEO_ROOT=${GEO_ROOT:-/scratch/yp2841/geometry-vla}
ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-${GEO_ROOT}/RoboTwin}
TASKS_STR=${TASKS:-"open_laptop stack_bowls_two"}
TASK_CONFIG=${TASK_CONFIG:-demo_clean_camaware}
EXPERT_DATA_NUM=${EXPERT_DATA_NUM:-50}
NORM_CONFIG_NAME=${NORM_CONFIG_NAME:-pi0_robotwin_cam_baseline}
NORM_MAX_FRAMES=${NORM_MAX_FRAMES:-50000}
NORM_BATCH_SIZE=${NORM_BATCH_SIZE:-32}
SUBMIT=${SUBMIT:-0}
DO_ASSETS=${DO_ASSETS:-1}

ASSETS_SCRIPT=${ASSETS_SCRIPT:-${REPO_ROOT}/scripts/sbatch/download_robotwin_assets_cpu.sbatch}
COLLECT_SCRIPT=${COLLECT_SCRIPT:-${REPO_ROOT}/scripts/sbatch/collect_robotwin_raw_l40s.sbatch}
CONVERT_SCRIPT=${CONVERT_SCRIPT:-${REPO_ROOT}/scripts/sbatch/convert_robotwin_cam_cpu.sbatch}
NORM_SCRIPT=${NORM_SCRIPT:-${REPO_ROOT}/scripts/sbatch/compute_norm_stats_robotwin.sbatch}
GT_CACHE_SCRIPT=${GT_CACHE_SCRIPT:-${REPO_ROOT}/scripts/sbatch/cache_robotwin_gt_targets_cpu.sbatch}

CPU_ACCOUNT=${CPU_ACCOUNT:-torch_pr_637_general}
CPU_PARTITION=${CPU_PARTITION:-cs}
COLLECT_ACCOUNT=${COLLECT_ACCOUNT:-torch_pr_637_general}
COLLECT_PARTITION=${COLLECT_PARTITION:-l40s_public}
COLLECT_GRES=${COLLECT_GRES:-gpu:l40s:1}

submit_or_print() {
  if [[ "${SUBMIT}" == "1" ]]; then
    sbatch --parsable "$@"
  else
    printf 'DRY RUN: sbatch'
    printf ' %q' "$@"
    printf '\n'
    echo "DRYRUN_JOBID"
  fi
}

read -r -a TASKS_ARR <<< "${TASKS_STR}"

assets_dep=()
if [[ "${DO_ASSETS}" == "1" ]]; then
  assets_job=$(submit_or_print -A "${CPU_ACCOUNT}" -p "${CPU_PARTITION}" --export=ALL,GEO_ROOT="${GEO_ROOT}",ROBOTWIN_ROOT="${ROBOTWIN_ROOT}" "${ASSETS_SCRIPT}")
  echo "assets job: ${assets_job}"
  if [[ "${SUBMIT}" == "1" ]]; then
    assets_dep=(--dependency="afterok:${assets_job}")
  fi
fi

for task in "${TASKS_ARR[@]}"; do
  repo_id="robotwin/${task}_${TASK_CONFIG}_${EXPERT_DATA_NUM}"
  raw_dir="${ROBOTWIN_ROOT}/data/${task}/${TASK_CONFIG}"

  collect_job=$(submit_or_print "${assets_dep[@]}" -A "${COLLECT_ACCOUNT}" -p "${COLLECT_PARTITION}" --gres="${COLLECT_GRES}" --export=ALL,ROBOTWIN_ROOT="${ROBOTWIN_ROOT}",TASK_NAME="${task}",TASK_CONFIG="${TASK_CONFIG}",GPU_ID=0 "${COLLECT_SCRIPT}")
  echo "${task} collect job: ${collect_job}"

  dep=()
  if [[ "${SUBMIT}" == "1" ]]; then
    dep=(--dependency="afterok:${collect_job}")
  fi

  convert_job=$(submit_or_print "${dep[@]}" -A "${CPU_ACCOUNT}" -p "${CPU_PARTITION}" --export=ALL,TASK_NAME="${task}",TASK_CONFIG="${TASK_CONFIG}",EXPERT_DATA_NUM="${EXPERT_DATA_NUM}",RAW_ROOT="${ROBOTWIN_ROOT}/data",REPO_ID="${repo_id}" "${CONVERT_SCRIPT}")
  echo "${task} convert job: ${convert_job}"

  conv_dep=()
  if [[ "${SUBMIT}" == "1" ]]; then
    conv_dep=(--dependency="afterok:${convert_job}")
  fi

  norm_job=$(submit_or_print "${conv_dep[@]}" -A "${CPU_ACCOUNT}" -p "${CPU_PARTITION}" --export=ALL,CONFIG_NAME="${NORM_CONFIG_NAME}",REPO_ID="${repo_id}",MAX_FRAMES="${NORM_MAX_FRAMES}",BATCH_SIZE="${NORM_BATCH_SIZE}" "${NORM_SCRIPT}")
  echo "${task} norm job: ${norm_job}"

  gt_job=$(submit_or_print "${dep[@]}" -A "${CPU_ACCOUNT}" -p "${CPU_PARTITION}" --export=ALL,TASK_NAME="${task}",TASK_CONFIG="${TASK_CONFIG}",EXPERT_DATA_NUM="${EXPERT_DATA_NUM}",ROBOTWIN_ROOT="${ROBOTWIN_ROOT}",RAW_DIR="${raw_dir}",REPO_ID="${repo_id}" "${GT_CACHE_SCRIPT}")
  echo "${task} gt cache job: ${gt_job}"
done

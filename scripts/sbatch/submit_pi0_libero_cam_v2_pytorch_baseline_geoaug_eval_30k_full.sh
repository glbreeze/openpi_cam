#!/bin/bash

set -euo pipefail

resolve_repo_root() {
  local candidate
  local script_root=""

  if script_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd); then
    :
  else
    script_root=""
  fi

  for candidate in "${REPO_ROOT:-}" "${OPENPI_CAM_ROOT:-}" "${PWD:-}" "${script_root}"; do
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

resolve_geoaug_exp_name() {
  local repo_root=$1
  local config_root="${repo_root}/checkpoints/pi0_libero_cam_v2_pytorch_baseline_geoaug"
  local candidate

  for candidate in \
    "pi0_libero_cam_v2_pytorch_baseline_geoaug_v1" \
    "pi0_libero_cam_v2_pytorch_baseline_geoaug_2gpu_v1"; do
    if [[ -d "${config_root}/${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  printf '%s\n' "pi0_libero_cam_v2_pytorch_baseline_geoaug_v1"
}

REPO_ROOT=${REPO_ROOT:-$(resolve_repo_root)} || {
  echo "Unable to locate openpi_cam repo root. Set REPO_ROOT or run from inside the repo." >&2
  exit 1
}

TARGET_SCRIPT="${REPO_ROOT}/scripts/sbatch/eval_pi0_libero_cam_v2_pytorch_baseline_geoaug_full.sbatch"
EXP_NAME=${EXP_NAME:-$(resolve_geoaug_exp_name "${REPO_ROOT}")}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/pi0_libero_cam_v2_pytorch_baseline_geoaug/${EXP_NAME}}
CHECKPOINT_STEP=${CHECKPOINT_STEP:-30000}
SUITES=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-50}
TASK_ID_START=${TASK_ID_START:-0}
TASK_ID_END=${TASK_ID_END:--1}
BASE_PORT=${BASE_PORT:-19000}
LOG_ROOT=${LOG_ROOT:-${REPO_ROOT}/log/libero_eval/${EXP_NAME}/full_30k}

SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-}
SBATCH_PARTITION=${SBATCH_PARTITION:-}
SBATCH_GRES=${SBATCH_GRES:-}
SBATCH_TIME=${SBATCH_TIME:-}
SBATCH_CPUS=${SBATCH_CPUS:-}
SBATCH_MEM=${SBATCH_MEM:-}

mkdir -p "${LOG_ROOT}"

read -r -a suite_list <<< "${SUITES}"
manifest="${LOG_ROOT}/full_eval_30k_manifest_$(date +%Y%m%d_%H%M%S).tsv"
printf 'suite\tstep\tjob_id\tlog_root\n' > "${manifest}"

echo "Submitting full LIBERO geoaug eval at 30k"
echo "repo root: ${REPO_ROOT}"
echo "target script: ${TARGET_SCRIPT}"
echo "config: pi0_libero_cam_v2_pytorch_baseline_geoaug"
echo "exp name: ${EXP_NAME}"
echo "checkpoint root: ${CHECKPOINT_ROOT}"
echo "checkpoint step: ${CHECKPOINT_STEP}"
echo "suites: ${suite_list[*]}"
echo "num_trials_per_task: ${NUM_TRIALS_PER_TASK}"
echo "task range: [${TASK_ID_START}, ${TASK_ID_END})"
echo "log root: ${LOG_ROOT}"
echo "manifest: ${manifest}"
echo "default account: ${SBATCH_ACCOUNT:-torch_pr_926_general}"
echo "default partition: ${SBATCH_PARTITION:-l40s_public}"
echo "default gres: ${SBATCH_GRES:-gpu:l40s:1}"

for suite_idx in "${!suite_list[@]}"; do
  suite="${suite_list[$suite_idx]}"
  port=$((BASE_PORT + suite_idx))
  step_log_root="${LOG_ROOT}/${suite}/step_${CHECKPOINT_STEP}"
  mkdir -p "${step_log_root}"

  SBATCH_ARGS=()
  [[ -n "${SBATCH_ACCOUNT}" ]] && SBATCH_ARGS+=(--account="${SBATCH_ACCOUNT}")
  [[ -n "${SBATCH_PARTITION}" ]] && SBATCH_ARGS+=(--partition="${SBATCH_PARTITION}")
  [[ -n "${SBATCH_GRES}" ]] && SBATCH_ARGS+=(--gres="${SBATCH_GRES}")
  [[ -n "${SBATCH_TIME}" ]] && SBATCH_ARGS+=(--time="${SBATCH_TIME}")
  [[ -n "${SBATCH_CPUS}" ]] && SBATCH_ARGS+=(--cpus-per-task="${SBATCH_CPUS}")
  [[ -n "${SBATCH_MEM}" ]] && SBATCH_ARGS+=(--mem="${SBATCH_MEM}")
  SBATCH_ARGS+=(--job-name="pi0_lgeo_${suite}_30k")
  SBATCH_ARGS+=(--output="${step_log_root}/slurm-%j.out" --error="${step_log_root}/slurm-%j.err")

  job_id=$(
    sbatch --parsable "${SBATCH_ARGS[@]}" \
      --export=ALL,CONFIG_NAME=pi0_libero_cam_v2_pytorch_baseline_geoaug,EXP_NAME="${EXP_NAME}",CHECKPOINT_ROOT="${CHECKPOINT_ROOT}",CHECKPOINT_STEP="${CHECKPOINT_STEP}",SUITE_NAME="${suite}",NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}",TASK_ID_START="${TASK_ID_START}",TASK_ID_END="${TASK_ID_END}",SERVER_PORT="${port}",LOG_ROOT="${step_log_root}",MODEL_NAME="${EXP_NAME}_${suite}_ckpt${CHECKPOINT_STEP}" \
      "${TARGET_SCRIPT}"
  )

  printf '%s\t%s\t%s\t%s\n' "${suite}" "${CHECKPOINT_STEP}" "${job_id}" "${step_log_root}" >> "${manifest}"
  printf '%s step %s -> job %s\n' "${suite}" "${CHECKPOINT_STEP}" "${job_id}"
done

echo "Manifest: ${manifest}"

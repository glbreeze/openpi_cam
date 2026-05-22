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

REPO_ROOT=${REPO_ROOT:-$(resolve_repo_root)} || {
  echo "Unable to locate openpi_cam repo root. Set REPO_ROOT or run from inside the repo." >&2
  exit 1
}

TARGET_SCRIPT="${REPO_ROOT}/scripts/sbatch/train_pi0_ur5_real_robot_cross_attn_fg_distill_stage2_hard_4gpu.sbatch"

SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-}
SBATCH_PARTITION=${SBATCH_PARTITION:-}
SBATCH_GRES=${SBATCH_GRES:-}
SBATCH_CPUS=${SBATCH_CPUS:-}
SBATCH_MEM=${SBATCH_MEM:-}
DEPENDENCY_KIND=${DEPENDENCY_KIND:-afterany}

COMMON_ARGS=()
[[ -n "${SBATCH_ACCOUNT}" ]] && COMMON_ARGS+=(--account="${SBATCH_ACCOUNT}")
[[ -n "${SBATCH_PARTITION}" ]] && COMMON_ARGS+=(--partition="${SBATCH_PARTITION}")
[[ -n "${SBATCH_GRES}" ]] && COMMON_ARGS+=(--gres="${SBATCH_GRES}")
[[ -n "${SBATCH_CPUS}" ]] && COMMON_ARGS+=(--cpus-per-task="${SBATCH_CPUS}")
[[ -n "${SBATCH_MEM}" ]] && COMMON_ARGS+=(--mem="${SBATCH_MEM}")

labels=("default" "10h" "12h" "1d")
times=("" "10:00:00" "12:00:00" "1-00:00:00")

previous_job_id=""

echo "Submitting chained UR5 stage-2 4-GPU resume variants"
echo "target script: ${TARGET_SCRIPT}"
echo "exp name: ${EXP_NAME:-pi0_ur5_real_robot_cross_attn_fg_distill_hard_stage2_4gpu}"
echo "resume: ${RESUME:-false}"
echo "dependency kind: ${DEPENDENCY_KIND}"
echo "variants: ${labels[*]}"

for i in "${!labels[@]}"; do
  label=${labels[$i]}
  time_override=${times[$i]}

  sbatch_args=("${COMMON_ARGS[@]}")
  if [[ -n "${time_override}" ]]; then
    sbatch_args+=(--time="${time_override}")
  fi
  if [[ -n "${previous_job_id}" ]]; then
    sbatch_args+=(--dependency="${DEPENDENCY_KIND}:${previous_job_id}")
  fi

  echo
  echo "Submitting variant: ${label}"
  if [[ -n "${time_override}" ]]; then
    echo "time override: ${time_override}"
  else
    echo "time override: <wrapper default>"
  fi
  if [[ -n "${previous_job_id}" ]]; then
    echo "dependency: ${DEPENDENCY_KIND}:${previous_job_id}"
  fi

  job_id=$(sbatch --parsable "${sbatch_args[@]}" "${TARGET_SCRIPT}")
  echo "submitted ${label}: ${job_id}"
  previous_job_id=${job_id}
done


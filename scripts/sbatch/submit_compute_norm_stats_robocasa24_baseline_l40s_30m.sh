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
  echo "Unable to locate repo root. Set REPO_ROOT or run from inside the repo." >&2
  exit 1
}
GEO_ROOT=${GEO_ROOT:-$(cd -- "${REPO_ROOT}/.." && pwd)}
CONFIG_NAME=${CONFIG_NAME:-pi0_robocasa24_all24_baseline}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-torch_pr_926_general}
SLURM_PARTITION=${SLURM_PARTITION:-l40s_public}
SLURM_TIME=${SLURM_TIME:-}
SBATCH_SCRIPT="${REPO_ROOT}/scripts/sbatch/compute_norm_stats_robocasa24_baseline_l40s_30m.sbatch"

if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
  echo "Missing sbatch script at ${SBATCH_SCRIPT}" >&2
  exit 1
fi

mkdir -p "${REPO_ROOT}/log/compute_norm_stats_robocasa24"

echo "Submitting RoboCasa24 baseline norm stats job"
echo "repo root: ${REPO_ROOT}"
echo "geo root: ${GEO_ROOT}"
echo "config: ${CONFIG_NAME}"
echo "slurm account: ${SLURM_ACCOUNT}"
echo "slurm partition: ${SLURM_PARTITION}"
echo "slurm time: ${SLURM_TIME:-<sbatch default>}"
echo "sbatch script: ${SBATCH_SCRIPT}"

sbatch_time_args=()
if [[ -n "${SLURM_TIME}" ]]; then
  sbatch_time_args+=(--time="${SLURM_TIME}")
fi

job_id=$(
  sbatch --parsable \
    --account="${SLURM_ACCOUNT}" \
    --partition="${SLURM_PARTITION}" \
    "${sbatch_time_args[@]}" \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",GEO_ROOT="${GEO_ROOT}",CONFIG_NAME="${CONFIG_NAME}" \
    "${SBATCH_SCRIPT}"
)

printf 'norm_stats_job=%s\n' "${job_id}"

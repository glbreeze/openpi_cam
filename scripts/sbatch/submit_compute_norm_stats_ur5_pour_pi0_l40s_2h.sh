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
SBATCH_SCRIPT="${REPO_ROOT}/scripts/sbatch/compute_norm_stats_ur5_pour_pi0_l40s_2h.sbatch"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-torch_pr_69_general}"
SLURM_PARTITION="${SLURM_PARTITION:-l40s_public}"
DATASET_REPO_ID="${DATASET_REPO_ID:-ur5_place_and_pour_nuts_camera_shifts}"
NORM_ASSET_ID="${NORM_ASSET_ID:-ur5_place_and_pour_nuts_camera_shifts}"
CONFIG_NAME="${CONFIG_NAME:-pi0_ur5_real_robot_pytorch_baseline}"

mkdir -p "${REPO_ROOT}/log/compute_norm_stats_ur5_pour"

echo "Submitting UR5 pour pi0 norm stats job"
echo "repo root: ${REPO_ROOT}"
echo "geo root: ${GEO_ROOT}"
echo "config: ${CONFIG_NAME}"
echo "dataset repo id: ${DATASET_REPO_ID}"
echo "norm asset id: ${NORM_ASSET_ID}"
echo "output dir: ${GEO_ROOT}/pi0_ur5_real_robot/${NORM_ASSET_ID}"
echo "slurm account: ${SLURM_ACCOUNT}"
echo "slurm partition: ${SLURM_PARTITION}"

job_id=$(
  sbatch --parsable \
    --account="${SLURM_ACCOUNT}" \
    --partition="${SLURM_PARTITION}" \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",OPENPI_GEO_ROOT="${GEO_ROOT}",CONFIG_NAME="${CONFIG_NAME}",DATASET_REPO_ID="${DATASET_REPO_ID}",NORM_ASSET_ID="${NORM_ASSET_ID}" \
    "${SBATCH_SCRIPT}"
)

printf 'norm_stats_pi0_job=%s\n' "${job_id}"

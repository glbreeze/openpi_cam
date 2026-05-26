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

if [[ -z "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}" ]]; then
  echo "Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN before submitting." >&2
  echo "Example: HF_TOKEN=<token> bash ${BASH_SOURCE[0]}" >&2
  exit 1
fi

export RELEASE_DIR="${RELEASE_DIR:-/scratch/${USER}/hf_release/GeoPi0-new}"
export HF_REPO_ID="${HF_REPO_ID:-ColinSkywalker/Pi0-Tube-GT}"
export HF_REPO_TYPE="${HF_REPO_TYPE:-model}"
export HF_UPLOAD_COMMIT_MESSAGE="${HF_UPLOAD_COMMIT_MESSAGE:-Upload GeoPi0-new release folder}"
export HF_UPLOAD_NUM_WORKERS="${HF_UPLOAD_NUM_WORKERS:-8}"

export SBATCH_PARTITION="${SBATCH_PARTITION:-l40s_public}"
export SBATCH_GRES="${SBATCH_GRES:-gpu:l40s:1}"
export SBATCH_TIME="${SBATCH_TIME:-01:00:00}"
export SBATCH_CPUS="${SBATCH_CPUS:-8}"
export SBATCH_MEM="${SBATCH_MEM:-64G}"

echo "Submitting GeoPi0-new upload to Hugging Face"
echo "repo root: ${REPO_ROOT}"
echo "release dir: ${RELEASE_DIR}"
echo "hf repo id: ${HF_REPO_ID}"
echo "partition: ${SBATCH_PARTITION}"
echo "gres: ${SBATCH_GRES}"
echo "time: ${SBATCH_TIME}"

bash "${REPO_ROOT}/scripts/sbatch/submit_upload_geopi0_real_robot_hf.sh"

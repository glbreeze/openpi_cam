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

TARGET_SCRIPT="${REPO_ROOT}/scripts/sbatch/upload_pour_pi05_gt_hf.sbatch"

SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-}
SBATCH_PARTITION=${SBATCH_PARTITION:-}
SBATCH_TIME=${SBATCH_TIME:-}
SBATCH_CPUS=${SBATCH_CPUS:-}
SBATCH_MEM=${SBATCH_MEM:-}

SBATCH_ARGS=()
[[ -n "${SBATCH_ACCOUNT}" ]] && SBATCH_ARGS+=(--account="${SBATCH_ACCOUNT}")
[[ -n "${SBATCH_PARTITION}" ]] && SBATCH_ARGS+=(--partition="${SBATCH_PARTITION}")
[[ -n "${SBATCH_TIME}" ]] && SBATCH_ARGS+=(--time="${SBATCH_TIME}")
[[ -n "${SBATCH_CPUS}" ]] && SBATCH_ARGS+=(--cpus-per-task="${SBATCH_CPUS}")
[[ -n "${SBATCH_MEM}" ]] && SBATCH_ARGS+=(--mem="${SBATCH_MEM}")

echo "Submitting Hugging Face upload job for pour_pi05_gt"
echo "repo root: ${REPO_ROOT}"
echo "target script: ${TARGET_SCRIPT}"
echo "release dir: ${RELEASE_DIR:-/scratch/${USER}/hf_release/pour_pi05_gt}"
echo "hf repo id: ${HF_REPO_ID:-ColinSkywalker/pour_pi05_gt}"
echo "hf repo type: ${HF_REPO_TYPE:-model}"
echo "revision: ${HF_UPLOAD_REVISION:-<default>}"
echo "num workers: ${HF_UPLOAD_NUM_WORKERS:-4}"
echo "include patterns: ${HF_UPLOAD_INCLUDE:-<all>}"
echo "exclude patterns: ${HF_UPLOAD_EXCLUDE:-<none>}"
echo "default account: ${SBATCH_ACCOUNT:-torch_pr_69_tandon_advanced}"
echo "default partition: ${SBATCH_PARTITION:-cpu_short}"
echo "default cpus: ${SBATCH_CPUS:-4}"
echo "default mem: ${SBATCH_MEM:-16G}"
echo "default time: ${SBATCH_TIME:-02:00:00}"
echo "token source: $([[ -n "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}" ]] && echo env || echo existing_login)"

sbatch "${SBATCH_ARGS[@]}" "${TARGET_SCRIPT}"

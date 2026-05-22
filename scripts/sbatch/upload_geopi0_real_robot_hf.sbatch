#!/bin/bash
#SBATCH --job-name=hf_geopi0_upload
#SBATCH --output=/scratch/%u/openpi_cam/log/hf_upload/slurm-%j.out
#SBATCH --error=/scratch/%u/openpi_cam/log/hf_upload/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=h100_tandon,h200_tandon,a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --export=ALL

set -euo pipefail

resolve_repo_root() {
  local candidate
  local script_root=""

  if script_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd); then
    :
  else
    script_root=""
  fi

  for candidate in "${REPO_ROOT:-}" "${OPENPI_CAM_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "${PWD:-}" "${script_root}"; do
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

if [[ -z "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(resolve_repo_root || true)"
fi
if [[ -z "${REPO_ROOT}" && -d "/scratch/${USER}/openpi_cam" ]]; then
  REPO_ROOT="/scratch/${USER}/openpi_cam"
fi
if [[ -z "${REPO_ROOT}" ]]; then
  echo "Unable to locate repo root." >&2
  exit 1
fi

cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/env/activate_env.sh"

RELEASE_DIR="${RELEASE_DIR:-/scratch/${USER}/hf_release/GeoPi0-real-robot}"
HF_REPO_ID="${HF_REPO_ID:-ColinSkywalker/GeoPi0-real-robot}"
HF_REPO_TYPE="${HF_REPO_TYPE:-model}"
HF_UPLOAD_NUM_WORKERS="${HF_UPLOAD_NUM_WORKERS:-8}"
HF_UPLOAD_COMMIT_MESSAGE="${HF_UPLOAD_COMMIT_MESSAGE:-Upload GeoPi0-real-robot release folder}"
HF_UPLOAD_REVISION="${HF_UPLOAD_REVISION:-}"
HF_CREATE_PRIVATE_REPO="${HF_CREATE_PRIVATE_REPO:-false}"
HF_UPLOAD_CACHE_ROOT="${HF_UPLOAD_CACHE_ROOT:-/scratch/${USER}/tmp/hf_upload_cache}"

HF_TOKEN_VALUE="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"

LOG_DIR="/scratch/${USER}/openpi_cam/log/hf_upload"
mkdir -p "${LOG_DIR}"

if [[ ! -d "${RELEASE_DIR}" ]]; then
  echo "Release directory not found: ${RELEASE_DIR}" >&2
  exit 1
fi

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli not found in PATH" >&2
  exit 1
fi

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs not found in PATH" >&2
  exit 1
fi

JOB_CACHE_ROOT="${HF_UPLOAD_CACHE_ROOT%/}/${SLURM_JOB_ID:-manual}"
HF_HOME_DIR="${JOB_CACHE_ROOT}/hf_home"
HF_HUB_CACHE_DIR="${JOB_CACHE_ROOT}/hub"
XDG_CACHE_HOME_DIR="${JOB_CACHE_ROOT}/xdg"
TMPDIR_DIR="${JOB_CACHE_ROOT}/tmp"
UPLOAD_RELEASE_DIR="${JOB_CACHE_ROOT}/release_snapshot"

mkdir -p "${HF_HOME_DIR}" "${HF_HUB_CACHE_DIR}" "${XDG_CACHE_HOME_DIR}" "${TMPDIR_DIR}"

# Upload from a hardlinked snapshot so resumable HF metadata lives outside the
# canonical release folder and concurrent resubmissions cannot corrupt it.
rm -rf "${UPLOAD_RELEASE_DIR}"
mkdir -p "${UPLOAD_RELEASE_DIR}"
cp -al "${RELEASE_DIR}/." "${UPLOAD_RELEASE_DIR}/"
rm -rf "${UPLOAD_RELEASE_DIR}/.cache"

export HF_HOME="${HF_HOME_DIR}"
export HF_HUB_CACHE="${HF_HUB_CACHE_DIR}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME_DIR}"
export TMPDIR="${TMPDIR_DIR}"

echo "===== HF UPLOAD ENV CHECK ====="
echo "repo root: ${REPO_ROOT}"
echo "release dir: ${RELEASE_DIR}"
echo "upload snapshot dir: ${UPLOAD_RELEASE_DIR}"
echo "hf repo id: ${HF_REPO_ID}"
echo "hf repo type: ${HF_REPO_TYPE}"
echo "num workers: ${HF_UPLOAD_NUM_WORKERS}"
echo "commit message: ${HF_UPLOAD_COMMIT_MESSAGE}"
echo "revision: ${HF_UPLOAD_REVISION:-<default>}"
echo "create private repo: ${HF_CREATE_PRIVATE_REPO}"
echo "token source: $([[ -n "${HF_TOKEN_VALUE}" ]] && echo env || echo existing_login)"
echo "hf home: ${HF_HOME}"
echo "hf hub cache: ${HF_HUB_CACHE}"
echo "xdg cache home: ${XDG_CACHE_HOME}"
echo "tmpdir: ${TMPDIR}"
echo "python: $(command -v python)"
echo "huggingface-cli: $(command -v huggingface-cli)"
echo "git-lfs: $(command -v git-lfs)"
hostname
date

UPLOAD_ARGS=(
  upload-large-folder
  "${HF_REPO_ID}"
  "${UPLOAD_RELEASE_DIR}"
  --repo-type "${HF_REPO_TYPE}"
  --num-workers "${HF_UPLOAD_NUM_WORKERS}"
)

if [[ -n "${HF_UPLOAD_REVISION}" ]]; then
  UPLOAD_ARGS+=(--revision "${HF_UPLOAD_REVISION}")
fi

if [[ "${HF_CREATE_PRIVATE_REPO}" == "true" ]]; then
  UPLOAD_ARGS+=(--private)
fi

if [[ -n "${HF_TOKEN_VALUE}" ]]; then
  UPLOAD_ARGS+=(--token "${HF_TOKEN_VALUE}")
fi

echo "Running: huggingface-cli ${UPLOAD_ARGS[*]}"
huggingface-cli "${UPLOAD_ARGS[@]}"

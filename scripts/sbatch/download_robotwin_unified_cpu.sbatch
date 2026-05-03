#!/bin/bash
#SBATCH --job-name=dl_robotwin
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/robotwin_dataset_download/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/robotwin_dataset_download/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --account=torch_pr_637_general
#SBATCH --partition=cs
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

  for candidate in "${REPO_ROOT:-}" "${OPENPI_CAM_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "${script_root}"; do
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
  echo "Unable to locate repo root. Set REPO_ROOT or submit from inside the repo." >&2
  exit 1
}
GEO_ROOT=${GEO_ROOT:-$(cd -- "${REPO_ROOT}/.." && pwd)}

JOB_TMP_BASE="${SLURM_TMPDIR:-/tmp/$USER}"
JOB_TMP_ROOT="${JOB_TMP_BASE}/robotwin_dl_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOB_TMP_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export HOME="${JOB_TMP_ROOT}/home"
export XDG_CACHE_HOME="${JOB_TMP_ROOT}/xdg_cache"
export XDG_CONFIG_HOME="${JOB_TMP_ROOT}/xdg_config"
export XDG_STATE_HOME="${JOB_TMP_ROOT}/xdg_state"

mkdir -p \
  "${TMPDIR}" \
  "${HOME}" \
  "${XDG_CACHE_HOME}" \
  "${XDG_CONFIG_HOME}" \
  "${XDG_STATE_HOME}" \
  /scratch/yp2841/geometry-vla/openpi_cam/log/robotwin_dataset_download
chmod 700 "${TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" || true

cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/env/activate_env.sh"

export HF_LEROBOT_HOME="${GEO_ROOT}"
export HF_HOME="${GEO_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"

REPO_ID=${REPO_ID:-lerobot/robotwin_unified}
TARGET_DIR="${HF_LEROBOT_HOME}/${REPO_ID}"
export REPO_ID TARGET_DIR

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "$(dirname -- "${TARGET_DIR}")"

echo "===== ROBO TWIN DATASET DOWNLOAD ====="
echo "repo root: ${REPO_ROOT}"
echo "geo root: ${GEO_ROOT}"
echo "hf lerobot home: ${HF_LEROBOT_HOME}"
echo "repo id: ${REPO_ID}"
echo "target dir: ${TARGET_DIR}"
hostname
python -V

python - <<'PY'
from pathlib import Path
import os

from huggingface_hub import snapshot_download

repo_id = os.environ["REPO_ID"]
target_dir = Path(os.environ["TARGET_DIR"])
target_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=str(target_dir),
    local_dir_use_symlinks=False,
    resume_download=True,
)
PY

if [[ ! -d "${TARGET_DIR}/data" || ! -f "${TARGET_DIR}/meta/info.json" ]]; then
  echo "Dataset download finished but expected LeRobot structure is missing at ${TARGET_DIR}" >&2
  exit 1
fi

echo "Dataset ready at ${TARGET_DIR}"

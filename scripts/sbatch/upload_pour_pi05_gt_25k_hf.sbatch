#!/bin/bash
#SBATCH --job-name=hf_pour_pi05_gt_25k
#SBATCH --output=/scratch/%u/openpi_cam/log/hf_upload/slurm-%j.out
#SBATCH --error=/scratch/%u/openpi_cam/log/hf_upload/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=cpu_short
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

RELEASE_DIR="${RELEASE_DIR:-/scratch/${USER}/hf_release/pour_pi05_gt}"
HF_REPO_ID="${HF_REPO_ID:-ColinSkywalker/pour_pi05_gt}"
HF_REPO_TYPE="${HF_REPO_TYPE:-model}"
HF_UPLOAD_REVISION="${HF_UPLOAD_REVISION:-}"
HF_UPLOAD_COMMIT_MESSAGE="${HF_UPLOAD_COMMIT_MESSAGE:-Upload pour_pi05_gt 25000 checkpoint and remove placeholder}"

HF_TOKEN_VALUE="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"

export RELEASE_DIR HF_REPO_ID HF_REPO_TYPE HF_UPLOAD_REVISION HF_UPLOAD_COMMIT_MESSAGE

if [[ ! -d "${RELEASE_DIR}" ]]; then
  echo "Release directory not found: ${RELEASE_DIR}" >&2
  exit 1
fi

if [[ -z "${HF_TOKEN_VALUE}" ]]; then
  echo "Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN before running." >&2
  exit 1
fi

echo "===== HF 25K UPLOAD ENV CHECK ====="
echo "repo root: ${REPO_ROOT}"
echo "release dir: ${RELEASE_DIR}"
echo "hf repo id: ${HF_REPO_ID}"
echo "hf repo type: ${HF_REPO_TYPE}"
echo "revision: ${HF_UPLOAD_REVISION:-<default>}"
echo "commit message: ${HF_UPLOAD_COMMIT_MESSAGE}"
echo "python: $(command -v python)"
hostname
date

python - <<'PY'
import os
from huggingface_hub import HfApi

repo_id = os.environ["HF_REPO_ID"] if "HF_REPO_ID" in os.environ else f"{os.environ['USER']}/pour_pi05_gt"
repo_type = os.environ.get("HF_REPO_TYPE", "model")
folder_path = os.environ.get("RELEASE_DIR", f"/scratch/{os.environ['USER']}/hf_release/pour_pi05_gt")
revision = os.environ.get("HF_UPLOAD_REVISION") or None
commit_message = os.environ.get(
    "HF_UPLOAD_COMMIT_MESSAGE",
    "Upload pour_pi05_gt 25000 checkpoint and remove placeholder",
)
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

api = HfApi(token=token)
api.upload_folder(
    repo_id=repo_id,
    repo_type=repo_type,
    folder_path=folder_path,
    revision=revision,
    commit_message=commit_message,
    allow_patterns=[
        "checkpoints/25000/model.safetensors",
        "checkpoints/25000/metadata.pt",
        "checkpoints/25000/assets/*",
        "checkpoints/25000/assets/**/*",
    ],
    delete_patterns=[
        "checkpoints/25000/PENDING.txt",
    ],
    token=token,
)
print("Uploaded 25000 checkpoint subtree and deleted remote placeholder if present.")
PY

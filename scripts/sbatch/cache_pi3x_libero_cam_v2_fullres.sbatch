#!/bin/bash
#SBATCH --job-name=cache_pi3x_v2_224
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/cache_pi3x_224/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/cache_pi3x_224/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=l40s_public
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

export OPENPI_GEO_ROOT="${OPENPI_GEO_ROOT:-${GEO_ROOT}}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/scratch/yp2841/cache/lerobot}"
export HF_HOME="${HF_HOME:-${GEO_ROOT}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export OPENPI_CACHE_DIR="${OPENPI_CACHE_DIR:-${GEO_ROOT}/.cache/openpi}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${OPENPI_CACHE_DIR}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${GEO_ROOT}/.cache/xdg}"
unset TRANSFORMERS_CACHE

REPO_ID=${REPO_ID:-glbreeze/libero_cam_v2}
DATA_ROOT="${HF_LEROBOT_HOME}/${REPO_ID}"
PI3X_REPO="${PI3X_REPO:-${GEO_ROOT}/Pi3X_Libero}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${OPENPI_CACHE_DIR}/pi3x_targets_224/libero_cam_v2}"
BATCH_SIZE="${BATCH_SIZE:-16}"
OUTPUT_RESOLUTION="${OUTPUT_RESOLUTION:-224}"
LINK_PATH="${LINK_PATH:-${HOME}/.cache/openpi/pi3x_targets_224/libero_cam_v2}"

mkdir -p \
  "${HF_HOME}" \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${OPENPI_CACHE_DIR}" \
  "${XDG_CACHE_HOME}" \
  "$(dirname -- "${OUTPUT_ROOT}")" \
  "$(dirname -- "${LINK_PATH}")"

if [[ ! -d "${DATA_ROOT}/data" || ! -f "${DATA_ROOT}/meta/info.json" ]]; then
  echo "Missing LeRobot dataset under ${DATA_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${PI3X_REPO}" ]]; then
  echo "Missing Pi3X repo at ${PI3X_REPO}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/env/activate_env.sh"

cp -rf src/openpi/models_pytorch/transformers_replace/. .venv/lib/python3.11/site-packages/transformers/

echo "===== PI3X FULL-RES CACHE ENV ====="
echo "repo root: ${REPO_ROOT}"
echo "hf lerobot home: ${HF_LEROBOT_HOME}"
echo "repo id: ${REPO_ID}"
echo "data root: ${DATA_ROOT}"
echo "pi3x repo: ${PI3X_REPO}"
echo "output root: ${OUTPUT_ROOT}"
echo "link path: ${LINK_PATH}"
echo "batch size: ${BATCH_SIZE}"
echo "output resolution: ${OUTPUT_RESOLUTION}"
hostname
python -V

python - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id="yyfz233/Pi3X", repo_type="model")
print(f"Pi3X weights ready at: {path}")
PY

python scripts/cache_pi3x_targets.py \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --pi3x-repo "${PI3X_REPO}" \
  --output-resolution "${OUTPUT_RESOLUTION}" \
  --batch-size "${BATCH_SIZE}"

ln -sfn "${OUTPUT_ROOT}" "${LINK_PATH}"
echo "Symlinked ${LINK_PATH} -> ${OUTPUT_ROOT}"

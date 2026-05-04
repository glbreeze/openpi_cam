#!/bin/bash
#SBATCH --job-name=norm_libcam_v3
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/compute_norm_stats_libero_cam_v3/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/compute_norm_stats_libero_cam_v3/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=torch_pr_676_general
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
JOB_TMP_ROOT="${JOB_TMP_BASE}/openpi_norm_stats_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOB_TMP_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export DATASETS_TMPDIR="${JOB_TMP_ROOT}/datasets_tmp"
export HOME="${JOB_TMP_ROOT}/home"
export XDG_CACHE_HOME="${JOB_TMP_ROOT}/xdg_cache"
export XDG_CONFIG_HOME="${JOB_TMP_ROOT}/xdg_config"
export XDG_STATE_HOME="${JOB_TMP_ROOT}/xdg_state"

mkdir -p "${TMPDIR}" "${DATASETS_TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}"
chmod 700 "${TMPDIR}" "${DATASETS_TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" || true

cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/env/activate_env.sh"

export OPENPI_GEO_ROOT="${GEO_ROOT}"
export HF_LEROBOT_HOME="/scratch/yp2841/.huggingface/lerobot"
export HF_HOME="${GEO_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export OPENPI_CACHE_DIR="${GEO_ROOT}/.cache/openpi"
export OPENPI_DATA_HOME="${OPENPI_CACHE_DIR}"
unset TRANSFORMERS_CACHE

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${OPENPI_CACHE_DIR}"

# `pi0_libero_cam_pytorch_prope_ray_view_distill_fullres` resolves
# `pi3x_targets_root` via `~/.cache/openpi/...` at Python runtime. This sbatch
# script overrides HOME into a job-local tmp dir, so mirror the real cache into
# that tmp HOME to keep `expanduser("~/.cache/...")` pointing at the existing
# scratch-backed Pi3X target cache.
JOB_HOME_OPENPI_CACHE="${HOME}/.cache/openpi"
mkdir -p "${JOB_HOME_OPENPI_CACHE}"
if [[ -L "${JOB_HOME_OPENPI_CACHE}/pi3x_targets_224" || -d "${JOB_HOME_OPENPI_CACHE}/pi3x_targets_224" ]]; then
  rm -rf "${JOB_HOME_OPENPI_CACHE}/pi3x_targets_224"
fi
ln -s "${OPENPI_CACHE_DIR}/pi3x_targets_224" "${JOB_HOME_OPENPI_CACHE}/pi3x_targets_224"

DATASET_DIR="${HF_LEROBOT_HOME}/glbreeze/libero_object_cam_v3"
OUTPUT_DIR="${GEO_ROOT}/pi0_libero/glbreeze/libero_object_cam_v3"

if [[ ! -d "${DATASET_DIR}/data" || ! -f "${DATASET_DIR}/meta/info.json" ]]; then
  echo "Missing local LeRobot dataset at ${DATASET_DIR}" >&2
  exit 1
fi

echo "===== NORM STATS ENV ====="
echo "repo root: ${REPO_ROOT}"
echo "openpi geo root: ${OPENPI_GEO_ROOT}"
echo "hf lerobot home: ${HF_LEROBOT_HOME}"
echo "dataset dir: ${DATASET_DIR}"
echo "output dir: ${OUTPUT_DIR}"
hostname
python -V

python scripts/compute_norm_stats.py --config-name=pi0_libero_cam_pytorch_prope_ray_view_distill_fullres

if [[ ! -f "${OUTPUT_DIR}/norm_stats.json" ]]; then
  echo "Expected norm stats missing at ${OUTPUT_DIR}/norm_stats.json" >&2
  exit 1
fi

echo "Norm stats ready at ${OUTPUT_DIR}/norm_stats.json"

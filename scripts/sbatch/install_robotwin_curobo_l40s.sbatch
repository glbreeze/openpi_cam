#!/bin/bash
#SBATCH --job-name=install_curobo
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/install_robotwin_curobo/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/install_robotwin_curobo/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_637_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --export=ALL

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/scratch/yp2841/geometry-vla/openpi_cam}
GEO_ROOT=${GEO_ROOT:-/scratch/yp2841/geometry-vla}
ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-${GEO_ROOT}/RoboTwin}
ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-/scratch/yp2841/tmp/robotwin2/bin/python}
CUROBO_REF=${CUROBO_REF:-v0.7.8}
CUROBO_SRC=${CUROBO_SRC:-${ROBOTWIN_ROOT}/envs/curobo_${CUROBO_REF}}

JOB_TMP_BASE="${SLURM_TMPDIR:-/tmp/$USER}"
JOB_TMP_ROOT="${JOB_TMP_BASE}/install_curobo_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOB_TMP_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export HOME="${JOB_TMP_ROOT}/home"
export XDG_CACHE_HOME="${JOB_TMP_ROOT}/xdg_cache"
export XDG_CONFIG_HOME="${JOB_TMP_ROOT}/xdg_config"
export XDG_STATE_HOME="${JOB_TMP_ROOT}/xdg_state"
export PIP_CACHE_DIR="${GEO_ROOT}/.cache/pip"
export TORCH_EXTENSIONS_DIR="${GEO_ROOT}/.cache/torch_extensions/robotwin_curobo"
export MAX_JOBS=${MAX_JOBS:-8}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.0;8.9;9.0"}

mkdir -p \
  "${TMPDIR}" \
  "${HOME}" \
  "${XDG_CACHE_HOME}" \
  "${XDG_CONFIG_HOME}" \
  "${XDG_STATE_HOME}" \
  "${PIP_CACHE_DIR}" \
  "${TORCH_EXTENSIONS_DIR}" \
  /scratch/yp2841/geometry-vla/openpi_cam/log/install_robotwin_curobo
chmod 700 "${TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" || true

if [[ ! -x "${ROBOTWIN_PYTHON}" ]]; then
  echo "Missing executable ROBOTWIN_PYTHON at ${ROBOTWIN_PYTHON}" >&2
  exit 1
fi

ROBOTWIN_VENV=$(cd -- "$(dirname -- "${ROBOTWIN_PYTHON}")/.." && pwd)
export VIRTUAL_ENV="${ROBOTWIN_VENV}"
export CUDA_HOME="${CUDA_HOME:-${ROBOTWIN_VENV}}"
export PATH="${ROBOTWIN_VENV}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

echo "===== INSTALL ROBOTWIN CUROBO ====="
echo "robotwin root : ${ROBOTWIN_ROOT}"
echo "python        : $(command -v python)"
echo "pip           : $(command -v pip)"
echo "cuda home     : ${CUDA_HOME}"
echo "curobo ref    : ${CUROBO_REF}"
echo "curobo src    : ${CUROBO_SRC}"
echo "torch arch    : ${TORCH_CUDA_ARCH_LIST}"
hostname
python -V
python - <<'PY'
import torch
print("torch", torch.__version__, "torch_cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
PY
nvcc --version || true

mkdir -p "$(dirname -- "${CUROBO_SRC}")"
if [[ ! -d "${CUROBO_SRC}/.git" ]]; then
  if [[ -e "${CUROBO_SRC}" ]]; then
    backup="${CUROBO_SRC}.preinstall.${SLURM_JOB_ID:-$$}"
    echo "Existing non-git curobo path found; moving to ${backup}"
    mv "${CUROBO_SRC}" "${backup}"
  fi
  git clone --branch "${CUROBO_REF}" --depth 1 https://github.com/NVlabs/curobo.git "${CUROBO_SRC}"
else
  git -C "${CUROBO_SRC}" fetch --depth 1 origin "refs/tags/${CUROBO_REF}:refs/tags/${CUROBO_REF}" || true
  git -C "${CUROBO_SRC}" checkout "${CUROBO_REF}"
fi

cd "${CUROBO_SRC}"
python -m pip install -U pip setuptools wheel ninja
python -m pip install -e . --no-build-isolation

cd "${ROBOTWIN_ROOT}"
python - <<'PY'
import curobo
from curobo.types.math import Pose
print("curobo_import_ok", getattr(curobo, "__file__", "<namespace>"))
print("pose_class_ok", Pose)
PY

echo "Curobo installed and import smoke test passed."

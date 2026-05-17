#!/bin/bash
#SBATCH --job-name=robotwin_assets
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/robotwin_assets/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/robotwin_assets/slurm-%j.err
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00

set -euo pipefail

GEO_ROOT=${GEO_ROOT:-/scratch/yp2841/geometry-vla}
ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-${GEO_ROOT}/RoboTwin}
ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-/scratch/yp2841/tmp/robotwin2/bin/python}

JOB_TMP_ROOT=${JOB_TMP_ROOT:-/scratch/yp2841/tmp/robotwin_assets_${SLURM_JOB_ID:-$$}}
export TMPDIR=${JOB_TMP_ROOT}/tmp
export HOME=${JOB_TMP_ROOT}/home
export XDG_CACHE_HOME=${JOB_TMP_ROOT}/xdg
export HF_HOME=${JOB_TMP_ROOT}/huggingface
export PIP_CACHE_DIR=${JOB_TMP_ROOT}/pip
export MPLCONFIGDIR=${JOB_TMP_ROOT}/matplotlib

mkdir -p \
  "${TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${HF_HOME}" "${PIP_CACHE_DIR}" "${MPLCONFIGDIR}" \
  /scratch/yp2841/geometry-vla/openpi_cam/log/robotwin_assets

if [[ ! -x "${ROBOTWIN_PYTHON}" ]]; then
  echo "Missing RoboTwin python at ${ROBOTWIN_PYTHON}" >&2
  exit 1
fi

if [[ ! -d "${ROBOTWIN_ROOT}/assets" ]]; then
  echo "Missing RoboTwin repo/assets at ${ROBOTWIN_ROOT}" >&2
  exit 1
fi

cd "${ROBOTWIN_ROOT}"

echo "robotwin root : ${ROBOTWIN_ROOT}"
echo "python        : ${ROBOTWIN_PYTHON}"
echo "hf cache      : ${HF_HOME}"

cd assets
for archive in background_texture.zip embodiments.zip objects.zip; do
  if [[ ! -f "${archive}" && -f "${ROBOTWIN_ROOT}/${archive}" ]]; then
    mv "${ROBOTWIN_ROOT}/${archive}" "${archive}"
  fi
done

if [[ ! -f background_texture.zip || ! -f embodiments.zip || ! -f objects.zip ]]; then
  "${ROBOTWIN_PYTHON}" _download.py
fi

for archive in background_texture.zip embodiments.zip objects.zip; do
  if [[ ! -f "${archive}" ]]; then
    echo "Expected downloaded archive missing: ${archive}" >&2
    exit 1
  fi
  unzip -oq "${archive}"
  rm -f "${archive}"
done

cd "${ROBOTWIN_ROOT}"
"${ROBOTWIN_PYTHON}" script/update_embodiment_config_path.py

du -sh assets
find assets -maxdepth 2 -type d | sort | head -80

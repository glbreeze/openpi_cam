#!/bin/bash
#SBATCH --job-name=robotwin_curobo
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/robotwin_curobo/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/robotwin_curobo/slurm-%j.err
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -euo pipefail

GEO_ROOT=${GEO_ROOT:-/scratch/yp2841/geometry-vla}
ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-${GEO_ROOT}/RoboTwin}
ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-/scratch/yp2841/tmp/robotwin2/bin/python}

JOB_TMP_ROOT=${JOB_TMP_ROOT:-/scratch/yp2841/tmp/robotwin_curobo_${SLURM_JOB_ID:-$$}}
export TMPDIR=${JOB_TMP_ROOT}/tmp
export HOME=${JOB_TMP_ROOT}/home
export XDG_CACHE_HOME=${JOB_TMP_ROOT}/xdg
export PIP_CACHE_DIR=${JOB_TMP_ROOT}/pip

mkdir -p \
  "${TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}" \
  /scratch/yp2841/geometry-vla/openpi_cam/log/robotwin_curobo

cd "${ROBOTWIN_ROOT}/envs/curobo"
"${ROBOTWIN_PYTHON}" -m pip install -e . --no-build-isolation

"${ROBOTWIN_PYTHON}" - <<'PY'
import curobo
import warp
import quaternion
import yourdfpy
print("curobo ok", curobo.__file__)
print("warp ok", warp.__version__)
print("numpy-quaternion ok")
print("yourdfpy ok")
PY

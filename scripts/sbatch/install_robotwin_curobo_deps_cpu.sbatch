#!/bin/bash
#SBATCH --job-name=robotwin_curobo_deps
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/robotwin_curobo/slurm-deps-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/robotwin_curobo/slurm-deps-%j.err
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -euo pipefail

ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-/scratch/yp2841/tmp/robotwin2/bin/python}
CUROBO_SRC=${CUROBO_SRC:-/scratch/yp2841/geometry-vla/RoboTwin/envs/curobo/src}

JOB_TMP_ROOT=${JOB_TMP_ROOT:-/scratch/yp2841/tmp/robotwin_curobo_deps_${SLURM_JOB_ID:-$$}}
export TMPDIR=${JOB_TMP_ROOT}/tmp
export HOME=${JOB_TMP_ROOT}/home
export XDG_CACHE_HOME=${JOB_TMP_ROOT}/xdg
export PIP_CACHE_DIR=${JOB_TMP_ROOT}/pip

mkdir -p \
  "${TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}" \
  /scratch/yp2841/geometry-vla/openpi_cam/log/robotwin_curobo

"${ROBOTWIN_PYTHON}" -m pip install \
  "warp-lang>=0.9.0" \
  numpy-quaternion \
  setuptools_scm \
  yourdfpy \
  importlib_resources

PYTHONPATH="${CUROBO_SRC}:${PYTHONPATH:-}" "${ROBOTWIN_PYTHON}" - <<'PY'
import curobo
import warp
import quaternion
import yourdfpy
from curobo.types.math import Pose
print("curobo source ok", curobo.__file__)
print("warp ok", warp.__version__)
print("Pose ok", Pose)
PY

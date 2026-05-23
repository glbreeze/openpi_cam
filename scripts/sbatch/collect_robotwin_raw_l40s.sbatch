#!/bin/bash
#SBATCH --job-name=robotwin_collect
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/collect_robotwin_raw/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/collect_robotwin_raw/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --account=torch_pr_637_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --export=ALL

set -euo pipefail

CONDA_SH=${CONDA_SH:-/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh}
ROBOTWIN_CONDA_ENV=${ROBOTWIN_CONDA_ENV:-robotwin}
ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-/scratch/yp2841/tmp/robotwin2/bin/python}
ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-/scratch/yp2841/geometry-vla/RoboTwin}
TASK_NAME=${TASK_NAME:-beat_block_hammer}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
GPU_ID=${GPU_ID:-0}

mkdir -p /scratch/yp2841/geometry-vla/openpi_cam/log/collect_robotwin_raw

JOB_TMP_BASE="${SLURM_TMPDIR:-/tmp/$USER}"
JOB_TMP_ROOT="${JOB_TMP_BASE}/robotwin_collect_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOB_TMP_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export HOME="${JOB_TMP_ROOT}/home"
export XDG_CACHE_HOME="${JOB_TMP_ROOT}/xdg_cache"
export XDG_CONFIG_HOME="${JOB_TMP_ROOT}/xdg_config"
export XDG_STATE_HOME="${JOB_TMP_ROOT}/xdg_state"
export MPLCONFIGDIR="${JOB_TMP_ROOT}/matplotlib"
export JOB_BIN="${JOB_TMP_ROOT}/bin"
mkdir -p "${TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" "${MPLCONFIGDIR}" "${JOB_BIN}"
chmod 700 "${TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" "${MPLCONFIGDIR}" "${JOB_BIN}" || true

if [[ ! -d "${ROBOTWIN_ROOT}" ]]; then
  echo "Missing RoboTwin repo at ${ROBOTWIN_ROOT}" >&2
  exit 1
fi

if [[ -x "${ROBOTWIN_PYTHON}" ]]; then
  ROBOTWIN_VENV=$(cd -- "$(dirname -- "${ROBOTWIN_PYTHON}")/.." && pwd)
  export VIRTUAL_ENV="${ROBOTWIN_VENV}"
  export PATH="${ROBOTWIN_VENV}/bin:${PATH}"
else
  if [[ ! -f "${CONDA_SH}" ]]; then
    echo "Missing conda.sh at ${CONDA_SH} and ROBOTWIN_PYTHON is not executable at ${ROBOTWIN_PYTHON}" >&2
    exit 1
  fi
  source "${CONDA_SH}"
  conda activate "${ROBOTWIN_CONDA_ENV}"
fi

cd "${ROBOTWIN_ROOT}"

echo "===== ROBOTWIN RAW COLLECTION ====="
echo "robotwin root: ${ROBOTWIN_ROOT}"
echo "python: $(command -v python)"
echo "task name: ${TASK_NAME}"
echo "task config: ${TASK_CONFIG}"
echo "gpu id: ${GPU_ID}"
echo "slurm cuda visible devices: ${CUDA_VISIBLE_DEVICES:-<unset>}"
hostname
python -V

FFMPEG_EXE=$(python - <<'PY'
try:
    import imageio_ffmpeg
    print(imageio_ffmpeg.get_ffmpeg_exe())
except Exception:
    print("")
PY
)
if [[ -n "${FFMPEG_EXE}" && -x "${FFMPEG_EXE}" ]]; then
  ln -sf "${FFMPEG_EXE}" "${JOB_BIN}/ffmpeg"
  export PATH="${JOB_BIN}:${PATH}"
  echo "ffmpeg: ${JOB_BIN}/ffmpeg -> ${FFMPEG_EXE}"
else
  echo "warning: imageio_ffmpeg ffmpeg binary not found; PATH ffmpeg will be used if available"
fi

./script/.update_path.sh > /dev/null 2>&1 || true

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
echo "effective cuda visible devices: ${CUDA_VISIBLE_DEVICES:-<unset>}"

rm -rf "data/${TASK_NAME}/${TASK_CONFIG}/.cache"
PYTHONWARNINGS=ignore::UserWarning PYTHONUNBUFFERED=1 python -u script/collect_data.py "${TASK_NAME}" "${TASK_CONFIG}"
rm -rf "data/${TASK_NAME}/${TASK_CONFIG}/.cache"

RAW_DIR="${ROBOTWIN_ROOT}/data/${TASK_NAME}/${TASK_CONFIG}"
if [[ ! -d "${RAW_DIR}/data" || ! -d "${RAW_DIR}/instructions" ]]; then
  echo "Expected raw collection output missing under ${RAW_DIR}" >&2
  exit 1
fi

echo "Collected raw RoboTwin data at ${RAW_DIR}"

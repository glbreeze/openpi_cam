#!/bin/bash
#SBATCH --job-name=setup_rc24_env
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/robocasa24_env_setup/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/robocasa24_env_setup/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=cs
#SBATCH --export=ALL

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/scratch/yp2841/geometry-vla/openpi_cam}
GEO_ROOT=${GEO_ROOT:-$(cd -- "${REPO_ROOT}/.." && pwd)}
mkdir -p "${REPO_ROOT}/log/robocasa24_env_setup"
cd "${REPO_ROOT}"

export TMPDIR="${TMPDIR:-/scratch/${USER}/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/${USER}/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${XDG_CACHE_HOME}/pip}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${XDG_CACHE_HOME}/torch}"
export PYTHONNOUSERSITE=1
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}"

CONDA_SH=${CONDA_SH:-/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh}
ROBOCASA_ENV=${ROBOCASA_ENV:-/scratch/${USER}/.venvs/robocasa24}
ROBOCASA24_SOURCE_ROOT=${ROBOCASA24_SOURCE_ROOT:-/scratch/yz11445/openpi_cam/third_party/robocasa_v02}
ROBOSUITE151_SOURCE_ROOT=${ROBOSUITE151_SOURCE_ROOT:-/scratch/yz11445/openpi_cam/third_party/robosuite_v151}
ROBOCASA24_ASSETS_ROOT=${ROBOCASA24_ASSETS_ROOT:-${ROBOCASA24_SOURCE_ROOT}/robocasa/models/assets}
LOCAL_SOURCE_ROOT=${LOCAL_SOURCE_ROOT:-/scratch/${USER}/src/robocasa24_infer}
LOCAL_ROBOCASA24_SOURCE_ROOT=${LOCAL_ROBOCASA24_SOURCE_ROOT:-${LOCAL_SOURCE_ROOT}/robocasa_v02}
LOCAL_ROBOSUITE151_SOURCE_ROOT=${LOCAL_ROBOSUITE151_SOURCE_ROOT:-${LOCAL_SOURCE_ROOT}/robosuite_v151}

[[ -f "${CONDA_SH}" ]] || { echo "Missing conda.sh: ${CONDA_SH}" >&2; exit 1; }
[[ -d "${ROBOCASA24_SOURCE_ROOT}" ]] || { echo "Missing RoboCasa v0.2 source: ${ROBOCASA24_SOURCE_ROOT}" >&2; exit 1; }
[[ -d "${ROBOSUITE151_SOURCE_ROOT}" ]] || { echo "Missing robosuite 1.5.1 source: ${ROBOSUITE151_SOURCE_ROOT}" >&2; exit 1; }
[[ -d "${ROBOCASA24_ASSETS_ROOT}" ]] || { echo "Missing RoboCasa assets: ${ROBOCASA24_ASSETS_ROOT}" >&2; exit 1; }

echo "===== SETUP ROBOCASA24 INFER ENV ====="
echo "env           : ${ROBOCASA_ENV}"
echo "robocasa src  : ${ROBOCASA24_SOURCE_ROOT}"
echo "robosuite src : ${ROBOSUITE151_SOURCE_ROOT}"
echo "local src     : ${LOCAL_SOURCE_ROOT}"
hostname
date

mkdir -p "${LOCAL_SOURCE_ROOT}"
rsync -a --delete \
  --exclude '/robocasa/models/assets' \
  --exclude '*.egg-info' \
  --exclude '__pycache__' \
  "${ROBOCASA24_SOURCE_ROOT}/" "${LOCAL_ROBOCASA24_SOURCE_ROOT}/"
rsync -a --delete \
  --exclude '*.egg-info' \
  --exclude '__pycache__' \
  "${ROBOSUITE151_SOURCE_ROOT}/" "${LOCAL_ROBOSUITE151_SOURCE_ROOT}/"
ln -sfn "${ROBOCASA24_ASSETS_ROOT}" "${LOCAL_ROBOCASA24_SOURCE_ROOT}/robocasa/models/assets"

source "${CONDA_SH}"
if [[ ! -d "${ROBOCASA_ENV}/conda-meta" ]]; then
  conda create -y --prefix "${ROBOCASA_ENV}" python=3.10 pip
fi
conda activate "${ROBOCASA_ENV}"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-deps -e "${LOCAL_ROBOSUITE151_SOURCE_ROOT}" -e "${LOCAL_ROBOCASA24_SOURCE_ROOT}"
python -m pip install \
  numpy==1.23.3 \
  numba==0.56.4 \
  scipy==1.13.1 \
  mujoco==3.2.6 \
  mink==0.0.13 \
  pygame \
  Pillow \
  opencv-python \
  pyyaml \
  pynput \
  tqdm \
  termcolor \
  imageio \
  imageio-ffmpeg \
  h5py \
  lxml \
  hidapi \
  robosuite_models \
  dm-tree \
  msgpack \
  tree \
  websockets \
  typing_extensions

export NUMBA_DISABLE_JIT=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONPATH="${REPO_ROOT}/packages/openpi-client/src:${REPO_ROOT}:${LOCAL_ROBOCASA24_SOURCE_ROOT}:${LOCAL_ROBOSUITE151_SOURCE_ROOT}:${PYTHONPATH:-}"

python -c "import robocasa, robosuite, mujoco, imageio; print('imports_ok', robosuite.__version__)"
python -c "from robocasa.utils.dataset_registry import SINGLE_STAGE_TASK_DATASETS; tasks=('PnPCounterToCab','PnPCabToCounter','PnPCounterToSink','PnPSinkToCounter','PnPCounterToMicrowave','PnPMicrowaveToCounter','PnPCounterToStove','PnPStoveToCounter','OpenSingleDoor','CloseSingleDoor','OpenDoubleDoor','CloseDoubleDoor','OpenDrawer','CloseDrawer','TurnOnSinkFaucet','TurnOffSinkFaucet','TurnSinkSpout','TurnOnStove','TurnOffStove','CoffeeSetupMug','CoffeeServeMug','CoffeePressButton','TurnOnMicrowave','TurnOffMicrowave'); missing=[t for t in tasks if t not in SINGLE_STAGE_TASK_DATASETS]; print('missing_tasks', missing); raise SystemExit(1 if missing else 0)"

echo "RoboCasa24 infer env ready: ${ROBOCASA_ENV}"

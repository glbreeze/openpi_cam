#!/bin/bash
#SBATCH --job-name=pi0_libero_infer
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_libero_infer/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_libero_infer/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --account=torch_pr_637_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
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

find_latest_experiment() {
  local ckpt_root="$1"
  find "${ckpt_root}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %f\n' \
    | sort -n \
    | tail -1 \
    | awk '{print $2}'
}

find_latest_step() {
  local exp_root="$1"
  find "${exp_root}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
    | grep -E '^[0-9]+$' \
    | sort -n \
    | tail -1
}

REPO_ROOT=${REPO_ROOT:-$(resolve_repo_root)} || {
  echo "Unable to locate repo root. Set REPO_ROOT or submit from inside the repo." >&2
  exit 1
}
GEO_ROOT=${GEO_ROOT:-$(cd -- "${REPO_ROOT}/.." && pwd)}

CONFIG_NAME=${CONFIG_NAME:-pi0_libero_pytorch_full_finetune}
CKPT_ROOT=${CKPT_ROOT:-${REPO_ROOT}/checkpoints/${CONFIG_NAME}}
CKPT_EXP_NAME=${CKPT_EXP_NAME:-}
CKPT_STEP=${CKPT_STEP:-}
CKPT_DIR=${CKPT_DIR:-}

PORT=${PORT:-8000}
CLIENT_HOST=${CLIENT_HOST:-127.0.0.1}
TASK_SUITE_NAME=${TASK_SUITE_NAME:-libero_10}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-50}
SEED=${SEED:-7}
RESIZE_SIZE=${RESIZE_SIZE:-224}
REPLAN_STEPS=${REPLAN_STEPS:-5}
NUM_STEPS_WAIT=${NUM_STEPS_WAIT:-10}
CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-16}

LIBERO_ROOT=${LIBERO_ROOT:-/scratch/yp2841/LIBERO-plus}

LOG_ROOT=${LOG_ROOT:-${REPO_ROOT}/log/pi0_libero_infer}
VIDEO_OUT_ROOT=${VIDEO_OUT_ROOT:-${LOG_ROOT}/videos/${SLURM_JOB_ID:-manual}}
CACHE_ROOT=${CACHE_ROOT:-${GEO_ROOT}/.cache/huggingface}
OPENPI_CACHE_ROOT=${OPENPI_CACHE_ROOT:-${GEO_ROOT}/.cache/openpi}

JOB_TMP_BASE="${SLURM_TMPDIR:-/tmp/$USER}"
JOB_TMP_ROOT="${JOB_TMP_BASE}/openpi_cam_infer_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOB_TMP_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export DATASETS_TMPDIR="${JOB_TMP_ROOT}/datasets_tmp"
export HOME="${JOB_TMP_ROOT}/home"
export XDG_CACHE_HOME="${JOB_TMP_ROOT}/xdg_cache"
export XDG_CONFIG_HOME="${JOB_TMP_ROOT}/xdg_config"
export XDG_STATE_HOME="${JOB_TMP_ROOT}/xdg_state"

mkdir -p \
  "${TMPDIR}" \
  "${DATASETS_TMPDIR}" \
  "${HOME}" \
  "${XDG_CACHE_HOME}" \
  "${XDG_CONFIG_HOME}" \
  "${XDG_STATE_HOME}" \
  "${LOG_ROOT}" \
  "${VIDEO_OUT_ROOT}"
chmod 700 "${TMPDIR}" "${DATASETS_TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" || true

cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/env/activate_env.sh"

export HF_HOME="${CACHE_ROOT}"
export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
export HF_HUB_CACHE="${CACHE_ROOT}/hub"
export OPENPI_CACHE_DIR="${OPENPI_CACHE_ROOT}"
export OPENPI_DATA_HOME="${OPENPI_CACHE_ROOT}"
unset TRANSFORMERS_CACHE

mkdir -p \
  "${HF_HOME}" \
  "${HF_DATASETS_CACHE}" \
  "${HF_HUB_CACHE}" \
  "${OPENPI_CACHE_DIR}"

if [[ ! -d "${LIBERO_ROOT}" || -z "$(find "${LIBERO_ROOT}" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)" ]]; then
  echo "Missing local LIBERO checkout at ${LIBERO_ROOT}" >&2
  exit 1
fi

if [[ -z "${CKPT_DIR}" ]]; then
  if [[ ! -d "${CKPT_ROOT}" ]]; then
    echo "Missing checkpoint root at ${CKPT_ROOT}" >&2
    exit 1
  fi

  if [[ -z "${CKPT_EXP_NAME}" ]]; then
    CKPT_EXP_NAME=$(find_latest_experiment "${CKPT_ROOT}")
  fi

  if [[ -z "${CKPT_EXP_NAME}" ]]; then
    echo "Unable to find an experiment directory under ${CKPT_ROOT}" >&2
    exit 1
  fi

  EXP_ROOT="${CKPT_ROOT}/${CKPT_EXP_NAME}"
  if [[ ! -d "${EXP_ROOT}" ]]; then
    echo "Missing experiment directory at ${EXP_ROOT}" >&2
    exit 1
  fi

  if [[ -z "${CKPT_STEP}" ]]; then
    CKPT_STEP=$(find_latest_step "${EXP_ROOT}")
  fi

  if [[ -z "${CKPT_STEP}" ]]; then
    echo "Unable to find a numeric checkpoint step under ${EXP_ROOT}" >&2
    exit 1
  fi

  CKPT_DIR="${EXP_ROOT}/${CKPT_STEP}"
fi

if [[ ! -f "${CKPT_DIR}/model.safetensors" ]]; then
  echo "Missing model.safetensors at ${CKPT_DIR}" >&2
  exit 1
fi

if [[ ! -d "${CKPT_DIR}/assets" ]]; then
  echo "Missing assets directory at ${CKPT_DIR}/assets" >&2
  exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${CPUS_PER_TASK}}"
export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export PYTHONPATH="${PYTHONPATH:-}"

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${TMPDIR}/libero_config}"
mkdir -p "${LIBERO_CONFIG_PATH}"
cat >"${LIBERO_CONFIG_PATH}/config.yaml" <<CFG
benchmark_root: ${LIBERO_ROOT}/libero/libero
bddl_files: ${LIBERO_ROOT}/libero/libero/bddl_files
init_states: ${LIBERO_ROOT}/libero/libero/init_files
datasets: ${LIBERO_ROOT}/libero/datasets
assets: ${LIBERO_ROOT}/libero/libero/assets
CFG

echo "===== ENV CHECK ====="
echo "repo root: ${REPO_ROOT}"
echo "config: ${CONFIG_NAME}"
echo "checkpoint root: ${CKPT_ROOT}"
echo "checkpoint exp: ${CKPT_EXP_NAME:-direct}"
echo "checkpoint step: ${CKPT_STEP:-direct}"
echo "checkpoint dir: ${CKPT_DIR}"
echo "task suite: ${TASK_SUITE_NAME}"
echo "trials per task: ${NUM_TRIALS_PER_TASK}"
echo "video out root: ${VIDEO_OUT_ROOT}"
echo "libero root: ${LIBERO_ROOT}"
which python
python -V
hostname
nvidia-smi -L || true

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Stopping policy server (pid=${SERVER_PID})..."
    kill "${SERVER_PID}" || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting policy server..."
python scripts/serve_policy.py \
  --env LIBERO \
  --port "${PORT}" \
  policy:checkpoint \
  --policy.config "${CONFIG_NAME}" \
  --policy.dir "${CKPT_DIR}" &
SERVER_PID=$!

echo "Waiting for policy server on ${CLIENT_HOST}:${PORT}..."
server_ready=0
for _ in $(seq 1 300); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "ERROR: policy server exited before becoming ready." >&2
    exit 1
  fi

  if python - "${CLIENT_HOST}" "${PORT}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

try:
    with socket.create_connection((host, port), timeout=2):
        pass
except OSError:
    raise SystemExit(1)
PY
  then
    server_ready=1
    break
  fi

  sleep 2
done

if [[ "${server_ready}" -ne 1 ]]; then
  echo "Policy server did not become ready within 600 seconds." >&2
  exit 1
fi

export PYTHONPATH="${LIBERO_ROOT}:${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "Client env:"
which python
python -V

python examples/libero/main.py \
  --args.host "${CLIENT_HOST}" \
  --args.port "${PORT}" \
  --args.task-suite-name "${TASK_SUITE_NAME}" \
  --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
  --args.video-out-path "${VIDEO_OUT_ROOT}" \
  --args.seed "${SEED}" \
  --args.resize-size "${RESIZE_SIZE}" \
  --args.replan-steps "${REPLAN_STEPS}" \
  --args.num-steps-wait "${NUM_STEPS_WAIT}"

echo "LIBERO inference complete."

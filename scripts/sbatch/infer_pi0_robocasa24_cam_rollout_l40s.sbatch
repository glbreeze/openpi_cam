#!/bin/bash
#SBATCH --job-name=pi0_rc24_infer
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/robocasa24_infer/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/robocasa24_infer/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --account=torch_pr_676_general
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

latest_checkpoint_step() {
  local root="$1"
  find "${root}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
    | grep -E '^[0-9]+$' \
    | sort -n \
    | tail -n 1
}

REPO_ROOT=${REPO_ROOT:-$(resolve_repo_root)} || {
  echo "Unable to locate repo root. Set REPO_ROOT or submit from inside the repo." >&2
  exit 1
}
GEO_ROOT=${GEO_ROOT:-$(cd -- "${REPO_ROOT}/.." && pwd)}
mkdir -p "${REPO_ROOT}/log/robocasa24_infer/server" "${REPO_ROOT}/log/robocasa24_infer/videos"

cd "${REPO_ROOT}"

JOB_TMP_BASE="${SLURM_TMPDIR:-/scratch/${USER}/tmp}"
JOB_TMP_ROOT="${JOB_TMP_BASE}/robocasa24_infer_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOB_TMP_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export ROBOCASA_HOME="${ROBOCASA_HOME:-/scratch/${USER}/.cache/robocasa24_home}"
export HOME="${ROBOCASA_HOME}"
export XDG_CACHE_HOME="${GEO_ROOT}/.cache"
export XDG_CONFIG_HOME="${JOB_TMP_ROOT}/xdg_config"
export XDG_STATE_HOME="${JOB_TMP_ROOT}/xdg_state"
export MPLCONFIGDIR="${JOB_TMP_ROOT}/mplconfig"
mkdir -p "${TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" "${MPLCONFIGDIR}"
chmod 700 "${TMPDIR}" "${HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" "${MPLCONFIGDIR}" || true

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${GEO_ROOT}}"
export HF_HOME="${HF_HOME:-${GEO_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export OPENPI_CACHE_DIR="${OPENPI_CACHE_DIR:-${GEO_ROOT}/.cache/openpi}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${OPENPI_CACHE_DIR}}"
export OPENPI_GEO_ROOT="${OPENPI_GEO_ROOT:-${GEO_ROOT}}"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export OPENPI_DISABLE_TORCH_COMPILE="${OPENPI_DISABLE_TORCH_COMPILE:-1}"
export PYTHONNOUSERSITE=1
export ROBOCASA_TMP_XML_DIR="${ROBOCASA_TMP_XML_DIR:-${JOB_TMP_ROOT}/robocasa_mjcf_xml}"
unset TRANSFORMERS_CACHE

CONFIG_NAME=${CONFIG_NAME:-pi0_robocasa24_all24_cam_prope_ray_view_distill_fullres_stage2}
EXP_NAME=${EXP_NAME:-stage2_all24_libero4suite_pi3x224}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/${CONFIG_NAME}/${EXP_NAME}}
CHECKPOINT_STEP=${CHECKPOINT_STEP:-latest}
if [[ -n "${CHECKPOINT_DIR:-}" ]]; then
  ckpt_dir="${CHECKPOINT_DIR}"
else
  if [[ "${CHECKPOINT_STEP}" == "latest" ]]; then
    CHECKPOINT_STEP=$(latest_checkpoint_step "${CHECKPOINT_ROOT}")
  fi
  ckpt_dir="${CHECKPOINT_ROOT}/${CHECKPOINT_STEP}"
fi

ROBOCASA_ENV=${ROBOCASA_ENV:-/scratch/${USER}/.venvs/robocasa24}
if [[ ! -x "${ROBOCASA_ENV}/bin/python" && -x /scratch/yz11445/.venvs/robocasa24/bin/python ]]; then
  ROBOCASA_ENV=/scratch/yz11445/.venvs/robocasa24
fi
ROBOCASA_PYTHON="${ROBOCASA_ENV}/bin/python"
ROBOCASA_CLIENT_PYDEPS=${ROBOCASA_CLIENT_PYDEPS:-/scratch/${USER}/tmp/robocasa24_pydeps}
ROBOCASA24_SOURCE_ROOT=${ROBOCASA24_SOURCE_ROOT:-/scratch/${USER}/src/robocasa24_infer/robocasa_v02}
ROBOSUITE151_SOURCE_ROOT=${ROBOSUITE151_SOURCE_ROOT:-/scratch/${USER}/src/robocasa24_infer/robosuite_v151}
if [[ ! -d "${ROBOCASA24_SOURCE_ROOT}" ]]; then
  ROBOCASA24_SOURCE_ROOT=/scratch/yz11445/openpi_cam/third_party/robocasa_v02
fi
if [[ ! -d "${ROBOSUITE151_SOURCE_ROOT}" ]]; then
  ROBOSUITE151_SOURCE_ROOT=/scratch/yz11445/openpi_cam/third_party/robosuite_v151
fi

EVAL_BACKEND=${EVAL_BACKEND:-robocasa-v0}
EVAL_TASK=${EVAL_TASK:-OpenDrawer}
PROMPT=${PROMPT:-auto}
NUM_EPISODES=${NUM_EPISODES:-10}
MAX_STEPS=${MAX_STEPS:-500}
CHUNK_SIZE=${CHUNK_SIZE:-50}
SEED=${SEED:-0}
RENDER_IMAGE_SIZE=${RENDER_IMAGE_SIZE:-128}
VIDEO_EVERY=${VIDEO_EVERY:-0}
PORT=${PORT:-$((18400 + (${SLURM_JOB_ID:-0} % 20000)))}
SERVER_START_TIMEOUT=${SERVER_START_TIMEOUT:-600}
SERVER_BATCH_MAX_SIZE=${SERVER_BATCH_MAX_SIZE:-1}
SERVER_BATCH_TIMEOUT_MS=${SERVER_BATCH_TIMEOUT_MS:-5}
RESULT_JSON=${RESULT_JSON:-${REPO_ROOT}/log/robocasa24_infer/${EXP_NAME}_${CHECKPOINT_STEP}_${EVAL_TASK}_${SLURM_JOB_ID:-manual}.json}
VIDEO_DIR=${VIDEO_DIR:-${REPO_ROOT}/log/robocasa24_infer/videos/${EXP_NAME}_${CHECKPOINT_STEP}_${EVAL_TASK}_${SLURM_JOB_ID:-manual}}
SERVER_LOG=${SERVER_LOG:-${REPO_ROOT}/log/robocasa24_infer/server/serve_${EXP_NAME}_${CHECKPOINT_STEP}_${SLURM_JOB_ID:-manual}.log}
DEBUG_JSON=${DEBUG_JSON:-}
DEMO_DATASET=${DEMO_DATASET:-}
DEMO_KEYS=${DEMO_KEYS:-}
DEMO_POLICY_IMAGES=${DEMO_POLICY_IMAGES:-0}
DEMO_CAMERA_CACHE=${DEMO_CAMERA_CACHE:-}
SEND_OPENCV_EXTRINSICS=${SEND_OPENCV_EXTRINSICS:-1}
RESULT_JSON=${RESULT_JSON//%j/${SLURM_JOB_ID:-manual}}
RESULT_JSON=${RESULT_JSON//%u/${USER:-user}}
VIDEO_DIR=${VIDEO_DIR//%j/${SLURM_JOB_ID:-manual}}
VIDEO_DIR=${VIDEO_DIR//%u/${USER:-user}}
SERVER_LOG=${SERVER_LOG//%j/${SLURM_JOB_ID:-manual}}
SERVER_LOG=${SERVER_LOG//%u/${USER:-user}}
DEBUG_JSON=${DEBUG_JSON//%j/${SLURM_JOB_ID:-manual}}
DEBUG_JSON=${DEBUG_JSON//%u/${USER:-user}}
DEMO_CAMERA_CACHE=${DEMO_CAMERA_CACHE//%j/${SLURM_JOB_ID:-manual}}
DEMO_CAMERA_CACHE=${DEMO_CAMERA_CACHE//%u/${USER:-user}}

[[ -f "${ckpt_dir}/model.safetensors" ]] || { echo "Missing checkpoint: ${ckpt_dir}/model.safetensors" >&2; exit 1; }
[[ -x "${ROBOCASA_PYTHON}" ]] || { echo "Missing RoboCasa env python: ${ROBOCASA_PYTHON}" >&2; exit 1; }

echo "===== ROBOCASA24 CAM-AWARE INFER ====="
echo "repo root      : ${REPO_ROOT}"
echo "config         : ${CONFIG_NAME}"
echo "checkpoint     : ${ckpt_dir}"
echo "robocasa env   : ${ROBOCASA_ENV}"
echo "client pydeps  : ${ROBOCASA_CLIENT_PYDEPS}"
echo "robocasa src   : ${ROBOCASA24_SOURCE_ROOT}"
echo "robosuite src  : ${ROBOSUITE151_SOURCE_ROOT}"
echo "backend        : ${EVAL_BACKEND}"
echo "task           : ${EVAL_TASK}"
echo "prompt         : ${PROMPT}"
echo "episodes       : ${NUM_EPISODES}"
echo "render size    : ${RENDER_IMAGE_SIZE}"
echo "demo dataset   : ${DEMO_DATASET:-none}"
echo "demo keys      : ${DEMO_KEYS:-none}"
echo "demo images    : ${DEMO_POLICY_IMAGES}"
echo "demo cam cache : ${DEMO_CAMERA_CACHE:-none}"
echo "send opencv T  : ${SEND_OPENCV_EXTRINSICS:-0}"
echo "result json    : ${RESULT_JSON}"
hostname
python -V || true
nvidia-smi -L || true
date

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

source "${REPO_ROOT}/scripts/env/activate_env.sh"
PYTHONUNBUFFERED=1 uv run python -u scripts/serve_policy.py \
  --port "${PORT}" \
  --batch_max_size "${SERVER_BATCH_MAX_SIZE}" \
  --batch_timeout_ms "${SERVER_BATCH_TIMEOUT_MS}" \
  policy:checkpoint \
  --policy.config "${CONFIG_NAME}" \
  --policy.dir "${ckpt_dir}" > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

deadline=$(( SECONDS + SERVER_START_TIMEOUT ))
while ! python -c "import socket; s=socket.socket(); s.settimeout(0.5); s.connect(('127.0.0.1', ${PORT}))" 2>/dev/null; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Server exited before port ${PORT} was ready. Last log:" >&2
    tail -n 80 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  if (( SECONDS > deadline )); then
    echo "Timed out after ${SERVER_START_TIMEOUT}s waiting for port ${PORT}" >&2
    tail -n 80 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  sleep 1
done
echo "Policy server up on port ${PORT}."

export PATH="${ROBOCASA_ENV}/bin:${PATH}"
client_pydeps_prefix=""
if [[ -d "${ROBOCASA_CLIENT_PYDEPS}" ]]; then
  client_pydeps_prefix="${ROBOCASA_CLIENT_PYDEPS}:"
fi
export PYTHONPATH="${client_pydeps_prefix}${REPO_ROOT}/packages/openpi-client/src:${REPO_ROOT}:${ROBOCASA24_SOURCE_ROOT}:${ROBOSUITE151_SOURCE_ROOT}:${PYTHONPATH:-}"

debug_args=()
if [[ -n "${DEBUG_JSON}" ]]; then
  debug_args+=(--debug-out "${DEBUG_JSON}")
fi
if [[ -n "${FORCE_GRIPPER_AFTER:-}" ]]; then
  debug_args+=(--force-gripper-after "${FORCE_GRIPPER_AFTER}")
fi
if [[ -n "${FORCE_GRIPPER_VALUE:-}" ]]; then
  debug_args+=(--force-gripper-value "${FORCE_GRIPPER_VALUE}")
fi
if [[ "${SEND_OPENCV_EXTRINSICS:-0}" == "1" ]]; then
  debug_args+=(--send-opencv-extrinsics)
fi
if [[ -n "${DEMO_DATASET}" ]]; then
  debug_args+=(--demo-dataset "${DEMO_DATASET}")
fi
if [[ -n "${DEMO_KEYS}" ]]; then
  debug_args+=(--demo-keys "${DEMO_KEYS}")
fi
if [[ "${DEMO_POLICY_IMAGES}" == "1" ]]; then
  debug_args+=(--demo-policy-images)
fi
if [[ -n "${DEMO_CAMERA_CACHE}" ]]; then
  debug_args+=(--demo-camera-cache "${DEMO_CAMERA_CACHE}")
fi

"${ROBOCASA_PYTHON}" scripts/debug/eval_robocasa365_remote.py \
  --backend "${EVAL_BACKEND}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --env "${EVAL_TASK}" \
  --prompt "${PROMPT}" \
  --episodes "${NUM_EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --chunk-size "${CHUNK_SIZE}" \
  --seed "${SEED}" \
  --image-size 224 \
  --render-image-size "${RENDER_IMAGE_SIZE}" \
  --require-camera-matrices \
  --video-dir "${VIDEO_DIR}" \
  --video-every "${VIDEO_EVERY}" \
  --out "${RESULT_JSON}" \
  "${debug_args[@]}"

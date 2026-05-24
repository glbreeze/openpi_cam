#!/bin/bash
#SBATCH --job-name=robocasa_big_infer
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/robocasa365_big_infer/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/robocasa365_big_infer/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
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

REPO_ROOT=${REPO_ROOT:-$(resolve_repo_root)} || {
  echo "Unable to locate repo root. Set REPO_ROOT or submit from inside the repo." >&2
  exit 1
}
GEO_ROOT=${GEO_ROOT:-$(cd -- "${REPO_ROOT}/.." && pwd)}

JOB_TMP_BASE="${SLURM_TMPDIR:-/scratch/${USER}/tmp}"
JOB_TMP_ROOT="${JOB_TMP_BASE}/robocasa365_big_infer_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOB_TMP_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export ROBOCASA_HOME="${ROBOCASA_HOME:-/scratch/${USER}/.cache/robocasa365_home}"
export HOME="${ROBOCASA_HOME}"
export XDG_CACHE_HOME="${GEO_ROOT}/.cache"
export XDG_CONFIG_HOME="${JOB_TMP_ROOT}/xdg_config"
export XDG_STATE_HOME="${JOB_TMP_ROOT}/xdg_state"
export MPLCONFIGDIR="${JOB_TMP_ROOT}/mplconfig"

mkdir -p \
  "${TMPDIR}" \
  "${HOME}" \
  "${XDG_CACHE_HOME}" \
  "${XDG_CONFIG_HOME}" \
  "${XDG_STATE_HOME}" \
  "${MPLCONFIGDIR}" \
  "${REPO_ROOT}/log/robocasa365_big_infer/server" \
  "${REPO_ROOT}/log/robocasa365_big_infer/videos"
chmod 700 "${TMPDIR}" "${HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" "${MPLCONFIGDIR}" || true

cd "${REPO_ROOT}"

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${GEO_ROOT}}"
export HF_HOME="${HF_HOME:-${GEO_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export OPENPI_CACHE_DIR="${OPENPI_CACHE_DIR:-${GEO_ROOT}/.cache/openpi}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${OPENPI_CACHE_DIR}}"
export OPENPI_GEO_ROOT="${OPENPI_GEO_ROOT:-${GEO_ROOT}}"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export TOKENIZERS_PARALLELISM=false
unset TRANSFORMERS_CACHE

CONFIG_NAME=${CONFIG_NAME:-pi0_robocasa365_baseline}
EXP_NAME=${EXP_NAME:-robocasa365_opendrawer_baseline_8908268}
CHECKPOINT_STEP=${CHECKPOINT_STEP:-30000}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${REPO_ROOT}/checkpoints/${CONFIG_NAME}/${EXP_NAME}/${CHECKPOINT_STEP}}
PORT=${PORT:-$((18000 + (${SLURM_JOB_ID:-0} % 20000)))}

ROBOCASA_ENV=${ROBOCASA_ENV:-/scratch/${USER}/.venvs/robocasa365}
CONDA_SH=${CONDA_SH:-/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh}
EVAL_ENV=${EVAL_ENV:-robocasa/OpenDrawer}
EVAL_SPLIT=${EVAL_SPLIT:-target}
PROMPT=${PROMPT:-auto}
NUM_EPISODES=${NUM_EPISODES:-50}
MAX_STEPS=${MAX_STEPS:-500}
CHUNK_SIZE=${CHUNK_SIZE:-50}
SEED=${SEED:-0}
VIDEO_EVERY=${VIDEO_EVERY:-0}
RESULT_JSON=${RESULT_JSON:-${REPO_ROOT}/log/robocasa365_big_infer/result_${SLURM_JOB_ID:-manual}.json}
VIDEO_DIR=${VIDEO_DIR:-${REPO_ROOT}/log/robocasa365_big_infer/videos/${SLURM_JOB_ID:-manual}}
SERVER_LOG=${SERVER_LOG:-${REPO_ROOT}/log/robocasa365_big_infer/server/serve_${SLURM_JOB_ID:-manual}.log}
DEBUG_JSON=${DEBUG_JSON:-}
FORCE_GRIPPER_AFTER=${FORCE_GRIPPER_AFTER:-}
FORCE_GRIPPER_VALUE=${FORCE_GRIPPER_VALUE:-1.0}

if [[ ! -f "${CHECKPOINT_DIR}/model.safetensors" ]]; then
  echo "Missing checkpoint model at ${CHECKPOINT_DIR}/model.safetensors" >&2
  exit 1
fi
if [[ ! -f "${ROBOCASA_ENV}/bin/activate" && ! -d "${ROBOCASA_ENV}/conda-meta" ]]; then
  echo "Missing RoboCasa env at ${ROBOCASA_ENV}. Run scripts/sbatch/install_robocasa365_env_cpu.sbatch first." >&2
  exit 1
fi

echo "===== ROBOCASA365 BIG INFER ====="
echo "repo root      : ${REPO_ROOT}"
echo "config         : ${CONFIG_NAME}"
echo "checkpoint dir : ${CHECKPOINT_DIR}"
echo "eval env       : ${EVAL_ENV}"
echo "episodes       : ${NUM_EPISODES}"
echo "max steps      : ${MAX_STEPS}"
echo "chunk size     : ${CHUNK_SIZE}"
echo "result json    : ${RESULT_JSON}"
hostname
python -V || true
nvidia-smi
date

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

source "${REPO_ROOT}/scripts/env/activate_env.sh"
uv run python scripts/serve_policy.py \
  --port "${PORT}" \
  --default-prompt "${PROMPT}" \
  policy:checkpoint \
  --policy.config "${CONFIG_NAME}" \
  --policy.dir "${CHECKPOINT_DIR}" > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

if [[ -f "${ROBOCASA_ENV}/bin/activate" ]]; then
  source "${ROBOCASA_ENV}/bin/activate"
else
  if [[ ! -f "${CONDA_SH}" ]]; then
    echo "Missing conda.sh at ${CONDA_SH}" >&2
    exit 1
  fi
  source "${CONDA_SH}"
  conda activate "${ROBOCASA_ENV}"
fi
export PYTHONPATH="${REPO_ROOT}/packages/openpi-client/src:${REPO_ROOT}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

debug_args=()
if [[ -n "${DEBUG_JSON}" ]]; then
  debug_args+=(--debug-out "${DEBUG_JSON}")
fi
if [[ -n "${FORCE_GRIPPER_AFTER}" ]]; then
  debug_args+=(--force-gripper-after "${FORCE_GRIPPER_AFTER}")
  debug_args+=(--force-gripper-value "${FORCE_GRIPPER_VALUE}")
fi

python scripts/debug/eval_robocasa365_remote.py \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --env "${EVAL_ENV}" \
  --split "${EVAL_SPLIT}" \
  --prompt "${PROMPT}" \
  --episodes "${NUM_EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --chunk-size "${CHUNK_SIZE}" \
  --seed "${SEED}" \
  --video-dir "${VIDEO_DIR}" \
  --video-every "${VIDEO_EVERY}" \
  --out "${RESULT_JSON}" \
  "${debug_args[@]}"

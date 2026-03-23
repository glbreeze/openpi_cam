#!/bin/bash
#SBATCH --job-name=eval_libero_object_cam
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/libero_object_eval/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/libero_object_eval/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=48:00:00
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

if [[ -z "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(resolve_repo_root || true)"
fi
if [[ -z "${REPO_ROOT}" && -d "/scratch/yp2841/geometry-vla/openpi_cam" ]]; then
  REPO_ROOT="/scratch/yp2841/geometry-vla/openpi_cam"
fi
if [[ -z "${REPO_ROOT}" ]]; then
  echo "Unable to locate repo root. Set REPO_ROOT or submit from inside the repo." >&2
  exit 1
fi

GEO_ROOT=${GEO_ROOT:-$(cd -- "${REPO_ROOT}/.." && pwd)}
CHECKPOINT_ROOT="${REPO_ROOT}/checkpoints/pi0_libero_cam_pytorch_full_finetune/pi0_libero_object_cam_4711127"
CHECKPOINT_ASSET_ID=${CHECKPOINT_ASSET_ID:-glbreeze/libero_object_cam}
SERVE_ASSET_ID=${SERVE_ASSET_ID:-glbreeze/libero_cam}

SUITE_NAME=${SUITE_NAME:-libero_object}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-50}
REPLAN_STEPS=${REPLAN_STEPS:-5}
RESIZE_SIZE=${RESIZE_SIZE:-224}
SEED=${SEED:-7}
SERVER_START_TIMEOUT=${SERVER_START_TIMEOUT:-600}

PORT_BASE=${PORT_BASE:-19000}
CHECKPOINT_STEP=${CHECKPOINT_STEP:-5000}
PORT=${PORT:-${PORT_BASE}}

JOB_TMP_BASE="${SLURM_TMPDIR:-/tmp/$USER}"
JOB_TMP_ROOT="${JOB_TMP_BASE}/openpi_cam_eval_${SLURM_JOB_ID:-$$}"
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

export HF_HOME="${GEO_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export OPENPI_CACHE_DIR="${GEO_ROOT}/.cache/openpi"
export OPENPI_DATA_HOME="${OPENPI_CACHE_DIR}"
export PYTHONPATH="${REPO_ROOT}/third_party/libero:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH="${JOB_TMP_ROOT}/libero_config"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
unset TRANSFORMERS_CACHE

mkdir -p "${REPO_ROOT}/log/libero_object_eval"
mkdir -p "${LIBERO_CONFIG_PATH}"
SERVER_LOG_ROOT="${REPO_ROOT}/log/libero_object_eval/server_logs/cam"
mkdir -p "${SERVER_LOG_ROOT}"

cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: ${REPO_ROOT}/third_party/libero/libero/libero
bddl_files: ${REPO_ROOT}/third_party/libero/libero/libero/bddl_files
init_states: ${REPO_ROOT}/third_party/libero/libero/libero/init_files
datasets: ${GEO_ROOT}/libero_cam_rlds
assets: ${REPO_ROOT}/third_party/libero/libero/libero/assets
EOF

wait_for_server() {
  local port=$1
  local server_pid=$2
  local server_log=$3
  local retries="${SERVER_START_TIMEOUT}"
  while [[ $retries -gt 0 ]]; do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "Server process ${server_pid} exited before port ${port} became ready"
      echo "Last server log lines from ${server_log}:"
      tail -n 80 "${server_log}" || true
      return 1
    fi
    if python - "$port" <<'PY'
import socket,sys
port=int(sys.argv[1])
sock=socket.socket()
sock.settimeout(0.5)
try:
    sock.connect(("127.0.0.1", port))
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
    then
      return 0
    fi
    sleep 1
    retries=$((retries - 1))
  done
  echo "Timed out after ${SERVER_START_TIMEOUT}s waiting for port ${port}"
  echo "Last server log lines from ${server_log}:"
  tail -n 80 "${server_log}" || true
  return 1
}

ckpt_dir="${CHECKPOINT_ROOT}/${CHECKPOINT_STEP}"
if [[ ! -d "${ckpt_dir}" ]]; then
  echo "Missing checkpoint: ${ckpt_dir}" >&2
  exit 1
fi

video_out="${REPO_ROOT}/data/libero/videos/libero_object/cam/${CHECKPOINT_STEP}"
server_log="${SERVER_LOG_ROOT}/serve_${CHECKPOINT_STEP}.log"
mkdir -p "${video_out}"
mkdir -p "${ckpt_dir}/assets/$(dirname "${SERVE_ASSET_ID}")"
ln -sfn "$(basename "${CHECKPOINT_ASSET_ID}")" "${ckpt_dir}/assets/${SERVE_ASSET_ID}"

echo "Evaluating checkpoint step ${CHECKPOINT_STEP}"
echo "Starting policy server for checkpoint ${ckpt_dir} on port ${PORT}"
echo "Server log: ${server_log}"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python -u scripts/serve_policy.py \
  --port "${PORT}" \
  --batch_max_size 1 \
  --batch_timeout_ms 0.0 \
  policy:checkpoint \
  --policy.config pi0_libero_cam_pytorch_full_finetune \
  --policy.dir "${ckpt_dir}" > "${server_log}" 2>&1 &
server_pid=$!

cleanup() {
  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT

if ! wait_for_server "${PORT}" "${server_pid}" "${server_log}"; then
  echo "Server failed to start on port ${PORT}, killing PID ${server_pid}"
  exit 1
fi

echo "Running LIBERO eval for checkpoint ${CHECKPOINT_STEP}"
python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port "${PORT}" \
  --args.task-suite-name "${SUITE_NAME}" \
  --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
  --args.replan-steps "${REPLAN_STEPS}" \
  --args.resize-size "${RESIZE_SIZE}" \
  --args.seed "${SEED}" \
  --args.video-out-path "${video_out}"

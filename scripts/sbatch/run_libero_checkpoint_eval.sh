#!/usr/bin/env bash

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

wait_for_server() {
  local port=$1
  local server_pid=$2
  local server_log=$3
  local retries="${SERVER_START_TIMEOUT}"

  while [[ "${retries}" -gt 0 ]]; do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "Server process ${server_pid} exited before port ${port} became ready"
      echo "Last server log lines from ${server_log}:"
      tail -n 80 "${server_log}" || true
      return 1
    fi

    if python - "${port}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket()
sock.settimeout(0.5)
try:
    sock.connect(("127.0.0.1", port))
    raise SystemExit(0)
except Exception:
    raise SystemExit(1)
finally:
    sock.close()
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

get_task_count() {
  local suite_name="$1"
  "${LIBERO_PYTHON}" - "${suite_name}" <<'PY'
import contextlib
import io
import sys

from libero.libero import benchmark

suite_name = sys.argv[1]
with contextlib.redirect_stdout(io.StringIO()):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_count = benchmark_dict[suite_name]().n_tasks
print(task_count)
PY
}

resolve_latest_step() {
  local ckpt_root=$1
  find "${ckpt_root}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
    | awk '/^[0-9]+$/ { print $1 }' \
    | sort -n \
    | tail -n 1
}

resolve_libero_datasets_root() {
  local candidate
  local -a candidates=()

  if [[ -n "${LIBERO_DATASETS_ROOT:-}" ]]; then
    candidates+=("${LIBERO_DATASETS_ROOT}")
  fi

  candidates+=(
    "${GEO_ROOT}/libero_cam_rlds"
    "${GEO_ROOT}/tmp/libero_datasets/datasets"
    "${REPO_ROOT}/third_party/libero/libero/datasets"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -d "${candidate}" ]] && find "${candidate}" -mindepth 1 -print -quit | grep -q .; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  printf '%s\n' "${GEO_ROOT}/libero_cam_rlds"
}

if [[ -z "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(resolve_repo_root || true)"
fi
if [[ -z "${REPO_ROOT}" && -d "/scratch/${USER}/openpi_cam" ]]; then
  REPO_ROOT="/scratch/${USER}/openpi_cam"
fi
if [[ -z "${REPO_ROOT}" ]]; then
  echo "Unable to locate repo root. Set REPO_ROOT or submit from inside the repo." >&2
  exit 1
fi

GEO_ROOT=${GEO_ROOT:-$(cd -- "${REPO_ROOT}/.." && pwd)}
CONFIG_NAME=${CONFIG_NAME:-pi0_libero_object_pytorch_baseline}
EXP_NAME=${EXP_NAME:-pi0_libero_object_cam_v3_pytorch_baseline_v1}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/pi0_libero_object_pytorch_baseline/${EXP_NAME}}
CHECKPOINT_STEP=${CHECKPOINT_STEP:-latest}

if [[ "${CHECKPOINT_STEP}" == "latest" || -z "${CHECKPOINT_STEP}" ]]; then
  CHECKPOINT_STEP=$(resolve_latest_step "${CHECKPOINT_ROOT}")
fi
if [[ -z "${CHECKPOINT_STEP}" ]]; then
  echo "Unable to resolve checkpoint step under ${CHECKPOINT_ROOT}" >&2
  exit 1
fi

SUITE_NAME=${SUITE_NAME:-libero_object}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-50}
TASK_ID_START=${TASK_ID_START:-0}
TASK_ID_END=${TASK_ID_END:--1}
REPLAN_STEPS=${REPLAN_STEPS:-5}
RESIZE_SIZE=${RESIZE_SIZE:-224}
NUM_STEPS_WAIT=${NUM_STEPS_WAIT:-10}
SEED=${SEED:-7}
SERVER_PORT=${SERVER_PORT:-18000}
SERVER_BATCH_MAX_SIZE=${SERVER_BATCH_MAX_SIZE:-1}
SERVER_BATCH_TIMEOUT_MS=${SERVER_BATCH_TIMEOUT_MS:-0}
NUM_PARALLEL_CLIENTS=${NUM_PARALLEL_CLIENTS:-1}
# Model startup can be slow on busy nodes, especially with larger checkpoints.
# Give the server more time before we declare the eval job failed.
SERVER_START_TIMEOUT=${SERVER_START_TIMEOUT:-1200}

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
export MPLCONFIGDIR="${JOB_TMP_ROOT}/mplconfig"

mkdir -p "${TMPDIR}" "${DATASETS_TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" "${MPLCONFIGDIR}"
chmod 700 "${TMPDIR}" "${DATASETS_TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" "${MPLCONFIGDIR}" || true

cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/env/activate_env.sh"

REPO_PYTHON="${REPO_ROOT}/.venv/bin/python"
LIBERO_PYTHON="${REPO_ROOT}/examples/libero/.venv/bin/python"

if [[ ! -x "${REPO_PYTHON}" ]]; then
  echo "Missing repo python: ${REPO_PYTHON}" >&2
  exit 1
fi
if [[ ! -x "${LIBERO_PYTHON}" ]]; then
  echo "Missing LIBERO eval python: ${LIBERO_PYTHON}" >&2
  exit 1
fi

export HF_HOME="${GEO_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export OPENPI_CACHE_DIR="${GEO_ROOT}/.cache/openpi"
export OPENPI_DATA_HOME="${OPENPI_CACHE_DIR}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/libero:${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export LIBERO_CONFIG_PATH="${JOB_TMP_ROOT}/libero_config"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
unset TRANSFORMERS_CACHE

LIBERO_DATASETS_ROOT=${LIBERO_DATASETS_ROOT:-$(resolve_libero_datasets_root)}

MODEL_NAME=${MODEL_NAME:-${EXP_NAME}}
LOG_ROOT=${LOG_ROOT:-${REPO_ROOT}/log/libero_eval/${EXP_NAME}}
SUMMARY_OUT_PATH=${SUMMARY_OUT_PATH:-${LOG_ROOT}/summary_${CHECKPOINT_STEP}_${SLURM_JOB_ID:-manual}.json}
VIDEO_OUT_PATH=${VIDEO_OUT_PATH:-${LOG_ROOT}/videos/${CHECKPOINT_STEP}/${SLURM_JOB_ID:-manual}}
SERVER_LOG_ROOT="${LOG_ROOT}/server_logs"
SERVER_LOG="${SERVER_LOG_ROOT}/serve_${CHECKPOINT_STEP}_${SLURM_JOB_ID:-manual}.log"

mkdir -p "${LOG_ROOT}" "${SERVER_LOG_ROOT}" "${VIDEO_OUT_PATH}" "${LIBERO_CONFIG_PATH}"

cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: ${REPO_ROOT}/third_party/libero/libero/libero
bddl_files: ${REPO_ROOT}/third_party/libero/libero/libero/bddl_files
init_states: ${REPO_ROOT}/third_party/libero/libero/libero/init_files
datasets: ${LIBERO_DATASETS_ROOT}
assets: ${REPO_ROOT}/third_party/libero/libero/libero/assets
EOF

CKPT_DIR="${CHECKPOINT_ROOT}/${CHECKPOINT_STEP}"
if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "Missing checkpoint directory: ${CKPT_DIR}" >&2
  exit 1
fi

echo "===== LIBERO EVAL ENV CHECK ====="
echo "repo root: ${REPO_ROOT}"
echo "model name: ${MODEL_NAME}"
echo "config: ${CONFIG_NAME}"
echo "exp: ${EXP_NAME}"
echo "checkpoint root: ${CHECKPOINT_ROOT}"
echo "checkpoint step: ${CHECKPOINT_STEP}"
echo "checkpoint dir: ${CKPT_DIR}"
echo "suite: ${SUITE_NAME}"
echo "libero datasets root: ${LIBERO_DATASETS_ROOT}"
echo "num_trials_per_task: ${NUM_TRIALS_PER_TASK}"
echo "task range request: [${TASK_ID_START}, ${TASK_ID_END})"
echo "num_parallel_clients: ${NUM_PARALLEL_CLIENTS}"
echo "server port: ${SERVER_PORT}"
echo "summary path: ${SUMMARY_OUT_PATH}"
echo "video path: ${VIDEO_OUT_PATH}"
echo "server log: ${SERVER_LOG}"
echo "server python: ${REPO_PYTHON}"
"${REPO_PYTHON}" -V
echo "client python: ${LIBERO_PYTHON}"
"${LIBERO_PYTHON}" -V
hostname
nvidia-smi -L || true

echo "Starting policy server for checkpoint ${CKPT_DIR} on port ${SERVER_PORT}"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 "${REPO_PYTHON}" -u scripts/serve_policy.py \
  --port "${SERVER_PORT}" \
  --batch_max_size "${SERVER_BATCH_MAX_SIZE}" \
  --batch_timeout_ms "${SERVER_BATCH_TIMEOUT_MS}" \
  policy:checkpoint \
  --policy.config "${CONFIG_NAME}" \
  --policy.dir "${CKPT_DIR}" > "${SERVER_LOG}" 2>&1 &
server_pid=$!

cleanup() {
  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT

if ! wait_for_server "${SERVER_PORT}" "${server_pid}" "${SERVER_LOG}"; then
  echo "Server failed to start on port ${SERVER_PORT}, killing PID ${server_pid}"
  exit 1
fi

task_count=$(get_task_count "${SUITE_NAME}")
effective_task_start=${TASK_ID_START}
if (( effective_task_start < 0 )); then
  effective_task_start=0
fi
effective_task_end=${TASK_ID_END}
if (( effective_task_end < 0 || effective_task_end > task_count )); then
  effective_task_end=${task_count}
fi
if (( effective_task_start >= effective_task_end )); then
  echo "Invalid effective task range [${effective_task_start}, ${effective_task_end}) for suite ${SUITE_NAME}" >&2
  exit 1
fi

effective_clients=${NUM_PARALLEL_CLIENTS}
task_span=$((effective_task_end - effective_task_start))
if (( effective_clients < 1 )); then
  effective_clients=1
fi
if (( effective_clients > task_span )); then
  effective_clients=${task_span}
fi

echo "task count: ${task_count}"
echo "effective task range: [${effective_task_start}, ${effective_task_end})"
echo "effective parallel clients: ${effective_clients}"

if (( effective_clients == 1 )); then
  "${LIBERO_PYTHON}" examples/libero/main.py \
    --args.host 127.0.0.1 \
    --args.port "${SERVER_PORT}" \
    --args.model-name "${MODEL_NAME}" \
    --args.task-suite-name "${SUITE_NAME}" \
    --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
    --args.task-id-start "${effective_task_start}" \
    --args.task-id-end "${effective_task_end}" \
    --args.replan-steps "${REPLAN_STEPS}" \
    --args.resize-size "${RESIZE_SIZE}" \
    --args.num-steps-wait "${NUM_STEPS_WAIT}" \
    --args.seed "${SEED}" \
    --args.video-out-path "${VIDEO_OUT_PATH}" \
    --args.summary-out-path "${SUMMARY_OUT_PATH}"
else
  SHARD_SUMMARY_ROOT="${LOG_ROOT}/shard_summaries/${CHECKPOINT_STEP}/${SLURM_JOB_ID:-manual}"
  mkdir -p "${SHARD_SUMMARY_ROOT}"

  CLIENT_PIDS=()
  client_rc=0

  base_chunk=$((task_span / effective_clients))
  extra=$((task_span % effective_clients))
  next_task_start=${effective_task_start}

  for ((client_idx = 0; client_idx < effective_clients; client_idx++)); do
    chunk_size=${base_chunk}
    if (( client_idx < extra )); then
      chunk_size=$((chunk_size + 1))
    fi
    shard_task_start=${next_task_start}
    shard_task_end=$((shard_task_start + chunk_size))
    next_task_start=${shard_task_end}

    shard_summary="${SHARD_SUMMARY_ROOT}/shard_${client_idx}.json"
    shard_video_out="${VIDEO_OUT_PATH}/shard_${client_idx}"

    echo "Launching eval shard ${client_idx}: [${shard_task_start}, ${shard_task_end})"
    "${LIBERO_PYTHON}" examples/libero/main.py \
      --args.host 127.0.0.1 \
      --args.port "${SERVER_PORT}" \
      --args.model-name "${MODEL_NAME}" \
      --args.task-suite-name "${SUITE_NAME}" \
      --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
      --args.task-id-start "${shard_task_start}" \
      --args.task-id-end "${shard_task_end}" \
      --args.replan-steps "${REPLAN_STEPS}" \
      --args.resize-size "${RESIZE_SIZE}" \
      --args.num-steps-wait "${NUM_STEPS_WAIT}" \
      --args.seed "${SEED}" \
      --args.video-out-path "${shard_video_out}" \
      --args.summary-out-path "${shard_summary}" &
    CLIENT_PIDS+=("$!")
  done

  for pid in "${CLIENT_PIDS[@]}"; do
    wait "${pid}" || client_rc=$?
  done

  if [[ "${client_rc}" -ne 0 ]]; then
    echo "At least one eval shard failed." >&2
    exit "${client_rc}"
  fi

  "${REPO_PYTHON}" - "${SHARD_SUMMARY_ROOT}" "${SUMMARY_OUT_PATH}" <<'PY'
import json
import pathlib
import sys

shard_dir = pathlib.Path(sys.argv[1])
output_path = pathlib.Path(sys.argv[2])
paths = sorted(shard_dir.glob("shard_*.json"))
if not paths:
    raise SystemExit(f"No shard summaries found in {shard_dir}")

summaries = [json.loads(path.read_text()) for path in paths]
records = []
for summary in summaries:
    records.extend(summary.get("records", []))

total_episodes = sum(int(summary.get("total_episodes", 0)) for summary in summaries)
total_successes = sum(int(summary.get("total_successes", 0)) for summary in summaries)

aggregate = {
    "model_name": summaries[0].get("model_name"),
    "task_suite_name": summaries[0]["task_suite_name"],
    "task_range": [
        min(summary["task_range"][0] for summary in summaries),
        max(summary["task_range"][1] for summary in summaries),
    ],
    "num_trials_per_task": summaries[0]["num_trials_per_task"],
    "total_episodes": total_episodes,
    "total_successes": total_successes,
    "total_success_rate": total_successes / total_episodes if total_episodes else 0.0,
    "records": sorted(records, key=lambda item: item["task_id"]),
    "shard_summaries": [str(path) for path in paths],
}

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(aggregate, indent=2) + "\n")
print(output_path)
PY
fi

echo "Evaluation complete."
echo "Summary written to: ${SUMMARY_OUT_PATH}"

#!/bin/bash
#SBATCH --job-name=libero_obj_multicam_infer
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/libero_object_multicam_eval/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/libero_object_multicam_eval/slurm-%j.err
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
  python - "${suite_name}" <<'PY'
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
MODEL_NAME=${MODEL_NAME:-baseline}
CONFIG_NAME=${CONFIG_NAME:-pi0_libero_pytorch_full_finetune}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/pi0_libero_pytorch_full_finetune/pi0_libero_object_baseline_full}
CHECKPOINT_STEP=${CHECKPOINT_STEP:-30000}
CHECKPOINT_ASSET_ID=${CHECKPOINT_ASSET_ID:-glbreeze/libero_object}
SERVE_ASSET_ID=${SERVE_ASSET_ID:-glbreeze/libero}

SUITE_NAME=${SUITE_NAME:-libero_object}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-50}
TASK_ID_START=${TASK_ID_START:-0}
TASK_ID_END=${TASK_ID_END:--1}
NUM_PARALLEL_CLIENTS=${NUM_PARALLEL_CLIENTS:-4}
REPLAN_STEPS=${REPLAN_STEPS:-5}
RESIZE_SIZE=${RESIZE_SIZE:-224}
NUM_STEPS_WAIT=${NUM_STEPS_WAIT:-10}
SEED=${SEED:-7}

CAMERA_VARIATION_CONFIG=${CAMERA_VARIATION_CONFIG:-${GEO_ROOT}/LIBERO-Camera/configs/camera_variation/libero_object_left_variation.json}
CAMERA_VARIATION_COUNT=${CAMERA_VARIATION_COUNT:-4}
CAMERA_VARIATION_SEED=${CAMERA_VARIATION_SEED:-0}
CAMERA_VARIATION_INCLUDE_ORIGINAL=${CAMERA_VARIATION_INCLUDE_ORIGINAL:-true}
CAMERA_VARIATION_TARGET_CAMERA=${CAMERA_VARIATION_TARGET_CAMERA:-agentview}
SAVE_VIDEOS=${SAVE_VIDEOS:-false}

PORT_BASE=${PORT_BASE:-19200}
PORT=${PORT:-${PORT_BASE}}
SERVER_START_TIMEOUT=${SERVER_START_TIMEOUT:-600}
SERVER_BATCH_TIMEOUT_MS=${SERVER_BATCH_TIMEOUT_MS:-5}
SERVER_BATCH_MAX_SIZE=${SERVER_BATCH_MAX_SIZE:-${NUM_PARALLEL_CLIENTS}}

JOB_TMP_BASE="${SLURM_TMPDIR:-/tmp/$USER}"
JOB_TMP_ROOT="${JOB_TMP_BASE}/openpi_cam_multicam_eval_${SLURM_JOB_ID:-$$}"
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
source "${REPO_ROOT}/scripts/env/activate_env.sh"

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

LOG_ROOT=${LOG_ROOT:-${REPO_ROOT}/log/libero_object_multicam_eval/${MODEL_NAME}}
SERVER_LOG_ROOT="${LOG_ROOT}/server_logs"
SHARD_SUMMARY_ROOT="${LOG_ROOT}/summary/${CHECKPOINT_STEP}/${SLURM_JOB_ID:-manual}"
AGGREGATE_SUMMARY_PATH=${SUMMARY_OUT_PATH:-${LOG_ROOT}/summary/aggregate_${CHECKPOINT_STEP}_${SLURM_JOB_ID:-manual}.json}
VIDEO_OUT_ROOT=${VIDEO_OUT_ROOT:-${REPO_ROOT}/data/libero/videos/libero_object/multicam/${MODEL_NAME}/${CHECKPOINT_STEP}/${SLURM_JOB_ID:-manual}}

mkdir -p "${REPO_ROOT}/log/libero_object_multicam_eval" "${LOG_ROOT}" "${SERVER_LOG_ROOT}" "${SHARD_SUMMARY_ROOT}" "${VIDEO_OUT_ROOT}" "${LIBERO_CONFIG_PATH}"

cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: ${REPO_ROOT}/third_party/libero/libero/libero
bddl_files: ${REPO_ROOT}/third_party/libero/libero/libero/bddl_files
init_states: ${REPO_ROOT}/third_party/libero/libero/libero/init_files
datasets: ${GEO_ROOT}/libero_cam_rlds
assets: ${REPO_ROOT}/third_party/libero/libero/libero/assets
EOF

if [[ ! -f "${CAMERA_VARIATION_CONFIG}" ]]; then
  echo "Missing camera variation config: ${CAMERA_VARIATION_CONFIG}" >&2
  exit 1
fi

ckpt_dir="${CHECKPOINT_ROOT}/${CHECKPOINT_STEP}"
if [[ ! -d "${ckpt_dir}" ]]; then
  echo "Missing checkpoint: ${ckpt_dir}" >&2
  exit 1
fi

mkdir -p "${ckpt_dir}/assets/$(dirname "${SERVE_ASSET_ID}")"
ln -sfn "$(basename "${CHECKPOINT_ASSET_ID}")" "${ckpt_dir}/assets/${SERVE_ASSET_ID}"

server_log="${SERVER_LOG_ROOT}/serve_${CHECKPOINT_STEP}_${SLURM_JOB_ID:-manual}.log"

echo "===== MULTICAM INFER ENV CHECK ====="
echo "repo root: ${REPO_ROOT}"
echo "model name: ${MODEL_NAME}"
echo "checkpoint dir: ${ckpt_dir}"
echo "config: ${CONFIG_NAME}"
echo "suite: ${SUITE_NAME}"
echo "task range request: [${TASK_ID_START}, ${TASK_ID_END})"
echo "trials per task: ${NUM_TRIALS_PER_TASK}"
echo "parallel clients: ${NUM_PARALLEL_CLIENTS}"
echo "camera variation config: ${CAMERA_VARIATION_CONFIG}"
echo "camera variation count: ${CAMERA_VARIATION_COUNT}"
echo "camera variation include original: ${CAMERA_VARIATION_INCLUDE_ORIGINAL}"
echo "camera target camera: ${CAMERA_VARIATION_TARGET_CAMERA}"
echo "save videos: ${SAVE_VIDEOS}"
echo "server batch max size: ${SERVER_BATCH_MAX_SIZE}"
echo "port: ${PORT}"
echo "summary root: ${SHARD_SUMMARY_ROOT}"
echo "aggregate summary: ${AGGREGATE_SUMMARY_PATH}"
which python
python -V
hostname
nvidia-smi -L || true

echo "Starting policy server for checkpoint ${ckpt_dir} on port ${PORT}"
echo "Server log: ${server_log}"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python -u scripts/serve_policy.py \
  --port "${PORT}" \
  --batch_max_size "${SERVER_BATCH_MAX_SIZE}" \
  --batch_timeout_ms "${SERVER_BATCH_TIMEOUT_MS}" \
  policy:checkpoint \
  --policy.config "${CONFIG_NAME}" \
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

task_count=$(get_task_count "${SUITE_NAME}")
task_count=$(printf '%s\n' "${task_count}" | tail -n 1 | tr -d '[:space:]')
if [[ ! "${task_count}" =~ ^[0-9]+$ ]] || [[ "${task_count}" -le 0 ]]; then
  echo "Failed to determine valid task count for suite ${SUITE_NAME}: ${task_count}" >&2
  exit 1
fi

global_task_start=${TASK_ID_START}
if [[ "${TASK_ID_END}" -lt 0 ]]; then
  global_task_end=${task_count}
else
  global_task_end=${TASK_ID_END}
fi

if [[ "${global_task_start}" -lt 0 ]]; then
  global_task_start=0
fi
if [[ "${global_task_end}" -gt "${task_count}" ]]; then
  global_task_end=${task_count}
fi
if [[ "${global_task_start}" -ge "${global_task_end}" ]]; then
  echo "Invalid effective task range [${global_task_start}, ${global_task_end})" >&2
  exit 1
fi

range_task_count=$((global_task_end - global_task_start))
if [[ "${NUM_PARALLEL_CLIENTS}" -lt 1 ]]; then
  echo "NUM_PARALLEL_CLIENTS must be >= 1" >&2
  exit 1
fi
if [[ "${NUM_PARALLEL_CLIENTS}" -gt "${range_task_count}" ]]; then
  NUM_PARALLEL_CLIENTS=${range_task_count}
fi

echo "suite task count: ${task_count}"
echo "effective task range: [${global_task_start}, ${global_task_end})"
echo "effective parallel clients: ${NUM_PARALLEL_CLIENTS}"

camera_include_original_flag=()
case "${CAMERA_VARIATION_INCLUDE_ORIGINAL}" in
  true|TRUE|True|1|yes|YES|Yes)
    camera_include_original_flag=(--args.camera-variation-include-original)
    ;;
  false|FALSE|False|0|no|NO|No)
    camera_include_original_flag=(--args.no-camera-variation-include-original)
    ;;
  *)
    echo "Unsupported CAMERA_VARIATION_INCLUDE_ORIGINAL value: ${CAMERA_VARIATION_INCLUDE_ORIGINAL}" >&2
    exit 1
    ;;
esac

save_videos_flag=()
case "${SAVE_VIDEOS}" in
  true|TRUE|True|1|yes|YES|Yes)
    save_videos_flag=(--args.save-videos)
    ;;
  false|FALSE|False|0|no|NO|No)
    save_videos_flag=()
    ;;
  *)
    echo "Unsupported SAVE_VIDEOS value: ${SAVE_VIDEOS}" >&2
    exit 1
    ;;
esac

CLIENT_PIDS=()
client_rc=0
for client_idx in $(seq 0 $((NUM_PARALLEL_CLIENTS - 1))); do
  offset_start=$(( client_idx * range_task_count / NUM_PARALLEL_CLIENTS ))
  offset_end=$(( (client_idx + 1) * range_task_count / NUM_PARALLEL_CLIENTS ))
  task_start=$(( global_task_start + offset_start ))
  task_end=$(( global_task_start + offset_end ))
  if [[ "${task_start}" -ge "${task_end}" ]]; then
    continue
  fi

  client_video_out="${VIDEO_OUT_ROOT}/shard_${client_idx}"
  shard_summary="${SHARD_SUMMARY_ROOT}/shard_${client_idx}.json"
  mkdir -p "${client_video_out}"
  echo "Starting client ${client_idx} for task range [${task_start}, ${task_end})"
  python examples/libero/main_multicam.py \
    --args.host 127.0.0.1 \
    --args.port "${PORT}" \
    --args.model-name "${MODEL_NAME}" \
    --args.task-suite-name "${SUITE_NAME}" \
    --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
    --args.task-id-start "${task_start}" \
    --args.task-id-end "${task_end}" \
    --args.replan-steps "${REPLAN_STEPS}" \
    --args.resize-size "${RESIZE_SIZE}" \
    --args.num-steps-wait "${NUM_STEPS_WAIT}" \
    --args.seed "${SEED}" \
    --args.camera-variation-config "${CAMERA_VARIATION_CONFIG}" \
    --args.camera-variation-count "${CAMERA_VARIATION_COUNT}" \
    --args.camera-variation-seed "${CAMERA_VARIATION_SEED}" \
    "${camera_include_original_flag[@]}" \
    --args.camera-variation-target-camera "${CAMERA_VARIATION_TARGET_CAMERA}" \
    "${save_videos_flag[@]}" \
    --args.video-out-path "${client_video_out}" \
    --args.summary-out-path "${shard_summary}" &
  CLIENT_PIDS+=("$!")
done

for pid in "${CLIENT_PIDS[@]}"; do
  wait "${pid}" || client_rc=$?
done

if [[ "${client_rc}" -ne 0 ]]; then
  echo "At least one inference client failed." >&2
  exit "${client_rc}"
fi

python - "${SHARD_SUMMARY_ROOT}" "${AGGREGATE_SUMMARY_PATH}" <<'PY'
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

per_camera = {}
per_task = {}
for record in records:
    camera_label = record["camera_label"]
    task_key = f"{record['task_id']}::{camera_label}"

    camera_entry = per_camera.setdefault(
        camera_label,
        {
            "camera_label": camera_label,
            "variation_id": record.get("variation_id"),
            "camera_pos": record.get("camera_pos"),
            "camera_quat": record.get("camera_quat"),
            "delta_pos": record.get("delta_pos"),
            "delta_rpy_deg": record.get("delta_rpy_deg"),
            "episodes": 0,
            "successes": 0,
        },
    )
    camera_entry["episodes"] += int(record["episodes"])
    camera_entry["successes"] += int(record["successes"])

    task_entry = per_task.setdefault(
        task_key,
        {
            "task_id": int(record["task_id"]),
            "task_description": record["task_description"],
            "camera_label": camera_label,
            "variation_id": record.get("variation_id"),
            "episodes": 0,
            "successes": 0,
            "camera_pos": record.get("camera_pos"),
            "camera_quat": record.get("camera_quat"),
            "delta_pos": record.get("delta_pos"),
            "delta_rpy_deg": record.get("delta_rpy_deg"),
        },
    )
    task_entry["episodes"] += int(record["episodes"])
    task_entry["successes"] += int(record["successes"])

for entry in per_camera.values():
    entry["success_rate"] = entry["successes"] / entry["episodes"] if entry["episodes"] else 0.0
for entry in per_task.values():
    entry["success_rate"] = entry["successes"] / entry["episodes"] if entry["episodes"] else 0.0

aggregate = {
    "model_name": summaries[0].get("model_name"),
    "task_suite_name": summaries[0]["task_suite_name"],
    "task_range": [
        min(summary["task_range"][0] for summary in summaries),
        max(summary["task_range"][1] for summary in summaries),
    ],
    "num_trials_per_task": summaries[0]["num_trials_per_task"],
    "camera_variation_config": summaries[0]["camera_variation_config"],
    "camera_variation_count": summaries[0]["camera_variation_count"],
    "camera_variation_include_original": summaries[0]["camera_variation_include_original"],
    "camera_labels": sorted(per_camera),
    "total_episodes": total_episodes,
    "total_successes": total_successes,
    "total_success_rate": total_successes / total_episodes if total_episodes else 0.0,
    "per_camera": [per_camera[key] for key in sorted(per_camera)],
    "per_task_camera": sorted(per_task.values(), key=lambda item: (item["task_id"], item["camera_label"])),
    "shard_summaries": [str(path) for path in paths],
}

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(aggregate, indent=2) + "\n")
print(output_path)
PY

echo "Multi-camera parallel inference complete."
echo "Aggregate summary written to: ${AGGREGATE_SUMMARY_PATH}"

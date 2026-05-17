#!/bin/bash
#SBATCH --job-name=eval_libero
#SBATCH --output=/scratch/lg154/Research/openpi/logs/eval_libero/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/openpi/logs/eval_libero/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --account=torch_pr_769_tandon_advanced 
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --export=ALL

# Single-suite LIBERO eval runner (lg154 setup).
#
# Spins up one policy server on localhost and fans out NUM_PARALLEL_CLIENTS
# rollout clients against it, sharded over the suite's task range. Each shard
# writes its own JSON summary; an aggregate JSON is written at the end.
#
# Required env (set by the dispatcher / caller):
#   CONFIG_NAME, EXP_NAME, STEP, MODEL_NAME, SUITE_NAME
# Optional overrides (defaults shown):
#   CHECKPOINT_ASSET_ID=glbreeze/libero_cam_v2      # asset_id training wrote to ckpt
#   SERVE_ASSET_ID=glbreeze/libero_object_cam_v3    # asset_id serve_policy expects
#   NUM_TRIALS_PER_TASK=50, NUM_PARALLEL_CLIENTS=4
#   REPLAN_STEPS=5, RESIZE_SIZE=224, NUM_STEPS_WAIT=10, SEED=7
#   PORT=19200, SERVER_START_TIMEOUT=600

set -euo pipefail

# ------------- inputs / defaults --------------------------------------------
: "${CONFIG_NAME:?required from caller}"
: "${EXP_NAME:?required from caller}"
: "${STEP:?required from caller}"
: "${MODEL_NAME:?required from caller}"
: "${SUITE_NAME:?required from caller}"

REPO_ROOT=/scratch/lg154/Research/openpi
GEO_ROOT=$(cd -- "${REPO_ROOT}/.." && pwd)
CHECKPOINT_ROOT="${REPO_ROOT}/checkpoints/${CONFIG_NAME}/${EXP_NAME}"
ckpt_dir="${CHECKPOINT_ROOT}/${STEP}"

CHECKPOINT_ASSET_ID=${CHECKPOINT_ASSET_ID:-glbreeze/libero_cam_v2}
SERVE_ASSET_ID=${SERVE_ASSET_ID:-glbreeze/libero_object_cam_v3}

NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-50}
NUM_PARALLEL_CLIENTS=${NUM_PARALLEL_CLIENTS:-4}
REPLAN_STEPS=${REPLAN_STEPS:-5}
RESIZE_SIZE=${RESIZE_SIZE:-224}
NUM_STEPS_WAIT=${NUM_STEPS_WAIT:-10}
SEED=${SEED:-7}

PORT=${PORT:-19200}
SERVER_START_TIMEOUT=${SERVER_START_TIMEOUT:-600}
SERVER_BATCH_MAX_SIZE=${SERVER_BATCH_MAX_SIZE:-${NUM_PARALLEL_CLIENTS}}
SERVER_BATCH_TIMEOUT_MS=${SERVER_BATCH_TIMEOUT_MS:-5}

LOG_ROOT="${REPO_ROOT}/logs/eval_libero/${MODEL_NAME}/${SUITE_NAME}"
SHARD_SUMMARY_ROOT="${LOG_ROOT}/summary/${STEP}/${SLURM_JOB_ID:-manual}"
AGGREGATE_SUMMARY_PATH="${LOG_ROOT}/summary/aggregate_${STEP}_${SLURM_JOB_ID:-manual}.json"
SERVER_LOG="${LOG_ROOT}/server_logs/serve_${STEP}_${SLURM_JOB_ID:-manual}.log"
VIDEO_OUT_ROOT="${REPO_ROOT}/data/libero/videos/${MODEL_NAME}/${SUITE_NAME}/${STEP}/${SLURM_JOB_ID:-manual}"

[[ -d "${ckpt_dir}" ]] || { echo "Missing checkpoint: ${ckpt_dir}" >&2; exit 1; }

# ------------- per-job scratch + env ----------------------------------------
JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp/$USER}/openpi_eval_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOB_TMP_ROOT}/tmp" TMP="${JOB_TMP_ROOT}/tmp" TEMP="${JOB_TMP_ROOT}/tmp"
export HOME="${JOB_TMP_ROOT}/home"
export XDG_CACHE_HOME="${JOB_TMP_ROOT}/xdg_cache"
export XDG_CONFIG_HOME="${JOB_TMP_ROOT}/xdg_config"
export XDG_STATE_HOME="${JOB_TMP_ROOT}/xdg_state"
export MPLCONFIGDIR="${JOB_TMP_ROOT}/mplconfig"
export LIBERO_CONFIG_PATH="${JOB_TMP_ROOT}/libero_config"

mkdir -p "${TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" \
         "${XDG_STATE_HOME}" "${MPLCONFIGDIR}" "${LIBERO_CONFIG_PATH}" \
         "$(dirname "${SERVER_LOG}")" "${SHARD_SUMMARY_ROOT}" "${VIDEO_OUT_ROOT}"

cd "${REPO_ROOT}"
source "${REPO_ROOT}/.venv/bin/activate"

export HF_HOME="${GEO_ROOT}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export OPENPI_DATA_HOME="${GEO_ROOT}/.cache/openpi"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${GEO_ROOT}}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/libero:${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}

# LIBERO config yaml the benchmark loader reads.
cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: ${REPO_ROOT}/third_party/libero/libero/libero
bddl_files: ${REPO_ROOT}/third_party/libero/libero/libero/bddl_files
init_states: ${REPO_ROOT}/third_party/libero/libero/libero/init_files
datasets: ${GEO_ROOT}/libero_cam_rlds
assets: ${REPO_ROOT}/third_party/libero/libero/libero/assets
EOF

# Alias the ckpt's saved asset dir so serve_policy can find norm_stats under
# whatever asset_id the config registry uses.
mkdir -p "${ckpt_dir}/assets/$(dirname "${SERVE_ASSET_ID}")"
ln -sfn "$(basename "${CHECKPOINT_ASSET_ID}")" "${ckpt_dir}/assets/${SERVE_ASSET_ID}"

# ------------- launch policy server -----------------------------------------
echo "===== EVAL ENV ====="
echo "ckpt:    ${ckpt_dir}"
echo "config:  ${CONFIG_NAME}"
echo "suite:   ${SUITE_NAME}"
echo "trials:  ${NUM_TRIALS_PER_TASK} per task, ${NUM_PARALLEL_CLIENTS} parallel clients"
echo "summary: ${AGGREGATE_SUMMARY_PATH}"
which python; python -V; hostname; nvidia-smi -L || true

echo "Starting policy server on port ${PORT} (log: ${SERVER_LOG})"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python -u scripts/serve_policy.py \
  --port "${PORT}" \
  --batch_max_size "${SERVER_BATCH_MAX_SIZE}" \
  --batch_timeout_ms "${SERVER_BATCH_TIMEOUT_MS}" \
  policy:checkpoint \
  --policy.config "${CONFIG_NAME}" \
  --policy.dir "${ckpt_dir}" > "${SERVER_LOG}" 2>&1 &
server_pid=$!
trap 'kill "${server_pid}" 2>/dev/null || true; wait "${server_pid}" 2>/dev/null || true' EXIT

# Wait for the server's port to accept connections, or die.
deadline=$(( SECONDS + SERVER_START_TIMEOUT ))
while ! python -c "import socket,sys; s=socket.socket(); s.settimeout(0.5); s.connect(('127.0.0.1', ${PORT}))" 2>/dev/null; do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
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
echo "Policy server up."

# ------------- shard the suite across N clients ------------------------------
task_count=$(python -c "
import contextlib, io
from libero.libero import benchmark
with contextlib.redirect_stdout(io.StringIO()):
    bd = benchmark.get_benchmark_dict()
    n = bd['${SUITE_NAME}']().n_tasks
print(n)
")
[[ "${task_count}" =~ ^[0-9]+$ && "${task_count}" -gt 0 ]] \
  || { echo "Bad task count for ${SUITE_NAME}: ${task_count}" >&2; exit 1; }
(( NUM_PARALLEL_CLIENTS > task_count )) && NUM_PARALLEL_CLIENTS=${task_count}
echo "Suite has ${task_count} tasks; sharding across ${NUM_PARALLEL_CLIENTS} clients."

client_pids=()
for i in $(seq 0 $((NUM_PARALLEL_CLIENTS - 1))); do
  task_start=$(( i * task_count / NUM_PARALLEL_CLIENTS ))
  task_end=$(( (i + 1) * task_count / NUM_PARALLEL_CLIENTS ))
  (( task_start >= task_end )) && continue

  shard_video="${VIDEO_OUT_ROOT}/shard_${i}"
  shard_summary="${SHARD_SUMMARY_ROOT}/shard_${i}.json"
  mkdir -p "${shard_video}"

  echo "Client ${i}: tasks [${task_start}, ${task_end})"
  python examples/libero/main.py \
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
    --args.video-out-path "${shard_video}" \
    --args.summary-out-path "${shard_summary}" &
  client_pids+=("$!")
done

client_rc=0
for pid in "${client_pids[@]}"; do
  wait "${pid}" || client_rc=$?
done
(( client_rc == 0 )) || { echo "At least one client failed (rc=${client_rc})" >&2; exit "${client_rc}"; }

# ------------- aggregate shard summaries -> one JSON -------------------------
python - "${SHARD_SUMMARY_ROOT}" "${AGGREGATE_SUMMARY_PATH}" <<'PY'
import json, pathlib, sys

shard_dir = pathlib.Path(sys.argv[1])
out_path = pathlib.Path(sys.argv[2])
paths = sorted(shard_dir.glob("shard_*.json"))
if not paths:
    raise SystemExit(f"No shard summaries in {shard_dir}")

summaries = [json.loads(p.read_text()) for p in paths]
records = [r for s in summaries for r in s.get("records", [])]
total_eps = sum(int(s.get("total_episodes", 0)) for s in summaries)
total_succ = sum(int(s.get("total_successes", 0)) for s in summaries)

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps({
    "model_name": summaries[0].get("model_name"),
    "task_suite_name": summaries[0]["task_suite_name"],
    "task_range": [
        min(s["task_range"][0] for s in summaries),
        max(s["task_range"][1] for s in summaries),
    ],
    "num_trials_per_task": summaries[0]["num_trials_per_task"],
    "total_episodes": total_eps,
    "total_successes": total_succ,
    "total_success_rate": total_succ / total_eps if total_eps else 0.0,
    "records": records,
    "shard_summaries": [str(p) for p in paths],
}, indent=2) + "\n")
print(out_path)
PY

echo "Eval complete."
echo "Aggregate summary: ${AGGREGATE_SUMMARY_PATH}"

#!/bin/bash
#SBATCH --job-name=rc24_20k_eval
#SBATCH --output=/scratch/%u/openpi_cam/log/robocasa24_20k_eval/slurm-%j.out
#SBATCH --error=/scratch/%u/openpi_cam/log/robocasa24_20k_eval/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --account=torch_pr_926_general
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

CONFIG_NAME=${CONFIG_NAME:-pi0_robocasa24_all24_baseline}
EXP_NAME=${EXP_NAME:-pi0_robocasa24_all24_baseline_tandon_2gpu_v1}
CHECKPOINT_STEP=${CHECKPOINT_STEP:-20000}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/scratch/${USER}/tmp/openpi_cam/checkpoints/${CONFIG_NAME}/${EXP_NAME}}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${CHECKPOINT_ROOT}/${CHECKPOINT_STEP}}
NORM_ASSET_ID=${NORM_ASSET_ID:-robocasa24/all24_human_camaware}
NORM_STATS_ROOT=${NORM_STATS_ROOT:-${GEO_ROOT}/pi0_robocasa24}
NORM_STATS_DIR="${NORM_STATS_ROOT}/${NORM_ASSET_ID}"

ROBOCASA_ENV=${ROBOCASA_ENV:-/scratch/${USER}/.venvs/robocasa24}
CONDA_SH=${CONDA_SH:-}
EVAL_TASKS=${EVAL_TASKS:-all24}
EVAL_SPLIT=${EVAL_SPLIT:-target}
PROMPT=${PROMPT:-auto}
NUM_EPISODES_PER_TASK=${NUM_EPISODES_PER_TASK:-1}
MAX_STEPS=${MAX_STEPS:-500}
CHUNK_SIZE=${CHUNK_SIZE:-50}
SEED=${SEED:-0}
VIDEO_EVERY=${VIDEO_EVERY:-0}
PORT=${PORT:-$((18000 + (${SLURM_JOB_ID:-0} % 20000)))}

LOG_ROOT=${LOG_ROOT:-${REPO_ROOT}/log/robocasa24_20k_eval/${EXP_NAME}_${CHECKPOINT_STEP}/${SLURM_JOB_ID:-manual}}
RESULT_DIR=${RESULT_DIR:-${LOG_ROOT}/results}
VIDEO_DIR=${VIDEO_DIR:-${LOG_ROOT}/videos}
SERVER_LOG=${SERVER_LOG:-${LOG_ROOT}/server/serve_${SLURM_JOB_ID:-manual}.log}
AGGREGATE_JSON=${AGGREGATE_JSON:-${LOG_ROOT}/aggregate.json}

JOB_TMP_BASE="${SLURM_TMPDIR:-/scratch/${USER}/tmp}"
JOB_TMP_ROOT="${JOB_TMP_BASE}/robocasa24_20k_eval_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOB_TMP_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export ROBOCASA_HOME="${ROBOCASA_HOME:-/scratch/${USER}/.cache/robocasa24_home}"
export HOME="${ROBOCASA_HOME}"
export XDG_CACHE_HOME="${GEO_ROOT}/.cache"
export XDG_CONFIG_HOME="${JOB_TMP_ROOT}/xdg_config"
export XDG_STATE_HOME="${JOB_TMP_ROOT}/xdg_state"
export MPLCONFIGDIR="${JOB_TMP_ROOT}/mplconfig"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${GEO_ROOT}}"
export HF_HOME="${HF_HOME:-${GEO_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export OPENPI_CACHE_DIR="${OPENPI_CACHE_DIR:-${GEO_ROOT}/.cache/openpi}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${OPENPI_CACHE_DIR}}"
export OPENPI_GEO_ROOT="${OPENPI_GEO_ROOT:-${GEO_ROOT}}"
export TOKENIZERS_PARALLELISM=false
unset TRANSFORMERS_CACHE

mkdir -p \
  "${TMPDIR}" \
  "${HOME}" \
  "${XDG_CACHE_HOME}" \
  "${XDG_CONFIG_HOME}" \
  "${XDG_STATE_HOME}" \
  "${MPLCONFIGDIR}" \
  "${RESULT_DIR}" \
  "${VIDEO_DIR}" \
  "$(dirname -- "${SERVER_LOG}")"
chmod 700 "${TMPDIR}" "${HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}" "${MPLCONFIGDIR}" || true

if [[ ! -f "${CHECKPOINT_DIR}/model.safetensors" ]]; then
  echo "Missing checkpoint model at ${CHECKPOINT_DIR}/model.safetensors" >&2
  exit 1
fi
if [[ ! -f "${NORM_STATS_DIR}/norm_stats.json" ]]; then
  echo "Missing norm stats at ${NORM_STATS_DIR}/norm_stats.json" >&2
  exit 1
fi

asset_link="${CHECKPOINT_DIR}/assets/${NORM_ASSET_ID}"
mkdir -p "$(dirname -- "${asset_link}")"
if [[ -e "${asset_link}" && ! -L "${asset_link}" ]]; then
  if [[ ! -f "${asset_link}/norm_stats.json" ]]; then
    echo "Checkpoint asset path exists but has no norm_stats.json: ${asset_link}" >&2
    exit 1
  fi
else
  ln -sfn "${NORM_STATS_DIR}" "${asset_link}"
fi

TASKS_24=(
  PnPCounterToCab PnPCabToCounter PnPCounterToSink PnPSinkToCounter
  PnPCounterToMicrowave PnPMicrowaveToCounter PnPCounterToStove
  PnPStoveToCounter
  OpenSingleDoor CloseSingleDoor OpenDoubleDoor CloseDoubleDoor
  OpenDrawer CloseDrawer
  TurnOnSinkFaucet TurnOffSinkFaucet TurnSinkSpout
  TurnOnStove TurnOffStove
  CoffeeSetupMug CoffeeServeMug CoffeePressButton
  TurnOnMicrowave TurnOffMicrowave
)

eval_tasks=()
if [[ "${EVAL_TASKS}" == "all24" ]]; then
  eval_tasks=("${TASKS_24[@]}")
else
  IFS=', ' read -r -a eval_tasks <<< "${EVAL_TASKS}"
fi

if [[ "${#eval_tasks[@]}" -eq 0 ]]; then
  echo "No eval tasks requested. Set EVAL_TASKS=all24 or a comma/space-separated task list." >&2
  exit 1
fi

find_conda_sh() {
  local candidate
  for candidate in \
    "${CONDA_SH}" \
    "${GEO_ROOT}/miniforge3/etc/profile.d/conda.sh" \
    "${GEO_ROOT}/miniconda3/etc/profile.d/conda.sh" \
    "/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh"; do
    [[ -n "${candidate}" ]] || continue
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

find_robocasa_env() {
  local candidate
  local conda_sh
  for candidate in \
    "${ROBOCASA_ENV}" \
    "/scratch/${USER}/.venvs/robocasa24" \
    "/scratch/${USER}/.venvs/robocasa365" \
    "/scratch/${USER}/.conda/envs/robocasa24" \
    "/scratch/${USER}/.conda/envs/robocasa365" \
    "robocasa24" \
    "robocasa365"; do
    [[ -n "${candidate}" ]] || continue
    if [[ -f "${candidate}/bin/activate" || -d "${candidate}/conda-meta" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
    if [[ "${candidate}" != */* ]]; then
      conda_sh=$(find_conda_sh 2>/dev/null || true)
      if [[ -n "${conda_sh}" ]]; then
        source "${conda_sh}" >/dev/null 2>&1
        if conda env list | awk '{print $1}' | grep -Fxq "${candidate}"; then
          printf '%s\n' "${candidate}"
          return 0
        fi
      fi
    fi
  done
  return 1
}

activate_robocasa_env() {
  local env_spec
  env_spec=$(find_robocasa_env) || {
    echo "Could not locate a RoboCasa Python env. Set ROBOCASA_ENV to a conda env name or an env path." >&2
    return 1
  }

  if [[ -f "${env_spec}/bin/activate" ]]; then
    source "${env_spec}/bin/activate"
    ROBOCASA_ENV="${env_spec}"
    return 0
  fi

  local conda_sh
  conda_sh=$(find_conda_sh) || {
    echo "Could not find conda.sh. Set CONDA_SH or use ROBOCASA_ENV=/path/to/env." >&2
    return 1
  }
  source "${conda_sh}"
  conda activate "${env_spec}"
  ROBOCASA_ENV="${env_spec}"
}

verify_robocasa_env() {
  python - <<'PY'
import importlib
import sys

missing = [name for name in ("robocasa", "robosuite", "gymnasium") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing_python_modules={' '.join(missing)}")
print(sys.executable)
PY
}

wait_for_server() {
  local port="$1"
  local server_pid="$2"
  local server_log="$3"

  for _ in $(seq 1 180); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "Policy server exited early. Last server log:"
      tail -120 "${server_log}" || true
      return 1
    fi
    if "${REPO_ROOT}/.venv/bin/python" - "${port}" <<'PY'
import socket
import sys

sock = socket.socket()
sock.settimeout(1)
try:
    sock.connect(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
    then
      return 0
    fi
    sleep 5
  done

  echo "Timed out waiting for policy server. Last server log:"
  tail -120 "${server_log}" || true
  return 1
}

echo "===== ROBOCASA24 20K BASELINE EVAL ====="
echo "repo root        : ${REPO_ROOT}"
echo "config           : ${CONFIG_NAME}"
echo "exp name         : ${EXP_NAME}"
echo "checkpoint step  : ${CHECKPOINT_STEP}"
echo "checkpoint dir   : ${CHECKPOINT_DIR}"
echo "norm stats       : ${NORM_STATS_DIR}"
echo "robocasa env     : ${ROBOCASA_ENV}"
echo "eval tasks       : ${EVAL_TASKS}"
echo "episodes/task    : ${NUM_EPISODES_PER_TASK}"
echo "max steps        : ${MAX_STEPS}"
echo "chunk size       : ${CHUNK_SIZE}"
echo "split            : ${EVAL_SPLIT}"
echo "result dir       : ${RESULT_DIR}"
hostname
nvidia-smi
date

cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/env/activate_env.sh"
export OPENPI_POLICY_ASSET_ID="${NORM_ASSET_ID}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "Starting policy server on port ${PORT}"
activate_robocasa_env
if ! robocasa_python=$(verify_robocasa_env 2>&1); then
  echo "Activated RoboCasa env '${ROBOCASA_ENV}', but required imports failed: ${robocasa_python}" >&2
  echo "Set ROBOCASA_ENV to an env that can import robocasa, robosuite, and gymnasium." >&2
  exit 1
fi
echo "robocasa python  : ${robocasa_python}"
export PYTHONPATH="${REPO_ROOT}/packages/openpi-client/src:${REPO_ROOT}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

echo "Starting policy server on port ${PORT}"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 "${REPO_ROOT}/.venv/bin/python" -u scripts/serve_policy.py \
  --port "${PORT}" \
  --default-prompt "${PROMPT}" \
  policy:checkpoint \
  --policy.config "${CONFIG_NAME}" \
  --policy.dir "${CHECKPOINT_DIR}" > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

wait_for_server "${PORT}" "${SERVER_PID}" "${SERVER_LOG}"

failed_tasks=()
for task in "${eval_tasks[@]}"; do
  [[ -n "${task}" ]] || continue
  env_name="${task}"
  if [[ "${env_name}" != robocasa/* ]]; then
    env_name="robocasa/${env_name}"
  fi
  task_slug="${env_name//\//_}"
  result_json="${RESULT_DIR}/${task_slug}.json"
  task_video_dir="${VIDEO_DIR}/${task_slug}"

  echo "===== Evaluating ${env_name} ====="
  if ! python scripts/debug/eval_robocasa365_remote.py \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --env "${env_name}" \
    --split "${EVAL_SPLIT}" \
    --prompt "${PROMPT}" \
    --episodes "${NUM_EPISODES_PER_TASK}" \
    --max-steps "${MAX_STEPS}" \
    --chunk-size "${CHUNK_SIZE}" \
    --seed "${SEED}" \
    --video-dir "${task_video_dir}" \
    --video-every "${VIDEO_EVERY}" \
    --no-include-right-camera \
    --out "${result_json}"; then
    failed_tasks+=("${env_name}")
  fi
done

python - "${RESULT_DIR}" "${AGGREGATE_JSON}" "${CONFIG_NAME}" "${EXP_NAME}" "${CHECKPOINT_STEP}" <<'PY'
import json
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
aggregate_path = Path(sys.argv[2])
config_name = sys.argv[3]
exp_name = sys.argv[4]
checkpoint_step = int(sys.argv[5])

task_results = []
total_episodes = 0
total_successes = 0
for path in sorted(result_dir.glob("*.json")):
    data = json.loads(path.read_text())
    episodes = data.get("episodes", [])
    successes = sum(1 for episode in episodes if episode.get("success"))
    total_episodes += len(episodes)
    total_successes += successes
    task_results.append(
        {
            "file": str(path),
            "env": data.get("env"),
            "split": data.get("split"),
            "episodes_completed": len(episodes),
            "successes": successes,
            "success_rate": data.get("success_rate"),
        }
    )

aggregate = {
    "config": config_name,
    "exp_name": exp_name,
    "checkpoint_step": checkpoint_step,
    "tasks_completed": len(task_results),
    "episodes_completed": total_episodes,
    "successes": total_successes,
    "success_rate": (total_successes / total_episodes) if total_episodes else None,
    "tasks": task_results,
}
aggregate_path.parent.mkdir(parents=True, exist_ok=True)
aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)
PY

if [[ "${#failed_tasks[@]}" -gt 0 ]]; then
  echo "Failed tasks: ${failed_tasks[*]}" >&2
  exit 1
fi

echo "RoboCasa24 eval complete. Aggregate: ${AGGREGATE_JSON}"

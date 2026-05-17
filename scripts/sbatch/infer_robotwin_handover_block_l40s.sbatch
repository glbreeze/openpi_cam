#!/bin/bash
#SBATCH --job-name=robotwin_infer
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/robotwin_infer/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/robotwin_infer/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_676_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1

set -euo pipefail

GEO_ROOT=${GEO_ROOT:-/scratch/yp2841/geometry-vla}
REPO_ROOT=${REPO_ROOT:-${GEO_ROOT}/openpi_cam}
ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-${GEO_ROOT}/RoboTwin}
ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-/scratch/yp2841/tmp/robotwin2/bin/python}

TASK=${TASK:-handover_block}
TASK_CONFIG=${TASK_CONFIG:-demo_clean_camaware_infer}
ASSET_TASK_CONFIG=${ASSET_TASK_CONFIG:-demo_clean_camaware}
CONFIG_NAME=${CONFIG_NAME:-pi0_robotwin_cam_prope_ray_view_distill_fullres_stage2}
EXP_NAME=${EXP_NAME:-stage2_${TASK}_adaptfix}
CHECKPOINT_STEP=${CHECKPOINT_STEP:-30000}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${REPO_ROOT}/checkpoints/${CONFIG_NAME}/${EXP_NAME}/${CHECKPOINT_STEP}}
ASSET_ID=${ASSET_ID:-robotwin/${TASK}_${ASSET_TASK_CONFIG}_50}
PORT=${PORT:-$((18000 + (${SLURM_JOB_ID:-0} % 10000)))}
PI0_STEP=${PI0_STEP:-50}
SEED=${SEED:-0}
INSTRUCTION_TYPE=${INSTRUCTION_TYPE:-unseen}

JOB_TMP_ROOT=${JOB_TMP_ROOT:-/scratch/yp2841/tmp/robotwin_infer_${SLURM_JOB_ID:-$$}}
export TMPDIR=${JOB_TMP_ROOT}/tmp
export HOME=${JOB_TMP_ROOT}/home
export XDG_CACHE_HOME=${JOB_TMP_ROOT}/xdg
export HF_HOME=${JOB_TMP_ROOT}/huggingface
export TORCH_HOME=${JOB_TMP_ROOT}/torch
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-${JOB_TMP_ROOT}/torch_extensions}
export PIP_CACHE_DIR=${JOB_TMP_ROOT}/pip
export MPLCONFIGDIR=${JOB_TMP_ROOT}/matplotlib
export CUDA_HOME=${CUDA_HOME:-/scratch/yp2841/tmp/robotwin2}
export PATH=/scratch/yp2841/tmp/robotwin2/bin:${PATH}
export TOKENIZERS_PARALLELISM=false

mkdir -p \
  "${TMPDIR}" "${HOME}" "${XDG_CACHE_HOME}" "${HF_HOME}" "${TORCH_HOME}" "${TORCH_EXTENSIONS_DIR}" "${PIP_CACHE_DIR}" "${MPLCONFIGDIR}" \
  "${REPO_ROOT}/log/robotwin_infer" "${REPO_ROOT}/log/robotwin_infer/server"

for required in \
  "${CHECKPOINT_DIR}/model.safetensors" \
  "${CHECKPOINT_DIR}/assets/${ASSET_ID}/norm_stats.json" \
  "${ROBOTWIN_ROOT}/assets/background_texture" \
  "${ROBOTWIN_ROOT}/assets/embodiments" \
  "${ROBOTWIN_ROOT}/assets/objects" \
  "${ROBOTWIN_PYTHON}" \
  "${CUDA_HOME}/bin/nvcc" \
  "${CUDA_HOME}/bin/ninja"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}" >&2
    exit 1
  fi
done

if [[ ! -f "${ROBOTWIN_ROOT}/task_config/${TASK_CONFIG}.yml" ]]; then
  cp "${REPO_ROOT}/scripts/robotwin_task_configs/${TASK_CONFIG}.yml" "${ROBOTWIN_ROOT}/task_config/${TASK_CONFIG}.yml"
fi

EVAL_CONFIG=${JOB_TMP_ROOT}/openpi_remote_eval.yml
cat > "${EVAL_CONFIG}" <<EOF
policy_name: openpi_remote
task_name: ${TASK}
task_config: ${TASK_CONFIG}
ckpt_setting: ${EXP_NAME}_${CHECKPOINT_STEP}
seed: ${SEED}
instruction_type: ${INSTRUCTION_TYPE}
server_host: 127.0.0.1
server_port: ${PORT}
pi0_step: ${PI0_STEP}
eval_video_log: false
EOF

wait_for_server() {
  local port="$1"
  local server_pid="$2"
  local server_log="$3"
  for _ in $(seq 1 240); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "Policy server exited early. Last server log:"
      tail -120 "${server_log}" || true
      return 1
    fi
    if grep -q "server listening on .*:${port}" "${server_log}" 2>/dev/null && "${REPO_ROOT}/.venv/bin/python" - "${port}" <<'PY'
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

cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/env/activate_env.sh"
export OPENPI_POLICY_ASSET_ID=${ASSET_ID}

server_log="${REPO_ROOT}/log/robotwin_infer/server/serve_${SLURM_JOB_ID:-manual}.log"
echo "Starting policy server on port ${PORT}"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python -u scripts/serve_policy.py \
  --port "${PORT}" \
  policy:checkpoint \
  --policy.config "${CONFIG_NAME}" \
  --policy.dir "${CHECKPOINT_DIR}" > "${server_log}" 2>&1 &
server_pid=$!

cleanup() {
  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT

wait_for_server "${PORT}" "${server_pid}" "${server_log}"

export PYTHONPATH="${REPO_ROOT}/scripts/robotwin_runtime:${ROBOTWIN_ROOT}/envs/curobo/src:${REPO_ROOT}/scripts/robotwin_eval_policies:${REPO_ROOT}/packages/openpi-client/src:${ROBOTWIN_ROOT}:${ROBOTWIN_ROOT}/policy:${ROBOTWIN_ROOT}/description/utils:${PYTHONPATH:-}"

cd "${ROBOTWIN_ROOT}"
echo "Starting RoboTwin eval for ${TASK}/${TASK_CONFIG}"
"${ROBOTWIN_PYTHON}" script/eval_policy.py --config "${EVAL_CONFIG}"

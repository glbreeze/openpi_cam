#!/bin/bash
#SBATCH --job-name=pi0_rc24_dsact
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/robocasa24_infer/dataset_policy_actions-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/robocasa24_infer/dataset_policy_actions-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --account=torch_pr_676_tandon_priority
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:a100:1
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
cd "${REPO_ROOT}"

JOB_TMP_BASE="${SLURM_TMPDIR:-/scratch/${USER}/tmp}"
JOB_TMP_ROOT="${JOB_TMP_BASE}/robocasa24_dsact_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOB_TMP_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export HOME="${HOME:-/scratch/${USER}}"
export XDG_CACHE_HOME="${GEO_ROOT}/.cache"
export HF_HOME="${HF_HOME:-${GEO_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export OPENPI_CACHE_DIR="${OPENPI_CACHE_DIR:-${GEO_ROOT}/.cache/openpi}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${OPENPI_CACHE_DIR}}"
export OPENPI_GEO_ROOT="${OPENPI_GEO_ROOT:-${GEO_ROOT}}"
export TOKENIZERS_PARALLELISM=false
export OPENPI_DISABLE_TORCH_COMPILE="${OPENPI_DISABLE_TORCH_COMPILE:-1}"
export PYTHONNOUSERSITE=1
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${HF_HOME}" "${OPENPI_CACHE_DIR}" log/robocasa24_infer/server

CONFIG_NAME=${CONFIG_NAME:-pi0_robocasa24_all24_cam_prope_ray_view_distill_fullres_stage2_gtonly}
EXP_NAME=${EXP_NAME:-stage2_all24_gtonly_224}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/${CONFIG_NAME}/${EXP_NAME}}
CHECKPOINT_STEP=${CHECKPOINT_STEP:-30000}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${CHECKPOINT_ROOT}/${CHECKPOINT_STEP}}
DATASET_ROOT=${DATASET_ROOT:-/scratch/yz11445/robocasa24/all24_human_camaware}
EPISODES=${EPISODES:-646,647,655,692,699}
FRAMES=${FRAMES:-0,25,50,75,100,125,150,175}
EXTRINSICS_MODES=${EXTRINSICS_MODES:-dataset,unflip}
IMAGE_MODES=${IMAGE_MODES:-as_is}
SAMPLES_PER_FRAME=${SAMPLES_PER_FRAME:-3}
PORT=${PORT:-$((21400 + (${SLURM_JOB_ID:-0} % 20000)))}
SERVER_START_TIMEOUT=${SERVER_START_TIMEOUT:-600}
SERVER_LOG=${SERVER_LOG:-${REPO_ROOT}/log/robocasa24_infer/server/serve_dsact_${EXP_NAME}_${CHECKPOINT_STEP}_${SLURM_JOB_ID:-manual}.log}
RESULT_JSON=${RESULT_JSON:-${REPO_ROOT}/log/robocasa24_infer/dataset_policy_actions_${EXP_NAME}_${CHECKPOINT_STEP}_${SLURM_JOB_ID:-manual}.json}

[[ -f "${CHECKPOINT_DIR}/model.safetensors" ]] || { echo "Missing checkpoint: ${CHECKPOINT_DIR}/model.safetensors" >&2; exit 1; }

echo "===== ROBOCASA24 DATASET POLICY ACTION CHECK ====="
echo "repo root      : ${REPO_ROOT}"
echo "config         : ${CONFIG_NAME}"
echo "checkpoint     : ${CHECKPOINT_DIR}"
echo "dataset        : ${DATASET_ROOT}"
echo "episodes       : ${EPISODES}"
echo "frames         : ${FRAMES}"
echo "extrinsics     : ${EXTRINSICS_MODES}"
echo "image modes    : ${IMAGE_MODES}"
echo "samples/frame  : ${SAMPLES_PER_FRAME}"
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
  --batch_max_size 1 \
  --batch_timeout_ms 5 \
  policy:checkpoint \
  --policy.config "${CONFIG_NAME}" \
  --policy.dir "${CHECKPOINT_DIR}" > "${SERVER_LOG}" 2>&1 &
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

PYTHONUNBUFFERED=1 uv run python -u scripts/debug/check_robocasa_policy_on_dataset.py \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --dataset-root "${DATASET_ROOT}" \
  --episodes "${EPISODES}" \
  --frames "${FRAMES}" \
  --extrinsics-modes "${EXTRINSICS_MODES}" \
  --image-modes "${IMAGE_MODES}" \
  --image-size 224 \
  --chunk-size 10 \
  --samples-per-frame "${SAMPLES_PER_FRAME}" \
  --out "${RESULT_JSON}"

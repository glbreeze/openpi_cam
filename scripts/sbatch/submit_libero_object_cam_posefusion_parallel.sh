#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SBATCH_SCRIPT="${SCRIPT_DIR}/infer_libero_object_cam_posefusion_parallel.sbatch"

CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/scratch/yp2841/geometry-vla/openpi_cam/checkpoints/pi0_libero_cam_pytorch_full_finetune/pi0_libero_object_cam_posefusion}
CHECKPOINT_STEP=${CHECKPOINT_STEP:-30000}
SUITE_NAME=${SUITE_NAME:-libero_object}
NUM_PARALLEL_CLIENTS=${NUM_PARALLEL_CLIENTS:-4}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-50}
TASK_ID_START=${TASK_ID_START:-0}
TASK_ID_END=${TASK_ID_END:--1}
PORT_BASE=${PORT_BASE:-19100}

echo "Submitting posefusion parallel inference"
echo "checkpoint root: ${CHECKPOINT_ROOT}"
echo "checkpoint step: ${CHECKPOINT_STEP}"
echo "suite: ${SUITE_NAME}"
echo "task range: [${TASK_ID_START}, ${TASK_ID_END})"
echo "parallel clients: ${NUM_PARALLEL_CLIENTS}"
echo "trials per task: ${NUM_TRIALS_PER_TASK}"

job_id=$(
  sbatch --parsable \
    --export=ALL,CHECKPOINT_ROOT="${CHECKPOINT_ROOT}",CHECKPOINT_STEP="${CHECKPOINT_STEP}",SUITE_NAME="${SUITE_NAME}",NUM_PARALLEL_CLIENTS="${NUM_PARALLEL_CLIENTS}",NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}",TASK_ID_START="${TASK_ID_START}",TASK_ID_END="${TASK_ID_END}",PORT_BASE="${PORT_BASE}" \
    "${SBATCH_SCRIPT}"
)

printf 'submitted_job=%s\n' "${job_id}"

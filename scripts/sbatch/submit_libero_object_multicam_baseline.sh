#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
LOG_DIR="${REPO_ROOT}/log/libero_object_multicam_eval/baseline"

mkdir -p "${LOG_DIR}"

CHECKPOINT_STEP=${CHECKPOINT_STEP:-30000}

sbatch \
  --job-name=libero_obj_multicam_base \
  --output="${LOG_DIR}/slurm-%j.out" \
  --error="${LOG_DIR}/slurm-%j.err" \
  --export=ALL,\
MODEL_NAME=baseline,\
CONFIG_NAME=pi0_libero_pytorch_full_finetune,\
CHECKPOINT_ROOT="${REPO_ROOT}/checkpoints/pi0_libero_pytorch_full_finetune/pi0_libero_object_baseline_full",\
CHECKPOINT_STEP="${CHECKPOINT_STEP}",\
CHECKPOINT_ASSET_ID=glbreeze/libero_object,\
SERVE_ASSET_ID=glbreeze/libero,\
PORT_BASE=19200 \
  "${SCRIPT_DIR}/infer_libero_object_multicam_parallel.sbatch"

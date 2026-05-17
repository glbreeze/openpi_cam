#!/usr/bin/env bash
# Fan out a 4-suite LIBERO eval (spatial, object, goal, 10) for one checkpoint.
# Submits 4 sbatch jobs that share `scripts/sbatch_lg/infer_libero.sbatch`.
#
# Usage (defaults run the L1 stage 2 ckpt at step 30000):
#   bash scripts/sbatch_lg/submit_eval_4suites.sh
#
# Override any of these per-run by exporting / inlining env vars:
#   CONFIG_NAME=... EXP_NAME=... STEP=... MODEL_NAME=... \
#     bash scripts/sbatch_lg/submit_eval_4suites.sh
#
# Each job runs on one L40S, fans out NUM_PARALLEL_CLIENTS=4 rollout clients
# against one policy server, and writes its summary to
#   logs/eval_libero/${MODEL_NAME}/${SUITE_NAME}/summary/aggregate_${STEP}_${jobid}.json

set -euo pipefail

REPO_ROOT=/scratch/lg154/Research/openpi
SBATCH_SCRIPT="${REPO_ROOT}/scripts/sbatch_lg/infer_libero.sbatch"

# --- defaults (override via env on the command line) ----------------------
export CONFIG_NAME=${CONFIG_NAME:-pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage2}
export EXP_NAME=${EXP_NAME:-pi0_libero_cam_v2_prope_ray_view_distill_fullres_stage2_l1}
export STEP=${STEP:-30000}
export MODEL_NAME=${MODEL_NAME:-stage2_l1}
# Asset_id overrides if the config you serve doesn't match the defaults:
# export CHECKPOINT_ASSET_ID=glbreeze/libero_cam_v2
# export SERVE_ASSET_ID=glbreeze/libero_object_cam_v3
# --------------------------------------------------------------------------

mkdir -p "${REPO_ROOT}/logs/eval_libero"

# Sanity check the checkpoint exists before queueing 4 jobs.
ckpt_dir="${REPO_ROOT}/checkpoints/${CONFIG_NAME}/${EXP_NAME}/${STEP}"
if [[ ! -f "${ckpt_dir}/model.safetensors" ]]; then
  echo "Checkpoint not found: ${ckpt_dir}/model.safetensors" >&2
  exit 1
fi

SUITES=(libero_spatial libero_object libero_goal libero_10)
for SUITE in "${SUITES[@]}"; do
  export SUITE_NAME="${SUITE}"
  sbatch \
    --job-name="eval_${MODEL_NAME}_${SUITE}" \
    "${SBATCH_SCRIPT}"
done

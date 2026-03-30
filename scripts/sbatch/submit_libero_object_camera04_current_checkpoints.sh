#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
GEO_ROOT=$(cd -- "${REPO_ROOT}/.." && pwd)

# "camera 04" refers to the 5th predefined left-variation pose, i.e. camvar_04.
CAMERA04_CONFIG=${CAMERA04_CONFIG:-${GEO_ROOT}/LIBERO-Camera/configs/camera_variation/libero_object_left_camvar_04_only.json}
CHECKPOINT_STEP=${CHECKPOINT_STEP:-30000}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-50}
NUM_PARALLEL_CLIENTS=${NUM_PARALLEL_CLIENTS:-4}

mkdir -p \
  "${REPO_ROOT}/log/libero_object_multicam_eval/current_camera04_pi0" \
  "${REPO_ROOT}/log/libero_object_multicam_eval/current_camera04_posefusion"

pi0_job=$(
  sbatch --parsable \
    --job-name=eval_obj_camera04_pi0 \
    --output="${REPO_ROOT}/log/libero_object_multicam_eval/current_camera04_pi0/slurm-%j.out" \
    --error="${REPO_ROOT}/log/libero_object_multicam_eval/current_camera04_pi0/slurm-%j.err" \
    --export=ALL,MODEL_NAME=current_camera04_pi0,CONFIG_NAME=pi0_libero_pytorch_full_finetune,CHECKPOINT_ROOT="${REPO_ROOT}/checkpoints/pi0_libero_pytorch_full_finetune/pi0_libero_object_cam_pi0_baseline",CHECKPOINT_STEP="${CHECKPOINT_STEP}",CHECKPOINT_ASSET_ID=glbreeze/libero_object_cam,SERVE_ASSET_ID=glbreeze/libero,CAMERA_VARIATION_CONFIG="${CAMERA04_CONFIG}",CAMERA_VARIATION_COUNT=1,CAMERA_VARIATION_INCLUDE_ORIGINAL=false,NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}",NUM_PARALLEL_CLIENTS="${NUM_PARALLEL_CLIENTS}",PORT_BASE=19640 \
    "${SCRIPT_DIR}/infer_libero_object_multicam_parallel.sbatch"
)

pose_job=$(
  sbatch --parsable \
    --job-name=eval_obj_camera04_pose \
    --output="${REPO_ROOT}/log/libero_object_multicam_eval/current_camera04_posefusion/slurm-%j.out" \
    --error="${REPO_ROOT}/log/libero_object_multicam_eval/current_camera04_posefusion/slurm-%j.err" \
    --export=ALL,MODEL_NAME=current_camera04_posefusion,CONFIG_NAME=pi0_libero_cam_pytorch_full_finetune,CHECKPOINT_ROOT="${REPO_ROOT}/checkpoints/pi0_libero_cam_pytorch_full_finetune/pi0_libero_object_cam_posefusion",CHECKPOINT_STEP="${CHECKPOINT_STEP}",CHECKPOINT_ASSET_ID=glbreeze/libero_object_cam,SERVE_ASSET_ID=glbreeze/libero_cam,CAMERA_VARIATION_CONFIG="${CAMERA04_CONFIG}",CAMERA_VARIATION_COUNT=1,CAMERA_VARIATION_INCLUDE_ORIGINAL=false,NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}",NUM_PARALLEL_CLIENTS="${NUM_PARALLEL_CLIENTS}",PORT_BASE=19660 \
    "${SCRIPT_DIR}/infer_libero_object_multicam_parallel.sbatch"
)

printf 'pi0_camera04=%s\npose_camera04=%s\n' "${pi0_job}" "${pose_job}"

#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
GEO_ROOT=$(cd -- "${REPO_ROOT}/.." && pwd)

ABC_REPO_ID=${ABC_REPO_ID:-glbreeze/libero_object_cam_abc}
ABC_INCLUDE_CAMERA_LABELS=${ABC_INCLUDE_CAMERA_LABELS:-camvar_00+camvar_01+camvar_02}
ABC_CONVERT_MODE=${ABC_CONVERT_MODE:-all_views}
ABC_PI0_EXP_NAME=${ABC_PI0_EXP_NAME:-pi0_libero_object_cam_abc_pi0}
ABC_POSE_EXP_NAME=${ABC_POSE_EXP_NAME:-pi0_libero_object_cam_abc_posefusion}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-30000}
CHECKPOINT_STEP=${CHECKPOINT_STEP:-${NUM_TRAIN_STEPS}}

D_CAMERA_CONFIG=${D_CAMERA_CONFIG:-${GEO_ROOT}/LIBERO-Camera/configs/camera_variation/libero_object_left_camvar_03_only.json}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-50}
NUM_PARALLEL_CLIENTS=${NUM_PARALLEL_CLIENTS:-4}

mkdir -p \
  "${REPO_ROOT}/log/libero_object_convert" \
  "${REPO_ROOT}/log/pi0_libero_object_ft" \
  "${REPO_ROOT}/log/libero_object_multicam_eval/abc_to_d_pi0" \
  "${REPO_ROOT}/log/libero_object_multicam_eval/abc_to_d_posefusion"

echo "Submitting LIBERO object ABC->D experiment"
echo "repo root: ${REPO_ROOT}"
echo "abc repo id: ${ABC_REPO_ID}"
echo "train camera labels: ${ABC_INCLUDE_CAMERA_LABELS}"
echo "eval camera config: ${D_CAMERA_CONFIG}"

convert_job=$(
  sbatch --parsable \
    --job-name=convert_libero_obj_abc \
    --export=ALL,RAW_HDF5_ROOT="${GEO_ROOT}/libero_object_camera_left",OUTPUT_REPO_ID="${ABC_REPO_ID}",MODE="${ABC_CONVERT_MODE}",INCLUDE_CAMERA_LABELS="${ABC_INCLUDE_CAMERA_LABELS}" \
    "${SCRIPT_DIR}/convert_libero_object_hdf5_selected_views.sbatch"
)

pi0_train_job=$(
  sbatch --parsable \
    --dependency="afterok:${convert_job}" \
    --job-name=pi0_obj_abc_pi0 \
    --export=ALL,USE_CAM=false,DATASET_REPO_ID="${ABC_REPO_ID}",NORM_ASSET_ID="${ABC_REPO_ID}",EXP_NAME="${ABC_PI0_EXP_NAME}",NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS}",WANDB_PROJECT=openpi_cam_libero_object \
    "${SCRIPT_DIR}/train_pi0_libero_object_ft.sbatch"
)

pose_train_job=$(
  sbatch --parsable \
    --dependency="afterok:${convert_job}" \
    --job-name=pi0_obj_abc_pose \
    --export=ALL,USE_CAM=true,DATASET_REPO_ID="${ABC_REPO_ID}",NORM_ASSET_ID="${ABC_REPO_ID}",EXP_NAME="${ABC_POSE_EXP_NAME}",NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS}",WANDB_PROJECT=openpi_cam_libero_object \
    "${SCRIPT_DIR}/train_pi0_libero_object_ft.sbatch"
)

pi0_eval_job=$(
  sbatch --parsable \
    --dependency="afterok:${pi0_train_job}" \
    --job-name=eval_obj_abc_to_d_pi0 \
    --output="${REPO_ROOT}/log/libero_object_multicam_eval/abc_to_d_pi0/slurm-%j.out" \
    --error="${REPO_ROOT}/log/libero_object_multicam_eval/abc_to_d_pi0/slurm-%j.err" \
    --export=ALL,MODEL_NAME=abc_to_d_pi0,CONFIG_NAME=pi0_libero_pytorch_full_finetune,CHECKPOINT_ROOT="${REPO_ROOT}/checkpoints/pi0_libero_pytorch_full_finetune/${ABC_PI0_EXP_NAME}",CHECKPOINT_STEP="${CHECKPOINT_STEP}",CHECKPOINT_ASSET_ID="${ABC_REPO_ID}",SERVE_ASSET_ID=glbreeze/libero,CAMERA_VARIATION_CONFIG="${D_CAMERA_CONFIG}",CAMERA_VARIATION_COUNT=1,CAMERA_VARIATION_INCLUDE_ORIGINAL=false,NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}",NUM_PARALLEL_CLIENTS="${NUM_PARALLEL_CLIENTS}",PORT_BASE=19600 \
    "${SCRIPT_DIR}/infer_libero_object_multicam_parallel.sbatch"
)

pose_eval_job=$(
  sbatch --parsable \
    --dependency="afterok:${pose_train_job}" \
    --job-name=eval_obj_abc_to_d_pose \
    --output="${REPO_ROOT}/log/libero_object_multicam_eval/abc_to_d_posefusion/slurm-%j.out" \
    --error="${REPO_ROOT}/log/libero_object_multicam_eval/abc_to_d_posefusion/slurm-%j.err" \
    --export=ALL,MODEL_NAME=abc_to_d_posefusion,CONFIG_NAME=pi0_libero_cam_pytorch_full_finetune,CHECKPOINT_ROOT="${REPO_ROOT}/checkpoints/pi0_libero_cam_pytorch_full_finetune/${ABC_POSE_EXP_NAME}",CHECKPOINT_STEP="${CHECKPOINT_STEP}",CHECKPOINT_ASSET_ID="${ABC_REPO_ID}",SERVE_ASSET_ID=glbreeze/libero_cam,CAMERA_VARIATION_CONFIG="${D_CAMERA_CONFIG}",CAMERA_VARIATION_COUNT=1,CAMERA_VARIATION_INCLUDE_ORIGINAL=false,NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}",NUM_PARALLEL_CLIENTS="${NUM_PARALLEL_CLIENTS}",PORT_BASE=19620 \
    "${SCRIPT_DIR}/infer_libero_object_multicam_parallel.sbatch"
)

printf 'convert=%s\npi0_train=%s\npose_train=%s\npi0_eval=%s\npose_eval=%s\n' \
  "${convert_job}" "${pi0_train_job}" "${pose_train_job}" "${pi0_eval_job}" "${pose_eval_job}"

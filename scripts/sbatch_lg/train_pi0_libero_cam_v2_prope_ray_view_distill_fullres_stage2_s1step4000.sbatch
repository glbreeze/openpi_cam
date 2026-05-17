#!/bin/bash
#SBATCH --job-name=pi0_cam_v2_s2_s1k4_h100
#SBATCH --output=/scratch/lg154/Research/openpi/logs/pi0_libero_cam_v2/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/openpi/logs/pi0_libero_cam_v2/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:2
#SBATCH --export=ALL

# Stage 2 variant: warm-start from MSE Stage 1's step-4000 checkpoint instead
# of step-5000. Step 4000 happened to have a slightly lower action_loss tail
# than 5000 in the MSE Stage 1 run, so this re-runs the same Stage 2 recipe
# with that earlier warmup. Output lands under a distinct EXP_NAME so the
# original Stage 2 checkpoint is preserved.

set -euo pipefail

export REPO_ROOT=/scratch/lg154/Research/openpi

# --- env consumed by train_pi0_libero.sbatch ---
export HF_LEROBOT_HOME=/scratch/lg154/cache/lerobot
export CONFIG_NAME=pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage2

export DATASET_REPO_ID=glbreeze/libero_cam_v2
export NORM_ASSET_ID=glbreeze/libero_cam_v2
export PI3X_TARGETS_ROOT_OVERRIDE=/scratch/lg154/.cache/openpi/pi3x_targets_224/libero_cam_v2

# Warm-start from MSE Stage 1 step 4000 (instead of the default 5000).
STAGE1_CONFIG=pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage1
STAGE1_EXP_NAME=pi0_libero_cam_v2_prope_ray_view_distill_fullres_stage1_h100_2gpu_b16_lg
STAGE1_FINAL_STEP=4000
export PYTORCH_WEIGHT_PATH_OVERRIDE="${REPO_ROOT}/checkpoints/${STAGE1_CONFIG}/${STAGE1_EXP_NAME}/${STAGE1_FINAL_STEP}"

export NUM_GPUS=2
export NUM_WORKERS=16
export BATCH_SIZE=16
export NUM_TRAIN_STEPS=30000
export SAVE_INTERVAL=1000
export KEEP_PERIOD=5000

# --- wandb ---
export WANDB_ENABLED=true
export WANDB_PROJECT=pi0_libero_lg
export WANDB_ENTITY=NYU-robotics
export WANDB_API_KEY_OVERRIDE="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"

# Distinct EXP_NAME -- writes to a separate checkpoint dir so the original
# Stage 2 run at `..._stage2_h100_2gpu_b16_lg` stays intact.
export EXP_NAME=pi0_libero_cam_v2_prope_ray_view_distill_fullres_stage2_s1step4000

# Pre-flight: confirm Stage 1's step-4000 checkpoint is on disk.
if [[ ! -f "${PYTORCH_WEIGHT_PATH_OVERRIDE}/model.safetensors" ]]; then
  echo "Stage 1 step-${STAGE1_FINAL_STEP} checkpoint not found at: ${PYTORCH_WEIGHT_PATH_OVERRIDE}" >&2
  echo "Re-run Stage 1 with KEEP_PERIOD<=${STAGE1_FINAL_STEP} so the step is preserved." >&2
  exit 1
fi

exec bash "${REPO_ROOT}/scripts/sbatch_lg/train_pi0_libero.sbatch"

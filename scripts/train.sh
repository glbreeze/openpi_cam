#!/bin/bash
#SBATCH --job-name=pi0_libero_object_ft
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_libero_object_ft/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_libero_object_ft/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=16:00:00
#SBATCH --account=torch_pr_637_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:h200:4

set -euo pipefail

# ---- user config ----

NUM_GPUS=4
CONFIG_NAME=pi0_libero_pytorch_full_finetune
DATASET_REPO_ID=glbreeze/libero_object
BASE_MODEL_DIR=/path/to/base_model
CHECKPOINT_DIR=/path/to/checkpoints

# ---- activate env ----

source /path/to/your/env/bin/activate

# ---- run training ----

srun python -m torch.distributed.run 
--standalone 
--nnodes=1 
--nproc_per_node=${NUM_GPUS} 
scripts/train_pytorch.py 
${CONFIG_NAME} 
--exp-name=my_experiment 
--checkpoint-base-dir=${CHECKPOINT_DIR} 
--batch-size=32 
--num-workers=16 
--num-train-steps=60000 
--pytorch_weight_path ${BASE_MODEL_DIR} 
--data.repo_id ${DATASET_REPO_ID}

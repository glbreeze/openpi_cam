#!/bin/bash
#SBATCH --job-name=pi0_robocasa24_baser
#SBATCH --output=/scratch/%u/openpi_cam/log/pi0_robocasa24_ft/slurm-%j.out
#SBATCH --error=/scratch/%u/openpi_cam/log/pi0_robocasa24_ft/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_69_tandon_advanced
#SBATCH --partition=h100_tandon,h200_tandon,a100_tandon
#SBATCH --gres=gpu:4
#SBATCH --export=ALL

set -euo pipefail

export RESUME=true
exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/train_pi0_robocasa24_all24_pytorch_baseline.sbatch"

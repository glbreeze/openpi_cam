
# init conda env 
~/miniconda3/bin/conda init
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openpi
# note: rerun-sdk for lerobot has been excluded from package

# download pretrained model 
gsutil cp -r gs://openpi-assets/checkpoints/pi0_base ./ckpt

gsutil -m \
  -o "GSUtil:parallel_composite_upload_threshold=0" \
  -o "GSUtil:sliced_object_download_threshold=0" \
  cp -r gs://openpi-assets/checkpoints/pi0_base ./ckpt


# Convert your data to a LeRobot dataset. Can skip this step for fine-tuning on LIBERO. 
python examples/libero/convert_libero_data_to_lerobot.py --data_dir /scratch/lg154/Research/datasets/modified_libero_rlds


cpu() {srun --nodes=1 --ntasks=1 --cpus-per-task=${1:-4} --mem=${2:-32}GB --time=3:00:00 --pty /bin/bash}
gpu() {
    srun -p sfscai --nodes=1 --ntasks=1 --cpus-per-task=${1:-8} --mem=${2:-64}GB --time=3:00:00 --gres=gpu:1 --pty /bin/bash
}



#!/bin/bash
#SBATCH --job-name=libero_convert

#SBATCH --output=logs/dt_to_lerobot.out
#SBATCH --error=logs/dt_to_lerobot.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate openpi

export HF_LEROBOT_HOME=/path/to/where/you/want/output
python examples/libero/convert_libero_data_to_lerobot.py --data_dir /scratch/lg154/Research/datasets/modified_libero_rlds

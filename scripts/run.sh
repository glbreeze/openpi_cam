
# activate python env
source /home/asus/Research/openpi/.venv/bin/activate

# download libero plus 
# nohup huggingface-cli download Sylvest/libero_plus_rlds \
#   --repo-type dataset \
#   --local-dir ~/Research/datasets/libero_plus_rlds \
#   > libero_download.log 2>&1 &


# Convert data to a LeRobot dataset 
export HF_HOME=/home/asus/Research/datasets/huggingface
nohup python examples/libero/convert_libero_data_to_lerobot.py --data_dir /home/asus/Research/datasets/libero_cam_rlds > libero_to_lerobot.out 2>&1 &
# modified_libero_rlds | libero_mix | libero_cam_rlds

# Compute the normalization statistics for the training data
# go to "src/openpi/training/config.py" and change repo_id to repo_id='glbreeze/libero_cam'
nohup python scripts/compute_norm_stats.py --config-name pi0_libero > compute_stats.out 2>&1 &

# convert JAX model ckpt to PyTorch format
uv run examples/convert_jax_model_to_pytorch.py \
    --config_name <config name> \
    --checkpoint_dir /path/to/jax/base/model \
    --output_path /path/to/pytorch/base/model


# Single GPU training:
python scripts/train_pytorch.py pi0_libero --exp_name test_run \
  --pytorch_weight_path /home/asus/Research/openpi/ckpt/pytorch/pi0_base --batch_size 16 \
  --data.repo_id glbreeze/libero_cam

# Example:
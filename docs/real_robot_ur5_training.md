# UR5 Real-Robot Baseline Training

This repo now includes a PyTorch pi0 baseline config for the local LeRobot-format
dataset at:

`/scratch/yz11445/real_robot_data/ur5_lab_test_tube_camera_shifts`

Config name:

`pi0_ur5_real_robot_pytorch_baseline`

Default view selection:

- base image: `observation.images.context_left_rgb`
- wrist image: `observation.images.wrist_right_rgb`

The config trains the plain baseline pi0 model:

- no pose/ray/view camera-aware modules
- PyTorch training path
- absolute joint targets converted to delta on the first 6 joints only
- prompt loaded from the LeRobot task text

## Commands

Compute normalization stats first:

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_ur5_real_robot_pytorch_baseline
```

Then launch training:

```bash
uv run python scripts/train_pytorch.py pi0_ur5_real_robot_pytorch_baseline --exp_name ur5_real_robot_baseline_v1
```

## Notes

- The config uses the local dataset root directly; no Hugging Face upload or
  symlink into `HF_LEROBOT_HOME` is required.
- If you want a different third-person camera, change the `base_camera_key`
  field in `LeRobotRealRobotUR5DataConfig` inside
  `src/openpi/training/config.py`.

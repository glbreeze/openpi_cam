# pi0 LIBERO Object Cam V3 Full-Resolution GTDual Zero-Init V3

This directory contains loss logs for the GTDual zero-init two-stage run.

## Runs

- Stage 1
  - Exp: `pi0_libero_object_cam_v3_prope_ray_view_distill_fullres_stage1_gtdual_zeroinit_v3_l40s_4gpu_b32_5k`
  - Config: `pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage1_gtdual`
  - Slurm job: `8258893`
  - State: `COMPLETED`
  - Time: `2026-05-08T12:42:56` to `2026-05-08T16:12:26`
  - Partition/account: `l40s_public`, `torch_pr_637_tandon_advanced`
  - Hardware: `4 x NVIDIA L40S`
  - Steps: `5000`
  - Checkpoint: `checkpoints/pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage1_gtdual/pi0_libero_object_cam_v3_prope_ray_view_distill_fullres_stage1_gtdual_zeroinit_v3_l40s_4gpu_b32_5k/5000`

- Stage 2
  - Exp: `pi0_libero_object_cam_v3_prope_ray_view_distill_fullres_stage2_gtdual_zeroinit_v3_ahh_4gpu_b32_30k`
  - Config: `pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage2_gtdual`
  - Slurm job: `8284105`
  - State: `COMPLETED`
  - Time: `2026-05-08T17:11:15` to `2026-05-09T02:14:22`
  - Partition/account: `h200_tandon`, `torch_pr_637_tandon_advanced`
  - Hardware: `4 x NVIDIA H200`
  - Steps: `30000`
  - Initialized from stage-1 checkpoint: `checkpoints/pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage1_gtdual/pi0_libero_object_cam_v3_prope_ray_view_distill_fullres_stage1_gtdual_zeroinit_v3_l40s_4gpu_b32_5k/5000`
  - Final checkpoint: `checkpoints/pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage2_gtdual/pi0_libero_object_cam_v3_prope_ray_view_distill_fullres_stage2_gtdual_zeroinit_v3_ahh_4gpu_b32_30k/30000`

Both stages use:

- Dataset: `/scratch/yp2841/.huggingface/lerobot/glbreeze/libero_object_cam_v3`
- Pi3X targets: `/scratch/yp2841/geometry-vla/.cache/openpi/pi3x_targets_224/libero_object_cam_v3`
- GT point targets: `/scratch/yp2841/geometry-vla/.cache/openpi/gt_point_targets_224/libero_object_cam_v3_aligned`
- Batch size: `32`
- Save interval: `1000`
- Keep period: `5000`

## Files

- `stage1_gtdual_zeroinit_l40s_4gpu_b32_5k_all_loss.tsv`
- `stage2_gtdual_zeroinit_ahh_4gpu_b32_30k_all_loss.tsv`

Each TSV includes:

- `logged_avg`: metrics emitted by `scripts/train_pytorch.py` at the logging interval, including breakdown fields.
- `tqdm_step`: per-step loss/lr parsed from the tqdm progress output.

Important columns:

- `source_job_id`: Slurm job that produced the row.
- `job_state`: final state for that source job.
- `row_type`: `logged_avg` or `tqdm_step`.
- `step`: training step.
- `loss`, `lr`: total loss and learning rate.
- `action_loss`, `aux_loss`, `aux_gt_loss`, `aux_pi3x_loss`, `aux_xy_loss`, `aux_z_loss`: available on `logged_avg` rows.
- `pytorch_weight_path_override`: checkpoint used to initialize the run, when set.

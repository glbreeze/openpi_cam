# pi0 LIBERO Object Cam V3 Full-Resolution Two-Stage V1

This directory contains loss logs for the two-stage run:

- Stage 1: `pi0_libero_object_cam_v3_prope_ray_view_distill_fullres_twostage_v1_stage1_l40s`
- Stage 2: `pi0_libero_object_cam_v3_prope_ray_view_distill_fullres_twostage_v1_stage2_zeroinit_l40s`

Both runs use the LIBERO object camera-v3 dataset at:

`/scratch/yp2841/.huggingface/lerobot/glbreeze/libero_object_cam_v3`

## Files

- `stage1_l40s_all_loss.tsv`
  - Source Slurm job: `7203317`
  - State: `COMPLETED`
  - Time: `2026-04-26T08:26:57` to `2026-04-26T11:40:59`
  - Partition/account: `l40s_public`, `torch_pr_637_general`
  - Hardware: `4 x NVIDIA L40S`
  - Config: `pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage1`
  - Steps: `5000`
  - Checkpoint: `checkpoints/pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage1/pi0_libero_object_cam_v3_prope_ray_view_distill_fullres_twostage_v1_stage1_l40s/5000`

- `stage2_zeroinit_l40s_all_loss.tsv`
  - Source Slurm jobs: `7225039`, `7256288`
  - Job `7225039`: `TIMEOUT`, `2026-04-26T13:39:32` to `2026-04-26T21:39:46`, `a100_tandon`, `torch_pr_637_tandon_advanced`, `4 x NVIDIA A100-SXM4-80GB`
  - Job `7256288`: `COMPLETED`, `2026-04-27T06:01:52` to `2026-04-27T09:07:10`, `h200_tandon`, `torch_pr_637_tandon_advanced`, `4 x NVIDIA H200`
  - Config: `pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage2`
  - Steps: `30000`
  - Resumed from experiment step: `18000`
  - Stage-1 weight path: `checkpoints/pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage1/pi0_libero_object_cam_v3_prope_ray_view_distill_fullres_twostage_v1_stage1_l40s/5000`
  - Final checkpoint: `checkpoints/pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage2/pi0_libero_object_cam_v3_prope_ray_view_distill_fullres_twostage_v1_stage2_zeroinit_l40s/30000`

## TSV Schema

Each TSV includes two row types:

- `logged_avg`: metrics emitted by `scripts/train_pytorch.py` at the logging interval, including loss breakdown fields when available.
- `tqdm_step`: per-step loss/lr parsed from the tqdm progress output.

Important columns:

- `source_job_id`: Slurm job that produced the row.
- `job_state`: final state for that source job.
- `row_type`: `logged_avg` or `tqdm_step`.
- `step`: training step for the metric row.
- `loss`, `lr`: total loss and learning rate.
- `action_loss`, `aux_loss`, `aux_xy_loss`, `aux_z_loss`: available for `logged_avg` rows.
- `pytorch_weight_path_override`: checkpoint used to initialize the run, when set.

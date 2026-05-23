# PY Docs

## Pi0 LIBERO 4-Suite Experiment

- Dataset / norm asset: `glbreeze/libero_cam_v2`
- Training suites: `libero_object`, `libero_goal`, `libero_spatial`, `libero_10`
- Training steps: 30k final checkpoints
- Training batch size: 32 total, 4 GPUs, 8 per GPU
- Evaluation: canonical camera setting, 50 trials per task, repeated twice and averaged
- Evaluation suites: `libero_object`, `libero_goal`, `libero_spatial`, `libero_10`

### Configs

Baseline:
- Config: `pi0_libero_4suite_pytorch_baseline`
- Checkpoint: `checkpoints/pi0_libero_object_pytorch_baseline/pi0_libero_cam_v2_4suite_baseline_30k_any_4gpu_b32/30000`

Camera-aware:
- Stage 1 config: `pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_4suite_stage1`
- Stage 1 checkpoint: `checkpoints/pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_4suite_stage1/pi0_libero_cam_v2_prope_ray_view_distill_fullres_bestrecipe_4suite_stage1_any_4gpu_b32/5000`
- Stage 2 config: `pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_4suite_stage2`
- Stage 2 checkpoint: `checkpoints/pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_4suite_stage2/pi0_libero_cam_v2_prope_ray_view_distill_fullres_bestrecipe_4suite_stage2_zeroinit_any_4gpu_b32/30000`

GT-only camera-aware ablation:
- Stage 1 config: `pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_4suite_stage1_gtonly`
- Stage 1 checkpoint: `checkpoints/pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_4suite_stage1_gtonly/pi0_libero_cam_v2_prope_ray_view_distill_fullres_4suite_stage1_gtonly_769_any_4gpu_b32_5k/5000`
- Stage 2 config: `pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_4suite_stage2_gtonly`
- Stage 2 checkpoint root: `checkpoints/pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_4suite_stage2_gtonly/pi0_libero_cam_v2_prope_ray_view_distill_fullres_4suite_stage2_gtonly_769_any_4gpu_b32_30k`
- Difference vs. camera-aware: uses GT point targets from `/scratch/yp2841/geometry-vla/.cache/openpi/gt_point_targets_224/libero_cam_v2_aligned` instead of Pi3X targets; all other main 4-suite recipe settings are kept aligned.

### Results

Two-run average, 1000 episodes per suite and 4000 episodes total:

| Suite | Baseline | Camera-aware |
| --- | ---: | ---: |
| LIBERO Object | 89.40% | 93.40% |
| LIBERO Goal | 87.40% | 88.40% |
| LIBERO Spatial | 85.20% | 88.60% |
| LIBERO 10 | 76.50% | 77.60% |
| Overall | 84.62% | 87.00% |

### LIBERO-plus Camera Variation Results

Full official `Camera Viewpoints` category, 1 trial per task, 1599 tasks total. This run used the raw LIBERO-plus task descriptions; those include `view ... initstate ...` suffixes, so the low absolute rates are likely affected by both large OOD viewpoint shifts and prompt contamination.

| Suite | Baseline | Camera-aware |
| --- | ---: | ---: |
| LIBERO Object | 6/396 = 1.52% | 3/396 = 0.76% |
| LIBERO Goal | 10/408 = 2.45% | 9/408 = 2.21% |
| LIBERO Spatial | 7/376 = 1.86% | 7/376 = 1.86% |
| LIBERO 10 | 8/419 = 1.91% | 14/419 = 3.34% |
| Overall | 31/1599 = 1.94% | 33/1599 = 2.06% |

Small camera variation subset, 1 trial per selected task, with LIBERO-plus prompt suffix stripped. Task selection: `Camera Viewpoints` only, yaw within 15 degrees of canonical, vertical within 15 degrees, scale `100`, endpoint rotation `0`, endpoint vertical `0`.

| Suite | Baseline | Camera-aware |
| --- | ---: | ---: |
| LIBERO Object | 20/23 = 86.96% | 15/23 = 65.22% |
| LIBERO Goal | 26/37 = 70.27% | 21/37 = 56.76% |
| LIBERO Spatial | 14/23 = 60.87% | 6/23 = 26.09% |
| LIBERO 10 | 19/34 = 55.88% | 21/34 = 61.76% |
| Overall | 79/117 = 67.52% | 63/117 = 53.85% |

Interpretation: small viewpoint perturbations plus prompt stripping produce a meaningful eval signal, but the strict baseline is better overall on this subset. Camera-aware only wins on `libero_10` in this specific small-variation eval.


### Ablations

1. Camera-aware on Pi0.5 (`pi05=True`, otherwise identical to camera-aware):
   - Stage 1 config: `pi05_libero_cam_pytorch_prope_ray_view_distill_fullres_4suite_stage1`
   - Stage 2 config: `pi05_libero_cam_pytorch_prope_ray_view_distill_fullres_4suite_stage2`
   - Base weights: `$GEO_ROOT/pi05_base` (PyTorch). Convert from JAX with
     `examples/convert_jax_model_to_pytorch.py` if not present yet.
   - Training checkpoint found: `checkpoints/pi05_libero_cam_pytorch_prope_ray_view_distill_fullres_4suite_stage2/pi05_libero_cam_v2_prope_ray_view_distill_fullres_4suite_stage2_769_any_4gpu_b32_30k/30000`
   - No eval aggregate found in `log/` at the time this note was written.

2. Turn OFF ray-embed (PRoPE + cross-view + aux point head still on):
   - Stage 1 config: `pi0_libero_cam_pytorch_prope_view_distill_fullres_4suite_stage1`
   - Stage 2 config: `pi0_libero_cam_pytorch_prope_view_distill_fullres_4suite_stage2`
   - Diff vs. camera-aware: `ray_enc_type=False`; `ray_embed` dropped from
     `trainable_prefixes` in Stage 1.
   - No eval aggregate found in `log/` at the time this note was written.

3. Turn OFF PRoPE (ray-embed + cross-view + aux point head still on):
   - Stage 1 config: `pi0_libero_cam_pytorch_ray_view_distill_fullres_4suite_stage1`
   - Stage 2 config: `pi0_libero_cam_pytorch_ray_view_distill_fullres_4suite_stage2`
   - Diff vs. camera-aware: `pose_enc_type="null"` and
     `cross_view.prope_layer_idx=()`.
   - No eval aggregate found in `log/` at the time this note was written.

4. Strict Pi0 baseline AB:
   - Config: `pi0_libero_4suite_pytorch_baseline`
   - Checkpoint: `checkpoints/pi0_libero_object_pytorch_baseline/pi0_libero_cam_v2_4suite_baseline_30k_any_4gpu_b32/30000`
   - Used for the canonical baseline and LIBERO-plus baseline comparisons above.

5. One-stage camera-aware ablation:
   - Config: `pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_4suite_1stage`
   - Checkpoint: `checkpoints/pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_4suite_1stage/pi0_libero_cam_v2_prope_ray_view_distill_fullres_1stage_769_any_4gpu_b32_30k/30000`
   - Keeps the main camera-aware architecture and data recipe, but trains as one stage instead of the two-stage freeze/unfreeze recipe.
   - No eval aggregate found in `log/` at the time this note was written.

6. GT-only camera-aware ablation:
   - Stage 1 job `9191192`: completed successfully and saved step `5000`.
   - Stage 2 job `9191195`: running on `torch_pr_769_tandon_advanced`; latest observed progress was about `9000/30000` with checkpoints at `5000` and `8000`.
   - Latest observed training breakdown confirms GT-only targets: `aux_gt_frac=1.0000`, `aux_pi3x_frac=0.0000`.
   - No GT-only eval has been run yet because the stage-2 final checkpoint is not available yet.

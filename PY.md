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

### Results

Two-run average, 1000 episodes per suite and 4000 episodes total:

| Suite | Baseline | Camera-aware |
| --- | ---: | ---: |
| LIBERO Object | 89.40% | 93.40% |
| LIBERO Goal | 87.40% | 88.40% |
| LIBERO Spatial | 85.20% | 88.60% |
| LIBERO 10 | 76.50% | 77.60% |
| Overall | 84.62% | 87.00% |


### Ablations

1. Camera-aware on Pi0.5 (`pi05=True`, otherwise identical to camera-aware):
   - Stage 1 config: `pi05_libero_cam_pytorch_prope_ray_view_distill_fullres_4suite_stage1`
   - Stage 2 config: `pi05_libero_cam_pytorch_prope_ray_view_distill_fullres_4suite_stage2`
   - Base weights: `$GEO_ROOT/pi05_base` (PyTorch). Convert from JAX with
     `examples/convert_jax_model_to_pytorch.py` if not present yet.

2. Turn OFF ray-embed (PRoPE + cross-view + aux point head still on):
   - Stage 1 config: `pi0_libero_cam_pytorch_prope_view_distill_fullres_4suite_stage1`
   - Stage 2 config: `pi0_libero_cam_pytorch_prope_view_distill_fullres_4suite_stage2`
   - Diff vs. camera-aware: `ray_enc_type=False`; `ray_embed` dropped from
     `trainable_prefixes` in Stage 1.

3. Turn OFF PRoPE (ray-embed + cross-view + aux point head still on):
   - Stage 1 config: `pi0_libero_cam_pytorch_ray_view_distill_fullres_4suite_stage1`
   - Stage 2 config: `pi0_libero_cam_pytorch_ray_view_distill_fullres_4suite_stage2`
   - Diff vs. camera-aware: `pose_enc_type="null"` and
     `cross_view.prope_layer_idx=()`.

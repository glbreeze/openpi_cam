

| Run                                                           | Geom modules | Distillation | Total steps |
|---------------------------------------------------------------|--------------|--------------|-------------|
| train_pi0_libero_baseline.sbatch                              | ❌           | ❌           | 30k         |
| train_pi0_libero_cam_v2_prope_ray_view.sbatch (new)           | ✅           | ❌           | 30k         |
| ..._distill_fullres_stage1.sbatch + ..._stage2.sbatch         | ✅           | ✅           | 5k + 30k    |
# Pi3X Camera-Aware Pi0 System Architecture and Results

Last updated: 2026-05-04.

This document summarizes the current Pi3/Pi3X series only: the camera-aware
changes made on top of Pi0, the end-to-end training/evaluation chain, and the
latest experiment results. Older non-Pi3 camera baselines are intentionally not
included.

## 1. System Goal

The current system turns Pi0 into a camera-aware policy for LIBERO by adding
geometry signals at the visual-token level while keeping the action head and
policy API close to the original Pi0 path.

The high-level hypothesis is:

1. LIBERO provides calibrated camera intrinsics/extrinsics for the agent and
   wrist views.
2. Pi0's frozen/pretrained visual-language-action backbone does not explicitly
   know camera geometry.
3. Injecting camera rays, camera poses, and Pi3X point-head distillation into the
   visual tokens should make the learned representation more 3D-aware.
4. A two-stage schedule should first teach the new geometry modules, then
   fine-tune the full policy for action success.

## 2. End-to-End Chain

```text
LIBERO raw / rendered data
  -> LeRobot LIBERO camera dataset with RGB, intrinsics, extrinsics
  -> Pi3X teacher cache: xy, log_z, confidence per frame/view
  -> Pi0 training dataloader packs images, camera calibration, Pi3X targets
  -> Pi0 visual path injects rays, PRoPE pose, cross-view fusion
  -> auxiliary point head distills Pi3X targets from fused visual tokens
  -> Stage 1 trains new geometry/distillation modules only
  -> Stage 2 unfreezes full policy with small auxiliary geometry regularizer
  -> LIBERO policy inference passes calibrated camera intrinsics/extrinsics
  -> LIBERO-object / LIBERO-all evaluation
```

The current full 4-suite dataset is:

```text
/scratch/yp2841/cache/lerobot/glbreeze/libero_cam_v2
```

It contains all four LIBERO suites, 40 tasks, 1768 episodes, and 286537 frames.
The stored observations include 256x256 RGB images plus camera
intrinsics/extrinsics. It does not provide ground-truth metric depth or xyz, so
the current distillation target comes from Pi3X cache rather than GT depth.

## 3. Data And Policy Inputs

The LIBERO policy path uses two real views:

| Logical view | Source |
| --- | --- |
| `base_0_rgb` | agent / third-person camera |
| `left_wrist_0_rgb` | wrist camera |
| `right_wrist_0_rgb` | padded invalid slot |

The padded right-wrist slot is kept for model shape compatibility, but its mask
is invalid and its geometry contribution is zeroed.

The policy transform also carries camera calibration into training and
inference:

| Field | Purpose |
| --- | --- |
| `agent_extrinsic`, `wrist_extrinsic` | converted from MuJoCo convention to OpenCV convention |
| camera intrinsics `K` | adjusted for OpenPI image flip/resize preprocessing |
| `pi3x_target_xy`, `pi3x_target_logz`, `pi3x_target_conf` | optional Pi3X distillation targets |

Important alignment detail: the Pi3X cache is generated after matching OpenPI's
image preprocessing. The cache is therefore aligned with the actual image tokens
seen by SigLIP/vision tokens during training.

## 4. Pi0 Model Changes

The current Pi0 config adds the following geometry/distillation knobs:

| Config field | Meaning |
| --- | --- |
| `cross_view` | cross-view fusion block configuration |
| `aux_point_head` | Pi3X auxiliary point-head distillation head |
| `pose_enc_type` | currently `prope` for Pi3-style pose rotary encoding |
| `ray_enc_type` | enables camera-ray embedding |
| `view_enc_type` | optional learned view embedding, off in current Pi3 recipes |
| `disable_geometric_augs` | disables image geometric augmentations that would break cache alignment |
| `action_loss_weight` | lets Stage 1 downweight action loss while learning geometry modules |
| `ray_embed_pi3x_init_path` | optional Pi3X warm-start weights for ray embedding |
| `ray_embed_pi3x_init_scale` | scale for imported Pi3X ray weights |

The main architecture additions are:

| Module | Current role |
| --- | --- |
| `ray_embed` | Computes pixel rays from `K^-1 [u, v, 1]` and injects them into visual patch embeddings before SigLIP. |
| `PoseInjectBlock` / PRoPE | Uses OpenCV camera-to-world poses, inverted to world-to-camera for Pi3-style positional geometry. |
| `cross_view_fusion` | Fuses visual tokens across the two valid views using frame/global attention blocks. |
| `aux_point_head` | Predicts Pi3X-style `xy` and `log_z` from fused tokens for auxiliary supervision. |
| view masking | Prevents padded or invalid views from leaking geometry features or gradients. |

The base current recipe uses `aa_order="fg"` and `prope_layer_idx=(0,)`. The
deeper FGFG variant uses `aa_order="fgfg"` and `prope_layer_idx=(0, 1)`.

## 5. Pi3X Distillation Loss

Pi3X targets are cached per episode/view as:

| Key | Shape for full-res cache | Meaning |
| --- | --- | --- |
| `xy` | `(T, 224, 224, 2)` | Pi3X point-head image-plane coordinates |
| `log_z` | `(T, 224, 224, 1)` | Pi3X log-depth-like normalized output |
| `conf` | `(T, 224, 224, 1)` | Pi3X confidence logits |

These are Pi3X normalized teacher outputs, not metric ground-truth xyz/depth.

The current loss is soft confidence weighted:

```text
w_pix = sigmoid(conf_target) * valid_view_mask
aux_xy_loss = weighted MSE(xy_pred, xy_target)
aux_z_loss  = weighted MSE(logz_pred, logz_target)
aux_loss    = aux_loss_weight * (aux_xy_loss + aux_z_loss)
loss        = action_loss_weight * action_loss_raw + aux_loss
```

This replaced the earlier hard-gating version that used a threshold such as
`sigmoid(conf) > 0.1`. The latest full 4-suite Stage 1 run is therefore
`fg + soft confidence weighting`.

## 6. Current Training Recipes

| Recipe | Config | Key settings |
| --- | --- | --- |
| Stage 1 FG zero-init | `pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage1` | `fg`, PRoPE layer 0, ray embed enabled, new modules only, `action_loss_weight=0.1`, aux weight 1.0, 5k steps |
| Stage 1 FG Pi3X-ray | `pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage1_pi3xray` | Stage 1 plus `assets/pi3x_init/ray_embed.pt` warm start |
| Stage 1 FGFG Pi3X-ray | `pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage1_fgfg_pi3xray` | deeper `fgfg`, PRoPE layers 0 and 1, Pi3X ray init |
| Stage 2 FG | `pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage2` | full unfreeze, action weight 1.0, aux weight 0.05, 30k steps, initialized from Stage 1 |
| Stage 2 FGFG | `pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage2_fgfg` | deeper `fgfg`, full unfreeze, aux weight 0.05, initialized from FGFG Stage 1 |
| Zero-new-modules ablation | `pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage2_zero_new_modules` | tests whether gains remain when the new camera/Pi3X modules are disabled/zeroed |

The intended curriculum is:

1. Stage 1: train only `cross_view_fusion`, `ray_embed`, and `aux_point_head`.
2. Stage 2: initialize from Stage 1, unfreeze the full policy, restore full
   action loss, and keep Pi3X aux loss as a weak regularizer.

## 7. Training Runs

### LIBERO-object camera v3

| Job | Recipe | Dataset | Hardware | Status | Runtime | Final summary |
| --- | --- | --- | --- | --- | --- | --- |
| `7203317` | Stage 1 FG zero-init | `libero_object_cam_v3` | 4x L40S, batch 32 | completed | 03:14:02 | last-5 avg: action loss 0.0173, raw action 0.1730, aux 0.0936 |
| `7224086` | Stage 1 FG Pi3X-ray | `libero_object_cam_v3` | 4x L40S, batch 32 | completed | 03:08:35 | action 0.01726, raw action 0.17259, aux 0.08726, 1.90s/step |
| `7522058` | Stage 1 FGFG Pi3X-ray | `libero_object_cam_v3` | 4x L40S, batch 32 | completed | 03:19:25 | action 0.01715, raw action 0.17152, aux 0.10404, 1.96s/step |
| `7256288` | Stage 2 FG zero-init resume | `libero_object_cam_v3` | 4x H200, batch 32 | completed | 03:05:18 | action 0.0127, aux 0.00099, 0.74s/step |
| `7790605` | Stage 2 FGFG | `libero_object_cam_v3` | 2x H200, batch 16 | completed | 06:44:23 | action 0.0218, aux 0.00085, 0.71s/step |

### Full 4-suite LIBERO camera v2

| Job | Recipe | Dataset | Hardware | Status | Runtime | Final summary |
| --- | --- | --- | --- | --- | --- | --- |
| `7891401` | Stage 1 FG zero-init, soft weighting | `libero_cam_v2` | 1x A100, batch 16 | completed | 03:17:11 | action 0.01602, raw action 0.16021, aux 0.13319, 1.91s/step |
| `7891407` | dependent Stage 2 FG | `libero_cam_v2` | requested Tandon GPU | cancelled | 00:00:00 | no Stage 2 result yet |

The full 4-suite Stage 1 checkpoint exists at:

```text
checkpoints/pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage1/pi0_libero_cam_v2_prope_ray_view_distill_fullres_twostage_v1_stage1_a100_1gpu_b16/5000
```

The current blocker for full 4-suite results is not Stage 1. Stage 1 completed
normally. The missing piece is resubmitting Stage 2 from this checkpoint.

## 8. LIBERO-object Evaluation Results

All results below use the new Pi3/Pi3X camera-aware series on
`libero_object_cam_v3`, with 10 tasks and 50 trials per task, unless otherwise
noted.

### Stage 2 FG zero-init checkpoint sweep

| Checkpoint | Successes | Success rate | Aggregate file |
| --- | ---: | ---: | --- |
| 5k | 383 / 500 | 0.766 | `log/libero_object_eval/pi0_cam_v3_fr_s2_zero_5k_eval50_l40s/summary/aggregate_5000_7488906.json` |
| 10k | 425 / 500 | 0.850 | `log/libero_object_eval/pi0_cam_v3_fr_s2_zero_10k_eval50_l40s/summary/aggregate_10000_7488908.json` |
| 15k | 449 / 500 | 0.898 | `log/libero_object_eval/pi0_cam_v3_fr_s2_zero_15k_eval50_l40s/summary/aggregate_15000_7488907.json` |
| 20k | 452 / 500 | 0.904 | `log/libero_object_eval/pi0_cam_v3_fr_s2_zero_20k_eval50_l40s/summary/aggregate_20000_7488909.json` |
| 25k | 462 / 500 | 0.924 | `log/libero_object_eval/pi0_cam_v3_fr_s2_zero_25k_eval50_l40s/summary/aggregate_25000_7369651.json` |
| 30k | 466 / 500 | 0.932 | `log/libero_object_eval/pi0_cam_v3_fr_s2_zero_30k_eval50_a100/summary/aggregate_30000_7313465.json` |

Best observed result in this sweep: 30k, 93.2%.

### Zero-new-modules ablation

| Variant | Checkpoint | Successes | Success rate | Aggregate file |
| --- | ---: | ---: | ---: | --- |
| zero-new-modules, L40S eval | 30k | 408 / 500 | 0.816 | `log/libero_object_eval/pi0_cam_v3_fr_s2_30k_zero_newmods_eval50_l40s/summary/aggregate_30000_7390108.json` |
| zero-new-modules, A100 eval | 30k | 396 / 500 | 0.792 | `log/libero_object_eval/pi0_cam_v3_fr_s2_30k_zero_newmods_eval50_a100/summary/aggregate_30000_7390212.json` |

This ablation is important: disabling/zeroing the new geometry modules drops the
30k result from 93.2% to roughly 79-82%. That suggests the gain is not only from
backbone fine-tuning.

### Stage 2 FGFG results

| Run | Checkpoint | Successes | Success rate | Aggregate file |
| --- | ---: | ---: | ---: | --- |
| FGFG single fast eval | 30k | 450 / 500 | 0.900 | `log/libero_object_eval/pi0_cam_v3_fr_s2_fgfg_2gpu_b16_30k_eval50_h100_fast/summary/aggregate_30000_7816296.json` |
| FGFG sweep | 5k | 373 / 500 | 0.746 | `log/libero_object_eval/ckpt_sweep/pi0_cam_v3_fr_s2_fgfg_2gpu_b16_eval50_h100_sweep_serial/summary/aggregate_5000_7821559.json` |
| FGFG sweep | 10k | 416 / 500 | 0.832 | `log/libero_object_eval/ckpt_sweep/pi0_cam_v3_fr_s2_fgfg_2gpu_b16_eval50_h100_sweep_serial/summary/aggregate_10000_7821559.json` |
| FGFG sweep | 15k | 456 / 500 | 0.912 | `log/libero_object_eval/ckpt_sweep/pi0_cam_v3_fr_s2_fgfg_2gpu_b16_eval50_h100_sweep_serial/summary/aggregate_15000_7821559.json` |
| FGFG sweep | 20k | 459 / 500 | 0.918 | `log/libero_object_eval/ckpt_sweep/pi0_cam_v3_fr_s2_fgfg_2gpu_b16_eval50_h100_sweep_serial/summary/aggregate_20000_7821559.json` |
| FGFG sweep | 25k | 448 / 500 | 0.896 | `log/libero_object_eval/ckpt_sweep/pi0_cam_v3_fr_s2_fgfg_2gpu_b16_eval50_h100_sweep_serial/summary/aggregate_25000_7821559.json` |
| FGFG sweep | 30k | 444 / 500 | 0.888 | `log/libero_object_eval/ckpt_sweep/pi0_cam_v3_fr_s2_fgfg_2gpu_b16_eval50_h100_sweep_serial/summary/aggregate_30000_7821559.json` |

Best observed FGFG result: 20k, 91.8%. It does not beat the simpler FG Stage 2
30k result at 93.2%.

## 9. Current Interpretation

The strongest result so far is the simpler FG Stage 2 zero-init run at 30k,
466/500 = 93.2% on LIBERO-object. The zero-new-modules ablation is much worse,
so the camera/Pi3X geometry path is doing useful work.

The deeper FGFG variant is not currently better. Its best sweep point is 91.8%
at 20k, and later checkpoints regress. Given that FGFG also used a smaller
2-GPU/batch-16 Stage 2 run, the result is not a pure architecture comparison,
but there is no evidence yet that FGFG is the better default.

The full 4-suite `libero_cam_v2` pipeline looks structurally correct:

| Component | Status |
| --- | --- |
| 4-suite dataset | present and metadata-consistent |
| norm stats | generated under scratch OpenPI assets |
| Pi3X full-res cache | present for both agent and wrist views |
| Stage 1 | completed on 1x A100 |
| Stage 2 | not completed yet |
| downstream 4-suite eval | not available yet |

The next high-value action is to launch full 4-suite Stage 2 from the completed
v2 Stage 1 checkpoint, preferably on 4 GPUs when the scheduler allows it.

## 10. Practical Notes

1. Do not mix Pi3X caches across rebuilt LeRobot datasets unless episode/frame
   indexing is known to be identical.
2. Keep `disable_geometric_augs=True` for these recipes; random crops/rotations
   would invalidate the cached teacher-target alignment.
3. The current `libero_cam_v2` dataset does not include metric GT depth/xyz, so
   a 50/50 mix of Pi3X cache and GT xyz is not directly available from the
   current data without adding a depth/xyz generation path.
4. For the latest recipe, `fg + soft confidence weighting` is the active Stage 1
   configuration.
5. Based on current evidence, the default branch for continuing full 4-suite
   training should be FG rather than FGFG.

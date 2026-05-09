# RoboTwin Cam-Aware Code Patches (still TODO)

This is the punch list of in-repo code edits needed to wire RoboTwin2.0 into
the existing cam-aware + Pi3X mixed-GT distillation pipeline. The data
preprocessing (`convert_robotwin_cam_to_lerobot.py`, `cache_robotwin_gt_point_targets.py`,
the task-config yaml) and the runbook are already in this repo. The items
below are the remaining model/config edits.

The current `pi0_pytorch.py` already has a placeholder slot for `right_wrist`
extrinsics (line 484: `cam_pos = {"base": ..., "left_wrist": ..., "right_wrist": None}`).
We need to plumb a real value through.

## 1. `src/openpi/models/model.py` — extend `Observation`

Add two fields next to the existing `wrist_extrinsic`/`wrist_intrinsic`:

```python
right_wrist_extrinsic: at.Float[ArrayT, "*b 4 4"] | None = None
right_wrist_intrinsic: at.Float[ArrayT, "*b 3 3"] | None = None
```

Plumb them through:
- `Observation.from_dict` (line ~146): `right_wrist_extrinsic=data.get("right_wrist_extrinsic")`, etc.
- `preprocess_observation` (line ~240): pass through unchanged.

## 2. `src/openpi/models_pytorch/preprocessing_pytorch.py`

Update the camera field map:

```python
_IMAGE_TO_CAMERA_FIELDS = {
    "base_0_rgb": ("agent_intrinsic", "agent_extrinsic"),
    "left_wrist_0_rgb": ("wrist_intrinsic", "wrist_extrinsic"),
    "right_wrist_0_rgb": ("right_wrist_intrinsic", "right_wrist_extrinsic"),
}
```

In the resize / pixel-transform pass, ensure the `right_wrist` K is scaled the
same way as the others. Also emit `right_wrist_intrinsic` / `right_wrist_extrinsic`
in `updated_extrinsics` / `updated_intrinsics`.

## 3. `src/openpi/models_pytorch/pi0_pytorch.py:484`

```python
cam_pos = {
    "base": obs.agent_extrinsic,
    "left_wrist": obs.wrist_extrinsic,
    "right_wrist": obs.right_wrist_extrinsic,
}
cam_intr = {
    "base": obs.agent_intrinsic,
    "left_wrist": obs.wrist_intrinsic,
    "right_wrist": obs.right_wrist_intrinsic,
}
```

## 4. `src/openpi/policies/robotwin_policy.py` — add `RobotwinCamInputs`

Mirror `LiberoInputs`'s extrinsic/intrinsic plumbing, but for 3 cams. Key
mapping (RoboTwin LeRobot key -> openpi model field):

| LeRobot key | Model field |
|---|---|
| `observation.cam_high_extrinsic` | `agent_extrinsic` |
| `observation.cam_high_intrinsic` | `agent_intrinsic` |
| `observation.cam_left_wrist_extrinsic` | `wrist_extrinsic` |
| `observation.cam_left_wrist_intrinsic` | `wrist_intrinsic` |
| `observation.cam_right_wrist_extrinsic` | `right_wrist_extrinsic` |
| `observation.cam_right_wrist_intrinsic` | `right_wrist_intrinsic` |

The converter in `convert_robotwin_cam_to_lerobot.py` already writes camera-to-world
in OpenCV camera frame, so no MuJoCo-style flip is needed; the only adjustment
is `_adjust_K_for_openpi_image_flip` (`fx -> -fx`) on each K to match openpi's
flipped image preprocessing.

## 5. `src/openpi/training/config.py` — add data config + train configs

Add a `LeRobotRobotwinCamDataConfig` next to `LeRobotLiberoDataConfig`. It should
mirror the LIBERO version: `include_cam_extrinsics`, `pi3x_targets_root`,
`gt_point_targets_root`, `point_target_gt_ratio`, `point_target_mix_mode`. The
repack structure should map:

```python
{
    "observation/cam_high":         "observation.images.cam_high",
    "observation/cam_left_wrist":   "observation.images.cam_left_wrist",
    "observation/cam_right_wrist":  "observation.images.cam_right_wrist",
    "observation/state":            "observation.state",
    "actions":                      "action",
    "prompt":                       "task",
    # if include_cam_extrinsics:
    "observation/cam_high_extrinsic":         "observation.cam_high_extrinsic",
    "observation/cam_left_wrist_extrinsic":   "observation.cam_left_wrist_extrinsic",
    "observation/cam_right_wrist_extrinsic":  "observation.cam_right_wrist_extrinsic",
    "observation/cam_high_intrinsic":         "observation.cam_high_intrinsic",
    "observation/cam_left_wrist_intrinsic":   "observation.cam_left_wrist_intrinsic",
    "observation/cam_right_wrist_intrinsic":  "observation.cam_right_wrist_intrinsic",
}
```

Add 4 new `TrainConfig` entries (mirror the `pi0_libero_cam_pytorch_prope_ray_view*`
chain):

1. `pi0_robotwin_cam_prope_ray_view`                                  — bare cam-aware
2. `pi0_robotwin_cam_prope_ray_view_distill_fullres_stage1_gtdual`    — stage 1
3. `pi0_robotwin_cam_prope_ray_view_distill_fullres_stage2_gtdual`    — stage 2
4. `pi0_robotwin_cam_baseline`                                        — baseline (no PRoPE/ray)

For all of these set `model.action_dim=14, action_horizon=50`. Stage 1 uses
`trainable_prefixes=("cross_view_fusion", "ray_embed", "aux_point_head")`,
`action_loss_weight=0.1`, `aux_point_head.loss_weight=1.0`. Stage 2 unfreezes
everything, `action_loss_weight=1.0`, `aux_point_head.loss_weight=0.05`.

## 6. `scripts/cache_pi3x_targets.py` — accept 3-cam spec

The current `CAM_SPECS` constant lists 2 cams. Refactor it to read a CLI
`--cam-spec head:cam_high,left_wrist:cam_left_wrist,right_wrist:cam_right_wrist`
(parsing identical to the converter's `--cam-map`). The teacher forward already
runs per-cam in a loop, so this is a config-only change.

## 7. New sbatch scripts

Mirror the existing LIBERO sbatch scripts, swapping in the new config names
and dataset path:

```
scripts/sbatch/
    collect_robotwin_camaware_l40s.sbatch
    convert_robotwin_cam_to_lerobot_cpu.sbatch
    cache_pi3x_robotwin_cam_l40s.sbatch
    cache_robotwin_gt_point_targets_cpu.sbatch
    compute_norm_stats_robotwin_cam.sbatch
    train_pi0_robotwin_cam_distill_stage1_gtdual.sbatch
    train_pi0_robotwin_cam_distill_stage2_gtdual.sbatch
```

## Suggested commit/PR order

1. (small) policies + config schema: items 1-4 above.
2. (small) `cache_pi3x_targets.py` 3-cam refactor (item 6).
3. (medium) item 5: data + train configs.
4. (small) sbatch scripts.

After (1) compiles + `pytest src/openpi/models/` passes, run a smoke train on
the `pi0_robotwin_cam_prope_ray_view` config (no distillation) against a tiny
single-task LeRobot repo to validate the cam-aware plumbing end to end.

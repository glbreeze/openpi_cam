# Pi3X-distilled training for `pi0_libero_cam_pytorch_prope_ray_view`

This recipe trains the camera-aware `pi0_libero_cam_pytorch_prope_ray_view` config
with an auxiliary `PointHead` supervised by the [Pi3X](https://huggingface.co/yyfz233/Pi3X)
teacher at SigLIP patch resolution (16×16). The goal is to push geometry-aware
features into the SigLIP backbone via a cheap distillation head, without changing
the action-prediction objective.

## Pipeline overview

```
LIBERO HDF5  ──Phase 1──▶  rendered HDF5 with extrinsics/intrinsics  ──Phase 2──▶  LeRobot dataset
                                                                                        │
                                                                                        ▼
                                                                  scripts/cache_pi3x_targets.py
                                                                                        │
                                                                                        ▼
                                              ~/.cache/openpi/pi3x_targets/<dataset>/{agent,wrist}/episode_NNNNNN.npz
                                                                                        │
                                                                                        ▼
            scripts/compute_norm_stats.py  ──▶  scripts/train_pytorch.py  pi0_libero_cam_pytorch_prope_ray_view_distill
```

Each cached `.npz` holds three fp16 arrays at the model's exact patch grid:

| key      | shape           | meaning                                              |
|----------|-----------------|------------------------------------------------------|
| `xy`     | `(T, 16, 16, 2)`| direction, matches Pi3X point head output before exp |
| `log_z`  | `(T, 16, 16, 1)`| log depth, matches Pi3X point head output before exp |
| `conf`   | `(T, 16, 16, 1)`| confidence logits (pre-sigmoid)                      |

The teacher cache mirrors openpi's image preprocessing exactly (`[::-1, ::-1]` flip,
256→224 isotropic resize, `fx → -fx`, K scaled by 224/256), so teacher patch
features are pixel-aligned with the SigLIP patch grid that the student backbone
sees at training time.

## One-time setup

### 1. Convert the LIBERO dataset (LeRobot format)

```bash
# Phase 1 (libero conda env) renders HDF5 with extrinsics/intrinsics; Phase 2 converts to LeRobot.
REPO_ID=glbreeze/libero_object_cam_v3  bash scripts/regen_libero_object.sh
# If Phase 1 outputs already exist:
REPO_ID=glbreeze/libero_object_cam_v3  SKIP_PHASE1=1  bash scripts/regen_libero_object.sh
```

If you already have the dataset and only need the v2.1 LeRobot format without the
binary `obs.pcd`/`obs.conf` blobs that older converters wrote, regenerate with a
fresh `REPO_ID` and never overwrite a directory that another process may still be
appending to.

### 2. Cache Pi3X teacher targets

```bash
uv run scripts/cache_pi3x_targets.py \
    --data-root   ~/.cache/huggingface/lerobot/glbreeze/libero_object_cam_v3 \
    --output-root ~/.cache/openpi/pi3x_targets/libero_object_cam_v3 \
    --batch-size 32
```

- Walks every `episode_*.parquet` under `<data-root>/data/chunk-*/`.
- Defaults: ImageNet normalization (handled internally by Pi3X), bf16 autocast on
  Ampere+ GPUs, batch size 32.
- ~22 min for 460 LIBERO-Object episodes on a single 3090/4090. Resumable via
  per-episode skip-existing.

### 3. Compute action norm stats

```bash
uv run scripts/compute_norm_stats.py \
    --config-name pi0_libero_cam_pytorch_prope_ray_view_distill
```

Writes to `assets/<assets_dir>/<asset_id>/norm_stats.json`.

## Training

```bash
uv run scripts/train_pytorch.py \
    pi0_libero_cam_pytorch_prope_ray_view_distill \
    --exp_name pi3xd_w0p05_seed0
```

For multi-GPU DDP:

```bash
uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_pytorch.py \
    pi0_libero_cam_pytorch_prope_ray_view_distill \
    --exp_name pi3xd_w0p05_seed0
```

Resume with `--resume`.

## Loss-weight calibration (the most important knob)

`PointHead` is zero-initialized at `linear_out`, so at step 0:

- `xy_pred ≈ 0`, `z_pred ≈ 0`
- Empirically on `libero_object_cam_v3` (single-row sanity, 91% conf-mask
  coverage): `xy_loss ≈ 0.097`, `z_loss ≈ 6.32`. Without weighting that's
  **~6.4 per supervised view**. With `loss_weight=1.0` this dwarfs the action
  MSE (typically `~0.05`–`0.5` early in training).

**The recipe defaults `aux_point_head.loss_weight = 0.05`**, which makes the aux
contribution `~0.32` at step 0 — the same ballpark as the action MSE so neither
signal drowns out the other. Sweep `{0.0, 0.01, 0.05, 0.2}` on short 5k-step
runs to characterize. The `0.0` case equals the no-distill baseline and is a
useful sanity check that the head genuinely doesn't leak gradient when off.

To change the weight, edit the new training config:

```python
# src/openpi/training/config.py — the `_distill` entry
aux_point_head=point_head_config.AuxPointHeadConfig(enabled=True, loss_weight=0.05),
```

## Schedule

Mirrors the no-distill sibling so direct comparisons are clean:

| knob              | value                                                           |
|-------------------|-----------------------------------------------------------------|
| `num_train_steps` | `30_000`                                                        |
| optimizer / LR    | openpi defaults — PointHead joins the same param group          |
| batch size        | trainer auto-derives from GPU memory (typically 32 / GPU)       |
| dtype             | `bfloat16` autocast on the backbone; PointHead runs in `float32`|
| warmup            | inherited from the base recipe                                  |
| augmentations     | `disable_geometric_augs=True` (rays would otherwise mismatch)   |

## Comparison protocol (recommended)

Train both the no-distill baseline and the distill version with the **same seed,
same `num_train_steps`, same dataset**:

```bash
# Baseline (no aux head)
uv run scripts/train_pytorch.py pi0_libero_cam_pytorch_prope_ray_view --exp_name baseline_seed0

# Distillation (current recipe)
uv run scripts/train_pytorch.py pi0_libero_cam_pytorch_prope_ray_view_distill --exp_name pi3xd_w0p05_seed0
```

Compare:

- **action MSE curves** — distill should not regress action loss; if it does, drop
  `loss_weight`.
- **aux loss curve** — should decrease monotonically; flat curves mean the weight
  is too small.
- **LIBERO eval success rate** — the only signal that actually matters.

## What's wired (file map)

| file                                                    | role                                                                 |
|---------------------------------------------------------|----------------------------------------------------------------------|
| `scripts/cache_pi3x_targets.py`                         | one-shot teacher cache                                               |
| `src/openpi/models/point_head_config.py`                | `AuxPointHeadConfig(enabled, loss_weight, ...)`                      |
| `src/openpi/models_pytorch/layers/point_head.py`        | `PointHead` module (zero-init output)                                |
| `src/openpi/models/model.py`                            | `Observation.pi3x_target_xy/logz/conf` optional fields               |
| `src/openpi/models_pytorch/preprocessing_pytorch.py`    | propagates targets through `preprocess_observation_pytorch`          |
| `src/openpi/policies/libero_policy.py`                  | `Pi3xLiberoTargetLoader` transform; `LiberoInputs` pass-through      |
| `src/openpi/training/config.py`                         | `LeRobotLiberoDataConfig.pi3x_targets_root`; `_distill` train config |
| `src/openpi/models_pytorch/pi0_pytorch.py`              | masked MSE distillation loss in `forward`                            |

## Loss math

For each batch:

```
xy_pred, z_pred = aux_point_head(fused_tokens)             # (B, V_model, P, 2/1)
mask = (sigmoid(conf_target) > 0.1).float()                # Pi3X demo gating
mask = mask * image_mask[:, :V_target, None, None]         # drop padded views
denom = mask.sum().clamp_min(1.0)
xy_loss = ((xy_pred[:,:V_t] - xy_target)**2 * mask).sum() / denom / 2   # /channels
 z_loss = (( z_pred[:,:V_t] - logz_target)**2 * mask).sum() / denom
loss   += aux_point_head.loss_weight * (xy_loss + z_loss)
```

`V_target = 2` for LIBERO (agent + wrist; the padded `right_wrist` view has no
teacher target). The student predicts for all 3 views; only the first 2 are
supervised, the rest contributes nothing to the aux loss.

## Caveats

- The `PointHead` predicts in Pi3X's *normalized* coordinate system (rays in
  image-plane units; log-z in Pi3X's depth-normalization). It is *not* metric.
  The aux loss is a feature-shaping signal, not a useful 3D estimator on its own.
- Pi3X targets were cached with the dataset's stored intrinsics. If you change
  the dataset (different K, different image size), regenerate the cache.
- The cache is keyed by `(episode_index, frame_index)`. If the LeRobot dataset is
  rebuilt with a different episode order, the cache becomes invalid even if
  visual content matches.
- `disable_geometric_augs=True` is required: the random crop / rotate augs
  would shift pixels relative to the cached teacher rays, breaking alignment.

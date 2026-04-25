# HPC run guide — Pi3X-distilled `pi0_libero_cam_pytorch_prope_ray_view`

End-to-end workflow on HPC, assuming you've already moved the converted LIBERO
LeRobot dataset there and cloned `Pi3X_Libero`.

## 1. Pick scratch paths and set env

Put bulky things on fast scratch, not `$HOME`. Three env vars do most of the
work — set them in your `sbatch` wrapper or `~/.bashrc`:

- `OPENPI_GEO_ROOT` — parent of `pi0_base/` and `pi0_libero/` (assets dir).
- `HF_LEROBOT_HOME` — LeRobot dataset root.
- `HF_HUB_CACHE` — HuggingFace download cache (Pi3X teacher weights live here).

The training config has `pi3x_targets_root` hard-coded to
`~/.cache/openpi/pi3x_targets/{libero_object_cam_v3,...}`. Either edit the
config, or symlink that path to wherever you actually wrote the cache. Symlink
is more portable.

## 2. Place artifacts on HPC

| artifact | size | where |
|---|---|---|
| openpi repo (`py-torch` branch) | git | `git clone` |
| `Pi3X_Libero` repo | small | already cloned |
| pi0_base PyTorch ckpt | 6.6 GB | `$OPENPI_GEO_ROOT/pi0_base/{config.json, model.safetensors}` |
| LeRobot dataset (`libero_object_cam_v3`) | 9.1 GB | `$HF_LEROBOT_HOME/glbreeze/libero_object_cam_v3/` |
| Pi3X teacher weights | 5.4 GB | download via `hf download yyfz233/Pi3X` (one-time) |

## 3. Install env (once)

```
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

If `hf_transfer` stalls, fall back with `HF_HUB_ENABLE_HF_TRANSFER=0`.

## 4. Cache Pi3X targets at 224×224

Pick a scratch path with **≥ 60 GB free** for the cache. Run:

```
uv run scripts/cache_pi3x_targets.py \
    --data-root <lerobot dataset path> \
    --output-root <scratch>/pi3x_targets_224/libero_object_cam_v3 \
    --pi3x-repo  <Pi3X_Libero path> \
    --output-resolution 224 \
    --batch-size 16
```

Expect ~30 min on a single A100, ~52 GB written. Resumable.

Symlink so the training config's hard-coded path resolves:

```
ln -s <scratch>/pi3x_targets_224/libero_object_cam_v3 \
      ~/.cache/openpi/pi3x_targets_224/libero_object_cam_v3
```

(For the 16×16 variant config, replace `pi3x_targets_224` with `pi3x_targets`
on both sides.)

## 5. Norm stats (one-time, ~2 min)

```
uv run scripts/compute_norm_stats.py \
    --config-name=pi0_libero_cam_pytorch_prope_ray_view_distill_fullres
```

The output lands under the run's `assets_dirs` but the loader looks under
`$OPENPI_GEO_ROOT/pi0_libero/glbreeze/libero_object_cam_v3/`. Mirror the file
once — it's a couple of KB.

## 6. Smoke-test before the real run (3 steps, batch 4)

```
WANDB_MODE=disabled uv run python scripts/train_pytorch.py \
    pi0_libero_cam_pytorch_prope_ray_view_distill_fullres \
    --exp_name __smoke__ --num_train_steps=3 --batch_size=4 --num_workers=0 --overwrite
```

Pass criteria: loss decreases over 3 steps, no OOM, checkpoint at step 3.

## 7. Real training

Single GPU:

```
uv run python scripts/train_pytorch.py \
    pi0_libero_cam_pytorch_prope_ray_view_distill_fullres \
    --exp_name pi3xd_fullres_w0p05_seed0
```

Multi-GPU DDP:

```
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<N> \
    scripts/train_pytorch.py \
    pi0_libero_cam_pytorch_prope_ray_view_distill_fullres \
    --exp_name pi3xd_fullres_w0p05_seed0
```

With `--resume` to continue from the latest checkpoint.

## Tips

- **`OPENPI_GEO_ROOT` is required.** Without it, the training config resolves
  `pytorch_weight_path` and `assets.assets_dir` to paths that don't exist on
  HPC. Set it once in your `sbatch` wrapper.
- **Mismatch between `compute_norm_stats.py` write path and loader read
  path.** `compute_norm_stats.py` writes to `<config.assets_dirs>/<repo_id>/`
  but the loader (when an `assets.assets_dir` override exists) reads from
  `<assets.assets_dir>/<asset_id>/`. Mirror the file once after compute.
- **Cache disk reality.** 224×224 cache is ~52 GB; 16×16 is 268 MB. Pick the
  config name that matches whichever cache you've prepared (`_distill` for
  16×16, `_distill_fullres` for 224×224).
- **Memory at `batch_size=4` on the 16×16 path** uses ~33 GB peak (gradient
  checkpointing on). Full-res aux head adds +1.5M params and a big intermediate
  feature map. Tune `--batch_size` accordingly; use DDP for production.
- **Loss-weight calibration.** `loss_weight=0.05` is the calibrated default in
  both configs. If you change the resolution, the per-element MSE magnitude is
  similar (we mask + average per supervised element), so the same weight
  generally applies. Verify on smoke-test logs.
- **`disable_geometric_augs=True` is required** — random crop / rotate would
  shift pixels relative to the cached teacher rays.
- **Right-wrist view is unsupervised.** LIBERO has 3 image slots
  (`base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`); only the first two
  have teacher targets. The loss site slices `pred[:, :2]` automatically.
- **For the 16×16 variant**, change `--output-resolution 16` (or omit; 16 is
  the default) and use `pi0_libero_cam_pytorch_prope_ray_view_distill` as the
  config name.

## File map (what changed in this commit)

| file | role |
|---|---|
| `scripts/cache_pi3x_targets.py` | added `--output-resolution {16, 224}` flag |
| `src/openpi/models/point_head_config.py` | added `output_resolution: int = 16` field |
| `src/openpi/models_pytorch/layers/point_head.py` | added Pi3X-style ConvHead upsampler for full-res output |
| `src/openpi/models_pytorch/pi0_pytorch.py` | passes `output_resolution` to PointHead |
| `src/openpi/training/config.py` | added `pi0_libero_cam_pytorch_prope_ray_view_distill_fullres` config |

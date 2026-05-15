# RoboTwin2.0 Cam-Aware + Pi3X Mixed-GT Distillation Runbook

End-to-end recipe for running the cam-aware Pi0 + Pi3X mixed-GT distillation
training (the `pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage1_gtdual`
recipe) on **RoboTwin2.0**. The same architecture (PRoPE, ray_embed, AuxPointHead,
fgfg cross-view fusion) but on RoboTwin's bimanual 3-camera setup (head, left
wrist, right wrist) instead of LIBERO's agent + wrist setup.

This is the canonical reference. For HPC-specific tweaks see
[scripts/HPC_ROBOTWIN_CAM_RUNBOOK.md](../scripts/HPC_ROBOTWIN_CAM_RUNBOOK.md).

---

## 0. Layout

Recommended root: `${GEO_ROOT}` is the directory holding `openpi_cam/`,
`RoboTwin/`, `pi0_base/`, etc. Default is `/home/asus/Research/CamVLA`. The
training env auto-exports `HF_LEROBOT_HOME=${GEO_ROOT}` on `activate_env.sh`.

```
${GEO_ROOT}/
├── openpi_cam/                       # this repo (training)
├── RoboTwin/                         # official RoboTwin repo (sim, raw collection, eval)
├── Pi3X_Libero/                      # Pi3X teacher source (already in repo)
├── pi0_base/                         # local pi0 PyTorch checkpoint
├── pi0_libero/                       # asset roots (norm_stats live under here)
├── robotwin/                         # converted LeRobot v2.1 datasets
└── .cache/openpi/
    ├── pi3x_targets_224/<repo_id>/   # Pi3X teacher targets cache
    └── gt_point_targets_224/<repo_id>/   # Sapien-GT targets cache
```

---

## 1. Environments

Two envs, mirroring the LIBERO setup:

| Env | Python | Purpose |
|---|---|---|
| openpi venv (`openpi_cam/.venv/`) | 3.11 | training, conversion, norm stats |
| `robotwin` (conda) | 3.10 | RoboTwin sim, raw episode collection, GT-depth point cache |
| `pi3x` (conda) | 3.10 | Pi3X teacher cache (already set up) |

### 1.1. RoboTwin env (one-time)

```bash
conda create -n robotwin python=3.10 -y
conda activate robotwin

# Lightweight CUDA toolchain — only nvcc + a few dev libs (avoid the heavy full cuda-toolkit solve):
conda install -y -c "nvidia/label/cuda-12.1.0" \
    cuda-nvcc cuda-cudart-dev cuda-cccl \
    libcusparse-dev libcublas-dev libcusolver-dev libcurand-dev libcufft-dev cuda-libraries-dev

# RoboTwin pulls torch 2.4.1+cu121, sapien 3.0.0b1, mplib 0.2.1, etc.
cd ${GEO_ROOT}/RoboTwin
pip install -r script/requirements.txt

# pytorch3d compiled from source (~10–20 min):
export CUDA_HOME=$(conda info --base)/envs/robotwin
export TORCH_CUDA_ARCH_LIST="8.9"            # 4090 / 4090D 48G / 6000 Ada
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation

# Apply RoboTwin's sapien/mplib patches:
SAPIEN_LOC=$(pip show sapien | awk '/Location/{print $2}')/sapien
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "$SAPIEN_LOC/wrapper/urdf_loader.py"
MPLIB_LOC=$(pip show mplib | awk '/Location/{print $2}')/mplib
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' \
    "$MPLIB_LOC/planner.py"

# curobo from source (~minutes):
mkdir -p envs && cd envs
git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git
cd curobo && pip install -e . --no-build-isolation
cd ${GEO_ROOT}/RoboTwin

# === Required version pins (gotchas) ===
# warp-lang 1.4.x — newer 1.x removed `wp.torch` namespace which curobo v0.7.8 still uses.
pip install "warp-lang==1.4.2"

# setuptools<70 — sapien 3.0.0b1 imports pkg_resources which was dropped in 70.
pip install "setuptools<70"

# ffmpeg binary — RoboTwin's pkl→hdf5 step shells out to ffmpeg subprocess.
conda install -y -c conda-forge ffmpeg
```

Verify:

```bash
conda activate robotwin
python - <<'PY'
import torch
print(f"torch    : {torch.__version__}  cuda={torch.cuda.is_available()}  device={torch.cuda.get_device_name(0)}")
import sapien;  print(f"sapien   : {sapien.__version__}")
import mplib;   print(f"mplib    : {getattr(mplib, '__version__', '?')}")
import curobo;  print(f"curobo   : imported")
import pytorch3d._C; print(f"pytorch3d: {__import__('pytorch3d').__version__}")
PY
```

### 1.2. Asset download

Run once. ~16 GB unpacked under `RoboTwin/assets/{embodiments,objects,background_texture}`.

```bash
conda activate robotwin
cd ${GEO_ROOT}/RoboTwin
bash script/_download_assets.sh
```

---

## 2. Phase 1 — Re-collect raw episodes with depth + intrinsics + extrinsics

The official `demo_clean` task config does *not* dump depth. We need the
cam-aware variant that flips `data_type.depth: true`. This repo ships one:

```bash
cp ${GEO_ROOT}/openpi_cam/scripts/robotwin_task_configs/demo_clean_camaware.yml \
   ${GEO_ROOT}/RoboTwin/task_config/demo_clean_camaware.yml
```

This is `demo_clean.yml` with `data_type.depth: true`. Camera `intrinsic_cv`,
`extrinsic_cv`, and `cam2world_gl` are written **unconditionally** by
`cameras.get_config()` in `RoboTwin/envs/_base_task.py:447`, so flipping depth
is the only change needed.

Collect a single task (smoke):

```bash
conda activate robotwin
cd ${GEO_ROOT}/RoboTwin

TASK_NAME=beat_block_hammer
TASK_CONFIG=demo_clean_camaware
bash collect_data.sh $TASK_NAME $TASK_CONFIG 0     # GPU 0
```

This runs two phases inside `script/collect_data.py`:
1. Plan & validate seeds (50 successful episodes).
2. Replay each successful seed to write per-episode HDF5 + mp4.

Output:

```
RoboTwin/data/${TASK_NAME}/${TASK_CONFIG}/
├── data/episode{0..49}.hdf5         # ~117 frames each
├── instructions/episode{0..49}.json
├── video/episode{0..49}.mp4
└── seed.txt
```

Each HDF5 contains:

| Group | Field | Shape | Dtype | Notes |
|---|---|---|---|---|
| `/observation/<cam>` | `rgb` | (T,) | bytes | JPEG-encoded per frame |
| | `depth` | (T, 240, 320) | float64 | mm (`-z*1000`, Sapien convention) |
| | `intrinsic_cv` | (T, 3, 3) | float32 | OpenCV K |
| | `extrinsic_cv` | (T, 3, 4) | float32 | world→cam OpenCV |
| | `cam2world_gl` | (T, 4, 4) | float32 | OpenGL cam→world (debug) |
| `/joint_action` | `vector` | (T, 14) | float64 | `[L_arm(6); L_grip; R_arm(6); R_grip]` |
| `/joint_action` | `left_arm`, `left_gripper`, `right_arm`, `right_gripper` | per-arm |  | redundant with `vector` |
| `/endpose` | `{left,right}_endpose`, `{left,right}_gripper` |  |  | EE pose for reference |

Cameras present: `head_camera`, `left_camera`, `right_camera`, `front_camera`.
We use the first three (`front_camera` is ignored downstream).

---

## 3. Phase 2 — Convert raw HDF5 → LeRobot v2.1

Run from the **openpi venv** (LeRobot dependency is there).

```bash
source ${GEO_ROOT}/openpi_cam/scripts/env/activate_env.sh
cd ${GEO_ROOT}/openpi_cam

REPO_ID=robotwin/${TASK_NAME}_${TASK_CONFIG}_50
python scripts/convert_robotwin_cam_to_lerobot.py \
    --raw-dir ${GEO_ROOT}/RoboTwin/data/${TASK_NAME}/${TASK_CONFIG} \
    --repo-id $REPO_ID \
    --image-size 224 \
    --no-videos                          # see "video backend" gotcha below
```

What it does:
- Decodes each JPEG byte string → uint8 HxWx3, square-resizes 240×320 → 224×224.
- Inverts Sapien `extrinsic_cv` (world→cam) → camera-to-world (`T_wc`, OpenCV camera frame).
- Scales each `intrinsic_cv` (3,3) from src to 224×224.
- Maps `head_camera→cam_high`, `left_camera→cam_left_wrist`, `right_camera→cam_right_wrist`.

LeRobot v2.1 features written:

| Feature | Shape | Dtype |
|---|---|---|
| `observation.state` | (14,) | float32 |
| `action` | (14,) | float32 |
| `observation.images.{cam_high,cam_left_wrist,cam_right_wrist}` | (3, 224, 224) | image (PNG) |
| `observation.{cam_high,cam_left_wrist,cam_right_wrist}_extrinsic` | (4, 4) | float32 |
| `observation.{cam_high,cam_left_wrist,cam_right_wrist}_intrinsic` | (3, 3) | float32 |

Output: `${HF_LEROBOT_HOME}/${REPO_ID}/`. ~370 MB for 50 episodes × 5,737 frames.

> **Gotcha — video backend.** LeRobot v2.1 defaults to AV1 (`SVT-AV1`) when
> `use_videos=True`. The openpi venv's pyav couldn't decode RoboTwin's AV1
> output reliably. We bypass this with `--no-videos` (stores PNG image files
> directly). The data loader is also patched to pass `video_backend="pyav"` to
> `LeRobotDataset` so future video-backed datasets work without torchcodec
> system FFmpeg shared libs.

Verify:

```bash
python - <<'PY'
import json, pathlib
info = json.loads(pathlib.Path("${HF_LEROBOT_HOME}/${REPO_ID}/meta/info.json").read_text())
print("codebase_version:", info["codebase_version"])    # v2.1
print("total_episodes:  ", info["total_episodes"])      # 50
print("total_frames:    ", info["total_frames"])        # ~5700
print("features:", sorted(info["features"]))
PY
```

---

## 4. Phase 3 — Compute norm stats

Norm stats are needed once per dataset. The cam-aware-only config (no Pi3X/GT
loaders) is fastest:

```bash
source ${GEO_ROOT}/openpi_cam/.venv/bin/activate
cd ${GEO_ROOT}/openpi_cam
export HF_LEROBOT_HOME=${GEO_ROOT}
uv run scripts/compute_norm_stats.py --config-name pi0_robotwin_cam_prope_ray_view
```

`compute_norm_stats.py` writes to `assets/<config_name>/<repo_id>/norm_stats.json`
(controlled by `config.assets_dirs / data_config.repo_id`).

Training reads from `<data.assets.assets_dir>/<asset_id>/norm_stats.json`.
For our RoboTwin configs that's `${GEO_ROOT}/pi0_libero/<repo_id>/norm_stats.json`.

**Copy/symlink the file to the training-expected path:**

```bash
ASSET_ID="$REPO_ID"   # the same string e.g. robotwin/beat_block_hammer_demo_clean_camaware_50
mkdir -p ${GEO_ROOT}/pi0_libero/${ASSET_ID}
cp ${GEO_ROOT}/openpi_cam/assets/pi0_robotwin_cam_prope_ray_view/${REPO_ID}/norm_stats.json \
   ${GEO_ROOT}/pi0_libero/${ASSET_ID}/norm_stats.json
```

Sanity:

```bash
python -c "
import json, pathlib
p = pathlib.Path('${GEO_ROOT}/pi0_libero/${ASSET_ID}/norm_stats.json')
d = json.loads(p.read_text())
print('keys:', list(d['norm_stats'].keys()))
print('state mean[:3]:', d['norm_stats']['state']['mean'][:3])
"
```

---

## 5. Phase 4a — Pi3X teacher cache

Run from the **pi3x conda env**. Same script as LIBERO but pass a
`--cam-spec` matching RoboTwin's LeRobot keys.

```bash
conda activate pi3x

# pyarrow needs to be installed in pi3x env if not already.
pip show pyarrow >/dev/null || pip install pyarrow

cd ${GEO_ROOT}/openpi_cam
python scripts/cache_pi3x_targets.py \
    --data-root ${GEO_ROOT}/${REPO_ID} \
    --output-root ~/.cache/openpi/pi3x_targets_224/${REPO_ID/\//_} \
    --pi3x-repo  ${GEO_ROOT}/Pi3X_Libero \
    --src-hw 224 --target-hw 224 --output-resolution 224 \
    --batch-size 8 \
    --cam-spec 'agent:observation.images.cam_high:observation.cam_high_intrinsic,wrist:observation.images.cam_left_wrist:observation.cam_left_wrist_intrinsic'
```

Notes:
- `--cam-spec` defaults to the LIBERO 2-cam layout; pass the RoboTwin spec
  here (only `agent`/`wrist` because the dual-loss point loaders read
  `{root}/agent/` and `{root}/wrist/` — `right_wrist` gets no teacher signal,
  same as LIBERO's right_wrist behavior).
- Output: `{output_root}/{cam}/episode_{NNNNNN}.npz` containing
  `xy: (T, 224, 224, 2) fp16`, `log_z: (T, 224, 224, 1) fp16`,
  `conf: (T, 224, 224, 1) fp16`. ~85 MB / cam / episode-pair at 224.
- 50 episodes × 2 cams ≈ 4.3 GB total, ~3 min on a 4090.

---

## 5. Phase 4b — Sapien-GT point cache

Run from the **robotwin conda env** (only h5py + numpy + torch are needed).

```bash
conda activate robotwin
cd ${GEO_ROOT}/openpi_cam

python scripts/cache_robotwin_gt_point_targets.py \
    --raw-dir ${GEO_ROOT}/RoboTwin/data/${TASK_NAME}/${TASK_CONFIG} \
    --output-root ~/.cache/openpi/gt_point_targets_224/${REPO_ID/\//_} \
    --output-resolution 224 \
    --target-resolution 224
```

Reads each raw RoboTwin HDF5's `/observation/<cam>/{depth, intrinsic_cv}`,
applies the openpi flipped-image preprocessing (`[::-1,::-1]` + `fx → -fx`),
square-resizes 240×320 → 224×224, and writes the same `(xy, log_z, conf)` fp16
NPZ layout that the dual-loss loader consumes.

Default cam map:
`head_camera → cam_high`, `left_camera → cam_left_wrist`, `right_camera → cam_right_wrist`.

> **Rename to match the loader.** `MixedPointTargetLoader` /
> `DualPointTargetLoader` look up `{root}/agent/` and `{root}/wrist/` (the
> LIBERO names). Rename after caching:
>
> ```bash
> CACHE=~/.cache/openpi/gt_point_targets_224/${REPO_ID/\//_}
> mv $CACHE/cam_high          $CACHE/agent
> mv $CACHE/cam_left_wrist    $CACHE/wrist
> mv $CACHE/cam_right_wrist   $CACHE/cam_right_wrist_unused
> ```

50 episodes × 2 cams ≈ 1.6 GB, ~1 min.

---

## 6. Sanity checks

Final pre-flight before training. From the openpi venv:

```python
# verify_pipeline.py
import os, dataclasses
os.environ['HF_LEROBOT_HOME'] = '/home/asus/Research/CamVLA'
from openpi.training.config import _CONFIGS_DICT
from openpi.training import data_loader as dl
from openpi.models.model import Observation

# 1) Observation has the new optional right_wrist_* fields.
fields = {f.name for f in dataclasses.fields(Observation)}
assert {"right_wrist_extrinsic", "right_wrist_intrinsic"} <= fields

# 2) RoboTwin train config + sample.
cfg = _CONFIGS_DICT['pi0_robotwin_cam_prope_ray_view_distill_fullres_stage1_gtdual']
data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
assert data_cfg.norm_stats is not None, "norm_stats not found at data.assets.assets_dir/<asset_id>/"
ds = dl.transform_dataset(
    dl.create_torch_dataset(data_cfg, cfg.model.action_horizon, cfg.model),
    data_cfg,
)
sample = ds[0]
required = {
    "image", "image_mask", "state", "actions",
    "agent_extrinsic", "wrist_extrinsic", "right_wrist_extrinsic",
    "agent_intrinsic", "wrist_intrinsic", "right_wrist_intrinsic",
    "pi3x_target_xy", "pi3x_target_logz", "pi3x_target_conf",
    "point_target_xy", "point_target_logz", "point_target_conf",
    "point_target_source",
    "tokenized_prompt", "tokenized_prompt_mask",
}
assert required <= set(sample), f"missing: {required - set(sample)}"
print("pipeline OK")
```

Each `point_target_xy` / `pi3x_target_xy` should be `(2, 224, 224, 2)` and
all 3 cam extrinsics should be `(4, 4)`.

---

## 7. Phase 5 — Stage 1 training (5,000 steps, freeze backbone)

The recipe mirrors `pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage1_gtdual`:

- Architecture: `pose_enc=prope`, `ray_enc=True`, `view_enc=False`,
  cross-view fusion `aa_order="fgfg"` with PRoPE on layers `(0, 1)`.
- AuxPointHead enabled at `output_resolution=224`, `loss_weight=1.0`.
- `action_loss_weight=0.1` (training mainly the geometry head this stage).
- `ray_embed` warm-started from
  `assets/pi3x_init/ray_embed.pt` (first 1024 / 1152 output channels populated;
  the remaining 128 stay zero).
- Trainable prefixes: `cross_view_fusion`, `ray_embed`, `aux_point_head`.
- Loss: `α · L(pred, GT) + (1-α) · L(pred, Pi3X)` per cam, α=0.5.
- LR: cosine, warmup 500, peak 2.5e-5, decay over 5,000 to 2.5e-6.

Launch:

```bash
source ${GEO_ROOT}/openpi_cam/.venv/bin/activate
cd ${GEO_ROOT}/openpi_cam
export HF_LEROBOT_HOME=${GEO_ROOT}
export OPENPI_PI0_BASE_DIR=${GEO_ROOT}/pi0_base
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

uv run scripts/train_pytorch.py \
    pi0_robotwin_cam_prope_ray_view_distill_fullres_stage1_gtdual \
    --exp_name=stage1_${TASK_NAME}_${TASK_CONFIG}
```

Single-GPU defaults: `batch_size=8`, ~30–40 minutes to 5,000 steps on a 4090.
Memory peak: ~9 GB (backbone is frozen + gradient checkpointing on).

The expected step-0 loss breakdown (smoke-tested):

```
loss            ≈ 0.30
  action_loss   ≈ 0.02   (raw=0.18, weighted by 0.1)
  aux_loss      ≈ 0.30   (point-head total, weighted by 1.0)
    aux_gt_loss   ≈ 0.22   (GT branch, 50% mix)
    aux_pi3x_loss ≈ 0.08   (Pi3X branch, 50% mix)
    aux_xy_loss   ≈ 0.17
    aux_z_loss    ≈ 0.14
  aux_gt_frac   = 0.5    (configured)
  aux_pi3x_frac = 0.5    (configured)
```

Checkpoints land at:
```
checkpoints/pi0_robotwin_cam_prope_ray_view_distill_fullres_stage1_gtdual/<exp_name>/{1000,2000,3000,4000,5000}/
```

---

## 8. Phase 6 — Stage 2 training (30,000 steps, unfreeze)

Warm-start from the Stage 1 final checkpoint. Loss schedule shifts to
prioritize the action policy:

- `action_loss_weight=1.0`
- `aux_point_head.loss_weight=0.05`
- All parameters unfrozen.

```bash
S1_CKPT=${PWD}/checkpoints/pi0_robotwin_cam_prope_ray_view_distill_fullres_stage1_gtdual/stage1_${TASK_NAME}_${TASK_CONFIG}/5000

uv run scripts/train_pytorch.py \
    pi0_robotwin_cam_prope_ray_view_distill_fullres_stage2_gtdual \
    --exp_name=stage2_${TASK_NAME}_${TASK_CONFIG} \
    --pytorch_weight_path=$S1_CKPT
```

---

## 9. Phase 7 — Eval inside RoboTwin sim

Two-process setup: openpi venv serves the policy over a websocket, robotwin
env runs the eval client.

```bash
# Server (openpi venv):
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_robotwin_cam_prope_ray_view_distill_fullres_stage2_gtdual \
    --policy.dir=${PWD}/checkpoints/.../<exp_name>/30000

# Client (robotwin env): use RoboTwin's official eval entry point.
conda activate robotwin
cd ${GEO_ROOT}/RoboTwin
python script/eval_policy.py --task-name ${TASK_NAME} ...
```

---

## 10. Available train configs

All four entries live in `src/openpi/training/config.py`:

| Config name | Purpose | trainable_prefixes | action_loss_weight | aux_point_head.loss_weight |
|---|---|---|---|---|
| `pi0_robotwin_cam_prope_ray_view` | Bare cam-aware (no distillation) | (full) | 1.0 | n/a |
| `pi0_robotwin_cam_prope_ray_view_distill_fullres_stage1_gtdual` | Stage 1, 50/50 GT+Pi3X dual loss, freeze backbone | cross_view_fusion, ray_embed, aux_point_head | 0.1 | 1.0 |
| `pi0_robotwin_cam_prope_ray_view_distill_fullres_stage2_gtdual` | Stage 2, full unfreeze, warm-start from S1 | (full) | 1.0 | 0.05 |
| `pi0_robotwin_cam_baseline` | A/B baseline (no PRoPE/ray/cross-view/point-head) | (full) | 1.0 | n/a |

All four share `asset_id=robotwin/<task>_<task_config>_<N>` so `norm_stats.json`
is reusable.

---

## 11. Failure modes — quick triage

| Symptom | Root cause | Fix |
|---|---|---|
| `'Robot' object has no attribute 'left_planner'` (every episode in collect_data.py) | curobo CuroboPlanner ctor fails silently because `wp.torch` removed in newer warp-lang | `pip install "warp-lang==1.4.2"` |
| `ModuleNotFoundError: pkg_resources` on `import sapien` | `setuptools>=70` dropped pkg_resources | `pip install "setuptools<70"` |
| `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'` during `merge_pkl_to_hdf5_video` | conda env has no ffmpeg binary | `conda install -c conda-forge ffmpeg` |
| `RuntimeError: Could not load libtorchcodec` | torchcodec needs system FFmpeg shared libs | already worked around — `data_loader.py` passes `video_backend="pyav"` |
| `av.error.InvalidDataError: Invalid data found ...` (when LeRobot tries to read mp4) | LeRobot wrote AV1; pyav AV1 decode is unreliable | re-convert with `--no-videos` |
| Norm stats not found at training time | path mismatch: compute_norm_stats writes to `assets/<config_name>/<repo_id>/`, training reads from `data.assets.assets_dir/<asset_id>/` | copy as in §4 |
| `KeyError: 'observation.images.cam_high'` from `RobotwinInputs` | repack `RHS:LHS` order confused, or repack key form mismatched | verify `LeRobotRobotwinCamDataConfig.create()` repack uses dot-form keys on both sides |
| `FileNotFoundError: ...gt_point_targets_224/.../agent/episode_000000.npz` | GT cache directories not renamed to `agent/wrist` | rename per §5b |

---

## 12. Code-change punch list (already applied in this branch)

- `src/openpi/models/model.py` — `Observation.right_wrist_extrinsic / right_wrist_intrinsic` (default None, plumbed through `from_dict` and `preprocess_observation`).
- `src/openpi/models_pytorch/preprocessing_pytorch.py` — `_IMAGE_TO_CAMERA_FIELDS["right_wrist_0_rgb"]` mapped to the new fields; `updated_extrinsics`/`updated_intrinsics` propagate them.
- `src/openpi/models_pytorch/pi0_pytorch.py:484` — `cam_pos["right_wrist"]` and `cam_intr["right_wrist"]` now read from `obs.right_wrist_extrinsic` / `obs.right_wrist_intrinsic` (LIBERO falls back to None).
- `src/openpi/policies/robotwin_policy.py` — new `RobotwinCamInputs` (wraps `RobotwinInputs`) maps RoboTwin's 3 cams to model's `agent_*` / `wrist_*` / `right_wrist_*` slots; applies the openpi `fx → -fx` flip on K.
- `src/openpi/training/config.py` — new `LeRobotRobotwinCamDataConfig` (mirrors `LeRobotLiberoDataConfig`), 4 new TrainConfig entries.
- `src/openpi/training/data_loader.py` — pass `video_backend="pyav"` to `LeRobotDataset`.
- `scripts/cache_pi3x_targets.py` — `--cam-spec` CLI flag with `_parse_cam_spec` (default `None` → original 2-cam LIBERO behavior preserved).
- `scripts/convert_robotwin_cam_to_lerobot.py` — RoboTwin → LeRobot v2.1, decodes JPEG bytes, handles `(3,4)` extrinsics; `--no-videos` flag for image-only storage.
- `scripts/cache_robotwin_gt_point_targets.py` — Sapien depth → `(xy, log_z, conf)` cache.
- `scripts/robotwin_task_configs/demo_clean_camaware.yml` — drop-in RoboTwin task config that flips `data_type.depth: true`.

The LIBERO path is preserved: all `pi0_libero_cam_pytorch_prope_ray_view*` configs still build their data pipelines without modification.

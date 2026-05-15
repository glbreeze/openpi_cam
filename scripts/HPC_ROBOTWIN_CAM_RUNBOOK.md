# RoboTwin2.0 Cam-Aware + Pi3X Mixed-GT Distillation Runbook

End-to-end recipe for training the cam-aware Pi0 + Pi3X mixed-GT distillation
pipeline (your latest `pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage1_gtdual`
recipe) on **RoboTwin2.0**.

This runbook follows the same structure as `HPC_LIBERO_REGEN_RUNBOOK.md`, with
RoboTwin's bimanual 3-camera setup (`head_camera`, `left_camera`,
`right_camera`) substituted in for LIBERO's 2-camera setup.

## 0. Layout

Under `${GEO_ROOT}` (= the parent of `openpi_cam`):

```
${GEO_ROOT}/
├── openpi_cam/                       # this repo (training)
├── RoboTwin/                         # official RoboTwin repo (sim + collection + eval)
├── pi0_base/                         # local pi0 base checkpoint
├── pi0_libero/                       # asset_id roots for norm_stats.json (re-used)
└── .cache/
    ├── huggingface/lerobot/<repo_id>/    # converted LeRobot v3 datasets
    └── openpi/
        ├── pi3x_targets_224/<repo_id>/   # Pi3X teacher cache
        └── gt_point_targets_224/<repo_id>/ # Sapien-GT point-map cache
```

The training env reads `HF_LEROBOT_HOME=${GEO_ROOT}` (set by `scripts/env/activate_env.sh`).

## 1. Environments

Two envs, mirroring the LIBERO setup:

| Env | Python | Purpose |
|---|---|---|
| openpi venv (`.venv/`) | 3.11 | training, conversion, Pi3X cache |
| `robotwin` (conda) | 3.10 | RoboTwin sim, raw collection, GT cache, eval |

Local install (already kicked off; see `/tmp/robotwin_install.log`):

```bash
conda create -n robotwin python=3.10 -y
conda activate robotwin
conda install -y -c "nvidia/label/cuda-12.1.0" cuda-toolkit
cd ${GEO_ROOT}/RoboTwin
pip install -r script/requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
# patch sapien + mplib (see script/_install.sh)
cd envs && git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git
cd curobo && pip install -e . --no-build-isolation
```

Asset download (one-time, ~30+ GB):

```bash
cd ${GEO_ROOT}/RoboTwin
bash script/_download_assets.sh
```

## 2. Phase 1 — Re-collect raw episodes with depth + K + Tcw

The official RoboTwin pipeline does not collect depth by default. To match the
LIBERO-Camera fidelity we need a task config that enables it. Use:

```bash
cp ${GEO_ROOT}/openpi_cam/scripts/robotwin_task_configs/demo_clean_camaware.yml \
   ${GEO_ROOT}/RoboTwin/task_config/demo_clean_camaware.yml
```

This is a clone of `task_config/demo_clean.yml` with:

```yaml
data_type:
  rgb: true
  depth: true        # <-- enables Sapien depth dump per cam
  endpose: true
  qpos: true
```

Camera intrinsics (`intrinsic_cv`), extrinsics (`extrinsic_cv`), and
`cam2world_gl` are written unconditionally by `cameras.get_config()` in
`envs/_base_task.py:447`, so no other change is needed.

Run for the smoke task (single task, 50 demos):

```bash
conda activate robotwin
cd ${GEO_ROOT}/RoboTwin
TASK_NAME=beat_block_hammer
TASK_CONFIG=demo_clean_camaware
bash collect_data.sh ${TASK_NAME} ${TASK_CONFIG}
# Output: data/${TASK_NAME}/${TASK_CONFIG}/data/episode*.hdf5
#         data/${TASK_NAME}/${TASK_CONFIG}/instructions/episode*.json
# Per-frame keys (per cam): rgb, depth, intrinsic_cv, extrinsic_cv, cam2world_gl
```

## 3. Phase 2 — Convert to LeRobot v3 (with cam params)

```bash
source ${GEO_ROOT}/openpi_cam/scripts/env/activate_env.sh
cd ${GEO_ROOT}/openpi_cam

REPO_ID=robotwin/${TASK_NAME}_${TASK_CONFIG}_50_cam
python scripts/convert_robotwin_cam_to_lerobot.py \
    --raw-dir ${GEO_ROOT}/RoboTwin/data/${TASK_NAME}/${TASK_CONFIG} \
    --repo-id ${REPO_ID} \
    --image-size 224
```

Resulting LeRobot v3 features:

| Feature | Shape | Notes |
|---|---|---|
| `observation.images.cam_high` | (3, 224, 224) | head_camera |
| `observation.images.cam_left_wrist` | (3, 224, 224) | left_camera |
| `observation.images.cam_right_wrist` | (3, 224, 224) | right_camera |
| `observation.state` | (14,) | aloha-style 7+7 |
| `observation.joint_state` | (14,) | qpos |
| `action` | (14,) | next-step qpos |
| `observation.cam_high_extrinsic` | (4, 4) | OpenCV-frame `T_wc` |
| `observation.cam_left_wrist_extrinsic` | (4, 4) | |
| `observation.cam_right_wrist_extrinsic` | (4, 4) | |
| `observation.cam_high_intrinsic` | (3, 3) | natural Sapien/OpenCV K, positive `fx/fy` |
| `observation.cam_left_wrist_intrinsic` | (3, 3) | |
| `observation.cam_right_wrist_intrinsic` | (3, 3) | |

## 4. Phase 3 — Compute norm stats

```bash
cd ${GEO_ROOT}/openpi_cam
uv run scripts/compute_norm_stats.py \
    --config-name pi0_robotwin_cam_prope_ray_view_distill_fullres_stage1_gtdual
# Output: ${GEO_ROOT}/pi0_libero/${REPO_ID}/norm_stats.json
```

## 5. Phase 4a — Pi3X teacher cache (224×224, 3 cams)

```bash
conda activate pi3x
cd ${GEO_ROOT}/openpi_cam
python scripts/cache_pi3x_targets.py \
    --data-root ${HF_LEROBOT_HOME}/${REPO_ID} \
    --output-root ~/.cache/openpi/pi3x_targets_224/${REPO_ID} \
    --pi3x-repo ${GEO_ROOT}/Pi3X_Libero \
    --output-resolution 224 \
    --cam-spec head:cam_high,left_wrist:cam_left_wrist,right_wrist:cam_right_wrist
```

Each cam writes `{root}/{cam}/episode_NNNNNN.npz` with `(xy, log_z, conf)` fp16
tensors. ~52 GB / cam at 224.

## 6. Phase 4b — Sapien-GT point cache (224×224, 3 cams)

```bash
conda activate robotwin
cd ${GEO_ROOT}/openpi_cam
python scripts/cache_robotwin_gt_point_targets.py \
    --raw-dir ${GEO_ROOT}/RoboTwin/data/${TASK_NAME}/${TASK_CONFIG} \
    --output-root ~/.cache/openpi/gt_point_targets_224/${REPO_ID} \
    --output-resolution 224
```

Reads `/observation/<cam>/depth` + `intrinsic_cv` + `extrinsic_cv` from each raw
RoboTwin HDF5 episode; converts the 480×640 Sapien depth into the same
natural-orientation, 224×224 OpenCV-camera-frame point map that Pi3X emits, and
writes the same `(xy, log_z, conf)` fp16 layout.

## 7. Phase 5 — Stage 1 (5k steps, freeze backbone)

Mirrors `pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage1_gtdual`
but on the new RoboTwin dataset.

```bash
cd ${GEO_ROOT}/openpi_cam
uv run scripts/train_pytorch.py \
    pi0_robotwin_cam_prope_ray_view_distill_fullres_stage1_gtdual \
    --exp_name=stage1_${TASK_NAME}_${TASK_CONFIG}
# Trainable: cross_view_fusion, ray_embed, aux_point_head
# Loss: 0.1 * action + 1.0 * (alpha*L_GT + (1-alpha)*L_Pi3X), alpha=0.5
```

### HPC (sbatch)

```bash
sbatch scripts/sbatch/train_pi0_robotwin_cam_distill_stage1_gtdual.sbatch
```

## 8. Phase 6 — Stage 2 (30k steps, unfreeze)

```bash
uv run scripts/train_pytorch.py \
    pi0_robotwin_cam_prope_ray_view_distill_fullres_stage2_gtdual \
    --exp_name=stage2_${TASK_NAME}_${TASK_CONFIG} \
    --pytorch_weight_path=${PWD}/checkpoints/pi0_robotwin_cam_prope_ray_view_distill_fullres_stage1_gtdual/stage1_${TASK_NAME}_${TASK_CONFIG}/5000
# Loss: 1.0 * action + 0.05 * point_loss
```

## 9. Phase 7 — Eval inside RoboTwin sim

```bash
# Server (openpi venv):
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_robotwin_cam_prope_ray_view_distill_fullres_stage2_gtdual \
    --policy.dir=${PWD}/checkpoints/.../30000

# Client (robotwin env): use RoboTwin/script/eval_policy.py
conda activate robotwin
cd ${GEO_ROOT}/RoboTwin
python script/eval_policy.py --task-name ${TASK_NAME} ...
```

## 10. Verification checklist

- [ ] `script/_install.sh`, `script/_download_assets.sh` complete; `python -c "import sapien; import mplib; import curobo"` succeeds in `robotwin` env.
- [ ] Phase 1: HDF5 contains `/observation/<cam>/{rgb, depth, intrinsic_cv, extrinsic_cv}`.
- [ ] Phase 2: LeRobot v3 features include 3 extrinsics + 3 intrinsics.
- [ ] Phase 4a/b: per-cam `episode_*.npz` files exist with shape `(T, 224, 224, 2)` xy.
- [ ] Phase 5: stage1 loss curves: `loss/action`, `loss/point_gt`, `loss/point_pi3x` all decrease.
- [ ] Phase 6: stage2 starts from stage1 ckpt with all params unfrozen.
- [ ] Phase 7: eval rollouts run in RoboTwin sim end to end.

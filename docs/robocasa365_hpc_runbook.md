# RoboCasa365 HPC Runbook

How to bring up the camera-aware Pi3X-distilled Pi0 path on **NYU Tandon Torch
HPC** for RoboCasa365 (PandaOmron, single-arm Franka + mobile base, 365 atomic
kitchen tasks). Sibling to
[`robotwin_5tasks_hpc_runbook.md`](robotwin_5tasks_hpc_runbook.md).

State as of commit `e1ab1c3`:

- Cam-aware policy + train configs landed in `src/openpi/policies/robocasa_policy.py`
  and `training/config.py` (3 configs: `pi0_robocasa365_cam_prope_ray_view_distill_fullres_stage1/2`
  plus `pi0_robocasa365_baseline`).
- Sim env verified locally on a 4090 (see §1.4); the 23 GB of robocasa kitchen
  assets has been Globus-transferred to Torch under
  `/scratch/yp2841/geometry-vla/robocasa365_kitchen_assets/` (task
  `c0c35f86-51e2-11f1-afe3-0afffe4617ab`, 100 % success, 123 434 files).
- The LeRobot training data for RoboCasa365 has **not yet been transferred to
  Torch and has not been enriched with per-frame camera intrinsics +
  extrinsics**. Enrichment is the next blocker; see §3.
- Pi3X teacher cache has **not been computed yet**; see §4.

---

## 0. Hard prerequisites (must hold before any training launch)

| Item | Where | Status |
|---|---|---|
| Repo cloned at `git rev e1ab1c3` or later | `/scratch/yp2841/geometry-vla/openpi_cam/` | follow §1.1 |
| Kitchen assets (23 GB) | `/scratch/yp2841/geometry-vla/robocasa365_kitchen_assets/` | ✅ transferred |
| Pi0 base ckpt | `/scratch/yp2841/geometry-vla/pi0_base/` | already there from RoboTwin pipeline |
| Pi3X-ray-embed warm-start | `assets/pi3x_init/ray_embed.pt` (in repo) | ✅ |
| RoboCasa LeRobot dataset, enriched with K + T_wc | `/scratch/yp2841/geometry-vla/robocasa365/<task>_camaware/` | ❌ open — see §3 |
| Pi3X cache per task | `/scratch/yp2841/geometry-vla/.cache/openpi/pi3x_targets_224/robocasa365_<task>_camaware/` | ❌ open — see §4 |
| Norm stats per config | inside the asset/checkpoint tree | ❌ open — see §5 |

---

## 1. One-time setup on Torch

### 1.1 Repo

```bash
cd /scratch/yp2841/geometry-vla
git clone --branch py-torch git@github.com:glbreeze/openpi_cam.git
# or update an existing clone:
git -C openpi_cam fetch && git -C openpi_cam checkout py-torch && git -C openpi_cam pull
```

### 1.2 openpi conda env

If you do not already have an `openpi_cam` env from the RoboTwin runbook,
follow [`pi3x_distillation_hpc.md`](pi3x_distillation_hpc.md) §1. Otherwise
re-activate it. Confirm the new RoboCasa configs resolve:

```bash
cd /scratch/yp2841/geometry-vla/openpi_cam
uv run python -c "
from openpi.training import config
print([c.name for c in config._CONFIGS if 'robocasa' in c.name])
"
# expected:
# ['pi0_robocasa365_cam_prope_ray_view_distill_fullres_stage1',
#  'pi0_robocasa365_cam_prope_ray_view_distill_fullres_stage2',
#  'pi0_robocasa365_baseline']
```

### 1.3 robocasa365 sim env (separate from the trainer env)

The sim cannot share the trainer's Python env (numpy / mujoco versions
diverge). Create a sibling conda env on a node that has internet:

```bash
mamba create -n robocasa365 python=3.11 -y
mamba activate robocasa365

cd /scratch/yp2841/geometry-vla
git clone --depth 1 https://github.com/ARISE-Initiative/robosuite.git
git clone --depth 1 https://github.com/robocasa/robocasa.git robocasa365_src
pip install -e robosuite -e robocasa365_src
pip install robosuite_models 'mink==0.0.5' tyro imageio
# mink downgrades numpy; restore for robocasa:
pip install --upgrade 'numpy==2.2.5'

# Macros:
python robosuite/robosuite/scripts/setup_macros.py
python robocasa365_src/robocasa/scripts/setup_macros.py

# Symlink the transferred assets in lieu of re-downloading 23 GB from Box:
rm -rf robocasa365_src/robocasa/models/assets
ln -s /scratch/yp2841/geometry-vla/robocasa365_kitchen_assets \
      robocasa365_src/robocasa/models/assets
```

### 1.4 Verify the sim spins up

```bash
mamba activate robocasa365
cd /scratch/yp2841/geometry-vla/openpi_cam
python scripts/debug/verify_robocasa_setup.py --env robocasa/OpenDrawer \
  --out /scratch/yp2841/geometry-vla/robocasa365_setup_check
```

You should see:

```text
=== camera intrinsics + extrinsics ===
  robot0_agentview_left:  fovy=60.00deg  fx=221.70 fy=221.70 cx=128.0 cy=128.0
  robot0_agentview_right: fovy=60.00deg  fx=221.70 fy=221.70 cx=128.0 cy=128.0
  robot0_eye_in_hand:     fovy=75.00deg  fx=166.81 fy=166.81 cx=128.0 cy=128.0
=== step() ok ===
```

Eyeball `robocasa365_setup_check/*__gym_wrapper_humanview.png` — should look
right-side-up. The `*__raw_mujoco_buffer.png` versions are upside-down
(visual-debug artifact only). **The LeRobot stored frames decode
right-side-up too** (verified by decoding one MP4 frame with imageio), so
the eval client passes the gym frame through to the policy server with no
flip. This is opposite to my earlier inference from reading the converter
source; the conversion path apparently flips during `extract_trajectory`
or during MP4 encode.

---

## 2. Pick the task subset

The shipped `OpenDrawer` walk-through is the only single-command-downloadable
LeRobot dataset, but the full RoboCasa365 atomic set has ~25 tasks where the
base is stationary. For a first sweep that mirrors the LIBERO/RoboTwin budget,
**pick 5 atomic tasks with diverse manipulation primitives**. Suggested seed:

| Task | Why |
|---|---|
| `OpenDrawer` | already on disk, smallest smoke task |
| `PnPCounterToSink` | classic pick-and-place |
| `TurnOnStove` | rotational manipulation |
| `CloseSingleDoor` | revolute joint task |
| `CoffeeServeMug` | longer-horizon multi-step |

For each, download via:

```bash
mamba activate robocasa365
python -m robocasa.scripts.download_datasets \
    --tasks OpenDrawer PnPCounterToSink TurnOnStove CloseSingleDoor CoffeeServeMug \
    --split target --source human
# -> playground/Datasets/robocasa365/v1.0/target/atomic/<TASK>/<DATE>/lerobot/
```

Each LeRobot dataset has 256×256 RGB for all 3 cams + state + action + lang,
**but no camera intrinsics / extrinsics**. That gap is filled in §3.

---

## 3. Enrich LeRobot data with per-frame K + T_wc

> ⚠ **This script does not exist yet.** Pattern after
> `scripts/convert_robotwin_cam_to_lerobot.py`. Sketch:
>
> 1. Open the upstream LeRobot dataset for the task.
> 2. For each episode, replay the recorded action sequence through
>    `robocasa/<TaskName>` (sim env), and on each step pull `K` and `T_wc` for
>    each of the 3 cams from `env.unwrapped.env.sim` (formula already in
>    `scripts/debug/verify_robocasa_setup.py:_dump_cam_k_t_wc`).
> 3. Append `observation.agentview_left_extrinsic`, `..._intrinsic`, same for
>    `agentview_right` and `eye_in_hand` (6 new columns per frame) to a new
>    LeRobot v2.1 dataset under
>    `/scratch/yp2841/geometry-vla/robocasa365/<task>_camaware/`.
>
> The LeRobot `add_frame` API accepts arbitrary `(4,4) float32` / `(3,3) float32`
> arrays as long as the modality.json schema is updated to declare them.
>
> All three RoboCasa cams have **fixed intrinsics for the entire episode**
> (cam params come from the robot's URDF, not from runtime state). Per-frame
> extrinsics change because `agentview_*` are mounted to the robot base and the
> robot base translates/rotates; `eye_in_hand` is mounted to the wrist and
> moves continuously. So the K columns can be flat across frames (write the
> same matrix every frame), but the T_wc columns must be re-read per step.

Once the enriched datasets are on disk, smoke them with:

```bash
uv run python -c "
import datasets, glob
for p in sorted(glob.glob('/scratch/yp2841/geometry-vla/robocasa365/*_camaware')):
    ds = datasets.load_from_disk(p)
    print(p, ds.column_names[:8], '...')
"
```

---

## 4. Pi3X cache

Mirror the LIBERO / RoboTwin Pi3X cache pipeline. For each task:

```bash
# In the openpi_cam trainer env (NOT the robocasa365 env):
cd /scratch/yp2841/geometry-vla/openpi_cam
uv run python scripts/cache_pi3x_targets.py \
    --repo-id robocasa365/<task>_camaware \
    --output-dir $OPENPI_CACHE_DIR/pi3x_targets_224/robocasa365_<task>_camaware \
    --cam-spec \
        agent:robot0_agentview_left:agentview_left_intrinsic,\
        wrist:robot0_eye_in_hand:eye_in_hand_intrinsic,\
        right_wrist:robot0_agentview_right:agentview_right_intrinsic \
    --image-size 224
```

Set `OPENPI_CACHE_DIR=/scratch/yp2841/geometry-vla/.cache/openpi` so the cache
lands on scratch.

> ⚠ Before committing GPU hours to a full cache, run the Pi3X-vs-GT
> diagnostic on 5 random episodes per cam. RoboTwin's clean-background
> failure mode is a real risk on RoboCasa kitchens too. The diagnostic lives
> at `scripts/debug/diagnose_pi3x_target_quality.py` — see §6 of the
> RoboTwin runbook for usage. Acceptance threshold: `xy_cosine_mean ≥ 0.3`
> and `pearson_corr_per_frame_mean ≥ 0.2`. Below that, the Pi3X teacher is
> collapsing and the distillation signal is noise.

---

## 5. Norm stats

```bash
cd /scratch/yp2841/geometry-vla/openpi_cam
uv run python scripts/compute_norm_stats.py \
  --config-name pi0_robocasa365_cam_prope_ray_view_distill_fullres_stage1
uv run python scripts/compute_norm_stats.py \
  --config-name pi0_robocasa365_baseline
```

Stage 2 reuses stage 1's stats.

---

## 6. Training

### 6.1 Stage 1 — geometry modules only (5 000 steps, ~3 h on 1× A100/L40S)

```bash
cd /scratch/yp2841/geometry-vla/openpi_cam
uv run torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  scripts/train_pytorch.py \
  pi0_robocasa365_cam_prope_ray_view_distill_fullres_stage1 \
  --exp_name robocasa365_<task>_s1
```

### 6.2 Stage 2 — full unfreeze (30 000 steps)

```bash
STAGE1_FINAL=/scratch/yp2841/geometry-vla/openpi_cam/checkpoints/\
pi0_robocasa365_cam_prope_ray_view_distill_fullres_stage1/\
robocasa365_<task>_s1/5000

uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/train_pytorch.py \
  pi0_robocasa365_cam_prope_ray_view_distill_fullres_stage2 \
  --exp_name robocasa365_<task>_s2 \
  --pytorch_weight_path $STAGE1_FINAL
```

### 6.3 Baseline A/B (no cam-aware path)

```bash
uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/train_pytorch.py \
  pi0_robocasa365_baseline \
  --exp_name robocasa365_<task>_baseline
```

> ⚠ All three configs currently point at
> `robocasa365/OpenDrawer_target_human_camaware` as the repo_id. To run on a
> different task, either (a) pass `--data.repo_id robocasa365/<task>_camaware`
> at launch, or (b) clone the TrainConfig in `training/config.py` per task
> as the RoboTwin runbook does.

---

## 7. Evaluation

### 7.1 Policy server (trainer env)

```bash
mamba activate openpi_cam
cd /scratch/yp2841/geometry-vla/openpi_cam
uv run python scripts/serve_policy.py policy:checkpoint \
  --policy.config pi0_robocasa365_cam_prope_ray_view_distill_fullres_stage2 \
  --policy.dir /scratch/yp2841/geometry-vla/openpi_cam/checkpoints/.../30000 \
  --port 8000
```

### 7.2 Sim client (robocasa365 env)

The starVLA repo ships a websocket runner for the upstream robocasa benchmark
at `examples/Robocasa_365/eval_files/simulation_env.py`. The same runner
works for our policy if we replace its `PolicyWarper` with a thin client that
talks to openpi's websocket server. **This adapter does not exist yet** — it
needs to repack the 12-d action chunk that comes back from openpi into the
gym wrapper's action dict (keys `action.end_effector_position`,
`action.end_effector_rotation`, `action.gripper_close`, `action.base_motion`,
`action.control_mode`) per the LeRobot action layout `[base_motion(4),
control_mode(1), eef_pos(3), eef_rot(3), gripper_close(1)]`. Image
orientation is identity (gym frame = training frame).

Track this as TODO; pattern after `scripts/robotwin_eval_policies/` once the
training side is producing checkpoints.

---

## 8. Open items (track in PR descriptions, not here)

1. Per-task LeRobot enrichment script (§3).
2. Pi3X-vs-GT teacher-quality diagnostic on RoboCasa frames (§4).
3. Eval client adapter (§7.2).
4. Per-task TrainConfig clones with the right `repo_id` and norm-stat
   asset_id (§6).
5. (Decision) whether to reorder the 12-d LeRobot action layout to put the
   Franka 7-d block at dims `[0:7]` to better match Pi0 pretraining, or keep
   the upstream `[base, ctrl, eef, grip]` layout and rely on the model to
   learn it.

---

## 9. Reference

- Globus task carrying the kitchen assets: `c0c35f86-51e2-11f1-afe3-0afffe4617ab`
  (24.07 GB, 123 434 files, completed 2026-05-17 12:17 UTC).
- Commit `e1ab1c3` (`py-torch`) is the first revision with the RoboCasa
  policy + configs.
- Sibling docs:
  [`robotwin_5tasks_hpc_runbook.md`](robotwin_5tasks_hpc_runbook.md),
  [`robotwin_cam_pipeline.md`](robotwin_cam_pipeline.md),
  [`pi3x_distillation_hpc.md`](pi3x_distillation_hpc.md),
  [`pi3x_pi0_system_architecture_and_results.md`](pi3x_pi0_system_architecture_and_results.md).

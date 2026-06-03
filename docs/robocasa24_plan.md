# RoboCasa v0 (24-Task, 2024) Evaluation Plan

The concrete *how-to* for bringing up RoboCasa v0 — the older 24-task
benchmark — from scratch on the local workstation first, then mirroring to the
Torch HPC for the full sweep.

Scope baked into this plan:

- Local-first, then HPC for the multi-task scale-up.
- Train + eval from scratch on v0 (no zero-shot transfer of the 365 checkpoint).
- Starter task set: `OpenDrawer`, `CloseDrawer`, `OpenSingleDoor`.

The RoboCasa365 pipeline stays intact. Everything here lives under a
separate prefix and a separate conda env.

---

## 0. Layout

Code stays in `openpi_cam`. Data, assets, sim source, and caches live outside
the repo, following the `GEO_ROOT = parent-of-repo` convention used by
`scripts/env/activate_env.sh:10-50`.

### Local (this workstation)

```
~/Research/robocasa24/
├── src/
│   ├── robosuite/                # v1.5.1 — required by robocasa v0.2
│   └── robocasa/                 # v0.2 (24-task benchmark; uses robosuite 1.5 per its README)
├── assets/                       # robocasa kitchen assets (~24 GB)
├── raw/                          # downloaded HDF5 (robomimic-style)
│   ├── OpenDrawer/
│   ├── CloseDrawer/
│   └── OpenSingleDoor/
├── lerobot/                      # converted cam-aware LeRobot datasets
│   ├── OpenDrawer_mg_camaware/
│   ├── CloseDrawer_mg_camaware/
│   └── OpenSingleDoor_mg_camaware/
└── .cache/openpi/                # OPENPI_CACHE_DIR (norm_stats + pi3x targets)
```

Conda env: `~/miniconda3/envs/robocasa24` (Python 3.10; v0 tag predates 3.11).

### HPC (Torch, scale-up)

Mirror to the paths already reserved in
`robocasa_eval_debug_and_v0_migration.md:204-216`:

```
/scratch/yp2841/geometry-vla/robocasa24_src/
/scratch/yp2841/geometry-vla/robocasa24_assets/
/scratch/yp2841/geometry-vla/robocasa24_raw/
/scratch/yp2841/geometry-vla/robocasa24_lerobot/
/scratch/yp2841/geometry-vla/.cache/openpi/robocasa24/
/scratch/yp2841/.venvs/robocasa24/
```

Do **not** write env/data/logs under `/home/yp2841`.

---

## 1. Sim env bring-up (local, one-time)

```bash
# Use conda directly if mamba isn't on PATH in non-interactive shells.
source /home/asus/miniconda3/etc/profile.d/conda.sh
conda create -n robocasa24 python=3.10 -y
conda activate robocasa24

mkdir -p ~/Research/robocasa24/src
cd ~/Research/robocasa24/src

# robosuite v1.5.1 — robocasa v0.2's README (line 11) is explicit that v0.2
# uses robosuite v1.5 as the backend, NOT v1.4. robosuite_models also requires
# robosuite.models.bases (v1.5+).
git clone https://github.com/ARISE-Initiative/robosuite.git
(cd robosuite && git checkout v1.5.1)

git clone https://github.com/robocasa/robocasa.git
(cd robocasa && git checkout v0.2)

pip install -e robosuite -e robocasa
pip install robosuite_models 'mink==0.0.5' tyro imageio gymnasium

# Setup macros — robosuite prompts to overwrite if a stale macros_private.py
# from an earlier install exists; pipe "y" if so.
python robosuite/robosuite/scripts/setup_macros.py
python robocasa/robocasa/scripts/setup_macros.py

# Pull kitchen assets via the official downloader (~5 GB download, ~8 GB on
# disk after extraction). Prompts for confirmation; pipe "y".
yes y | python -m robocasa.scripts.download_kitchen_assets
```

Smoke check before touching anything else (note: RoboCasa v0.2 does **not**
register a `gymnasium` namespace — instantiate via `robosuite.make` directly):

```bash
python -c "
import robosuite_models, robocasa, robosuite
from robosuite.controllers import load_composite_controller_config
cfg = load_composite_controller_config(robot='PandaOmron')
env = robosuite.make(
    env_name='OpenDrawer', robots='PandaOmron', controller_configs=cfg,
    has_renderer=False, has_offscreen_renderer=False,
    use_camera_obs=False, ignore_done=True,
)
env.reset()
import numpy as np
o, r, done, info = env.step(np.zeros(env.action_dim))
print('action_dim:', env.action_dim, '— expected 12')
print('OK')
"
```

`action_dim=12` and a clean step confirm the env, controller, and assets are
all wired up. (In v1.5 the controller loader is
`load_composite_controller_config(robot=...)`, not the v1.4-era
`load_controller_config(default_controller=...)`.)

---

## 2. Dataset download (local)

**Chosen scope: all 24 tasks × `human_im` (~54 demos/task, ~1.3 GB/task,
~30 GB total).**

History of how we got here:

- Initially tried `mg_im` (3000 demos/task, ~28 GB per file). Box.com's
  per-connection rate cap (~3.4 MB/s) made the all-24 download
  ~55 hours single-stream; even with 6× parallel wget it's ~9 hours.
- `mg_im` is monolithic HDF5 — you cannot byte-range-select "just the
  cams we use." The full per-file size has to be downloaded regardless.
- For time-bound iteration, `human_im` is the right pragmatic choice:
  full 24 tasks downloads in ~30 min, gets the converter + training +
  eval loop validated end-to-end. The trade-off is fewer demos and
  numbers below paper baselines (which use Generated-3000).
- The converter scripts take a `--source-type {human_im, mg_im}` flag
  so re-pulling `mg_im` later is a one-line change; same scripts run
  on either source.

Note: `robocasa/scripts/` has no `__init__.py`, so `python -m
robocasa.scripts.download_datasets` will fail with `ModuleNotFoundError:
No module named 'robocasa.scripts'`. Invoke by file path. The script
prompts for confirmation; pipe `y`.

```bash
conda activate robocasa24
# Scoped to the 24 manipulation tasks only (skip NavigateKitchen + 5
# multi-stage composite tasks; the converter wouldn't use them anyway).
# The downloader's overwrite check skips files that already exist, so
# re-running after an interruption is safe and resumes-by-file.
yes y | python ~/Research/robocasa24/src/robocasa/robocasa/scripts/download_datasets.py \
  --ds_types human_im \
  --tasks \
    PnPCounterToCab PnPCabToCounter PnPCounterToSink PnPSinkToCounter \
    PnPCounterToMicrowave PnPMicrowaveToCounter PnPCounterToStove PnPStoveToCounter \
    OpenSingleDoor CloseSingleDoor OpenDoubleDoor CloseDoubleDoor \
    OpenDrawer CloseDrawer \
    TurnOnSinkFaucet TurnOffSinkFaucet TurnSinkSpout \
    TurnOnStove TurnOffStove \
    CoffeeSetupMug CoffeeServeMug CoffeePressButton \
    TurnOnMicrowave TurnOffMicrowave
```

Files land under
`~/Research/robocasa24/src/robocasa/datasets/v0.1/single_stage/<env>/<Task>/<date>/demo_gentex_im128_randcams.hdf5`
(registry default — *inside the cloned repo*). The converter reads from
this layout directly; no symlinks needed.

---

## 3. Schema inspection (do not skip)

The v0 HDF5 schema is **not assumed** to match the RoboCasa365 LeRobot
cam-aware layout. Before writing any converter, dump the structure of one
demo per task and confirm everything that the converter will rely on.

Write a one-off `scripts/debug/inspect_robocasa24_hdf5.py` that prints, per
task, for `/data/demo_0`:

- `actions.shape` and `action_dict` keys + per-key shapes
- `obs/*` keys, shapes, and image orientation (decode one frame and eyeball it)
- `states.shape` (needed for exact replay later)
- `.attrs["ep_meta"]` (language / task text)
- `/data.attrs["env_args"]` (controller config)

The action-layout question is the one that caused the 365 eval pain. The
working 365 layout is the 12-D
`[base_motion(4), control_mode(1), eef_pos(3), eef_rot(3), gripper(1)]`
documented in `robocasa_eval_debug_and_v0_migration.md:163-173`. If v0 stores
a different ordering or a different controller, every downstream config and
the eval adapter have to follow.

Also do one **stored-state replay** of a demo via the simulator before
trusting the converter — `env.reset(seed=i)` is *not* a faithful replay of
demo `i` (problem #6 in the migration doc, lines 76-86). Use the saved
`states` array and the env's `set_state` (or `sim.set_state_from_flattened`)
to step through one demo and confirm it succeeds in sim.

---

## 4. Converter (open code task)

### Locked-in design decisions

After surveying what's reusable in `openpi_cam` and inspecting the v0
HDF5 schema + the robocasa source, the converter is built around four
decisions:

1. **Image size: 224 × 224.** Matches Pi3 / Pi0 input size directly. v0
   HDF5 stores 128 × 128, so we upsample (bilinear) and scale K accordingly.
2. **Two views, not three.** Drop `agentview_right` entirely:
   - `observation.images.robot0_agentview_left`  → `base_0_rgb` (agent slot, mask=True)
   - `observation.images.robot0_eye_in_hand`     → `left_wrist_0_rgb` (wrist slot, mask=True)
   - `right_wrist_0_rgb` → pad image, `image_mask=False` (mirrors the LIBERO
     single-arm pattern in `libero_policy.py`).
3. **Per-frame K / T_wc via hybrid sim-replay.** v0 ships only the static
   `ep_meta["cam_configs"]` (pos / quat / parent_body, plus `fovy=60` for
   agent cams); the per-frame world poses must come from MuJoCo. Cleanest
   accurate path: build the sim once per episode from `@model_file`, then
   for each step call `sim.set_state_from_flattened(states[t])` and read
   `sim.data.cam_xmat` / `cam_xpos`. Apply `_mujoco_to_opencv_extrinsic`
   (already in `libero_policy.py`). Cost across the 24 tasks × ~3000 demos:
   roughly 5 h CPU single-threaded; parallelize over tasks if we want it
   tighter. For the 24 atomic tasks the base is stationary, so agentview
   T_wc is constant per episode (we still write per-frame for schema
   uniformity); only eye-in-hand actually changes per step.
4. **One giant multi-task LeRobot dataset, not one repo per task.** All
   24 tasks land in `$HF_LEROBOT_HOME/robocasa24/all24_<source>_camaware/`
   (where `<source>` is `human` for human_im, `mg` for mg_im),
   differentiated by `annotation.human.task_description` (pulled from
   `ep_meta["lang"]`, e.g. `"open the right drawer"`). Single norm_stats,
   single Pi3X cache (if/when we cam-aware-distill), single TrainConfig.
   Currently using `all24_human_camaware`.

### Reusable pieces (no rewrites needed)

| Need | Already in repo |
|---|---|
| LeRobot row → Pi0 input transform | `RobocasaCamInputs` in `src/openpi/policies/robocasa_policy.py:230` |
| Pi0 output → 12-d LeRobot action slice | `RobocasaOutputs` in `robocasa_policy.py:215` |
| Data factory (repack, delta, Pi3X / GT mix) | `LeRobotRobocasaCamDataConfig` in `src/openpi/training/config.py:494` |
| Closest converter pattern to follow | `scripts/convert_robotwin_cam_to_lerobot.py` |
| MuJoCo → OpenCV extrinsic swap | `libero_policy._mujoco_to_opencv_extrinsic` |

Only file change required on the model side is a ~6-line tweak to
`RobocasaCamInputs` so `agentview_right` is **optional**: when absent,
emit a zero pad with `image_mask["right_wrist_0_rgb"]=False` instead of
raising. No new DataConfig class.

### Target LeRobot row schema

(Must match what `LeRobotRobocasaCamDataConfig.create()` repacks — see
`config.py:541-552`.)

```
observation.images.robot0_agentview_left     (T, 224, 224, 3) uint8 video
observation.images.robot0_eye_in_hand        (T, 224, 224, 3) uint8 video
observation.state                            (T, 16) float32
  layout: [base_pos(3), base_quat(4), base_to_eef_pos(3),
           base_to_eef_quat(4), gripper_qpos(2)]
action                                       (T, 12) float32
  layout: [base_motion(4), control_mode(1), eef_pos(3), eef_rot(3), gripper(1)]
  ** v0 HDF5 stores [eef_pos(3), eef_rot(3), gripper(1), base_motion(4),
     control_mode(1)] — permutation [7,8,9,10, 11, 0,1,2, 3,4,5, 6] required **
annotation.human.task_description            str    (from ep_meta["lang"])
observation.agentview_left_intrinsic         (T, 3, 3) float32  (K scaled to 224)
observation.agentview_left_extrinsic         (T, 4, 4) float32  (T_wc, OpenCV frame)
observation.eye_in_hand_intrinsic            (T, 3, 3) float32
observation.eye_in_hand_extrinsic            (T, 4, 4) float32
```

Note: no `agentview_right_*` channels — both video and cam columns are
omitted for that cam, per decision (2).

### Why two stages

A single-script converter would need both `mujoco/robosuite/robocasa`
(for sim replay) and `lerobot` (for dataset writing) in the same env. But
`pip install --dry-run lerobot` in the `robocasa24` env shows that
lerobot would force `torch 2.12 → 2.10` and `numpy 1.23.3 → 2.2.6`, and
robocasa pins `numpy==1.23.3` (via numba). So we split:

- **Stage 1** in the `robocasa24` env: extract per-frame K + T_wc via sim
  replay; permute action; assemble state; write a small per-task HDF5
  cache (≈200 MB × 24 = ≈5 GB). The 28 GB source HDF5s are left
  untouched (stage 2 re-reads them for the raw RGB frames).
- **Stage 2** in the `openpi` venv: read source + cache → write the
  single multi-task LeRobot dataset.

Both scripts are resumable at the task level. Stage 1 writes through a
`.tmp` rename so a partial cache is never accepted on the next run.

### Stage 1 — `scripts/cache_robocasa24_cam_matrices.py`

For each of the 24 task HDF5s (defaults to `--source-type human_im`; pass
`--source-type mg_im` instead when you later want the larger MimicGen
source), for each demo:

1. **Construct sim** with `robocasa.utils.env_utils.create_env(...,
   robots="PandaMobile", camera_names=[agentview_left, eye_in_hand], ...)`
   once per task (env is reusable across demos).
2. **Demo init**: `robocasa.scripts.playback_dataset.reset_to(env, {
   "model": demo.attrs["model_file"],
   "ep_meta": demo.attrs["ep_meta"],
   "states": states[0]})` — swaps in the per-demo MJCF + layout/style
   metadata so kinematics resolve correctly.
3. **Frame loop** for `t in range(T)`:
   - `env.sim.set_state_from_flattened(states[t]); env.sim.forward()`
   - For each cam: `R = data.cam_xmat[id].reshape(3,3); p = data.cam_xpos[id]`
     → `T_wc` (4,4) → apply MuJoCo→OpenCV swap (negate cols 1, 2).
4. **K** per cam (constant per episode): from `model.cam_fovy[id]` +
   source image size (128). Stored once per demo; stage 2 scales to 224.
5. **Action permutation**:
   `lerobot_action = hdf5_action[:, [7,8,9,10, 11, 0,1,2, 3,4,5, 6]]`
6. **State assembly** per frame: concat
   `obs["robot0_base_pos"]`, `obs["robot0_base_quat"]`,
   `obs["robot0_base_to_eef_pos"]`, `obs["robot0_base_to_eef_quat"]`,
   `obs["robot0_gripper_qpos"]` → 16-d.
7. **Cache write**: per-demo group in
   `cam_cache_<Task>.h5:/demos/demo_<n>` with
   `{K_agent, K_wrist, T_wc_agent (T,4,4), T_wc_wrist (T,4,4),
   action (T,12), state (T,16), language str}`.

```bash
conda activate robocasa24
cd ~/Research/openpi_cam

# Smoke test on one task first (3 demos, human_im):
python scripts/cache_robocasa24_cam_matrices.py \
  --tasks OpenDrawer --max-demos-per-task 3

# Full run (all 24 tasks, human_im by default; expect ~5-10 min total
# since human_im is only ~54 demos/task):
python scripts/cache_robocasa24_cam_matrices.py

# Later, if you re-pull mg_im for paper-comparable numbers:
python scripts/cache_robocasa24_cam_matrices.py \
  --source-type mg_im \
  --output-dir ~/Research/robocasa24/cam_matrix_cache_mg
```

### Stage 2 — `scripts/convert_robocasa24_to_camaware_lerobot.py`

Per task, iterates the cache demos in numerical order; for each demo:

1. Read RGB frames from the source HDF5
   (`data/<demo>/obs/robot0_{agentview_left,eye_in_hand}_image`, both
   `(T, 128, 128, 3)` uint8).
2. Bilinear resize 128 → 224 and convert HWC → CHW (LeRobot stores CHW).
3. Pinhole rescale K: `K[0,0] *= 224/128; K[0,2] *= 224/128; K[1,1] *=
   224/128; K[1,2] *= 224/128`.
4. Per frame: `dataset.add_frame({observation.state, action, task,
   observation.images.robot0_{agentview_left,eye_in_hand},
   observation.{agentview_left,eye_in_hand}_intrinsic,
   observation.{...}_extrinsic})`.
5. `dataset.save_episode()` after each demo.

```bash
source ~/Research/openpi_cam/.venv/bin/activate
cd ~/Research/openpi_cam

# Smoke test (same 3 demos, human_im):
uv run scripts/convert_robocasa24_to_camaware_lerobot.py \
  --tasks OpenDrawer --max-demos-per-task 3

# Full run (human_im, all 24 tasks → robocasa24/all24_human_camaware):
uv run scripts/convert_robocasa24_to_camaware_lerobot.py

# Later, mg_im version (use a different repo-id so both can coexist):
uv run scripts/convert_robocasa24_to_camaware_lerobot.py \
  --source-type mg_im \
  --cache-dir ~/Research/robocasa24/cam_matrix_cache_mg \
  --repo-id robocasa24/all24_mg_camaware
```

Sanity check after the smoke test: load the dataset with
`LeRobotDataset(repo_id=..., root=...)`, assert the cam-aware columns
exist with shapes `(3,3)` / `(4,4)`, project one known 3-D fixture
point through `K @ inv(T_wc) @ X_world` and confirm it lands at a
plausible pixel on the rendered frame.

### End-to-end data flow

```
~30 GB source (24 tasks × ~1.3 GB human_im):
~/Research/robocasa24/src/robocasa/datasets/v0.1/single_stage/<env>/<Task>/<date>/
    demo_gentex_im128_randcams.hdf5         × 24 tasks
                          │
              ┌───────────┴────────────────┐
              │   (raw images stay here)   │
              │                            │
              │            scripts/cache_robocasa24_cam_matrices.py
              │            (robocasa24 env: MuJoCo replay only)
              │                  • per demo: reset_to({model, ep_meta, states[0]})
              │                  • for t: sim.set_state_from_flattened(states[t])
              │                          read cam_xmat/cam_xpos → T_wc (OpenCV)
              │                  • action[:, PERM]; state from 5 obs channels
              │                  ▼
              │   ~100 MB stage-1 cache (human_im; ~5 GB if mg_im later):
              │   ~/Research/robocasa24/cam_matrix_cache/cam_cache_<Task>.h5
              │       /demos/demo_<n>/{K_agent, K_wrist, T_wc_agent,
              │                        T_wc_wrist, action, state, @language}
              │                            │
              └────────────┬───────────────┘
                           │  scripts/convert_robocasa24_to_camaware_lerobot.py
                           │  (openpi venv: lerobot only, no mujoco/robosuite)
                           │     • read RGB from source HDF5
                           │     • read K/T_wc/action/state/language from cache
                           │     • resize 128 → 224; scale K
                           │     • drop agentview_right entirely
                           │     • dataset.add_frame(...); dataset.save_episode()
                           ▼
~2–4 GB LeRobot v2.1 (videos dominate the bytes; 24 tasks × ~54 demos
== ~1300 episodes for human_im):
$HF_LEROBOT_HOME/robocasa24/all24_human_camaware/
    data/chunk-*/episode_*.parquet
    videos/chunk-*/observation.images.robot0_{agentview_left,eye_in_hand}/.../*.mp4
    meta/{episodes.jsonl, info.json, tasks.jsonl, stats.json}
                           │
                           │  uv run scripts/compute_norm_stats.py
                           │     --config-name pi0_robocasa24_all24_baseline
                           ▼
assets/robocasa24/all24_human_camaware/norm_stats.json
                           │
                           │  (optional, cam-aware Pi3X distill variants only)
                           │  uv run scripts/cache_pi3x_targets.py
                           │     --repo-id robocasa24/all24_human_camaware
                           ▼
~/.cache/openpi/pi3x_targets_224/robocasa24_all24_human_camaware/
                           │
                           │  uv run torchrun ... scripts/train_pytorch.py
                           │     pi0_robocasa24_all24_baseline --exp_name <run>
                           ▼
checkpoints/pi0_robocasa24_all24_baseline/<run>/<step>/
```

`HF_LEROBOT_HOME` defaults to `GEO_ROOT` (parent of `openpi_cam`) per
`scripts/env/activate_env.sh:50`, so the LeRobot output naturally lands
outside the repo. `export HF_LEROBOT_HOME=~/Research` if you want a
specific location.

---

## 5. Training configs

No new DataConfig class — the existing `LeRobotRobocasaCamDataConfig`
already covers the two-cam (right-wrist-padded) case once
`RobocasaCamInputs` is tweaked to make `agentview_right` optional
(see §4).

Add to `src/openpi/training/config.py`:

- `pi0_robocasa24_all24_baseline` — clone of `pi0_robocasa365_baseline`
  (line 3090). `repo_id="robocasa24/all24_human_camaware"` (or
  `..._mg_camaware` once you re-convert from mg_im).
- (Optional, later) `pi0_robocasa24_all24_cam_prope_ray_view_distill_stage1`
  / `_stage2` — cam-aware Pi3X distillation variants, only after the
  baseline is working and the Pi3X teacher quality is verified on these
  scenes (`robocasa365_checklist.md:187-192`).

Norm stats:

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi0_robocasa24_all24_baseline
```

Subsetting (e.g., for the doors/drawers 6-task ablation) is done at the
loader level via episode-index filtering, not by re-converting — the
multi-task LeRobot dataset stays the single source of truth.

---

## 6. Train + eval (multi-task)

```bash
# Train (multi-GPU, all 24 tasks in one run; the prompt conditions the policy)
cd ~/Research/openpi_cam
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<N> \
  scripts/train_pytorch.py \
  pi0_robocasa24_all24_baseline \
  --exp_name robocasa24_all24_baseline
```

Eval (after checkpoint exists): fork
`scripts/debug/eval_robocasa365_remote.py` into
`scripts/debug/eval_robocasa24_remote.py`, looped over all 24 tasks (50
rollouts each, per the v0 paper protocol). The four eval-side fixes
from `robocasa_eval_debug_and_v0_migration.md:28-97` all carry over:

- **Image orientation**: gym frames pass through unflipped.
- **Gripper threshold**: monkey-patch `PandaOmronKeyConverter.unmap_action`
  to threshold at `0.0`, not `0.5`.
- **Control mode**: hard-lock `action.control_mode = -1.0` and
  `robot0_base_mode = -1.0`.
- **Base motion**: zero out base-motion dims with |val| < 0.02.

Re-verify all four for v0 before trusting them — at minimum the action
layout (if it differs) could invalidate the gripper/control_mode dim indices.

Eval shape:

```
log/robocasa24_eval/<exp>/<task>/result.json
log/robocasa24_eval/<exp>/<task>/debug.json
log/robocasa24_eval/<exp>/<task>/videos/
```

50 rollouts per task, binary task success.

---

## 7. HPC scale-up

Once the local conversion + smoke train is green, mirror to Torch:

1. `rsync` / Globus the single multi-task LeRobot dataset (~30–60 GB)
   to `/scratch/yp2841/geometry-vla/robocasa24_lerobot/all24_<source>_camaware/`.
2. Rebuild the `robocasa24` env under `/scratch/yp2841/.venvs/robocasa24/`
   following §1 (with the HPC asset-symlink trick from
   `robocasa365_checklist.md:55-59`).
3. Train `pi0_robocasa24_all24_baseline` on N×A100/H100 via SLURM (clone
   `scripts/sbatch/infer_pi0_robocasa365_baseline_rollout_l40s.sbatch`).
4. Eval all 24 tasks (50 rollouts each) in a single SLURM job array; one
   array task per kitchen task. Aggregate per category (Pick-and-Place,
   Doors/Drawers, Others) and overall — reporting layout in
   `robocasa_eval_debug_and_v0_migration.md:341-366`.
5. Ablation subsets (e.g., doors/drawers-only) are loader-side filters on
   `annotation.human.task_description`, not separate datasets.

---

## 8. Reporting language

Per `robocasa_eval_debug_and_v0_migration.md:130-147`, use precise wording.
**Crucially, name the source split** — `human_im` ("Human-50") and
`mg_im` ("Generated-3000") give very different numbers, and the
published RoboCasa baselines are almost always on Generated-3000.

Example (current human_im baseline):

> We evaluate Pi0 (cam-aware) on the RoboCasa v0 24-task benchmark, *Human-50
> split (~54 human teleop demos/task)*, 50 simulator rollouts per task.
> Success rates are reported per task and as Pick-and-Place / Doors-Drawers
> / Others category averages.

Avoid:
- Comparing a Human-50 result against any literature number unless the
  literature explicitly used the same split (most don't).
- Claiming "RoboCasa-24 SOTA" — paper-comparable numbers require
  Generated-3000 (mg_im).

---

## 9. Open items / risks

- **Action layout drift — RESOLVED.** v0 stores
  `[eef_pos(3), eef_rot(3), gripper(1), base_motion(4), control_mode(1)]`;
  LeRobot target is `[base_motion(4), control_mode(1), eef_pos(3),
  eef_rot(3), gripper(1)]`. Converter applies fixed permutation
  `[7,8,9,10, 11, 0,1,2, 3,4,5, 6]`. Eval client must permute back
  before sim.step.
- **Robot mismatch — v0 uses PandaMobile, not PandaOmron.** The eval
  adapter that we'll fork from `eval_robocasa365_remote.py` references
  `PandaOmronKeyConverter`; confirm whether v0 ships a PandaMobile
  equivalent (or use the same key-converter if the gym wrapper schema is
  identical).
- **`env_version=1.4.1` in the v0 HDF5 metadata** but we're running
  robosuite 1.5.1. Sim reconstruction reads `@model_file` (an explicit
  MJCF), so this should be fine for cam-pose extraction, but worth
  watching if replay-based action checks diverge.
- **Controller** — v0 uses `OSC_POSE` with `control_delta=True`. Matches
  the 365 baseline's `use_delta_joint_actions=False` choice (the v0
  actions are already deltas as stored). Don't enable the openpi delta
  transform on top.
- **Pi3X teacher collapse** on synthetic kitchens
  (`robocasa365_checklist.md:187-192`). Cam-aware Pi3X variants for v0 are
  *out of scope for the starter set*; baseline-only first. Add the Pi3X
  pre-flight diagnostic before any distillation run.

---

## 10. Related docs

- [`pi3x_pi0_system_architecture_and_results.md`](pi3x_pi0_system_architecture_and_results.md)
  — camera-aware Pi3X architecture overview.
- `scripts/env/activate_env.sh` — defines `GEO_ROOT` / `HF_LEROBOT_HOME`
  conventions used here.

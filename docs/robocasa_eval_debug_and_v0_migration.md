# RoboCasa Evaluation Debug Notes and v0 Migration Plan

This note records the RoboCasa365 OpenDrawer baseline failure analysis, the eval
adapter fix that moved success rate away from zero, and the work needed to port
the setup to the older RoboCasa v0 24-task benchmark.

## Current Scope

The result discussed here is not the full RoboCasa365 benchmark and not the
older RoboCasa 24-task benchmark. It is:

- Dataset/config: `robocasa365/OpenDrawer_target_human_camaware`
- Policy config: `pi0_robocasa365_baseline`
- Checkpoint: `robocasa365_opendrawer_baseline_8908268/30000`
- Eval env: `robocasa/OpenDrawer`
- Eval split: `target`
- Rollouts: 50 simulator rollouts, binary task success

Validated result after the eval-side fixes:

- Small sanity eval: 2 / 10 success, success rate 0.20
- Larger eval: 29 / 50 success, success rate 0.58

The 50-episode result is a single-task OpenDrawer specialist score. It should be
reported as "RoboCasa365 OpenDrawer target split, 50 rollouts", not as the full
RoboCasa365 aggregate benchmark score.

## Problems Found

### 1. Eval image orientation was wrong

The eval client previously flipped every gym observation with `img[::-1, :, :]`
before sending it to the policy. The newer checklist and direct frame checks
showed that both LeRobot training frames and RoboCasa gym wrapper observations
are already right-side-up for this pipeline. The extra eval flip created a
train/eval visual mismatch and was the most likely reason the initial rollout
success rate was 0.

Fix: send gym-wrapper camera frames unchanged.

### 2. Gripper threshold was too conservative for this adapter

RoboCasa's `PandaOmronKeyConverter.unmap_action` turns a scalar gripper value
into open/close commands. For eval, the threshold should be 0.0 rather than 0.5:

- `< 0.0` maps to open (`-1.0`)
- `>= 0.0` maps to close (`1.0`)

This matches the normalized model-output convention used by the current
OpenDrawer adapter.

### 3. Control mode must stay in manipulation mode

The atomic kitchen tasks should not enter navigation/base mode. The eval client
therefore hard-locks:

- `action.control_mode = -1.0`
- `robot0_base_mode = -1.0`

This removes a task-irrelevant mode switch from the policy output path.

### 4. Tiny base-motion noise can leak into the simulator

The policy emits a 4D base/torso action block even for atomic drawer tasks. Small
values are not meaningful for this task and can add unnecessary motion noise.

Fix: zero out base-motion components with absolute value below 0.02 during eval.

### 5. Forcing gripper close alone did not solve the zero-success issue

A debug run forced the executed gripper command to close after step 150. That did
produce close commands, but the drawer still did not move reliably. This showed
that the zero-success failure was not just a gripper-threshold issue; visual
orientation and action interpretation also had to be fixed.

### 6. Dataset episode index is not the same as eval seed

An expert-action replay attempt compared dataset episode indices to simulator
`reset(seed=episode_index)` rollouts. The initial states did not match. For
example, an eval seed and the corresponding dataset episode could have different
drawer side and robot base pose.

Conclusion: do not use `env.reset(seed=i)` as a direct replay of LeRobot episode
`i`. Use dataset-provided simulator state or official playback metadata when
testing exact demonstration replay.

### 7. Delta-vs-absolute action semantics need care

The RoboCasa controller config uses delta end-effector commands. However,
OpenPI's generic `DeltaActions` transform subtracts state dimensions and assumes
state/action layouts line up. The RoboCasa365 state and action layouts here do
not align that way, so enabling `use_delta_joint_actions=True` blindly is not a
safe fix.

For the current working baseline, keep the dataset/action layout unchanged and
fix the eval adapter instead.

## Eval Adapter Fix

The working eval client is `scripts/debug/eval_robocasa365_remote.py`.

Important behavior:

- Camera frames are resized and converted HWC -> CHW without vertical flip.
- `PandaOmronKeyConverter.unmap_action` is monkey-patched for eval only.
- Gripper threshold is `0.0`.
- `robot0_base_mode` is fixed to `-1.0`.
- `action.control_mode` is fixed to `-1.0`.
- `action.base_motion` uses a small dead-zone.
- The script can write per-policy-call debug records with state deltas, action
  statistics, executed action statistics, and task-specific drawer/eef metrics.

This is intentionally an eval-side adapter fix. It does not change the model or
the trained checkpoint.

## How to Re-run the Validated Eval

Use the Slurm wrapper:

```bash
sbatch \
  -A torch_pr_769_tandon_advanced \
  -p a100_tandon,h100_tandon,h200_tandon \
  --gres=gpu:1 \
  --export=ALL,OPENPI_DISABLE_TORCH_COMPILE=1,NUM_EPISODES=50,MAX_STEPS=500,VIDEO_EVERY=0,PROMPT=auto,DEBUG_JSON=/scratch/yp2841/geometry-vla/openpi_cam/log/robocasa365_big_infer/debug_passthrough_eval_50ep_multi.json,RESULT_JSON=/scratch/yp2841/geometry-vla/openpi_cam/log/robocasa365_big_infer/result_passthrough_eval_50ep_multi.json \
  scripts/sbatch/infer_pi0_robocasa365_baseline_rollout_l40s.sbatch
```

The script name says `l40s`, but the command-line `sbatch` account, partition,
and GRES override the script defaults.

## Reporting the Result

Use precise wording:

> We evaluate a single-task Pi0 OpenDrawer specialist on the RoboCasa365
> OpenDrawer target split for 50 simulator rollouts and report task success
> rate. The eval adapter fixes image orientation, gripper thresholding, and
> manipulation-mode locking. The resulting success rate is 29/50 = 58%.

Avoid writing:

- "RoboCasa365 benchmark score is 58%"
- "RoboCasa score is 58%"
- "24-task RoboCasa score is 58%"

Those would imply a multi-task aggregate that was not run.

## Should We Move to Older RoboCasa v0?

If the goal is comparison with older VLA and policy papers, yes. Many papers use
the older RoboCasa v0 24-task protocol:

- 24 manipulation atomic tasks
- 50 simulator trials per task
- category averages for Pick-and-Place, Doors/Drawers, Others
- overall average across tasks

RoboCasa365 is more useful for scale and generalist-policy claims, but it is
heavier and less directly comparable to the older 24-task paper ecosystem.

## v0 24-Task Candidate Set

Use the older benchmark's manipulation tasks, excluding navigation:

```text
PnPCounterToCab
PnPCabToCounter
PnPCounterToSink
PnPSinkToCounter
PnPCounterToMicrowave
PnPMicrowaveToCounter
PnPCounterToStove
PnPStoveToCounter
OpenSingleDoor
CloseSingleDoor
OpenDoubleDoor
CloseDoubleDoor
OpenDrawer
CloseDrawer
TurnOnSinkFaucet
TurnOffSinkFaucet
TurnSinkSpout
TurnOnStove
TurnOffStove
CoffeeSetupMug
CoffeeServeMug
CoffeePressButton
TurnOnMicrowave
TurnOffMicrowave
```

Start with 2-3 tasks, not all 24.

Recommended first tasks:

- `OpenDrawer`
- `CloseDrawer`
- `OpenSingleDoor`

These cover the drawer/door action family and keep debugging focused.

## v0 Environment Layout

Keep the existing RoboCasa365 environment intact. Create separate v0 paths:

```text
/scratch/yp2841/geometry-vla/robocasa24_src
/scratch/yp2841/geometry-vla/robocasa24_assets
/scratch/yp2841/geometry-vla/robocasa24_raw
/scratch/yp2841/geometry-vla/robocasa24_lerobot
/scratch/yp2841/geometry-vla/.cache/openpi/robocasa24
/scratch/yp2841/.venvs/robocasa24
```

Do not write environments, datasets, logs, or caches under `/home/yp2841`.

## v0 Data Format Checks

The older RoboCasa datasets are robomimic-style HDF5 files, not the
RoboCasa365 LeRobot cam-aware dataset used by the current baseline.

Before writing a full converter, inspect each downloaded task and print:

- hdf5 dataset root and demo count
- `/data/demo_*/actions.shape`
- `/data/demo_*/action_dict` keys and shapes
- `/data/demo_*/obs` image keys and shapes
- `/data/demo_*` metadata, especially language/task text
- `/data` attributes such as `env_args`
- image orientation for each camera
- whether one demo can be replayed in the simulator and achieve success

Expected useful keys usually look like:

```text
/data/demo_<n>/actions
/data/demo_<n>/action_dict
/data/demo_<n>/obs/...
/data/demo_<n>/states
/data/demo_<n>.attrs["ep_meta"]
/data.attrs["env_args"]
```

The exact schema must be verified from the downloaded v0 data; do not assume it
matches RoboCasa365.

## v0 Conversion Plan

Convert the old HDF5 data to an OpenPI-friendly LeRobot dataset with the same
surface schema as the working RoboCasa365 pipeline where possible:

```text
video.robot0_agentview_left
video.robot0_agentview_right
video.robot0_eye_in_hand
state.base_position
state.base_rotation
state.end_effector_position_relative
state.end_effector_rotation_relative
state.gripper_qpos
action.base_motion
action.control_mode
action.end_effector_position
action.end_effector_rotation
action.gripper_close
annotation.human.task_description
```

For each task:

1. Load HDF5 demos.
2. Verify action layout and controller semantics.
3. Decode or replay images.
4. Extract language/task description.
5. Add camera intrinsics/extrinsics if using the cam-aware policy variants.
6. Write a LeRobot dataset under:

```text
/scratch/yp2841/geometry-vla/robocasa24_lerobot/<TaskName>_<source>_camaware
```

Recommended source order:

- `mg_im` for generated-data benchmark comparability and more demos
- `human_im` for low-data human-demo ablations

## v0 Training Plan

Use staged training rather than jumping directly to 24 tasks.

### Stage 1: single-task smoke

Config:

```text
pi0_robocasa24_opendrawer_baseline
```

Suggested defaults:

- `repo_id=robocasa24/OpenDrawer_mg_camaware`
- `batch_size=8`
- `num_train_steps=30_000`
- one GPU

Goal: 50 rollout success rate is clearly non-zero.

### Stage 2: doors/drawers subset

Config:

```text
pi0_robocasa24_doorsdrawers_baseline
```

Tasks:

- `OpenSingleDoor`
- `CloseSingleDoor`
- `OpenDoubleDoor`
- `CloseDoubleDoor`
- `OpenDrawer`
- `CloseDrawer`

Goal: prompt-conditioned multi-task training works and category average is
stable.

### Stage 3: 24-task benchmark

Config:

```text
pi0_robocasa24_24task_baseline
```

Concatenate the task LeRobot datasets or add a small multi-dataset loader. The
lowest-risk initial implementation is a concatenated LeRobot dataset with
`annotation.human.task_description` preserved for every row.

## v0 Evaluation Plan

Extend the current websocket eval wrapper into a task loop:

```text
for task in 24_tasks:
  env = gym.make(f"robocasa/{task}")
  run 50 rollouts
  save task result json and debug json
```

Report:

- per-task success rate
- Pick-and-Place average
- Doors/Drawers average
- Others average
- overall average

Suggested output layout:

```text
log/robocasa24_eval/<exp>/<task>/result.json
log/robocasa24_eval/<exp>/<task>/debug.json
log/robocasa24_eval/<exp>/<task>/videos/
```

## Migration Risks

Main risks:

- v0 HDF5 action layout may differ from RoboCasa365 LeRobot action layout.
- v0 simulator/controller version may differ from the installed RoboCasa365 env.
- image keys and orientation may differ by dataset source.
- exact demo replay needs simulator states/metadata, not eval seed matching.
- 24-task evaluation cost is much larger than a single OpenDrawer eval.

Low-risk reused pieces:

- OpenPI policy server
- websocket eval architecture
- Slurm wrapper pattern
- gripper thresholding strategy
- manipulation-mode locking
- image-orientation verification method
- 50-rollout success-rate reporting

## Recommended Next Actions

1. Keep the RoboCasa365 OpenDrawer result as the current sanity baseline.
2. Create a separate `robocasa24` env and download only 2-3 v0 tasks.
3. Write a schema inspector for the v0 HDF5 files.
4. Replay one v0 demo exactly using stored simulator state.
5. Convert `OpenDrawer` to LeRobot cam-aware format.
6. Train and eval one v0 single-task baseline.
7. Expand to the doors/drawers subset.
8. Expand to the full 24-task benchmark only after the subset is stable.

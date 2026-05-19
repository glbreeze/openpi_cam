# RoboCasa365 Checklist

Companion to [`robocasa365_hpc_runbook.md`](robocasa365_hpc_runbook.md). The
runbook is reference; this is the **do-it-in-order task list** with current
state and known gotchas.

State as of commit `21fc7e8` (py-torch).

---

## A. Already done

- [x] `robocasa_policy.py` + `LeRobotRobocasaCamDataConfig` + 3 TrainConfigs
      (`pi0_robocasa365_cam_prope_ray_view_distill_fullres_stage1/2`,
      `pi0_robocasa365_baseline`) registered. All 30 LIBERO configs
      AST-identical to pre-change HEAD; LIBERO loader tests 8/8 pass.
- [x] Sim env installable on Torch via `pip install -e robosuite robocasa`
      after pulling repos in §1.3 of the runbook.
- [x] Kitchen assets transferred to Torch:
      `/scratch/yp2841/geometry-vla/robocasa365_kitchen_assets/`
      (24.07 GB, 123 434 files; Globus task `c0c35f86-51e2-11f1-afe3-0afffe4617ab`).
- [x] OpenDrawer LeRobot dataset transferred to Torch:
      `/scratch/yp2841/geometry-vla/robocasa365/OpenDrawer_target/20250816/lerobot/`
      (475.92 MB, 3 607 files; Globus task `262b5dd1-5205-11f1-ac9b-02535127e3d7`).
- [x] Runbook transferred to Torch:
      `/scratch/yp2841/geometry-vla/robocasa365_hpc_runbook.md`.
- [x] Empirical verification of image orientation: LeRobot v2.1 stored
      frames decode right-side-up, gym wrapper output is right-side-up,
      both agree → eval client passes frames through unchanged. K is
      identity passthrough; T_wc still needs MuJoCo→OpenCV swap.
- [x] Empirical verification of action distribution on OpenDrawer demos
      (10 eps, 2 836 frames):
      gripper is **hard-binary** (2044 at ≈-1, 0 in the middle, 792 at ≈+1);
      control_mode constant -1 in 99.5 % of frames;
      base_motion essentially zero (std < 0.003).

---

## B. Blockers before any training launch

In strict dependency order. Each step's output is the next step's input.

### B1. Bring up sim env on Torch (one-time)

- [ ] On a Torch login node with internet:
      ```bash
      cd /scratch/yp2841/geometry-vla
      git clone --depth 1 https://github.com/ARISE-Initiative/robosuite.git
      git clone --depth 1 https://github.com/robocasa/robocasa.git robocasa365_src
      mamba create -n robocasa365 python=3.11 -y && mamba activate robocasa365
      pip install -e robosuite -e robocasa365_src
      pip install robosuite_models 'mink==0.0.5' tyro imageio
      pip install --upgrade 'numpy==2.2.5'
      python robosuite/robosuite/scripts/setup_macros.py
      python robocasa365_src/robocasa/scripts/setup_macros.py
      rm -rf robocasa365_src/robocasa/models/assets
      ln -s /scratch/yp2841/geometry-vla/robocasa365_kitchen_assets \
            robocasa365_src/robocasa/models/assets
      ```
- [ ] Smoke check (still in `robocasa365` env):
      ```bash
      cd /scratch/yp2841/geometry-vla/openpi_cam
      python scripts/debug/verify_robocasa_setup.py --env robocasa/OpenDrawer
      ```
      Expect: fx==fy==221.70 for agentview, 166.81 for eye_in_hand;
      cx==cy==128; `step()` returns no error.

### B2. Build LeRobot enrichment script (open code task)

- [ ] **Write** `scripts/convert_robocasa_to_camaware_lerobot.py` (pattern after
      `scripts/convert_robotwin_cam_to_lerobot.py`). Per episode:
  - Replay actions through `robocasa/<TaskName>` env.
  - On each step, pull `K` from `sim.model.cam_fovy` and `T_wc` from
    `sim.data.cam_xmat` / `cam_xpos` for all 3 cams (formula in
    `scripts/debug/verify_robocasa_setup.py:_dump_cam_k_t_wc`).
  - Write 6 new columns per frame:
    `observation.<cam>_extrinsic` (4,4 float32) and
    `observation.<cam>_intrinsic` (3,3 float32) for
    `cam ∈ {agentview_left, agentview_right, eye_in_hand}`.
  - Output to `/scratch/yp2841/geometry-vla/robocasa365/<task>_camaware/`.
  - K is constant per episode → write same matrix every frame; T_wc must be
    re-read per step (eye_in_hand mounts to wrist; agentview cams mount to
    base which moves during composite tasks).
- [ ] Run for OpenDrawer first as the smoke task.
- [ ] Sanity check: load the enriched dataset and confirm new columns are
      present and have sensible shapes.

### B3. Pi3X target cache (after B2)

- [ ] In the openpi trainer env (not `robocasa365`):
      ```bash
      cd /scratch/yp2841/geometry-vla/openpi_cam
      OPENPI_CACHE_DIR=/scratch/yp2841/geometry-vla/.cache/openpi \
      uv run python scripts/cache_pi3x_targets.py \
        --repo-id robocasa365/OpenDrawer_target_human_camaware \
        --output-dir $OPENPI_CACHE_DIR/pi3x_targets_224/robocasa365_OpenDrawer_target_human_camaware \
        --cam-spec \
          agent:robot0_agentview_left:agentview_left_intrinsic,\
          wrist:robot0_eye_in_hand:eye_in_hand_intrinsic,\
          right_wrist:robot0_agentview_right:agentview_right_intrinsic \
        --image-size 224
      ```
- [ ] **Pre-flight teacher-quality check** (avoid wasting GPU on collapsed
      Pi3X targets like RoboTwin right_wrist):
      ```bash
      uv run python scripts/debug/diagnose_pi3x_target_quality.py \
        --pi3x-root $OPENPI_CACHE_DIR/pi3x_targets_224/robocasa365_OpenDrawer_target_human_camaware \
        --episodes 5 --views agent wrist right_wrist
      ```
      Acceptance: `xy_cosine_mean ≥ 0.3` AND
      `pearson_corr_per_frame_mean ≥ 0.2` per view. If below, Pi3X is
      collapsing on this scene; the distillation signal is noise. Stop and
      reconsider — likely need GT depth via custom enrichment.

### B4. Norm stats

- [ ] ```bash
      uv run python scripts/compute_norm_stats.py \
        --config-name pi0_robocasa365_cam_prope_ray_view_distill_fullres_stage1
      uv run python scripts/compute_norm_stats.py \
        --config-name pi0_robocasa365_baseline
      ```

### B5. Eval-time sim client adapter (can run in parallel with B2-B4)

- [ ] **Write** `scripts/robocasa_eval_policies/sim_client.py` (pattern after
      `scripts/robotwin_eval_policies/`). Required behaviors:
  - **Gripper threshold workaround**: the gym wrapper hard-thresholds
    `gripper_close` at +0.5 (line 112 of `gym_wrapper.py`). Training data
    is hard-binary at ±1 with zero values in (-0.8, +0.8). Adapter should
    binarize pi0's predicted gripper at **0.0** (not 0.5) before sending
    so under-fitting predictions still grip. See §C for full rationale.
  - **control_mode hard-lock to -1.0**: the field is constant -1 in 99.5 %
    of training frames; pi0 noise on this dim could flip the env into
    navigation mode mid-episode. Override pi0's prediction and always
    send -1.0.
  - **Optional base_motion dead-zone**: zero out base_motion dims unless
    `|pred| > 0.02` to suppress jitter.
  - **Action repack**: pi0 returns a 12-d vector in LeRobot order
    `[base_motion(4), control_mode(1), eef_pos(3), eef_rot(3), gripper(1)]`;
    convert to the dict expected by `PandaOmronKeyConverter.unmap_action`.
  - **Image passthrough**: send gym wrapper output unchanged (no flip —
    training and eval orientations agree).

---

## C. Known gotchas, indexed for later debugging

### C1. Gripper hard threshold (the "env noise" suspect)

`unmap_action` in `gym_wrapper.py` uses `pred < 0.5 ? -1 : +1`. Training data
is genuinely hard-binary at ±1 (verified: 2044 frames at -1, 792 at +1, **0
in between**), so the 0.5 threshold is fine when pi0 outputs near ±1.

Failure mode: an under-fitting baseline that smooths toward 0 will never
cross +0.5 → gripper never closes → 0 % success. Mitigation: binarize at
0.0 in the eval client (see B5).

### C2. Control_mode and base_motion are near-constants

Control_mode is -1.0 in 2,821 / 2,836 training frames (99.5 %); base_motion
dims have max\|val\| ≤ 0.16 and std ≤ 0.003. These dims carry essentially
zero training signal but consume pi0 output capacity. Lock them in the eval
client (see B5).

### C3. Action layout differs from Pi0 pretrain

LeRobot stores 12-d action as `[base_motion(4), control_mode(1), eef_pos(3),
eef_rot(3), gripper(1)]`. Pi0 single-arm pretraining starts at `eef_pos`,
so the warm-start mismatch could hurt the baseline. If baseline is bad,
consider reordering the 12-d to put `[eef_pos(3), eef_rot(3), gripper(1),
base(4), ctrl(1)]` at training time and reversing the order in
`RobocasaOutputs`.

### C4. Image orientation (resolved, do not re-debug)

- LeRobot v2.1 stored frames decode right-side-up (verified by direct MP4
  decode of `OpenDrawer/.../episode_000000.mp4`).
- Gym wrapper output is also right-side-up (applies one `[::-1, :, :]` to
  the raw MuJoCo OpenGL buffer).
- No flip exists in the current openpi `preprocess_observation` (JAX) or
  `preprocess_observation_pytorch`.
- Conclusion: K identity passthrough. T_wc still needs the OpenGL→OpenCV
  camera-frame swap (`_mujoco_to_opencv_extrinsic`) because MuJoCo
  `cam_xmat` is in OpenGL frame.

### C5. Pi3X collapse risk on clean synthetic kitchens

We hit this on RoboTwin right_wrist: clean synthetic backgrounds caused the
Pi3X teacher to produce a near-uniform output, killing the distillation
signal. RoboCasa kitchens are equally clean / synthetic. The pre-flight
diagnostic in B3 must pass before committing the full sweep.

### C6. Extrinsics are NOT in the shipped LeRobot data

The upstream LeRobot dataset has only RGB + state + action + lang. The 6
per-frame K/T columns are added by the B2 enrichment script. The cam-aware
TrainConfigs all set `include_cam_extrinsics=True`, which means the data
loader will refuse to start until B2 is done.

---

## D. Day-of training (after A-B complete)

### D1. Stage 1 (5 k steps, geometry modules only, ~3 h on 1× A100)

- [ ] ```bash
      cd /scratch/yp2841/geometry-vla/openpi_cam
      uv run torchrun --standalone --nnodes=1 --nproc_per_node=1 \
        scripts/train_pytorch.py \
        pi0_robocasa365_cam_prope_ray_view_distill_fullres_stage1 \
        --exp_name robocasa365_opendrawer_s1
      ```

### D2. Stage 2 (30 k steps, full unfreeze)

- [ ] ```bash
      STAGE1=/scratch/yp2841/geometry-vla/openpi_cam/checkpoints/\
      pi0_robocasa365_cam_prope_ray_view_distill_fullres_stage1/\
      robocasa365_opendrawer_s1/5000

      uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 \
        scripts/train_pytorch.py \
        pi0_robocasa365_cam_prope_ray_view_distill_fullres_stage2 \
        --exp_name robocasa365_opendrawer_s2 \
        --pytorch_weight_path $STAGE1
      ```

### D3. Baseline (30 k steps, A/B comparator — no cam-aware, no Pi3X)

- [ ] ```bash
      uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 \
        scripts/train_pytorch.py \
        pi0_robocasa365_baseline \
        --exp_name robocasa365_opendrawer_baseline
      ```

---

## E. Eval (after Stage 2 / baseline checkpoints exist)

- [ ] Policy server (openpi env):
      ```bash
      uv run python scripts/serve_policy.py policy:checkpoint \
        --policy.config pi0_robocasa365_cam_prope_ray_view_distill_fullres_stage2 \
        --policy.dir <ckpt>/30000 --port 8000
      ```
- [ ] Sim client (`robocasa365` env, B5 adapter):
      ```bash
      python scripts/robocasa_eval_policies/sim_client.py \
        --env robocasa/OpenDrawer --policy-host 127.0.0.1 --policy-port 8000 \
        --n-episodes 50
      ```
- [ ] Compare `robocasa365_opendrawer_s2` vs `robocasa365_opendrawer_baseline`
      success rates. Apples-to-apples: same dataset, same num_train_steps,
      only difference is the cam-aware + Pi3X-distill geometry path.

---

## F. Scale-up after OpenDrawer succeeds

- [ ] Add 4 more atomic tasks (suggested: `PnPCounterToSink`, `TurnOnStove`,
      `CloseSingleDoor`, `CoffeeServeMug`).
- [ ] Re-run B2-B5 per task. Clone the 3 TrainConfigs per task or pass
      `--data.repo_id robocasa365/<task>_camaware` at launch.

---

## G. References

- Runbook (sibling, more verbose): `docs/robocasa365_hpc_runbook.md`
- Commit `21fc7e8` (py-torch): first revision with the orientation
  correction (LeRobot is right-side-up; K identity passthrough).
- Commit `e1ab1c3` (py-torch): initial RoboCasa policy + configs + verifier.
- Globus tasks: `c0c35f86-...` (kitchen assets), `262b5dd1-...` (OpenDrawer
  LeRobot), `28b525e5-...` (runbook).

# HPC Runbook — LIBERO All-Suite Regeneration

**Goal:** produce a single LeRobot v2.1 dataset, `glbreeze/libero_cam_v2`, containing the four standard LIBERO suites (object, spatial, goal, 10) regenerated with the OpenVLA-parity pipeline:
- native 256×256 render
- no-op filter (1e-4 threshold, OpenVLA-equivalent)
- success-gated (failed replays dropped)
- `(obs_t, action_t)` pairing
- per-frame agent + wrist extrinsics, per-episode K, joint_state

**Final deliverable:** the LeRobot repo dir under `$HF_LEROBOT_HOME/glbreeze/libero_cam_v2/` containing `meta/` + `data/chunk-*/episode_*.parquet`. Plus `norm_stats.json` for the training config (Step 7).

**Expected scale:** ~1900 demos × ~150 frames ≈ ~340k frames, ~325 GB on disk for the LeRobot repo + ~125 GB for the intermediate HDF5.

---

## 0. Prerequisites

### Hardware
- 1+ GPU (MuJoCo offscreen render uses EGL). 1 GPU is enough; 4 GPUs cuts wall time roughly 4× via per-suite parallelism (Step 5).
- ~500 GB free disk on the volume that will host both `$OUT_HDF5_ROOT` and `$HF_LEROBOT_HOME`.

### Software
- Linux, CUDA-capable driver (only used for MuJoCo render; no CUDA Python deps).
- `conda` ≥ 4.12.
- `uv` ≥ 0.4 (`pip install uv` if missing; the openpi repo expects this).
- `git` + `git-lfs` (LFS only needed for openpi base weights, not for this regen — see Step 1.b).

### Authority
- Read access to the source LIBERO HDF5 dataset.
- Write access to the chosen `$OUT_HDF5_ROOT` and `$HF_LEROBOT_HOME` paths.
- No network access required after Step 1 — everything is local sim + local file I/O.

---

## 1. Repo + data setup

### 1.a Pull repos
```bash
mkdir -p "$WORK_DIR" && cd "$WORK_DIR"

# Required commits (or any later commit on the same branch):
git clone -b main git@github.com:pengyue-polaron/LIBERO-Camera.git
git -C LIBERO-Camera log --oneline -1   # expect b81da63 or later

git clone -b py-torch git@github.com:glbreeze/openpi_cam.git openpi
git -C openpi log --oneline -1          # expect 4d01062 or later
```

If the HPC has no internet, sync these repos in via rsync/scp from the source machine (`/home/asus/Research/LIBERO-Camera`, `/home/asus/Research/openpi`).

### 1.b Install deps

**openpi** (uv-managed, ~5 min, no GPU needed for install):
```bash
cd "$WORK_DIR/openpi"
GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```
The LFS skip is intentional — base model weights are not needed for data regen.

**libero conda env** (this is the one MuJoCo runs in):
```bash
cd "$WORK_DIR"
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git libero
cd libero && pip install -r requirements.txt   # inside a fresh conda env
# OR if a conda yml is provided in LIBERO-Camera:
#   conda env create -f LIBERO-Camera/libero/environment.yml
```

The env must satisfy:
- `python -c "import robosuite, h5py, mujoco; from libero.libero import get_libero_path"` exits cleanly.
- `python -c "from libero.libero.envs import OffScreenRenderEnv"` exits cleanly.

The expected env name is `libero`. Override at run time with `LIBERO_CONDA_ENV=<name>`.

### 1.c Source data layout

Required (read-only):
```
$SRC_BASE/
  libero_object/   *_demo.hdf5   (10 files)
  libero_spatial/  *_demo.hdf5   (10 files)
  libero_goal/     *_demo.hdf5   (10 files)
  libero_10/       *_demo.hdf5   (10 files)
```

Each `*_demo.hdf5` is the original LIBERO demonstration file (50 demos per file, 110-state vectors, no extrinsics). Total source is ~32 GB across the 4 suites.

Verify counts:
```bash
for s in libero_object libero_spatial libero_goal libero_10; do
    n=$(ls "$SRC_BASE/$s"/*_demo.hdf5 2>/dev/null | wc -l)
    echo "$s: $n files"
done
# Expect: libero_object: 10, libero_spatial: 10, libero_goal: 10, libero_10: 10
```

### 1.d Disk free check

```bash
df -BG "$OUT_HDF5_ROOT_PARENT" "$HF_LEROBOT_HOME_PARENT"
# Need ≥ 500 GB free on the volume(s) hosting OUT_HDF5_ROOT and HF_LEROBOT_HOME.
# (Both can share one volume.)
```

---

## 2. Choose paths

Set these env vars in the shell that will run the script. The script defaults are tuned for the source machine; on HPC, override:

```bash
export WORK_DIR=/scratch/$USER/libero_regen
export LIBERO_CAMERA_REPO=$WORK_DIR/LIBERO-Camera
export OPENPI_REPO=$WORK_DIR/openpi

export SRC_BASE=/scratch/$USER/data/libero_datasets        # has libero_*/ subdirs
export OUT_HDF5_ROOT=/scratch/$USER/data/libero_cam_v2     # regen target
export LIBERO_CFG_DIR=/scratch/$USER/data/libero_cam_v2_cfg
export HF_LEROBOT_HOME=/scratch/$USER/cache/lerobot         # LeRobot cache
export REPO_ID=glbreeze/libero_cam_v2

export LIBERO_CONDA_ENV=libero
export RENDER_RES=256
export SETTLE_STEPS=10
```

The orchestrator script is `$OPENPI_REPO/scripts/regen_libero_all.sh`. It already reads every variable above from the environment (see top of file).

---

## 3. Pre-flight (~30 s)

A no-touch dry run of the safety checks:

```bash
SKIP_PHASE1=1 SKIP_PHASE2=1 bash $OPENPI_REPO/scripts/regen_libero_all.sh
```

Exits 0 if every source dir is present and non-empty. Exits non-zero with a clear message otherwise. **Do not proceed past this until exit 0.**

---

## 4. Smoke run (~3 min)

End-to-end validation on tiny data, before launching the multi-hour job. Uses throwaway paths, never touches the real targets.

```bash
MAX_EPISODES=2 \
REPO_ID=glbreeze/_smoketest_libero_all \
OUT_HDF5_ROOT=/tmp/libero_all_smoke \
LIBERO_CFG_DIR=/tmp/libero_all_smoke_cfg \
HF_LEROBOT_HOME=/tmp/libero_all_smoke_lr \
bash $OPENPI_REPO/scripts/regen_libero_all.sh
```

Success criteria:
- Phase 1 prints `[regen done] <suite>:` for each of the 4 suites.
- Phase 2 prints `[convert done] LeRobot repo at: ...`.
- The smoke LeRobot repo exists with `meta/info.json` reporting `codebase_version: v2.1`.

Verify schema after smoke:
```bash
HF_LEROBOT_HOME=/tmp/libero_all_smoke_lr uv run --project $OPENPI_REPO python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('glbreeze/_smoketest_libero_all')
print('episodes:', ds.num_episodes, 'frames:', ds.num_frames)
assert 'joint_state' in ds.features
assert ds.features['image']['shape'] == (256, 256, 3)
assert ds.features['agent_extrinsic']['shape'] == (4, 4)
print('schema OK')
"
```

If smoke fails, the most likely causes are:
- BDDL paths in source HDF5 not resolvable from `LIBERO_CAMERA_REPO`. Check `LIBERO_CONFIG_PATH` was set correctly and that the libero env can find `libero/libero/bddl_files/` from the script's cwd.
- MUJOCO_GL not set to `egl` (the wrapper script sets it; if running create_dataset.py manually, set `MUJOCO_GL=egl`).
- libero conda env missing `robosuite` or `mujoco`.

Cleanup smoke artifacts:
```bash
rm -rf /tmp/libero_all_smoke /tmp/libero_all_smoke_cfg /tmp/libero_all_smoke_lr
```

---

## 5. Full run (~1.5 hr on a single H100; ~50 min on 4 GPUs in parallel)

### Single-GPU (simplest)

```bash
nohup bash $OPENPI_REPO/scripts/regen_libero_all.sh \
    > $WORK_DIR/regen_all.log 2>&1 &
echo $! > $WORK_DIR/regen_all.pid
```

Monitor:
```bash
tail -f $WORK_DIR/regen_all.log
```

### Multi-GPU parallel (recommended on HPC)

Run 4 suites simultaneously, then a single Phase 2 pass at the end:

```bash
# Phase 1, parallel across suites
for i in 0 1 2 3; do
    suite=( libero_object libero_spatial libero_goal libero_10 )
    SUITE=${suite[$i]}
    CUDA_VISIBLE_DEVICES=$i SUITES=$SUITE SKIP_PHASE2=1 \
        nohup bash $OPENPI_REPO/scripts/regen_libero_all.sh \
        > $WORK_DIR/regen_$SUITE.log 2>&1 &
    echo "launched $SUITE on GPU $i"
done
wait

# Phase 2 — single pass on combined root, no GPU needed
SKIP_PHASE1=1 bash $OPENPI_REPO/scripts/regen_libero_all.sh \
    | tee $WORK_DIR/convert_all.log
```

### SLURM template (if applicable)

```bash
#!/bin/bash
#SBATCH --job-name=libero_regen_all
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=libero_regen_all.%j.out

set -euo pipefail
source /etc/profile.d/conda.sh

# Set paths from Step 2 here
export WORK_DIR=/scratch/$USER/libero_regen
... (rest of Step 2 exports)

# Run full multi-GPU pipeline (paste the multi-GPU block above)
```

### Resumability

The orchestrator passes `--resume` to `create_dataset.py`, so:
- Re-running after a crash mid-Phase-1 skips already-finished HDF5 files.
- It is safe to kill and relaunch.
- Phase 2 itself is not resumable (it `rmtree`s the target repo at start). If it gets interrupted, just rerun with `SKIP_PHASE1=1`.

---

## 6. Verify the output (~5 min)

After the run completes, run these checks. **All must pass before declaring success.**

### 6.a HDF5 sanity (per-suite counts and attrs)

```bash
conda activate $LIBERO_CONDA_ENV
python <<'PY'
import h5py, glob, os
total = 0
for suite in ("libero_object", "libero_spatial", "libero_goal", "libero_10"):
    files = sorted(glob.glob(f"{os.environ['OUT_HDF5_ROOT']}/{suite}/*_demo.hdf5"))
    assert len(files) == 10, f"{suite}: expected 10 HDF5 files, found {len(files)}"
    suite_demos = 0
    for p in files:
        f = h5py.File(p, "r"); g = f["data"].attrs
        assert g["render_resolution"] == 256, f"{p}: render_resolution != 256"
        assert g["filter_noops"] == True, f"{p}: filter_noops != True"
        assert g["settle_steps"] == 10, f"{p}: settle_steps != 10"
        assert g["num_demos"] > 0, f"{p}: 0 demos written"
        suite_demos += int(g["num_demos"])
        f.close()
    print(f"{suite}: {len(files)} files, {suite_demos} demos written")
    total += suite_demos
print(f"TOTAL: {total} demos")
assert total >= 1700, f"too few demos kept ({total} < 1700) — check failure rate"
PY
conda deactivate
```

Expected total: ~1850 ± 50 demos (sum across 4 suites; ~92% success rate).

### 6.b LeRobot dataset sanity

```bash
uv run --project $OPENPI_REPO python <<'PY'
import json, os
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

repo = os.environ["REPO_ID"]
info = json.loads((HF_LEROBOT_HOME / repo / "meta" / "info.json").read_text())
assert info["codebase_version"] == "v2.1", info["codebase_version"]
assert info["fps"] == 10
assert info["robot_type"] == "panda"

ds = LeRobotDataset(repo)
assert ds.num_episodes >= 1700, ds.num_episodes
required_features = {
    "image", "wrist_image", "state", "joint_state", "actions",
    "agent_extrinsic", "wrist_extrinsic", "agent_intrinsic", "wrist_intrinsic",
}
assert required_features <= set(ds.features), required_features - set(ds.features)
assert ds.features["image"]["shape"] == (256, 256, 3)
assert ds.features["state"]["shape"] == (8,)
assert ds.features["joint_state"]["shape"] == (7,)
assert ds.features["actions"]["shape"] == (7,)
assert ds.features["agent_extrinsic"]["shape"] == (4, 4)

# Sample one frame end-to-end
f = ds[0]
print(f"sample image:           {tuple(f['image'].shape)}")
print(f"sample agent_intrinsic: \n{f['agent_intrinsic'].numpy()}")
print(f"OK: {ds.num_episodes} episodes, {ds.num_frames} frames")
PY
```

Expected:
- `codebase_version: v2.1`
- ~1850 episodes, ~340k frames
- `agent_intrinsic` ≈ `[[309.02, 0, 128], [0, 309.02, 128], [0, 0, 1]]`

### 6.c Geometric correctness (one episode, ~5 s)

The strongest single check — the EEF should project to a constant pixel in the wrist view across all frames of an episode (the wrist camera is rigidly mounted on the EEF). Sub-pixel scatter ⇒ K, extrinsic, and image are mutually consistent.

```bash
conda activate $LIBERO_CONDA_ENV
python <<'PY'
import h5py, numpy as np, os, glob
src = sorted(glob.glob(f"{os.environ['OUT_HDF5_ROOT']}/libero_object/*_demo.hdf5"))[0]
f = h5py.File(src, "r")
ep = f["data"][sorted(f["data"].keys())[0]]
ee = ep["obs/ee_states"][:]
wrist_ext = ep["obs/wrist_extrinsic"][:]
wrist_K = np.asarray(ep["obs"].attrs["wrist_intrinsic"])
H = int(ep["obs"].attrs["agent_image_size"][0])

def m2o(T): T=T.copy(); T[...,:3,1]*=-1; T[...,:3,2]*=-1; return T
def proj(p, K, To):
    R, t = To[:3,:3], To[:3,3]; pc = R.T @ (p - t)
    if pc[2] <= 0: return None
    uvw = K @ pc; return uvw[0]/uvw[2], uvw[1]/uvw[2]

T = len(ee)
projections = []
for t in [0, T//4, T//2, 3*T//4, T-1]:
    uv = proj(ee[t,:3], wrist_K, m2o(wrist_ext[t]))
    assert uv is not None, f"frame {t}: EEF behind wrist camera"
    u, v = uv[0], (H-1)-uv[1]
    projections.append((u, v))
    print(f"  frame {t:3d}: ({u:6.1f}, {v:6.1f})")

us = [p[0] for p in projections]; vs = [p[1] for p in projections]
u_scatter = max(us) - min(us); v_scatter = max(vs) - min(vs)
print(f"u scatter: {u_scatter:.2f} px,  v scatter: {v_scatter:.2f} px")
assert u_scatter < 2.0 and v_scatter < 2.0, "wrist projection drift > 2 px — geometry inconsistent!"
print("OK: rigid wrist mount geometry preserved")
f.close()
PY
conda deactivate
```

Expected: scatter < 1 px in both u and v. Anything > 2 px is a red flag — investigate before proceeding.

### 6.d Disk-size sanity

```bash
du -sh "$OUT_HDF5_ROOT" "$HF_LEROBOT_HOME/$REPO_ID"
# Expect: HDF5 ~125 GB, LeRobot ~325 GB
```

If LeRobot is < 200 GB, something is wrong (probably truncated / image encoding failed silently).

---

## 7. Compute norm stats (~10 min)

The training config needs per-feature normalization stats. They are NOT computed by the regen script.

First, edit `$OPENPI_REPO/src/openpi/training/config.py` so the consuming TrainConfig points at the new repo. There are two camera-aware blocks worth touching:

- `pi0_libero_cam` (legacy combined LIBERO config). Change:
  ```python
  repo_id=f"{HF_NAME}/libero_cam_v2",       # was "libero_cam"
  asset_id=f"{HF_NAME}/libero_cam_v2",      # was "libero_cam"
  ```
- `pi0_libero_cam_pytorch_prope_ray_view` (active PRoPE+ray config). Same edit, but its current `repo_id` is `libero_object_cam`. If you want this config to train on all four suites, change to `libero_cam_v2`. If you want to keep it object-only, leave it pointed at `libero_object_cam_v2` (a different repo, not produced by this run).

Then compute:
```bash
cd $OPENPI_REPO
uv run scripts/compute_norm_stats.py --config-name pi0_libero_cam
# (or whatever config name you wired up)
```

Output: `<assets_dir>/<asset_id>/norm_stats.json`. The path is printed at the end of the run.

Verify:
```bash
uv run --project $OPENPI_REPO python -c "
import json, pathlib
p = pathlib.Path('<the path printed by compute_norm_stats>')
stats = json.loads((p / 'norm_stats.json').read_text())
print('keys:', list(stats['norm_stats'].keys()))
print('state mean[:4]:', stats['norm_stats']['state']['mean'][:4])
print('actions mean :', stats['norm_stats']['actions']['mean'])
"
# Expect keys: ['state', 'actions']. State[0:3] mean should be ~workspace EEF positions
# (small numbers, magnitude < 0.5 m). Actions mean should be small (delta actions).
```

---

## 8. Hand-back

What to ship back to the source machine (or downstream user):

| Artifact | Path | Size | Required? |
|---|---|---|---|
| LeRobot v2.1 repo | `$HF_LEROBOT_HOME/$REPO_ID/` | ~325 GB | **Yes** — primary deliverable |
| `meta/info.json` (sanity-check copy) | `$HF_LEROBOT_HOME/$REPO_ID/meta/info.json` | <10 KB | as part of repo |
| Norm stats | `<assets_dir>/<asset_id>/norm_stats.json` | <10 KB | **Yes** — needed before training |
| HDF5 intermediate | `$OUT_HDF5_ROOT/` | ~125 GB | **Optional** — can be regenerated; keep only if you anticipate re-converting at a different `--image-size` or with `--mode all_views` |
| Run logs | `$WORK_DIR/regen_*.log`, `$WORK_DIR/convert_all.log` | <50 MB | for debugging |

Suggested transfer:
```bash
# From HPC, push LeRobot repo + norm stats:
rsync -av --info=progress2 \
    "$HF_LEROBOT_HOME/$REPO_ID/" \
    "user@source-machine:/home/asus/.cache/huggingface/lerobot/$REPO_ID/"

rsync -av "$ASSETS_DIR/$REPO_ID/norm_stats.json" \
    "user@source-machine:$ASSETS_DIR/$REPO_ID/norm_stats.json"
```

---

## 9. Failure modes — quick triage

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError: libero` in Phase 1 | wrong conda env | `conda activate $LIBERO_CONDA_ENV`; verify `python -c "from libero.libero import get_libero_path"` works |
| `RuntimeError: Failed to create OpenGL context` | MUJOCO_GL not set | the script sets `MUJOCO_GL=egl`; ensure no override; `nvidia-smi` shows GPU available |
| Phase 1 hangs on first demo | bddl path not found | verify `$LIBERO_CFG_DIR/config.yaml` `bddl_files:` points at a real dir; verify cwd for create_dataset.py is `$LIBERO_CAMERA_REPO` |
| Phase 2 errors `KeyError: 'agent_extrinsic'` | source HDF5 from old pipeline | regenerate Phase 1 fully — old HDF5 lacks the camera fields |
| `FileExistsError` on Phase 2 start | `$HF_LEROBOT_HOME/$REPO_ID` already exists | the script refuses to overwrite; `rm -rf` it or change `REPO_ID` |
| Geometric check (6.c) fails | the off-by-one fix isn't in `create_dataset.py` | `git -C $LIBERO_CAMERA_REPO log --oneline scripts/create_dataset.py` — must include commit `171d1fb` (Align create_dataset with OpenVLA) |
| LeRobot repo size much smaller than 325 GB | converter died mid-run | check `$WORK_DIR/convert_all.log` for tracebacks; `rm -rf $HF_LEROBOT_HOME/$REPO_ID` and re-run with `SKIP_PHASE1=1` |
| `compute_norm_stats.py` crashes | `repo_id` in config.py not updated to `libero_cam_v2` | re-edit the config block per Step 7 |

---

## 10. Reference: relevant repo files

For the HPC agent who needs to read further:

- `LIBERO-Camera/scripts/create_dataset.py` — the per-task regen entry point. CLI documented in `LIBERO-Camera/DATASET_CALIBRATION.md` §4.
- `LIBERO-Camera/DATASET_CALIBRATION.md` — full description of every change vs the legacy pipeline, including OpenVLA parity comparison.
- `openpi/scripts/regen_libero_object.sh` — single-suite version of the orchestrator (used to generate the existing `libero_object_cam_v2` repo as a reference).
- `openpi/scripts/regen_libero_all.sh` — the all-suite orchestrator this runbook drives.
- `openpi/examples/libero/convert_libero_hdf5_to_lerobot.py` — Phase 2 converter; `_iter_hdf5_files` does `rglob("*.hdf5")` so combining suites is just "point at the parent dir".
- `openpi/src/openpi/training/config.py` — TrainConfig blocks, in particular `pi0_libero_cam` (~L668) and `pi0_libero_cam_pytorch_prope_ray_view` (~L725).

---

## Success criteria (checklist for the agent)

- [ ] Step 3 pre-flight exits 0
- [ ] Step 4 smoke completes; LeRobot smoke repo has correct schema
- [ ] Phase 1 completes without traceback in any suite log
- [ ] Phase 2 completes; final line includes `[convert done] LeRobot repo at: ...`
- [ ] Step 6.a passes (HDF5 attrs uniform, ~1850 demos total)
- [ ] Step 6.b passes (LeRobot v2.1, all required features, K matches expected)
- [ ] Step 6.c passes (wrist projection scatter < 2 px)
- [ ] Step 6.d disk size ~325 GB LeRobot, ~125 GB HDF5
- [ ] Step 7 norm_stats.json exists and has reasonable means
- [ ] Step 8 artifacts shipped back

If all pass, hand back the LeRobot repo + norm_stats.json. If any fail, do **not** ship; report the failing step and the relevant log lines.

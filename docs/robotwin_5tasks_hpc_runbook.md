# RoboTwin 5-Task HPC Training Runbook

How to train the cam-aware Pi0 + Pi3X mixed-GT distillation recipe on 6 RoboTwin
tasks (5 selected for cam-aware benefit + the original `beat_block_hammer` smoke
task) on **NYU Tandon Torch HPC**.

This is the HPC follow-up to [`robotwin_cam_pipeline.md`](robotwin_cam_pipeline.md).
That doc covers data prep on a local 4090; this doc assumes the data has already
been transferred to Torch via Globus (task IDs in the bundle README) and walks
through training on Torch only.

---

## 0. Tasks covered

| Task | LeRobot eps / frames | Notes |
|---|---|---|
| `beat_block_hammer` | 50 / 5,737 | original smoke task |
| `handover_block` | 50 / 14,087 | dual-arm coordination |
| `stack_blocks_three` | 50 / 23,723 | precise vertical alignment |
| `place_container_plate` | 50 / 7,966 | fine 3D placement |
| `place_dual_shoes` | 50 / 11,490 | bilateral semantic constraint |
| `lift_pot` | 50 / 5,601 | bimanual coordinated lift |

For each task: LeRobot v2.1 dataset + Pi3X cache + Sapien-GT cache + norm_stats
are already on Torch under `/scratch/yp2841/geometry-vla/`.

---

## 1. Layout on Torch

After the Globus transfer (task `7634b636-4b56-11f1-ae1a-0afffe4617ab`):

```
/scratch/yp2841/geometry-vla/
├── openpi_cam/                                  # this repo (clone here)
├── pi0_base/                                    # pi0 PyTorch base ckpt
├── robotwin/                                    # 6 LeRobot v2.1 datasets
│   ├── beat_block_hammer_demo_clean_camaware_50/
│   ├── handover_block_demo_clean_camaware_50/
│   ├── stack_blocks_three_demo_clean_camaware_50/
│   ├── place_container_plate_demo_clean_camaware_50/
│   ├── place_dual_shoes_demo_clean_camaware_50/
│   └── lift_pot_demo_clean_camaware_50/
├── .cache/openpi/                               # caches
│   ├── pi3x_targets_224/robotwin_*_demo_clean_camaware_50/
│   └── gt_point_targets_224/robotwin_*_demo_clean_camaware_50/
└── pi0_libero/robotwin/                         # norm_stats (one per task)
    └── <task>_demo_clean_camaware_50/norm_stats.json
```

The data loader reads `.cache/openpi/...` from `OPENPI_CACHE_DIR` (set to
`${GEO_ROOT}/.cache/openpi/` by the existing sbatch templates), so no
symlinking is required.

---

## 2. One-time setup on Torch

### 2.a Clone the repo and install the env

```bash
cd /scratch/yp2841/geometry-vla
git clone -b py-torch git@github.com:glbreeze/openpi_cam.git
# or if you already have it: git -C openpi_cam pull
cd openpi_cam
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

The repo's `scripts/env/activate_env.sh` resolves `GEO_ROOT=$(dirname openpi_cam)`
and exports `HF_LEROBOT_HOME=${GEO_ROOT}` automatically.

### 2.b pi0 base checkpoint

Already at `/scratch/yp2841/geometry-vla/pi0_base/` (carried over from your
LIBERO setup). The training script reads `OPENPI_PI0_BASE_DIR=${GEO_ROOT}/pi0_base`.

### 2.c Pi3X ray_embed init

Already in the repo at `assets/pi3x_init/ray_embed.pt`.

### 2.d Verification

Per-task sanity check:

```bash
GEO_ROOT=/scratch/yp2841/geometry-vla
for T in beat_block_hammer handover_block stack_blocks_three place_container_plate place_dual_shoes lift_pot; do
    REPO_ID=robotwin/${T}_demo_clean_camaware_50
    SLUG=robotwin_${T}_demo_clean_camaware_50
    LR=$([ -f ${GEO_ROOT}/${REPO_ID}/meta/info.json ] && echo ✓ || echo ✗)
    PI=$([ "$(ls ${GEO_ROOT}/.cache/openpi/pi3x_targets_224/${SLUG}/agent 2>/dev/null | wc -l)" -ge 50 ] && echo ✓ || echo ✗)
    GT=$([ "$(ls ${GEO_ROOT}/.cache/openpi/gt_point_targets_224/${SLUG}/agent 2>/dev/null | wc -l)" -ge 50 ] && echo ✓ || echo ✗)
    NS=$([ -f ${GEO_ROOT}/pi0_libero/${REPO_ID}/norm_stats.json ] && echo ✓ || echo ✗)
    printf "%-25s LR:%s PI:%s GT:%s NS:%s\n" "$T" "$LR" "$PI" "$GT" "$NS"
done
```

All four checkmarks needed per task.

---

## 3. Choose the recipe

Two recipes are available, both implementing the cam-aware Pi0 + point-head
distillation chain. Pick **one** and stick with it across Stage 1 + Stage 2 for
a given task:

| Recipe (config name suffix) | Cross-view shape | Distillation | Mirrors LIBERO recipe |
|---|---|---|---|
| `_distill_fullres_stage{1,2}` | `aa_order="fg"`, `prope_layer_idx=(0,)` (single-block) | **Pi3X-only** (no GT) | `pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage{1,2}` |
| `_distill_fullres_stage{1,2}_gtdual` | `aa_order="fgfg"`, `prope_layer_idx=(0,1)` (two-block deeper) | **Pi3X + Sapien-GT dual loss** (α·L_GT + (1-α)·L_Pi3X, α=0.5) | `pi0_libero_cam_pytorch_prope_ray_view_distill_fullres_stage{1,2}_gtdual` |

The vanilla recipe is the closer apples-to-apples comparison vs your prior
LIBERO numbers; the gtdual variant adds simulator-GT depth supervision and a
deeper cross-view fusion. Both use the same Pi3X cache; gtdual additionally
reads the Sapien-GT cache.

Pick:
- **`stage{1,2}` (Pi3X-only)** if you want to match the LIBERO baseline recipe one-for-one.
- **`stage{1,2}_gtdual`** if you want the strongest result (deeper fusion + GT supervision).

The sections below describe Stage 1 / Stage 2 / launch loop / wall-time for
both recipes side by side.

## 3a. Stage 1 training (5,000 steps, freeze backbone)

### Vanilla recipe: `pi0_robotwin_cam_prope_ray_view_distill_fullres_stage1`

- `pose_enc=prope`, `ray_enc=True`, `view_enc=False`, single-block fg cross-view fusion with PRoPE on layer (0,)
- AuxPointHead at output_resolution=224, loss_weight=1.0
- `action_loss_weight=0.1`
- Trainable: `cross_view_fusion`, `ray_embed`, `aux_point_head`
- Loss: **L(Pi3X)** only (no GT)
- LR: cosine, warmup 500, peak 2.5e-5, decay over 5,000 to 2.5e-6

Submit per task with `scripts/sbatch/train_pi0_robotwin_s1_h200.sbatch`:

```bash
sbatch --export=ALL,TASK=handover_block        scripts/sbatch/train_pi0_robotwin_s1_h200.sbatch
sbatch --export=ALL,TASK=stack_blocks_three    scripts/sbatch/train_pi0_robotwin_s1_h200.sbatch
sbatch --export=ALL,TASK=place_container_plate scripts/sbatch/train_pi0_robotwin_s1_h200.sbatch
sbatch --export=ALL,TASK=place_dual_shoes      scripts/sbatch/train_pi0_robotwin_s1_h200.sbatch
sbatch --export=ALL,TASK=lift_pot              scripts/sbatch/train_pi0_robotwin_s1_h200.sbatch
sbatch --export=ALL,TASK=beat_block_hammer     scripts/sbatch/train_pi0_robotwin_s1_h200.sbatch
```

Stage 2 launches the same way with `train_pi0_robotwin_s2_h200.sbatch` after
Stage 1 succeeds (see §6 for the dependency-chain loop).

### gtdual variant: `pi0_robotwin_cam_prope_ray_view_distill_fullres_stage1_gtdual`

- `pose_enc=prope`, `ray_enc=True`, `view_enc=False`, fgfg cross-view fusion with PRoPE on layers (0,1)
- AuxPointHead at output_resolution=224, loss_weight=1.0
- `action_loss_weight=0.1`
- ray_embed warm-started from `assets/pi3x_init/ray_embed.pt`
- Trainable: `cross_view_fusion`, `ray_embed`, `aux_point_head`
- Loss: α·L(GT) + (1-α)·L(Pi3X), α=0.5
- LR: cosine, warmup 500, peak 2.5e-5, decay over 5,000 to 2.5e-6

### gtdual SBATCH template

Use the same shape as your existing
`train_pi0_libero_cam_v3_prope_ray_view_distill_fullres_stage1_gtdual_v3_h200_4gpu_b32.sbatch`,
just swap config + dataset/asset to RoboTwin per-task. A self-contained example:

```bash
#!/bin/bash
#SBATCH --job-name=pi0_robotwin_s1_gtdual
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_robotwin_s1/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_robotwin_s1/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_637_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:h200:4
#SBATCH --export=ALL

set -euo pipefail
TASK=${TASK:-handover_block}

REPO_ROOT=/scratch/yp2841/geometry-vla/openpi_cam
GEO_ROOT=/scratch/yp2841/geometry-vla
TASK_CONFIG=demo_clean_camaware
REPO_ID="robotwin/${TASK}_${TASK_CONFIG}_50"
SLUG="robotwin_${TASK}_${TASK_CONFIG}_50"

mkdir -p ${GEO_ROOT}/openpi_cam/log/pi0_robotwin_s1
cd ${REPO_ROOT}
source ${REPO_ROOT}/scripts/env/activate_env.sh

export HF_LEROBOT_HOME=${GEO_ROOT}
export OPENPI_PI0_BASE_DIR=${GEO_ROOT}/pi0_base
export OPENPI_CACHE_DIR=${GEO_ROOT}/.cache/openpi
export OPENPI_DATA_HOME=${OPENPI_CACHE_DIR}
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

# Optional: override wandb project/entity per your account
export WANDB_PROJECT=${WANDB_PROJECT:-pi0_robotwin_cam}
# export WANDB_ENTITY=NYU-robotics
# export WANDB_API_KEY=...

uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_pytorch.py \
    pi0_robotwin_cam_prope_ray_view_distill_fullres_stage1_gtdual \
    --exp_name=stage1_${TASK} \
    --batch_size=32 \
    --num_workers=16 \
    --num_train_steps=5000 \
    --save_interval=1000 \
    --keep_period=5000 \
    --data.repo_id ${REPO_ID} \
    --data.assets.assets_dir ${GEO_ROOT}/pi0_libero \
    --data.assets.asset_id ${REPO_ID} \
    --data.pi3x_targets_root  ${OPENPI_CACHE_DIR}/pi3x_targets_224/${SLUG} \
    --data.gt_point_targets_root ${OPENPI_CACHE_DIR}/gt_point_targets_224/${SLUG}
```

Save as `scripts/sbatch/train_pi0_robotwin_s1_gtdual_h200.sbatch`. Submit per task:

```bash
sbatch --export=ALL,TASK=handover_block        scripts/sbatch/train_pi0_robotwin_s1_gtdual_h200.sbatch
sbatch --export=ALL,TASK=stack_blocks_three    scripts/sbatch/train_pi0_robotwin_s1_gtdual_h200.sbatch
sbatch --export=ALL,TASK=place_container_plate scripts/sbatch/train_pi0_robotwin_s1_gtdual_h200.sbatch
sbatch --export=ALL,TASK=place_dual_shoes      scripts/sbatch/train_pi0_robotwin_s1_gtdual_h200.sbatch
sbatch --export=ALL,TASK=lift_pot              scripts/sbatch/train_pi0_robotwin_s1_gtdual_h200.sbatch
sbatch --export=ALL,TASK=beat_block_hammer     scripts/sbatch/train_pi0_robotwin_s1_gtdual_h200.sbatch
```

### 3.b Wall-time estimates (h200 × 4 GPUs, batch 32 effective)

5,000 steps with backbone frozen and gradient checkpointing on takes roughly:

| Task | Frames | Est. wall time | Reason |
|---|---|---|---|
| `lift_pot` | 5,601 | ~25-35 min | shortest |
| `beat_block_hammer` | 5,737 | ~25-35 min | |
| `place_container_plate` | 7,966 | ~30-40 min | |
| `place_dual_shoes` | 11,490 | ~30-45 min | |
| `handover_block` | 14,087 | ~30-45 min | |
| `stack_blocks_three` | 23,723 | ~40-60 min | longest, more dataloader work |

Stage-1 throughput is dataloader-bound on RoboTwin's small datasets, hence the
weak frame-count dependence.

### 3.c Smaller GPU pools

If you only have l40s available:

```
#SBATCH --partition=l40s_tandon
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
```

And drop `torchrun --nproc_per_node=4`, run `python scripts/train_pytorch.py` directly.
Halve `batch_size` to 16 (or 8), bump `num_workers` to 8.

### 3.d Output

Checkpoints land at:
```
checkpoints/pi0_robotwin_cam_prope_ray_view_distill_fullres_stage1_gtdual/stage1_${TASK}/{1000,2000,3000,4000,5000}/
```

`5000/` is the final Stage-1 checkpoint; `keep_period=5000` keeps it permanently
and may delete earlier ones.

---

## 4. Stage 2 training (30,000 steps, unfreeze)

Recipe: `pi0_robotwin_cam_prope_ray_view_distill_fullres_stage2_gtdual`.

- All parameters unfrozen
- `action_loss_weight=1.0`
- `aux_point_head.loss_weight=0.05`
- Warm-start from Stage 1 final checkpoint via `--pytorch_weight_path`

### 4.a SBATCH template

Mirror the Stage-1 template, just change config + steps + warm-start path:

```bash
#!/bin/bash
#SBATCH --job-name=pi0_robotwin_s2_gtdual
#SBATCH --output=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_robotwin_s2/slurm-%j.out
#SBATCH --error=/scratch/yp2841/geometry-vla/openpi_cam/log/pi0_robotwin_s2/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=18:00:00
#SBATCH --account=torch_pr_637_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:h200:4
#SBATCH --export=ALL

set -euo pipefail
TASK=${TASK:-handover_block}

REPO_ROOT=/scratch/yp2841/geometry-vla/openpi_cam
GEO_ROOT=/scratch/yp2841/geometry-vla
TASK_CONFIG=demo_clean_camaware
REPO_ID="robotwin/${TASK}_${TASK_CONFIG}_50"
SLUG="robotwin_${TASK}_${TASK_CONFIG}_50"

S1_CKPT=${REPO_ROOT}/checkpoints/pi0_robotwin_cam_prope_ray_view_distill_fullres_stage1_gtdual/stage1_${TASK}/5000

mkdir -p ${GEO_ROOT}/openpi_cam/log/pi0_robotwin_s2
cd ${REPO_ROOT}
source ${REPO_ROOT}/scripts/env/activate_env.sh

export HF_LEROBOT_HOME=${GEO_ROOT}
export OPENPI_PI0_BASE_DIR=${GEO_ROOT}/pi0_base
export OPENPI_CACHE_DIR=${GEO_ROOT}/.cache/openpi
export OPENPI_DATA_HOME=${OPENPI_CACHE_DIR}
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=${WANDB_PROJECT:-pi0_robotwin_cam}

uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_pytorch.py \
    pi0_robotwin_cam_prope_ray_view_distill_fullres_stage2_gtdual \
    --exp_name=stage2_${TASK} \
    --batch_size=32 \
    --num_workers=16 \
    --num_train_steps=30000 \
    --save_interval=2000 \
    --keep_period=10000 \
    --pytorch_weight_path=${S1_CKPT} \
    --data.repo_id ${REPO_ID} \
    --data.assets.assets_dir ${GEO_ROOT}/pi0_libero \
    --data.assets.asset_id ${REPO_ID} \
    --data.pi3x_targets_root  ${OPENPI_CACHE_DIR}/pi3x_targets_224/${SLUG} \
    --data.gt_point_targets_root ${OPENPI_CACHE_DIR}/gt_point_targets_224/${SLUG}
```

Submit:

```bash
sbatch --export=ALL,TASK=handover_block scripts/sbatch/train_pi0_robotwin_s2_gtdual_h200.sbatch
# ...etc
```

### 4.b Stage-2 dependencies

To make Stage 2 only kick off after Stage 1 succeeds, use a job dependency:

```bash
S1_JOB=$(sbatch --parsable --export=ALL,TASK=handover_block \
    scripts/sbatch/train_pi0_robotwin_s1_gtdual_h200.sbatch)
sbatch --dependency=afterok:${S1_JOB} --export=ALL,TASK=handover_block \
    scripts/sbatch/train_pi0_robotwin_s2_gtdual_h200.sbatch
```

### 4.c Wall-time estimates (h200 × 4 GPUs, batch 32)

30,000 steps with the full backbone unfrozen and gradient checkpointing:

| Task | Frames | Est. wall time |
|---|---|---|
| `lift_pot` / `beat_block_hammer` | 5.6k | 8-11 hours |
| `place_container_plate` | 8.0k | 9-12 hours |
| `place_dual_shoes` / `handover_block` | 11.5k / 14.1k | 10-14 hours |
| `stack_blocks_three` | 23.7k | 12-16 hours |

Set `--time=18:00:00` to be safe. If the job hits the wall, you can resume from
the latest saved checkpoint:

```bash
sbatch --export=ALL,TASK=...,RESUME=true scripts/sbatch/train_pi0_robotwin_s2_gtdual_h200.sbatch
```

(Add `--resume` to the train command in the sbatch when `${RESUME:-false}` is true.)

---

## 5. Baseline ablation (optional)

For an A/B comparison, run `pi0_robotwin_cam_baseline` (no PRoPE / ray / cross-view
fusion / point head) with the same data and steps. Same template, just change:

```
pi0_robotwin_cam_baseline
--exp_name=baseline_${TASK}
--num_train_steps=30000
# (and remove --pytorch_weight_path because there's no Stage 1)
```

---

## 6. Per-task launching all 6 in parallel (recommended)

If your queue allows multiple parallel jobs:

### Vanilla (Pi3X-only) recipe

```bash
cd /scratch/yp2841/geometry-vla/openpi_cam

for TASK in beat_block_hammer handover_block stack_blocks_three \
            place_container_plate place_dual_shoes lift_pot; do
    S1=$(sbatch --parsable --export=ALL,TASK=${TASK} \
        scripts/sbatch/train_pi0_robotwin_s1_h200.sbatch)
    sbatch --dependency=afterok:${S1} --export=ALL,TASK=${TASK} \
        scripts/sbatch/train_pi0_robotwin_s2_h200.sbatch
done
```

### gtdual recipe

```bash
for TASK in beat_block_hammer handover_block stack_blocks_three \
            place_container_plate place_dual_shoes lift_pot; do
    S1=$(sbatch --parsable --export=ALL,TASK=${TASK} \
        scripts/sbatch/train_pi0_robotwin_s1_gtdual_h200.sbatch)
    sbatch --dependency=afterok:${S1} --export=ALL,TASK=${TASK} \
        scripts/sbatch/train_pi0_robotwin_s2_gtdual_h200.sbatch
done
```

Either flow queues 12 SLURM jobs (6 stage-1 + 6 stage-2). With h200 nodes you'll
typically fit 2-4 concurrent runs. Stage-2 jobs queue automatically once their
stage-1 finishes.

---

## 7. Monitoring

```bash
squeue -u $USER                                  # active + queued jobs
sacct -u $USER --starttime $(date -d '24 hours ago' +%Y-%m-%dT%H:%M:%S) \
       --format=JobID,JobName%30,State,Elapsed,Start
tail -f /scratch/yp2841/geometry-vla/openpi_cam/log/pi0_robotwin_s1/slurm-<id>.out
```

Per-task wandb runs land under your project (default
`pi0_robotwin_cam`). Loss curves to watch:

- `loss/action`            (~0.18 → ~0.05–0.10 by step 30k)
- `loss/aux_gt`            (~0.22 → drops fastest in first 1-2k steps)
- `loss/aux_pi3x`          (~0.08 → drops slower; floors near Pi3X teacher noise)
- `loss/aux_xy`, `loss/aux_z` (point-head 2D + depth)
- `lr`                     (cosine schedule visible)

A healthy run shows all three loss components decreasing with no NaN spikes.

---

## 8. Eval (later, when Stage 2 finishes)

Two-process setup. Server runs in the openpi venv on Torch:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_robotwin_cam_prope_ray_view_distill_fullres_stage2_gtdual \
    --policy.dir=${PWD}/checkpoints/pi0_robotwin_cam_prope_ray_view_distill_fullres_stage2_gtdual/stage2_${TASK}/30000
```

Eval client runs in the `robotwin` conda env on a separate GPU node (or your
local 4090 box; eval doesn't need to be on Torch). Point it at the policy
server's host:port via the standard RoboTwin eval entry point:

```bash
conda activate robotwin
cd ${ROBOTWIN_REPO}
python script/eval_policy.py --task-name ${TASK} \
    --policy_server_host <server_node> --policy_server_port 8000 ...
```

(See `RoboTwin/script/eval_policy.py --help` for the per-task client args.)

---

## 9. Failure-mode triage

| Symptom | Fix |
|---|---|
| `FileNotFoundError: norm_stats.json` | check `${GEO_ROOT}/pi0_libero/robotwin/<task>_*/norm_stats.json` exists; otherwise re-compute with `uv run scripts/compute_norm_stats.py --config-name pi0_robotwin_cam_prope_ray_view --repo-id robotwin/<task>_demo_clean_camaware_50` |
| `FileNotFoundError: agent/episode_000000.npz` | the cache for that task isn't where the data loader expects. Confirm `${OPENPI_CACHE_DIR}/pi3x_targets_224/<slug>/{agent,wrist}/` and `gt_point_targets_224/<slug>/{agent,wrist}/` each contain 50 npz files |
| OOM | drop `--batch_size` to 16 or 8; increase `--num_workers`; ensure `disable_geometric_augs=True` in the config (already set) |
| `ImportError: torchcodec...` | already worked around: data loader passes `video_backend="pyav"`. If you somehow regress, the data is image-only PNG so torchcodec isn't actually used |
| Stage-2 not warm-starting | check that `--pytorch_weight_path=${S1_CKPT}` resolves to a directory with `model.safetensors`; the file format is the same as `pi0_base` |

---

## 10. Reference

- Pipeline + data prep details: [`robotwin_cam_pipeline.md`](robotwin_cam_pipeline.md)
- Code-change punch list: [`../scripts/ROBOTWIN_CAM_PATCHES.md`](../scripts/ROBOTWIN_CAM_PATCHES.md)
- LIBERO HPC parallel: [`../scripts/HPC_LIBERO_REGEN_RUNBOOK.md`](../scripts/HPC_LIBERO_REGEN_RUNBOOK.md)
- Bundle README (transfer source): `/home/asus/Research/CamVLA/robotwin_hpc_bundle/README.md`
- Globus transfer task: `7634b636-4b56-11f1-ae1a-0afffe4617ab` (succeeded 2026-05-09T04:17:41 UTC, 80.6 GB)

# RoboTwin Pi0 HPC Runbook

This runbook is for training `pi0` on RoboTwin from this repo on NYU HPC.

As of 2026-05-02, the official LeRobot RoboTwin docs say:

- dataset: `lerobot/robotwin_unified`
- format: LeRobot v3.0
- no conversion is needed for that dataset
- official RoboTwin env: Python 3.10 + `RoboTwin` repo + `bash script/_download_assets.sh`

Sources:

- https://huggingface.co/docs/lerobot/main/robotwin
- https://robotwin-platform.github.io/doc/usage/Pi0.html
- https://robotwin-platform.github.io/doc/usage/robotwin-install.html

## Directory Layout

Use this layout under `/scratch/yp2841/geometry-vla`:

```text
/scratch/yp2841/geometry-vla/
├── openpi_cam/                       # this repo
├── RoboTwin/                         # official RoboTwin repo
├── lerobot/
│   └── robotwin_unified/             # HF dataset snapshot
├── pi0_base/                         # local pi0 base checkpoint
├── pi0_libero/
│   └── lerobot/
│       └── robotwin_unified/
│           └── norm_stats.json
└── .cache/
    ├── huggingface/
    └── openpi/
```

This repo already defaults `HF_LEROBOT_HOME` to the parent of `openpi_cam`, so `repo_id="lerobot/robotwin_unified"` resolves to `/scratch/yp2841/geometry-vla/lerobot/robotwin_unified`.

## Environment Split

Keep two environments.

### 1. Training env

Use this repo's existing env:

```bash
cd /scratch/yp2841/geometry-vla/openpi_cam
source scripts/env/activate_env.sh
python -V
```

Expected:

- Python 3.11
- `OPENPI_PI0_BASE_DIR=/scratch/yp2841/geometry-vla/pi0_base`
- `HF_LEROBOT_HOME=/scratch/yp2841/geometry-vla`

### 2. Official RoboTwin env

Set this up separately for simulation and eval:

```bash
conda create -n robotwin python=3.10 -y
conda activate robotwin
cd /scratch/yp2841/geometry-vla
git clone https://github.com/RoboTwin-Platform/RoboTwin.git
cd RoboTwin
bash script/_install.sh
bash script/_download_assets.sh
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

If you need policy eval inside the `policy/pi0/.venv` flow from the official docs, follow the upstream Pi0 page. For this repo, training stays in `openpi_cam`; RoboTwin is only the simulator/eval repo.

### Raw data note

As of 2026-05-03, RoboTwin officially publishes the large-scale dataset as `lerobot/robotwin_unified`, but the official `pi0` conversion pipeline expects raw collected episodes under `data/${task_name}/${task_config}`. The official route to obtain that raw layout is `collect_data.sh`, not a standalone raw-data download script.

## Phase 1 Commands

### 1. Download the dataset on a CPU node

```bash
cd /scratch/yp2841/geometry-vla/openpi_cam
sbatch scripts/sbatch/download_robotwin_unified_cpu.sbatch
```

Expected target:

```text
/scratch/yp2841/geometry-vla/lerobot/robotwin_unified/
├── data/
└── meta/info.json
```

### 2. Compute norm stats on a CPU node

```bash
cd /scratch/yp2841/geometry-vla/openpi_cam
sbatch scripts/sbatch/compute_norm_stats_robotwin.sbatch
```

Expected output:

```text
/scratch/yp2841/geometry-vla/pi0_libero/lerobot/robotwin_unified/norm_stats.json
```

### 3. Run smoke training on one GPU

```bash
cd /scratch/yp2841/geometry-vla/openpi_cam
sbatch scripts/sbatch/train_pi0_robotwin_smoke.sbatch
```

This uses the local config `pi0_robotwin_smoke`.

## Fallback Path

If direct use of `lerobot/robotwin_unified` fails for any upstream LeRobot compatibility reason, switch to the official RoboTwin Pi0 pipeline.

In this repo, the official logic is mirrored from RoboTwin upstream into:

- [robotwin_official_process_data.py](/scratch/yp2841/geometry-vla/openpi_cam/scripts/robotwin_official_process_data.py:1)
- [convert_aloha_data_to_lerobot_robotwin_official.py](/scratch/yp2841/geometry-vla/openpi_cam/scripts/convert_aloha_data_to_lerobot_robotwin_official.py:1)
- [convert_robotwin_official_cpu.sbatch](/scratch/yp2841/geometry-vla/openpi_cam/scripts/sbatch/convert_robotwin_official_cpu.sbatch:1)

These mirror the official files as of 2026-05-03:

- `policy/pi0/process_data_pi0.sh`
- `policy/pi0/scripts/process_data.py`
- `policy/pi0/generate.sh`
- `policy/pi0/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py`

Expected raw data layout:

```text
/scratch/yp2841/geometry-vla/RoboTwin/data/
└── beat_block_hammer/
    └── demo_clean/
        ├── data/episode0.hdf5
        └── instructions/episode0.json
```

Collect raw episodes with official RoboTwin on L40S:

```bash
cd /scratch/yp2841/geometry-vla/openpi_cam
sbatch --export=ALL,TASK_NAME=beat_block_hammer,TASK_CONFIG=demo_clean,ROBOTWIN_CONDA_ENV=robotwin scripts/sbatch/collect_robotwin_raw_l40s.sbatch
```

Run official conversion on HPC:

```bash
cd /scratch/yp2841/geometry-vla/openpi_cam
sbatch \
  --export=ALL,TASK_NAME=beat_block_hammer,TASK_CONFIG=demo_clean,EXPERT_DATA_NUM=50,RAW_ROOT=/scratch/yp2841/geometry-vla/RoboTwin/data,REPO_ID=robotwin/beat_block_hammer_demo_clean_50 \
  scripts/sbatch/convert_robotwin_official_cpu.sbatch
```

Then compute stats against the converted local repo id:

```bash
cd /scratch/yp2841/geometry-vla/openpi_cam
sbatch --export=ALL,CONFIG_NAME=pi0_robotwin_smoke,REPO_ID=robotwin/beat_block_hammer_demo_clean_50 scripts/sbatch/compute_norm_stats_robotwin.sbatch
```

Then launch smoke training:

```bash
cd /scratch/yp2841/geometry-vla/openpi_cam
sbatch --export=ALL,DATASET_REPO_ID=robotwin/beat_block_hammer_demo_clean_50,NORM_ASSET_ID=robotwin/beat_block_hammer_demo_clean_50 scripts/sbatch/train_pi0_robotwin_smoke.sbatch
```

## Notes

- `open_laptop` is still excluded in the LeRobot RoboTwin doc because of the upstream `self.arm_tag` bug.
- This repo's PyTorch path does not support LoRA yet, so start with `pi0_robotwin` full finetuning or a freeze-backbone variant if you add one later.
- Heavy CPU work should stay on `sbatch`; the scripts added here follow that rule.

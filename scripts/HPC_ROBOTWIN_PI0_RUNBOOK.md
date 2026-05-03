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

If direct use of `lerobot/robotwin_unified` fails for any upstream LeRobot compatibility reason, switch to the official RoboTwin Pi0 pipeline:

1. `bash process_data_pi0.sh ${task_name} ${task_config} ${expert_data_num}`
2. `bash generate.sh ${hdf5_path} ${repo_id}`
3. point this repo's `repo_id` to the generated local LeRobot dataset

The official Pi0 doc places those scripts under `RoboTwin/policy/pi0/`.

## Notes

- `open_laptop` is still excluded in the LeRobot RoboTwin doc because of the upstream `self.arm_tag` bug.
- This repo's PyTorch path does not support LoRA yet, so start with `pi0_robotwin` full finetuning or a freeze-backbone variant if you add one later.
- Heavy CPU work should stay on `sbatch`; the scripts added here follow that rule.

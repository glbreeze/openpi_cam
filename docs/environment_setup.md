# Environment Setup and Migration Guide

This guide documents the environment needed to move this `real-robo` branch of
`openpi_cam` onto another Linux machine or HPC account.

The project is primarily a Python 3.11 / `uv` project. The current local
environment was inspected from this checkout and uses Python 3.11.15, PyTorch
2.7.1, JAX 0.5.3 with CUDA 12 wheels, and a local patch overlay for
Transformers 4.53.2.

## System Requirements

- Ubuntu 22.04 is the tested target OS from upstream `openpi`.
- NVIDIA GPU for training and inference.
- NVIDIA driver new enough for CUDA 12 wheels.
- Git and Git LFS.
- `uv` on `PATH`.
- Optional but recommended on clusters: SLURM with `sbatch`, `srun`, and a
  scratch filesystem for datasets, checkpoints, caches, and W&B files.
- Optional Docker environment: rootless Docker plus NVIDIA container toolkit.

You do not need a system CUDA toolkit for the normal `uv` install path. CUDA
runtime libraries are installed as Python wheels.

## Repository Checkout

Clone the branch and initialize submodules:

```bash
git clone --recurse-submodules <repo-url> openpi_cam
cd openpi_cam
git switch real-robo
git submodule update --init --recursive
```

Current git submodules in this checkout:

```text
third_party/aloha  d1dc83afd89ded4379851257fe5d85632d31d5ec
third_party/libero f78abd68ee283de9f9be3c8f7e2a9ad60246e95c
```

This branch also has local third-party simulator trees that are not listed as
git submodules in `.gitmodules`:

```text
third_party/robocasa_v02
third_party/robosuite_v151
```

When migrating, copy those directories or recreate them before running
RoboCasa / robosuite jobs.

## Main Python Environment

Use the repo-local virtualenv at `.venv`.

```bash
cd /path/to/openpi_cam

# Install uv if needed, then:
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

For this branch's PyTorch model code, apply the Transformers replacement files
after installing dependencies:

```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* \
  .venv/lib/python3.11/site-packages/transformers/
```

The repo also provides a migration-friendly setup helper:

```bash
bash scripts/env/setup_local_venv.sh
source scripts/env/activate_env.sh
```

`scripts/env/setup_local_venv.sh` creates `.venv`, sets local `uv` caches under
the repo, supports a parent-site-packages fallback, and validates the
Transformers overlay. `scripts/env/activate_env.sh` activates the discovered
venv and exports the common project paths.

## Core Project Dependencies

Declared in `pyproject.toml`:

```text
augmax>=0.3.4
dm-tree>=0.1.8
einops>=0.8.0
equinox>=0.11.8
flatbuffers>=24.3.25
flax==0.10.2
fsspec[gcs]>=2024.6.0
gym-aloha>=0.1.1
imageio>=2.36.1
jax[cuda12]==0.5.3
jaxtyping==0.2.36
lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
ml_collections==1.0.0
numpy>=1.22.4,<2.0.0
numpydantic>=1.6.6
opencv-python>=4.10.0.84
openpi-client
orbax-checkpoint==0.11.13
pillow>=11.0.0
sentencepiece>=0.2.0
torch==2.7.1
tqdm-loggable>=0.2
typing-extensions>=4.12.2
tyro>=0.9.5
wandb>=0.19.1
filelock>=3.16.1
beartype==0.19.0
treescope>=0.1.7
transformers==4.53.2
rich>=14.0.0
polars>=1.30.0
```

`uv` dependency groups:

```text
dev: pytest>=8.3.4, ruff>=0.8.6, pre-commit>=4.0.1, ipykernel>=6.29.5,
     ipywidgets>=8.1.5, matplotlib>=3.10.0, pynvml>=12.0.0
rlds: dlimp @ git+https://github.com/kvablack/dlimp@ad72ce3a9b414db2185bc0b38461d4101a65477a,
      tensorflow-cpu==2.15.0, tensorflow-datasets==4.9.9
```

`uv` overrides:

```text
ml-dtypes==0.4.1
tensorstore==0.1.74
```

`packages/openpi-client` dependencies:

```text
dm-tree>=0.1.8
msgpack>=1.0.5
numpy>=1.22.4,<2.0.0
pillow>=9.0.0
tree>=0.2.4
websockets>=11.0
```

## Example-Specific Environments

Several examples intentionally use separate Python versions and pinned
requirements. If migrating those workflows, install from the checked-in
`requirements.txt` in each example directory.

`examples/aloha_real` direct requirements from `requirements.in`:

```text
Pillow
dm_control
einops
h5py
matplotlib
modern_robotics
msgpack
numpy>=1.22.4,<2.0.0
opencv-python
packaging
pexpect
pyquaternion
pyrealsense2
pyyaml
requests
rospkg
tyro
websockets
```

`examples/aloha_sim` direct requirements from `requirements.in`:

```text
gym-aloha
imageio
matplotlib
msgpack
numpy>=1.22.4,<2.0.0
typing-extensions
tyro
websockets
```

`examples/libero` direct requirements from `requirements.in`:

```text
imageio[ffmpeg]
numpy==1.22.4
tqdm
tyro
PyYaml
opencv-python==4.6.0.66
torch==1.11.0+cu113
torchvision==0.12.0+cu113
torchaudio==0.11.0+cu113
robosuite==1.4.1
matplotlib==3.5.3
```

`examples/simple_client` direct requirements from `requirements.in`:

```text
numpy>=1.22.4,<2.0.0
rich
tqdm
tyro
polars
```

The exact transitive pins are already committed in:

```text
examples/aloha_real/requirements.txt
examples/aloha_sim/requirements.txt
examples/libero/requirements.txt
examples/simple_client/requirements.txt
```

## Third-Party Simulator Dependencies

`third_party/robosuite_v151/setup.py`:

```text
numpy>=1.13.3
numba>=0.49.1
scipy>=1.2.3
mujoco>=3.2.3
mink>=0.0.5
Pillow
opencv-python
pynput
termcolor
pytest
tqdm
```

`third_party/robocasa_v02/setup.py`:

```text
numpy==1.23.3
numba==0.56.4
scipy>=1.2.3
mujoco==3.2.6
pygame
Pillow
opencv-python
pyyaml
pynput
tqdm
termcolor
imageio
h5py
lxml
hidapi
tianshou==0.4.10
```

The main `.venv` currently has `numpy==1.26.4`, `numba==0.61.2`, and
`mujoco==2.3.7`, so RoboCasa's pinned setup requirements are not identical to
the active training environment. If running RoboCasa simulation tasks directly,
create a separate simulator env or install the simulator stack in a disposable
venv and test it before changing the training `.venv`.

LIBERO's upstream environment in `third_party/libero/README.md` is Python
3.8.13 with `third_party/libero/requirements.txt` plus:

```text
torch==1.11.0+cu113
torchvision==0.12.0+cu113
torchaudio==0.11.0
```

ALOHA's upstream environment in `third_party/aloha/README.md` is Python 3.8.10
with ROS Noetic plus:

```text
torch
torchvision
pyquaternion
pyyaml
rospkg
pexpect
mujoco==2.3.7
dm_control==1.0.14
opencv-python
matplotlib
einops
packaging
h5py
```

Keep these simulator / robot control environments separate from the main
`openpi` training environment unless you have verified the dependency conflicts.

## Common Environment Variables

The activation script sets these defaults:

```bash
export OPENPI_CAM_ROOT=/path/to/openpi_cam
export OPENPI_GEO_ROOT=/path/to/parent/of/openpi_cam
export OPENPI_PI0_BASE_DIR="${OPENPI_GEO_ROOT}/pi0_base"
export OPENPI_PI0_LIBERO_NORM_DIR="${OPENPI_GEO_ROOT}/pi0_libero"
export HF_LEROBOT_HOME="${OPENPI_GEO_ROOT}"
export PYTHONPATH="${OPENPI_CAM_ROOT}/src:${OPENPI_CAM_ROOT}/packages/openpi-client/src:${PYTHONPATH}"
```

Real-robot and SLURM scripts commonly also use:

```bash
export OPENPI_DATA_HOME="${OPENPI_GEO_ROOT}/.cache/openpi"
export OPENPI_CACHE_DIR="${OPENPI_GEO_ROOT}/.cache/openpi"
export HF_HOME="${OPENPI_GEO_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_ENABLE_HF_TRANSFER=0        # optional fallback if hf_transfer stalls
export WANDB_ENABLED=true                 # or false for offline/smoke runs
export WANDB_ENTITY=NYU-robotics
export WANDB_PROJECT=openpi_cam_real_robot
export WANDB_DIR="/scratch/${USER}/wandb"
export WANDB_CONFIG_DIR="/scratch/${USER}/.config/wandb"
export WANDB_CACHE_DIR="/scratch/${USER}/.cache/wandb"
export CHECKPOINT_BASE_DIR="/scratch/${USER}/tmp/openpi_cam/checkpoints"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 # JAX training only
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export OPENPI_DDP_FIND_UNUSED_PARAMETERS=true
export OPENPI_RESUME_WEIGHTS_ONLY=false
export OPENPI_MONITOR_GPU_USAGE=true
export OPENPI_GPU_MONITOR_INTERVAL_SEC=30
export OPENPI_GPU_MONITOR_WINDOW=8
export OPENPI_GPU_MONITOR_WARN_THRESHOLD=50
```

Camera-aware / distillation training scripts may additionally use:

```bash
export OPENPI_PI3X_TARGETS_224_BASE_DIR="${OPENPI_GEO_ROOT}/pi3x_targets_224"
export PI3X_TARGETS_ROOT_OVERRIDE=""
export GT_POINT_TARGETS_ROOT_OVERRIDE="${OPENPI_GEO_ROOT}/gt_point_targets_grid224/<dataset>"
export OPENPI_TRAINABLE_PREFIXES="paligemma_with_expert.cross_view_fusion,paligemma_with_expert.cam_pose_encoder,paligemma_with_expert.view_embedding,paligemma_with_expert.ray_embed"
export OPENPI_LR_MULTIPLIERS="<module_prefix>=<multiplier>,..."
export PYTORCH_WEIGHT_PATH_OVERRIDE="/path/to/checkpoint"
```

Dataset scripts commonly use:

```bash
export DATASET_DIR="/scratch/${USER}/real_robot_data/<dataset_name>"
export DATASET_REPO_ID="<dataset_name>"
export NORM_ASSET_ID="<dataset_name>"
```

## External Artifacts to Move or Recreate

These are not fully represented by Python dependency installation:

```text
pi0_base/ or pi05_base/ PyTorch checkpoint directories
  required files: config.json, model.safetensors

LeRobot datasets
  common local roots:
  /scratch/${USER}/real_robot_data/<dataset_name>
  ${HF_LEROBOT_HOME}/glbreeze/<dataset_name>

Normalization statistics
  usually under ${OPENPI_GEO_ROOT}/pi0_ur5_real_robot/<asset_id>
  or ${OPENPI_GEO_ROOT}/pi05_ur5_real_robot/<asset_id>

Pi3X / point target caches
  ${OPENPI_GEO_ROOT}/pi3x_targets_224/<dataset_name>
  ${OPENPI_GEO_ROOT}/gt_point_targets_grid224/<dataset_name>

Checkpoints and logs
  checkpoints/
  log/
  ${CHECKPOINT_BASE_DIR}

Hugging Face / Google Storage caches
  ${HF_HOME}
  ${HF_HUB_CACHE}
  ${OPENPI_DATA_HOME}
```

If artifacts are too large to copy, rebuild them on the new machine using the
existing scripts:

```bash
uv run scripts/compute_norm_stats.py --config-name <config_name>
uv run scripts/cache_pi3x_targets.py --data-root <dataset> --output-root <cache> --pi3x-repo <Pi3X_Libero> --output-resolution 224
```

## Smoke Tests After Migration

Basic import test:

```bash
source scripts/env/activate_env.sh
python - <<'PY'
import jax
import torch
import transformers
import openpi
print("jax", jax.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("openpi import ok")
PY
```

Validate the Transformers overlay:

```bash
python - <<'PY'
from transformers.models.siglip import check
assert check.check_whether_transformers_replace_is_installed_correctly()
print("transformers overlay ok")
PY
```

Real-robot baseline commands:

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_ur5_real_robot_pytorch_baseline
uv run python scripts/train_pytorch.py pi0_ur5_real_robot_pytorch_baseline --exp_name migration_smoke --num_train_steps=3 --batch_size=2 --num_workers=0 --overwrite
```

SLURM smoke scripts are available under `scripts/sbatch/*smoke*.sbatch`.

## Docker Environment

Docker is optional. If using it:

- Install Docker Engine, not Docker Desktop.
- Use rootless mode.
- Install NVIDIA container toolkit.
- Avoid the Snap Docker package because it is incompatible with the NVIDIA
  runtime.

Repo-level Docker command:

```bash
docker compose -f scripts/docker/compose.yml up --build
```

Example-specific Docker command:

```bash
docker compose -f examples/<example_name>/compose.yml up --build
```

## Exact Installed Package Inventory

The following is the inspected package inventory from the current `.venv`
excluding editable local packages. Reproduce with:

```bash
.venv/bin/python -m pip freeze --exclude-editable
```

```text
absl-py==2.3.0
aiohappyeyeballs==2.6.1
aiohttp==3.12.4
aiosignal==1.3.2
annotated-types==0.7.0
antlr4-python3-runtime==4.9.3
argcomplete==3.6.3
asttokens==3.0.0
attrs==25.3.0
augmax==0.4.1
av==14.4.0
beartype==0.19.0
beautifulsoup4==4.13.4
blinker==1.9.0
boto==2.49.0
cachetools==5.5.2
certifi==2025.4.26
cffi==1.17.1
cfgv==3.4.0
charset-normalizer==3.4.2
chex==0.1.89
click==8.2.1
cloudpickle==3.1.1
cmake==4.0.2
comm==0.2.2
contourpy==1.3.2
crc32c==2.7.1
crcmod==1.7
cryptography==46.0.7
cycler==0.12.1
datasets==3.6.0
debugpy==1.8.14
decorator==5.2.1
deepdiff==8.5.0
diffusers==0.33.1
dill==0.3.8
distlib==0.3.9
dm-control==1.0.14
dm-env==1.6
dm-tree==0.1.9
docker-pycreds==0.4.0
docstring_parser==0.16
donfig==0.8.1.post1
draccus==0.10.0
einops==0.8.1
equinox==0.12.2
etils==1.12.2
evdev==1.9.2
executing==2.2.0
Farama-Notifications==0.0.4
fasteners==0.20
filelock==3.18.0
Flask==3.1.1
flatbuffers==25.2.10
flax==0.10.2
fonttools==4.58.1
frozenlist==1.6.0
fsspec==2025.3.0
gcs-oauth2-boto-plugin==3.3
gcsfs==2025.3.0
gdown==5.2.0
gitdb==4.0.12
GitPython==3.1.44
glfw==2.9.0
google-api-core==2.24.2
google-apitools==0.5.35
google-auth==2.40.2
google-auth-httplib2==0.3.1
google-auth-oauthlib==1.2.2
google-cloud-core==2.4.3
google-cloud-storage==3.1.0
google-crc32c==1.7.1
google-reauth==0.1.1
google-resumable-media==2.7.2
googleapis-common-protos==1.70.0
gsutil==5.37
gym-aloha==0.1.1
gymnasium==0.29.1
h5py==3.13.0
hf-xet==1.1.2
hf_transfer==0.1.9
httplib2==0.20.4
huggingface-hub==0.32.3
humanize==4.12.3
identify==2.6.12
idna==3.10
imageio==2.37.0
imageio-ffmpeg==0.6.0
importlib_metadata==8.7.0
importlib_resources==6.5.2
iniconfig==2.1.0
inquirerpy==0.3.0
ipykernel==6.29.5
ipython==9.2.0
ipython_pygments_lexers==1.1.1
ipywidgets==8.1.7
itsdangerous==2.2.0
jax==0.5.3
jax-cuda12-pjrt==0.5.3
jax-cuda12-plugin==0.5.3
jaxlib==0.5.3
jaxtyping==0.2.36
jedi==0.19.2
Jinja2==3.1.6
jsonlines==4.0.0
jupyter_client==8.6.3
jupyter_core==5.8.1
jupyterlab_widgets==3.0.15
kiwisolver==1.4.8
labmaze==1.0.6
lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
llvmlite==0.44.0
lxml==5.4.0
markdown-it-py==3.0.0
MarkupSafe==3.0.2
matplotlib==3.10.3
matplotlib-inline==0.1.7
mdurl==0.1.2
mergedeep==1.3.4
ml-dtypes==0.4.1
ml_collections==1.0.0
monotonic==1.6
mpmath==1.3.0
msgpack==1.1.0
mujoco==2.3.7
multidict==6.4.4
multiprocess==0.70.16
mypy_extensions==1.1.0
nest-asyncio==1.6.0
networkx==3.5
nodeenv==1.9.1
numba==0.61.2
numcodecs==0.16.1
numpy==1.26.4
numpydantic==1.6.9
nvidia-cublas-cu12==12.6.4.1
nvidia-cuda-cupti-cu12==12.6.80
nvidia-cuda-nvcc-cu12==12.9.41
nvidia-cuda-nvrtc-cu12==12.6.77
nvidia-cuda-runtime-cu12==12.6.77
nvidia-cudnn-cu12==9.5.1.17
nvidia-cufft-cu12==11.3.0.4
nvidia-cufile-cu12==1.11.1.6
nvidia-curand-cu12==10.3.7.77
nvidia-cusolver-cu12==11.7.1.2
nvidia-cusparse-cu12==12.5.4.2
nvidia-cusparselt-cu12==0.6.3
nvidia-ml-py==12.575.51
nvidia-nccl-cu12==2.26.2
nvidia-nvjitlink-cu12==12.6.85
nvidia-nvtx-cu12==12.6.77
oauth2client==4.1.3
oauthlib==3.2.2
omegaconf==2.3.0
opencv-python==4.11.0.86
opencv-python-headless==4.11.0.86
opt_einsum==3.4.0
optax==0.2.4
orbax-checkpoint==0.11.13
orderly-set==5.4.1
packaging==25.0
pandas==2.2.3
parso==0.8.4
pexpect==4.9.0
pfzy==0.3.4
pillow==11.2.1
platformdirs==4.3.8
pluggy==1.6.0
polars==1.30.0
pre_commit==4.2.0
prompt_toolkit==3.0.51
propcache==0.3.1
proto-plus==1.26.1
protobuf==4.25.8
psutil==7.0.0
ptyprocess==0.7.0
pure_eval==0.2.3
pyarrow==20.0.0
pyasn1==0.6.1
pyasn1_modules==0.4.2
pycparser==2.22
pydantic==2.11.5
pydantic_core==2.33.2
Pygments==2.19.1
pymunk==7.0.0
pynput==1.8.1
pynvml==12.0.0
PyOpenGL==3.1.9
pyOpenSSL==26.0.7
pyparsing==3.2.3
PySocks==1.7.1
pytest==8.3.5
python-dateutil==2.9.0.post0
python-xlib==0.33
pytz==2025.2
pyu2f==0.1.5
PyYAML==6.0.2
pyyaml-include==1.4.1
pyzmq==26.4.0
regex==2024.11.6
requests==2.32.3
requests-oauthlib==2.0.0
rerun-sdk==0.23.1
retry_decorator==1.1.1
rich==14.0.0
rsa==4.9.1
ruff==0.11.12
safetensors==0.5.3
scipy==1.15.3
sentencepiece==0.2.0
sentry-sdk==2.29.1
setproctitle==1.3.6
shtab==1.7.2
simplejson==3.20.1
six==1.17.0
smmap==5.0.2
soupsieve==2.7
stack-data==0.6.3
svgwrite==1.4.3
sympy==1.14.0
tensorstore==0.1.74
termcolor==3.1.0
tokenizers==0.21.1
toml==0.10.2
toolz==1.0.0
torch==2.7.1
torchcodec==0.4.0
torchvision==0.22.1
tornado==6.5.1
tqdm==4.67.1
tqdm-loggable==0.2
traitlets==5.14.3
transformers==4.53.2
Tree==0.2.4
treescope==0.1.9
triton==3.3.1
typeguard==4.4.2
typing-inspect==0.9.0
typing-inspection==0.4.1
typing_extensions==4.13.2
tyro==0.9.22
tzdata==2025.2
urllib3==2.4.0
virtualenv==20.31.2
wadler_lindig==0.1.6
wandb==0.19.11
wcwidth==0.2.13
websockets==15.0.1
Werkzeug==3.1.3
widgetsnbextension==4.0.14
wrapt==1.14.1
xxhash==3.5.0
yarl==1.20.0
zarr==3.0.8
zipp==3.22.0
```

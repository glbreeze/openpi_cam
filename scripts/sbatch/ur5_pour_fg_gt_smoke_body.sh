#!/bin/bash

set -euo pipefail

export USE_CAM=false
export TRAIN_RECIPE=single_stage

export WANDB_ENABLED="${WANDB_ENABLED:-false}"
export WANDB_ENTITY="${WANDB_ENTITY:-NYU-robotics}"
export WANDB_PROJECT="${WANDB_PROJECT:-openpi_cam_real_robot}"
export WANDB_DIR="${WANDB_DIR:-/scratch/${USER}/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/scratch/${USER}/.config/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/scratch/${USER}/.cache/wandb}"

export OPENPI_GEO_ROOT="${OPENPI_GEO_ROOT:-/scratch/${USER}}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${OPENPI_GEO_ROOT}/.huggingface/lerobot}"

export DATASET_DIR="${DATASET_DIR:-/scratch/${USER}/real_robot_data/ur5_place_and_pour_nuts_camera_shifts}"
export DATASET_REPO_ID="${DATASET_REPO_ID:-ur5_place_and_pour_nuts_camera_shifts}"
export NORM_ASSET_ID="${NORM_ASSET_ID:-ur5_place_and_pour_nuts_camera_shifts}"
export PI3X_TARGETS_ROOT_OVERRIDE="${PI3X_TARGETS_ROOT_OVERRIDE:-}"
export GT_POINT_TARGETS_ROOT_OVERRIDE="${GT_POINT_TARGETS_ROOT_OVERRIDE:-${OPENPI_GEO_ROOT}/gt_point_targets_grid224/ur5_place_and_pour_nuts_camera_shifts}"
export CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-/scratch/${USER}/tmp/openpi_cam/checkpoints}"

export NUM_GPUS="${NUM_GPUS:-1}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export OPENPI_ENABLE_GPU_FILLER="${OPENPI_ENABLE_GPU_FILLER:-false}"
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"

STAGE1_NUM_TRAIN_STEPS="${STAGE1_NUM_TRAIN_STEPS:-2}"
STAGE1_CHECKPOINT_STEP="${STAGE1_CHECKPOINT_STEP:-${STAGE1_NUM_TRAIN_STEPS}}"
STAGE1_SAVE_INTERVAL="${STAGE1_SAVE_INTERVAL:-${STAGE1_NUM_TRAIN_STEPS}}"
STAGE1_KEEP_PERIOD="${STAGE1_KEEP_PERIOD:-${STAGE1_NUM_TRAIN_STEPS}}"
STAGE1_RESUME="${STAGE1_RESUME:-false}"

STAGE2_NUM_TRAIN_STEPS="${STAGE2_NUM_TRAIN_STEPS:-2}"
STAGE2_SAVE_INTERVAL="${STAGE2_SAVE_INTERVAL:-${STAGE2_NUM_TRAIN_STEPS}}"
STAGE2_KEEP_PERIOD="${STAGE2_KEEP_PERIOD:-${STAGE2_NUM_TRAIN_STEPS}}"
STAGE2_RESUME="${STAGE2_RESUME:-false}"

export STAGE1_CONFIG_NAME
export STAGE2_CONFIG_NAME
export SMOKE_EXPECT_PI05
export SMOKE_MODEL_TYPE

mkdir -p "${SMOKE_LOG_DIR}" "${CHECKPOINT_BASE_DIR}"

ARRAY_CACHE_ROOT="${DATASET_DIR}_array_cache"
NORM_STATS_DIR="${REAL_ROBOT_NORM_ROOT}/${NORM_ASSET_ID}"
if [[ ! -d "${DATASET_DIR}/data" || ! -f "${DATASET_DIR}/meta/info.json" || ! -f "${DATASET_DIR}/meta/episodes.jsonl" ]]; then
  echo "Missing local LeRobot pour dataset at ${DATASET_DIR}." >&2
  exit 1
fi
if [[ ! -f "${OPENPI_PI0_BASE_DIR}/model.safetensors" || ! -f "${OPENPI_PI0_BASE_DIR}/config.json" ]]; then
  echo "Missing base checkpoint under ${OPENPI_PI0_BASE_DIR}." >&2
  exit 1
fi
if [[ ! -f "${NORM_STATS_DIR}/norm_stats.json" ]]; then
  echo "Missing norm stats at ${NORM_STATS_DIR}/norm_stats.json." >&2
  exit 1
fi
if [[ ! -d "${ARRAY_CACHE_ROOT}/videos/chunk-000/observation.images.context_left_rgb" ||
      ! -d "${ARRAY_CACHE_ROOT}/videos/chunk-000/observation.images.wrist_right_rgb" ]]; then
  echo "Missing local video array cache for context_left_rgb and wrist_right_rgb under ${ARRAY_CACHE_ROOT}." >&2
  exit 1
fi
if [[ -n "${PI3X_TARGETS_ROOT_OVERRIDE}" ]]; then
  echo "PI3X_TARGETS_ROOT_OVERRIDE must be empty for this GT-only smoke, got: ${PI3X_TARGETS_ROOT_OVERRIDE}" >&2
  exit 1
fi
if [[ ! -d "${GT_POINT_TARGETS_ROOT_OVERRIDE}/base" ||
      ! -d "${GT_POINT_TARGETS_ROOT_OVERRIDE}/left_wrist" ]]; then
  echo "Missing GT target subdirs under ${GT_POINT_TARGETS_ROOT_OVERRIDE}; expected base/ and left_wrist/." >&2
  exit 1
fi

cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/env/activate_env.sh"

run_static_contract_checks() {
  python - <<'PY'
import json
import os
from pathlib import Path
import zipfile

import numpy as np
from numpy.lib import format as np_format

from openpi.training import config as _config
from openpi.training import data_loader as _data


def fail(message: str) -> None:
    raise SystemExit(message)


def read_npy_header(path: Path):
    with path.open("rb") as f:
        version = np_format.read_magic(f)
        if version == (1, 0):
            return np_format.read_array_header_1_0(f)
        if version == (2, 0):
            return np_format.read_array_header_2_0(f)
        raise ValueError(f"Unsupported npy version {version} in {path}")


def read_npz_member_header(path: Path, member: str):
    member_path = f"{member}.npy"
    with zipfile.ZipFile(path) as zf:
        if member_path not in zf.namelist():
            fail(f"{path} is missing member {member_path}")
        with zf.open(member_path) as f:
            version = np_format.read_magic(f)
            if version == (1, 0):
                return np_format.read_array_header_1_0(f)
            if version == (2, 0):
                return np_format.read_array_header_2_0(f)
            raise ValueError(f"Unsupported npy version {version} in {path}:{member_path}")


dataset_dir = Path(os.environ["DATASET_DIR"]).expanduser().resolve()
array_cache_root = Path(f"{dataset_dir}_array_cache")
gt_root = Path(os.environ["GT_POINT_TARGETS_ROOT_OVERRIDE"]).expanduser().resolve()
norm_stats_path = Path(os.environ["REAL_ROBOT_NORM_ROOT"]) / os.environ["NORM_ASSET_ID"] / "norm_stats.json"
expect_pi05 = os.environ["SMOKE_EXPECT_PI05"].lower() == "true"
if os.environ.get("PI3X_TARGETS_ROOT_OVERRIDE", ""):
    fail(f"Pi3X root must be empty for GT-only smoke, got {os.environ['PI3X_TARGETS_ROOT_OVERRIDE']}")
if not norm_stats_path.exists():
    fail(f"Missing norm stats file: {norm_stats_path}")

stage1 = _config.get_config(os.environ["STAGE1_CONFIG_NAME"])
stage2 = _config.get_config(os.environ["STAGE2_CONFIG_NAME"])

expected_repo_id = "ur5_place_and_pour_nuts_camera_shifts"
expected_point_cams = (("base", "base"), ("left_wrist", "left_wrist"))
expected_raw_cams = ("observation.images.context_left_rgb", "observation.images.wrist_right_rgb")

for label, cfg, action_weight, aux_weight in (
    ("stage1", stage1, 0.1, 1.0),
    ("stage2", stage2, 1.0, 0.05),
):
    data = cfg.data
    model = cfg.model
    aux = model.aux_point_head
    cross_view = model.cross_view

    if cfg.name != os.environ[f"{label.upper()}_CONFIG_NAME"]:
        fail(f"{label}: loaded wrong config name {cfg.name}")
    if bool(model.pi05) != expect_pi05:
        fail(f"{label}: pi05 mismatch, expected {expect_pi05}, got {model.pi05}")
    if data.repo_id != expected_repo_id:
        fail(f"{label}: repo_id mismatch {data.repo_id!r}")
    if data.assets.asset_id != expected_repo_id:
        fail(f"{label}: norm asset id mismatch {data.assets.asset_id!r}")
    if data.base_camera_key != expected_raw_cams[0]:
        fail(f"{label}: base camera mismatch {data.base_camera_key!r}")
    if data.wrist_camera_key != expected_raw_cams[1]:
        fail(f"{label}: wrist camera mismatch {data.wrist_camera_key!r}")
    if data.pi3x_targets_root is not None:
        fail(f"{label}: config still has pi3x_targets_root={data.pi3x_targets_root}")
    if Path(data.gt_point_targets_root).expanduser().resolve() != gt_root:
        fail(f"{label}: GT root mismatch {data.gt_point_targets_root!r} != {str(gt_root)!r}")
    if data.point_target_gt_ratio != 1.0:
        fail(f"{label}: point_target_gt_ratio must be 1.0, got {data.point_target_gt_ratio}")
    if tuple(data.point_target_cams) != expected_point_cams:
        fail(f"{label}: point_target_cams mismatch {data.point_target_cams!r}")
    if model.pose_enc_type != "prope" or not model.ray_enc_type or model.view_enc_type:
        fail(
            f"{label}: geometry config mismatch "
            f"pose={model.pose_enc_type!r} ray={model.ray_enc_type} view={model.view_enc_type}"
        )
    if cross_view.type != "standard" or cross_view.aa_order != "fg" or tuple(cross_view.prope_layer_idx) != (0,):
        fail(f"{label}: cross-view config mismatch {cross_view}")
    if not model.disable_geometric_augs:
        fail(f"{label}: disable_geometric_augs must be true")
    if model.ray_embed_pi3x_init_path is not None:
        fail(f"{label}: ray_embed must be zero-init, got init path {model.ray_embed_pi3x_init_path}")
    if abs(model.action_loss_weight - action_weight) > 1e-8:
        fail(f"{label}: action loss weight mismatch {model.action_loss_weight}")
    if not aux.enabled or aux.loss_type != "legacy_conf_mse" or aux.output_resolution != 224:
        fail(f"{label}: aux head config mismatch {aux}")
    if abs(aux.loss_weight - aux_weight) > 1e-8:
        fail(f"{label}: aux loss weight mismatch {aux.loss_weight}")

required_prefixes = {"cross_view_fusion", "ray_embed", "aux_point_head"}
if set(stage1.trainable_prefixes) != required_prefixes:
    fail(f"stage1: trainable prefixes must be {required_prefixes}; got {stage1.trainable_prefixes}")
if stage2.trainable_prefixes:
    fail(f"stage2: expected full training with no trainable-prefix restriction, got {stage2.trainable_prefixes}")

info = json.loads((dataset_dir / "meta/info.json").read_text())
features = info.get("features", {})
for camera in expected_raw_cams:
    if camera not in features:
        fail(f"Dataset info is missing camera feature {camera}")
    if features[camera].get("dtype") != "video":
        fail(f"Dataset feature {camera} is not video: {features[camera]}")
for key in ("observation.state", "action"):
    if key not in features:
        fail(f"Dataset info is missing feature {key}")

episode_lines = [line for line in (dataset_dir / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
episode_count = len(episode_lines)
if episode_count != int(info.get("total_episodes", episode_count)):
    fail(f"Episode count mismatch: episodes.jsonl={episode_count}, info.json={info.get('total_episodes')}")

raw_cache_dirs = {
    "context_left": array_cache_root / "videos/chunk-000/observation.images.context_left_rgb",
    "wrist_right": array_cache_root / "videos/chunk-000/observation.images.wrist_right_rgb",
}
gt_dirs = {
    "base": gt_root / "base",
    "left_wrist": gt_root / "left_wrist",
}
for label, directory in {**raw_cache_dirs, **gt_dirs}.items():
    if not directory.is_dir():
        fail(f"Missing directory for {label}: {directory}")

for ep in range(episode_count):
    expected_files = [
        raw_cache_dirs["context_left"] / f"episode_{ep:06d}.npy",
        raw_cache_dirs["wrist_right"] / f"episode_{ep:06d}.npy",
        gt_dirs["base"] / f"episode_{ep:06d}.npz",
        gt_dirs["left_wrist"] / f"episode_{ep:06d}.npz",
    ]
    missing = [str(path) for path in expected_files if not path.exists()]
    if missing:
        fail(f"Missing expected per-episode files for episode {ep}: {missing}")

for ep in sorted({0, episode_count // 2, episode_count - 1}):
    context_shape, _, context_dtype = read_npy_header(raw_cache_dirs["context_left"] / f"episode_{ep:06d}.npy")
    wrist_shape, _, wrist_dtype = read_npy_header(raw_cache_dirs["wrist_right"] / f"episode_{ep:06d}.npy")
    if len(context_shape) != 4 or len(wrist_shape) != 4:
        fail(f"Episode {ep}: cache arrays must be rank-4, got {context_shape} and {wrist_shape}")
    if context_dtype != np.dtype("uint8") or wrist_dtype != np.dtype("uint8"):
        fail(f"Episode {ep}: cache dtype must be uint8, got {context_dtype} and {wrist_dtype}")
    if context_shape[0] != wrist_shape[0]:
        fail(f"Episode {ep}: raw camera frame-count mismatch {context_shape[0]} vs {wrist_shape[0]}")
    for subdir, cache_shape in (("base", context_shape), ("left_wrist", wrist_shape)):
        npz_path = gt_dirs[subdir] / f"episode_{ep:06d}.npz"
        xy_shape, _, _ = read_npz_member_header(npz_path, "xy")
        logz_shape, _, _ = read_npz_member_header(npz_path, "log_z")
        conf_shape, _, _ = read_npz_member_header(npz_path, "conf")
        if xy_shape != (cache_shape[0], 224, 224, 2):
            fail(f"{npz_path}: xy shape {xy_shape} does not match expected {(cache_shape[0], 224, 224, 2)}")
        if logz_shape != (cache_shape[0], 224, 224, 1):
            fail(f"{npz_path}: log_z shape {logz_shape} does not match expected {(cache_shape[0], 224, 224, 1)}")
        if conf_shape != (cache_shape[0], 224, 224, 1):
            fail(f"{npz_path}: conf shape {conf_shape} does not match expected {(cache_shape[0], 224, 224, 1)}")

created_data = stage1.data.create(stage1.assets_dirs, stage1.model)
if Path(created_data.local_dataset_root).expanduser().resolve() != dataset_dir:
    fail(f"Created data config local_dataset_root mismatch: {created_data.local_dataset_root}")
if created_data.norm_stats is None:
    fail("Created data config did not load norm stats")
torch_dataset = _data.create_torch_dataset(created_data, stage1.model.action_horizon, stage1.model)
while isinstance(torch_dataset, _data.TransformedDataset):
    torch_dataset = torch_dataset._dataset
if not isinstance(torch_dataset, _data.LocalLeRobotDataset):
    fail(f"Expected LocalLeRobotDataset for local cache path, got {type(torch_dataset)}")
if not getattr(torch_dataset, "_array_cache_enabled", False):
    fail(f"LocalLeRobotDataset did not enable array cache at {array_cache_root}")
requested = getattr(torch_dataset, "_requested_video_keys", frozenset())
if requested != frozenset(expected_raw_cams):
    fail(f"Requested video keys mismatch: {sorted(requested)}")
for camera in expected_raw_cams:
    frames = torch_dataset._load_cached_video_frames(0, camera, [0.0])
    if frames is None:
        fail(f"Cache lookup returned None for {camera}; training would fall back to runtime video decode")
    if len(tuple(frames.shape)) != 4:
        fail(f"Cache lookup for {camera} returned bad shape {tuple(frames.shape)}")

print(
    "static_contract_checks_ok "
    f"episodes={episode_count} gt_root={gt_root} array_cache={array_cache_root} cameras={expected_raw_cams}"
)
PY
}

run_target_loader_passthrough_check() {
  python - <<'PY'
import os

import numpy as np

from openpi.models import model as _model
from openpi.policies.libero_policy import MixedPointTargetLoader
from openpi.policies.real_robot_policy import RealRobotUR5Inputs

model_type = getattr(_model.ModelType, os.environ["SMOKE_MODEL_TYPE"])
gt_root = os.environ["GT_POINT_TARGETS_ROOT_OVERRIDE"]
loader = MixedPointTargetLoader(
    pi3x_root=gt_root,
    gt_root=gt_root,
    gt_ratio=1.0,
    cam_to_npz_subdir=(("base", "base"), ("left_wrist", "left_wrist")),
)
data = {
    "episode_index": np.asarray(0, dtype=np.int64),
    "frame_index": np.asarray(0, dtype=np.int64),
    "observation/base_image": np.zeros((3, 480, 640), dtype=np.uint8),
    "observation/wrist_image": np.zeros((3, 480, 640), dtype=np.uint8),
    "observation/state": np.zeros(7, dtype=np.float32),
    "actions": np.zeros((10, 7), dtype=np.float32),
    "prompt": "smoke",
}
data = loader(data)
if float(np.asarray(data["point_target_source"]).item()) != 1.0:
    raise SystemExit(f"GT loader returned non-GT source: {data['point_target_source']}")
if data["point_target_xy"].shape != (2, 224, 224, 2):
    raise SystemExit(f"point_target_xy shape mismatch: {data['point_target_xy'].shape}")
if data["point_target_logz"].shape != (2, 224, 224, 1):
    raise SystemExit(f"point_target_logz shape mismatch: {data['point_target_logz'].shape}")
if data["point_target_conf"].shape != (2, 224, 224, 1):
    raise SystemExit(f"point_target_conf shape mismatch: {data['point_target_conf'].shape}")

out = RealRobotUR5Inputs(model_type)(data)
for key in ("point_target_xy", "point_target_logz", "point_target_conf", "point_target_source"):
    if key not in out:
        raise SystemExit(f"{key} was dropped by RealRobotUR5Inputs")

print("target_loader_passthrough_check_ok point_target_source=1.0")
PY
}

capture_stage1_initial_trainable_state() {
  local output_file=$1
  python - "${output_file}" <<'PY'
import os
from pathlib import Path
import sys

import numpy as np
import safetensors.torch
import torch

import openpi.models_pytorch.pi0_pytorch
from openpi.training import config as _config

output_file = Path(sys.argv[1])
cfg = _config.get_config(os.environ["STAGE1_CONFIG_NAME"])
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
model_cfg = cfg.model
object.__setattr__(model_cfg, "dtype", cfg.pytorch_training_precision)
model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg)

model_path = Path(os.environ["OPENPI_PI0_BASE_DIR"]) / "model.safetensors"
missing, unexpected = safetensors.torch.load_model(model, model_path, strict=False)
allowed_missing = ("cross_view_fusion", "view_embedding", "cam_pose_encoder", "ray_embed", "aux_point_head")
bad_missing = [key for key in missing if not any(token in key for token in allowed_missing)]
if bad_missing:
    raise SystemExit(f"Unexpected missing base checkpoint keys: {bad_missing[:8]}")
if unexpected:
    raise SystemExit(f"Unexpected base checkpoint keys: {unexpected[:8]}")

prefixes = tuple(cfg.trainable_prefixes)
state = {}
prefix_counts = {prefix: 0 for prefix in prefixes}
for key, tensor in model.state_dict().items():
    matched = [prefix for prefix in prefixes if prefix in key]
    if not matched:
        continue
    state[key] = tensor.detach().cpu().contiguous()
    for prefix in matched:
        prefix_counts[prefix] += 1

missing_prefixes = [prefix for prefix, count in prefix_counts.items() if count == 0]
if missing_prefixes:
    raise SystemExit(f"No initial tensors found for trainable prefixes: {missing_prefixes}")
output_file.parent.mkdir(parents=True, exist_ok=True)
safetensors.torch.save_file(state, output_file)
print(f"captured_stage1_initial_state {output_file} tensors={len(state)} prefix_counts={prefix_counts}")
PY
}

run_common_launcher_to_log() {
  local log_file=$1
  bash "${REPO_ROOT}/scripts/sbatch/train_pi0_libero_object_ft.sbatch" 2>&1 | tee "${log_file}"
}

validate_gt_aux_log() {
  local label=$1
  local log_file=$2
  python - "${label}" "${log_file}" <<'PY'
import re
import sys

label, log_file = sys.argv[1], sys.argv[2]
text = open(log_file, encoding="utf-8", errors="replace").read()
records = []
for line in text.splitlines():
    if "loss_breakdown " not in line:
        continue
    metrics = {key: float(value) for key, value in re.findall(r"([a-z0-9_]+)=([0-9.eE+-]+)", line)}
    if {"aux_loss", "aux_gt_frac", "aux_pi3x_frac"} <= metrics.keys():
        records.append(metrics)
if not records:
    raise SystemExit(f"{label}: no loss_breakdown with aux_loss/aux_gt_frac/aux_pi3x_frac found in {log_file}")

active = [
    record
    for record in records
    if record["aux_loss"] > 0.0 and record["aux_gt_frac"] >= 0.99 and record["aux_pi3x_frac"] <= 0.01
]
if not active:
    summary = [
        (record["aux_loss"], record["aux_gt_frac"], record["aux_pi3x_frac"])
        for record in records[:5]
    ]
    raise SystemExit(
        f"{label}: GT aux supervision inactive or mixed with Pi3X in {log_file}; observed {summary}"
    )
first = active[0]
print(
    f"{label}: GT aux supervision active; "
    f"aux_loss={first['aux_loss']:.6g}, aux_gt_frac={first['aux_gt_frac']:.3f}, "
    f"aux_pi3x_frac={first['aux_pi3x_frac']:.3f}"
)
PY
}

validate_trainable_log() {
  local label=$1
  local log_file=$2
  python - "${label}" "${log_file}" <<'PY'
import sys

label, log_file = sys.argv[1], sys.argv[2]
text = open(log_file, encoding="utf-8", errors="replace").read()
required = ("cross_view_fusion", "ray_embed", "aux_point_head")
missing = [item for item in required if item not in text]
if missing:
    raise SystemExit(f"{label}: trainable module log missing {missing}")
print(f"{label}: trainable module log check ok ({', '.join(required)})")
PY
}

validate_stage1_trainable_deltas() {
  local initial_file=$1
  local checkpoint_file=$2
  python - "${initial_file}" "${checkpoint_file}" <<'PY'
from pathlib import Path
import sys

import safetensors.torch
import torch

initial_file = Path(sys.argv[1])
checkpoint_file = Path(sys.argv[2])
initial = safetensors.torch.load_file(initial_file, device="cpu")
trained = safetensors.torch.load_file(checkpoint_file, device="cpu")
required = ("cross_view_fusion", "ray_embed", "aux_point_head")
for prefix in required:
    keys = sorted(key for key in initial if prefix in key and key in trained)
    if not keys:
        raise SystemExit(f"stage1: no common tensors for trainable prefix {prefix}")
    changed_tensors = 0
    total_abs_delta = 0.0
    max_abs_delta = 0.0
    for key in keys:
        before = initial[key].to(torch.float32)
        after = trained[key].to(torch.float32)
        if before.shape != after.shape:
            raise SystemExit(f"stage1: shape changed for {key}: {before.shape} -> {after.shape}")
        delta = (after - before).abs()
        tensor_max = float(delta.max().item()) if delta.numel() else 0.0
        total_abs_delta += float(delta.sum().item())
        max_abs_delta = max(max_abs_delta, tensor_max)
        if tensor_max > 0.0:
            changed_tensors += 1
    if changed_tensors == 0 or total_abs_delta <= 0.0:
        raise SystemExit(f"stage1: trainable prefix {prefix} did not change after training")
    print(
        f"stage1: trainable prefix {prefix} changed; "
        f"changed_tensors={changed_tensors}/{len(keys)} total_abs_delta={total_abs_delta:.6g} "
        f"max_abs_delta={max_abs_delta:.6g}"
    )
PY
}

validate_stage2_weight_log() {
  local label=$1
  local log_file=$2
  local expected_weight_path=$3
  python - "${label}" "${log_file}" "${expected_weight_path}" <<'PY'
import sys

label, log_file, expected = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(log_file, encoding="utf-8", errors="replace").read()
required_fragments = [
    f"stage2 weight init: {expected}",
    f"Loading weights from: {expected}",
    f"Loaded PyTorch weights from {expected}",
]
missing = [fragment for fragment in required_fragments if fragment not in text]
if missing:
    raise SystemExit(f"{label}: expected stage1 checkpoint handoff fragments missing: {missing}")
print(f"{label}: stage1 checkpoint handoff check ok ({expected})")
PY
}

echo "===== ${SMOKE_DISPLAY_NAME} UR5 POUR GT HARD ZERO-INIT STAGE1+STAGE2 SMOKE ====="
echo "repo root: ${REPO_ROOT}"
echo "dataset: ${DATASET_DIR}"
echo "array cache: ${ARRAY_CACHE_ROOT}"
echo "norm stats: ${NORM_STATS_DIR}"
echo "gt target root: ${GT_POINT_TARGETS_ROOT_OVERRIDE}"
echo "pi3x target root: ${PI3X_TARGETS_ROOT_OVERRIDE:-<none>}"
echo "camera combo: observation.images.context_left_rgb + observation.images.wrist_right_rgb"
echo "num_gpus: ${NUM_GPUS}"
echo "num_workers per GPU/rank: ${NUM_WORKERS}"
echo "batch_size: ${BATCH_SIZE}"
echo "stage1 config: ${STAGE1_CONFIG_NAME}"
echo "stage2 config: ${STAGE2_CONFIG_NAME}"

run_static_contract_checks
run_target_loader_passthrough_check

if [[ "${SMOKE_PREFLIGHT_ONLY:-false}" == "true" ]]; then
  echo "smoke_preflight_ok"
  exit 0
fi

STAGE1_LOG="${SMOKE_LOG_DIR}/stage1-${SLURM_JOB_ID:-manual}.log"
STAGE2_LOG="${SMOKE_LOG_DIR}/stage2-${SLURM_JOB_ID:-manual}.log"
STAGE1_INIT_STATE="${SMOKE_LOG_DIR}/stage1-initial-${SLURM_JOB_ID:-manual}.safetensors"

capture_stage1_initial_trainable_state "${STAGE1_INIT_STATE}"

export CONFIG_NAME="${STAGE1_CONFIG_NAME}"
export EXP_NAME="${STAGE1_EXP_NAME}"
export RESUME="${STAGE1_RESUME}"
export NUM_TRAIN_STEPS="${STAGE1_NUM_TRAIN_STEPS}"
export SAVE_INTERVAL="${STAGE1_SAVE_INTERVAL}"
export KEEP_PERIOD="${STAGE1_KEEP_PERIOD}"
export PYTORCH_WEIGHT_PATH_OVERRIDE="${OPENPI_PI0_BASE_DIR}"

echo "===== SMOKE STAGE 1 ====="
run_common_launcher_to_log "${STAGE1_LOG}"
validate_gt_aux_log "stage1" "${STAGE1_LOG}"
validate_trainable_log "stage1" "${STAGE1_LOG}"

STAGE1_CKPT_DIR="${CHECKPOINT_BASE_DIR}/${STAGE1_CONFIG_NAME}/${STAGE1_EXP_NAME}/${STAGE1_CHECKPOINT_STEP}"
if [[ ! -f "${STAGE1_CKPT_DIR}/model.safetensors" ]]; then
  echo "Expected Stage 1 checkpoint at ${STAGE1_CKPT_DIR}/model.safetensors" >&2
  exit 1
fi
validate_stage1_trainable_deltas "${STAGE1_INIT_STATE}" "${STAGE1_CKPT_DIR}/model.safetensors"

export CONFIG_NAME="${STAGE2_CONFIG_NAME}"
export EXP_NAME="${STAGE2_EXP_NAME}"
export RESUME="${STAGE2_RESUME}"
export NUM_TRAIN_STEPS="${STAGE2_NUM_TRAIN_STEPS}"
export SAVE_INTERVAL="${STAGE2_SAVE_INTERVAL}"
export KEEP_PERIOD="${STAGE2_KEEP_PERIOD}"
export PYTORCH_WEIGHT_PATH_OVERRIDE="${STAGE1_CKPT_DIR}"

echo "===== SMOKE STAGE 2 ====="
echo "stage2 weight init: ${PYTORCH_WEIGHT_PATH_OVERRIDE}"
run_common_launcher_to_log "${STAGE2_LOG}"
validate_stage2_weight_log "stage2" "${STAGE2_LOG}" "${STAGE1_CKPT_DIR}"
validate_gt_aux_log "stage2" "${STAGE2_LOG}"

STAGE2_CKPT_DIR="${CHECKPOINT_BASE_DIR}/${STAGE2_CONFIG_NAME}/${STAGE2_EXP_NAME}/${STAGE2_NUM_TRAIN_STEPS}"
if [[ ! -f "${STAGE2_CKPT_DIR}/model.safetensors" ]]; then
  echo "Expected Stage 2 checkpoint at ${STAGE2_CKPT_DIR}/model.safetensors" >&2
  exit 1
fi

echo "smoke_ok stage1_log=${STAGE1_LOG} stage2_log=${STAGE2_LOG}"

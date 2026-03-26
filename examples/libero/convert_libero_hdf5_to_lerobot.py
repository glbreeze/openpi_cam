"""Convert LIBERO-Camera HDF5 data directly into LeRobot format.

This script bypasses the TFDS / RLDS conversion path and reads LIBERO-Camera
HDF5 files produced by the camera-variation pipeline directly.

Expected input mapping:
  obs/agentview_rgb -> image
  obs/eye_in_hand_rgb -> wrist_image
  concat(obs/ee_states, obs/gripper_states) -> state
  actions -> actions
  problem_info.language_instruction -> task

Camera extrinsics are recomputed per frame from the MuJoCo runtime state so that:
  - agentview reflects camera-variation XML changes
  - robot0_eye_in_hand changes with robot motion
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import sys
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from openpi_client import image_tools
import tyro

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
GEO_ROOT = REPO_ROOT.parent
THIRD_PARTY_LIBERO = REPO_ROOT / "third_party" / "libero"
THIRD_PARTY_LIBERO_ROOT = THIRD_PARTY_LIBERO / "libero" / "libero"
_TASK_MAPPING = None
_LIBERO_UTILS = None
_ROBOSUITE_ASSETS_ROOT = None


def _ensure_libero_setup():
    global _TASK_MAPPING, _LIBERO_UTILS, _ROBOSUITE_ASSETS_ROOT

    if _TASK_MAPPING is not None and _LIBERO_UTILS is not None and _ROBOSUITE_ASSETS_ROOT is not None:
        return _TASK_MAPPING, _LIBERO_UTILS

    if str(THIRD_PARTY_LIBERO) not in sys.path:
        sys.path.insert(0, str(THIRD_PARTY_LIBERO))

    if "LIBERO_CONFIG_PATH" not in os.environ:
        default_libero_config = pathlib.Path.home() / ".libero_openpi_cam"
        os.environ["LIBERO_CONFIG_PATH"] = str(default_libero_config)
    else:
        default_libero_config = pathlib.Path(os.environ["LIBERO_CONFIG_PATH"])

    default_libero_config.mkdir(parents=True, exist_ok=True)
    libero_config_file = default_libero_config / "config.yaml"
    if not libero_config_file.exists():
        libero_config_file.write_text(
            "\n".join(
                [
                    f"benchmark_root: {THIRD_PARTY_LIBERO_ROOT}",
                    f"bddl_files: {THIRD_PARTY_LIBERO_ROOT / 'bddl_files'}",
                    f"init_states: {THIRD_PARTY_LIBERO_ROOT / 'init_files'}",
                    f"datasets: {GEO_ROOT / 'libero_cam_rlds'}",
                    f"assets: {THIRD_PARTY_LIBERO_ROOT / 'assets'}",
                ]
            )
            + "\n"
        )

    from libero.libero.envs import TASK_MAPPING  # noqa: PLC0415
    from libero.libero.utils import utils as libero_utils  # noqa: PLC0415
    import robosuite  # noqa: PLC0415

    _TASK_MAPPING = TASK_MAPPING
    _LIBERO_UTILS = libero_utils
    _ROBOSUITE_ASSETS_ROOT = pathlib.Path(robosuite.__file__).resolve().parent / "models" / "assets"
    return _TASK_MAPPING, _LIBERO_UTILS


def _as_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _load_json_attr(attrs: h5py.AttributeManager, key: str) -> dict:
    raw_value = attrs.get(key)
    if raw_value in (None, ""):
        raise ValueError(f"Missing required HDF5 attribute: {key}")
    return json.loads(_as_text(raw_value))


def _normalize_task(problem_info: dict) -> str:
    task = problem_info["language_instruction"]
    if isinstance(task, str):
        return task.strip().strip('"')
    if isinstance(task, (list, tuple)):
        return "".join(str(part) for part in task).strip().strip('"')
    raise TypeError(f"Unsupported language_instruction type: {type(task)!r}")


def _resolve_libero_path(raw_path: str) -> str:
    normalized = _as_text(raw_path).strip()
    if not normalized:
        return normalized

    candidate = pathlib.Path(normalized).expanduser()
    if candidate.exists():
        return str(candidate.resolve())

    posix_path = pathlib.PurePosixPath(normalized.replace("\\", "/"))
    parts = posix_path.parts
    fallback_candidates: list[pathlib.Path] = []

    for anchor in ("bddl_files", "init_files", "assets"):
        if anchor in parts:
            anchor_index = parts.index(anchor)
            fallback_candidates.append(THIRD_PARTY_LIBERO_ROOT / pathlib.Path(*parts[anchor_index:]))

    if len(parts) >= 2 and parts[0] == "libero" and parts[1] == "libero":
        fallback_candidates.append(THIRD_PARTY_LIBERO_ROOT / pathlib.Path(*parts[2:]))
    elif parts and parts[0] == "libero":
        fallback_candidates.append(THIRD_PARTY_LIBERO / pathlib.Path(*parts[1:]))

    fallback_candidates.append(THIRD_PARTY_LIBERO_ROOT / pathlib.Path(*parts))

    for fallback in fallback_candidates:
        if fallback.exists():
            return str(fallback.resolve())

    return normalized


def _build_env_from_data_group(data_group: h5py.Group):
    task_mapping, libero_utils = _ensure_libero_setup()
    env_kwargs = None
    problem_name = None

    if data_group.attrs.get("env_info") not in (None, ""):
        env_kwargs = _load_json_attr(data_group.attrs, "env_info")

    if data_group.attrs.get("problem_info") not in (None, ""):
        problem_info = _load_json_attr(data_group.attrs, "problem_info")
        problem_name = problem_info.get("problem_name")

    if env_kwargs is None or problem_name is None:
        env_args = _load_json_attr(data_group.attrs, "env_args")
        if env_kwargs is None:
            env_kwargs = env_args["env_kwargs"]
        if problem_name is None:
            problem_name = env_args["problem_name"]

    bddl_file_name = _as_text(data_group.attrs.get("bddl_file_name"))
    if not bddl_file_name:
        env_args = _load_json_attr(data_group.attrs, "env_args")
        bddl_file_name = env_args["bddl_file"]
    bddl_file_name = _resolve_libero_path(bddl_file_name)

    libero_utils.update_env_kwargs(
        env_kwargs,
        bddl_file_name=bddl_file_name,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        camera_depths=False,
        reward_shaping=True,
        control_freq=20,
        ignore_done=True,
    )
    return task_mapping[problem_name](**env_kwargs)


def _resolve_model_asset_path(raw_path: str) -> str:
    _ensure_libero_setup()

    normalized = _as_text(raw_path).strip()
    if not normalized:
        return normalized

    candidate = pathlib.Path(normalized).expanduser()
    if candidate.exists():
        return str(candidate.resolve())

    posix_path = pathlib.PurePosixPath(normalized.replace("\\", "/"))
    parts = posix_path.parts
    fallback_candidates: list[pathlib.Path] = []

    if "robosuite" in parts and "models" in parts and "assets" in parts:
        assets_index = parts.index("assets")
        fallback_candidates.append(_ROBOSUITE_ASSETS_ROOT / pathlib.Path(*parts[assets_index + 1 :]))
    if "chiliocosm" in parts and "assets" in parts:
        assets_index = parts.index("assets")
        fallback_candidates.append(THIRD_PARTY_LIBERO_ROOT / "assets" / pathlib.Path(*parts[assets_index + 1 :]))
        fallback_candidates.append(_ROBOSUITE_ASSETS_ROOT / pathlib.Path(*parts[assets_index + 1 :]))
    if len(parts) >= 3 and parts[0] == "libero" and parts[1] == "libero" and parts[2] == "assets":
        fallback_candidates.append(THIRD_PARTY_LIBERO_ROOT / "assets" / pathlib.Path(*parts[3:]))
    elif "libero" in parts and "assets" in parts:
        assets_index = parts.index("assets")
        fallback_candidates.append(THIRD_PARTY_LIBERO_ROOT / "assets" / pathlib.Path(*parts[assets_index + 1 :]))

    for fallback in fallback_candidates:
        if fallback.exists():
            return str(fallback.resolve())

    return normalized


def _rewrite_model_xml_paths(model_xml: str) -> str:
    def replace_file_attr(match: re.Match[str]) -> str:
        return f'file="{_resolve_model_asset_path(match.group(1))}"'

    return re.sub(r'file="([^"]+)"', replace_file_attr, model_xml)


def _get_camera_extrinsic(env, camera_name: str) -> np.ndarray:
    cam_id = env.sim.model.camera_name2id(camera_name)
    cam_pos = np.asarray(env.sim.data.cam_xpos[cam_id], dtype=np.float32)
    cam_rot = np.asarray(env.sim.data.cam_xmat[cam_id], dtype=np.float32).reshape(3, 3)

    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = cam_rot
    extrinsic[:3, 3] = cam_pos
    return extrinsic


def _reset_env_to_frame(env, model_xml: str, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reset_success = False
    while not reset_success:
        try:
            env.reset()
            reset_success = True
        except Exception:
            continue

    env.reset_from_xml_string(_rewrite_model_xml_paths(model_xml))
    env.sim.reset()
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    env._post_process()
    env._update_observables(force=True)

    return (
        _get_camera_extrinsic(env, "agentview"),
        _get_camera_extrinsic(env, "robot0_eye_in_hand"),
    )


def _preprocess_image(image: np.ndarray, image_size: int) -> np.ndarray:
    rotated = np.ascontiguousarray(np.asarray(image)[::-1, ::-1])
    resized = image_tools.resize_with_pad(rotated, image_size, image_size)
    return image_tools.convert_to_uint8(resized)


def _iter_hdf5_files(dataset_root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(dataset_root.rglob("*.hdf5"))


def _select_hdf5_files(
    dataset_root: pathlib.Path,
    mode: Literal["original_only", "all_views"],
) -> list[pathlib.Path]:
    files = _iter_hdf5_files(dataset_root)
    if mode == "original_only":
        return [path for path in files if "_camvar_" not in path.name]
    return files


def _select_shard_files(
    files: list[pathlib.Path],
    *,
    num_shards: int,
    shard_index: int,
) -> list[pathlib.Path]:
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not 0 <= shard_index < num_shards:
        raise ValueError(
            f"shard_index must satisfy 0 <= shard_index < num_shards, got shard_index={shard_index}, "
            f"num_shards={num_shards}"
        )
    if num_shards == 1:
        return files
    return [path for idx, path in enumerate(files) if idx % num_shards == shard_index]


def _default_repo_id(mode: Literal["original_only", "all_views"]) -> str:
    if mode == "original_only":
        return "glbreeze/libero_object"
    return "glbreeze/libero_object_cam"


def _create_dataset(repo_id: str, fps: int, image_size: int) -> LeRobotDataset:
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=fps,
        features={
            "image": {
                "dtype": "image",
                "shape": (image_size, image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (image_size, image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
            "agent_extrinsic": {
                "dtype": "float32",
                "shape": (4, 4),
                "names": ["row", "col"],
            },
            "wrist_extrinsic": {
                "dtype": "float32",
                "shape": (4, 4),
                "names": ["row", "col"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )


def _stage_selected_files(
    selected_files: list[pathlib.Path],
    *,
    staging_root: pathlib.Path,
) -> list[pathlib.Path]:
    staging_root.mkdir(parents=True, exist_ok=True)
    staged_files: list[pathlib.Path] = []
    for source_path in selected_files:
        destination_path = staging_root / source_path.name
        if not destination_path.exists():
            shutil.copy2(source_path, destination_path)
        staged_files.append(destination_path)
    return staged_files


def main(
    dataset_root: str,
    *,
    mode: Literal["original_only", "all_views"] = "all_views",
    repo_id: str | None = None,
    max_files: int = 0,
    max_episodes_per_file: int = 0,
    fps: int = 10,
    image_size: int = 256,
    num_shards: int = 1,
    shard_index: int = 0,
    stage_files_to_local: bool = False,
    push_to_hub: bool = False,
):
    dataset_root_path = pathlib.Path(dataset_root).expanduser().resolve()
    if not dataset_root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root_path}")

    selected_files = _select_hdf5_files(dataset_root_path, mode)
    selected_files = _select_shard_files(selected_files, num_shards=num_shards, shard_index=shard_index)
    if max_files > 0:
        selected_files = selected_files[:max_files]
    if not selected_files:
        raise ValueError(
            f"No HDF5 files found under {dataset_root_path} for mode={mode!r} "
            f"after shard filtering shard_index={shard_index}, num_shards={num_shards}"
        )

    if stage_files_to_local:
        staging_root = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / "libero_hdf5_stage" / f"shard_{shard_index:03d}"
        selected_files = _stage_selected_files(selected_files, staging_root=staging_root)
        print(f"[convert] staged_files_root={staging_root}")

    output_repo_id = repo_id or _default_repo_id(mode)
    dataset = _create_dataset(output_repo_id, fps=fps, image_size=image_size)

    total_episodes = 0
    total_frames = 0

    print(
        f"[convert] repo_id={output_repo_id} mode={mode} shard_index={shard_index}/{num_shards} "
        f"selected_files={len(selected_files)}"
    )

    for hdf5_path in selected_files:
        with h5py.File(hdf5_path, "r") as h5_file:
            data_group = h5_file["data"]
            problem_info = _load_json_attr(data_group.attrs, "problem_info")
            task = _normalize_task(problem_info)
            env = _build_env_from_data_group(data_group)

            try:
                episode_names = sorted(data_group.keys())
                if max_episodes_per_file > 0:
                    episode_names = episode_names[:max_episodes_per_file]

                for episode_name in episode_names:
                    episode_group = data_group[episode_name]
                    obs_group = episode_group["obs"]

                    agent_images = obs_group["agentview_rgb"][()]
                    wrist_images = obs_group["eye_in_hand_rgb"][()]
                    ee_states = np.asarray(obs_group["ee_states"][()], dtype=np.float32)
                    gripper_states = np.asarray(obs_group["gripper_states"][()], dtype=np.float32)
                    actions = np.asarray(episode_group["actions"][()], dtype=np.float32)
                    states = np.asarray(episode_group["states"][()])
                    model_xml = _as_text(episode_group.attrs["model_file"])

                    frame_count = len(actions)
                    if frame_count == 0:
                        continue
                    if any(len(array) != frame_count for array in (agent_images, wrist_images, ee_states, gripper_states, states)):
                        raise ValueError(f"Mismatched frame counts in {hdf5_path} / {episode_name}")

                    state = np.concatenate([ee_states, gripper_states], axis=-1).astype(np.float32)
                    if state.shape[-1] != 8:
                        raise ValueError(
                            f"Expected state dim 8, got {state.shape[-1]} in {hdf5_path} / {episode_name}"
                        )
                    if actions.shape[-1] != 7:
                        raise ValueError(
                            f"Expected action dim 7, got {actions.shape[-1]} in {hdf5_path} / {episode_name}"
                        )

                    for frame_index in range(frame_count):
                        agent_extrinsic, wrist_extrinsic = _reset_env_to_frame(
                            env,
                            model_xml=model_xml,
                            state=states[frame_index],
                        )
                        dataset.add_frame(
                            {
                                "image": _preprocess_image(agent_images[frame_index], image_size),
                                "wrist_image": _preprocess_image(wrist_images[frame_index], image_size),
                                "state": state[frame_index],
                                "actions": actions[frame_index],
                                "agent_extrinsic": agent_extrinsic,
                                "wrist_extrinsic": wrist_extrinsic,
                                "task": task,
                            }
                        )

                    dataset.save_episode()
                    total_episodes += 1
                    total_frames += frame_count
                    print(
                        f"[convert] file={hdf5_path.name} episode={episode_name} "
                        f"frames={frame_count} task={task}"
                    )
            finally:
                env.close()

    print(
        f"[convert] completed repo_id={output_repo_id} mode={mode} "
        f"files={len(selected_files)} episodes={total_episodes} frames={total_frames}"
    )

    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "hdf5"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)

"""Convert LIBERO-Camera HDF5 data directly into LeRobot format.

This script bypasses the TFDS / RLDS conversion path and reads LIBERO-Camera
HDF5 files produced by the camera-variation pipeline directly.

Expected input mapping (per-episode under /data/<episode>/obs):
  obs/agentview_rgb                -> image
  obs/eye_in_hand_rgb              -> wrist_image
  obs/agent_extrinsic  (T, 4, 4)   -> agent_extrinsic   (raw camera-to-world)
  obs/wrist_extrinsic  (T, 4, 4)   -> wrist_extrinsic
  obs.attrs["agent_intrinsic"] (3,3) -> agent_intrinsic (scaled to LeRobot image_size)
  obs.attrs["wrist_intrinsic"] (3,3) -> wrist_intrinsic
  obs.attrs["agent_image_size"] (H,W) -> reference resolution for K, used for scaling
  concat(obs/ee_states, obs/gripper_states) -> state
  actions                          -> actions
  problem_info.language_instruction -> task

K is scaled from the HDF5-native resolution (128 for LIBERO) to the LeRobot
output resolution (default 256) so downstream consumers receive K that matches
the stored image pixels. Extrinsics are stored raw (camera-to-world); any
image-flip convention correction is the model-side code's responsibility.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from openpi_client import image_tools
import tyro

_CAMVAR_PATH_RE = re.compile(r"_camvar_(?P<id>\d+)(?:_[A-Za-z0-9-]+)?\.hdf5$")


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


def _preprocess_image(image: np.ndarray, image_size: int) -> np.ndarray:
    rotated = np.ascontiguousarray(np.asarray(image)[::-1, ::-1])
    resized = image_tools.resize_with_pad(rotated, image_size, image_size)
    return image_tools.convert_to_uint8(resized)


def _scale_intrinsic(K: np.ndarray, src_h: int, src_w: int, dst_h: int, dst_w: int) -> np.ndarray:
    """Scale intrinsic K from (src_h, src_w) to (dst_h, dst_w) pixels.

    Works for the openpi LIBERO pipeline, where both source and target are
    square and resize_with_pad performs a uniform scale (no padding). The
    scaling multiplies fx, cx by dst_w/src_w and fy, cy by dst_h/src_h.
    """
    K_out = np.asarray(K, dtype=np.float32).copy()
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    K_out[0, 0] *= sx
    K_out[0, 2] *= sx
    K_out[1, 1] *= sy
    K_out[1, 2] *= sy
    return K_out


def _read_episode_camera_params(
    obs_group: h5py.Group,
    frame_count: int,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read per-frame extrinsics and per-episode intrinsics from one HDF5 obs group.

    Returns (agent_ext, wrist_ext, agent_K, wrist_K), where extrinsics are (T, 4, 4)
    and intrinsics are (3, 3) scaled to `image_size` pixels.
    """
    if "agent_extrinsic" not in obs_group or "wrist_extrinsic" not in obs_group:
        raise ValueError(
            "HDF5 obs group missing agent_extrinsic/wrist_extrinsic datasets — "
            "regenerate with an updated LIBERO-Camera/scripts/create_dataset.py"
        )
    if "agent_intrinsic" not in obs_group.attrs or "wrist_intrinsic" not in obs_group.attrs:
        raise ValueError(
            "HDF5 obs attrs missing agent_intrinsic/wrist_intrinsic — "
            "regenerate with an updated LIBERO-Camera/scripts/create_dataset.py"
        )

    agent_ext = np.asarray(obs_group["agent_extrinsic"][()], dtype=np.float32)
    wrist_ext = np.asarray(obs_group["wrist_extrinsic"][()], dtype=np.float32)
    if agent_ext.shape != (frame_count, 4, 4) or wrist_ext.shape != (frame_count, 4, 4):
        raise ValueError(
            f"Extrinsic shape mismatch: agent={agent_ext.shape}, wrist={wrist_ext.shape}, "
            f"expected ({frame_count}, 4, 4)"
        )

    agent_hw = np.asarray(obs_group.attrs.get("agent_image_size", (image_size, image_size)))
    wrist_hw = np.asarray(obs_group.attrs.get("wrist_image_size", (image_size, image_size)))
    agent_K = _scale_intrinsic(
        np.asarray(obs_group.attrs["agent_intrinsic"]),
        int(agent_hw[0]), int(agent_hw[1]), image_size, image_size,
    )
    wrist_K = _scale_intrinsic(
        np.asarray(obs_group.attrs["wrist_intrinsic"]),
        int(wrist_hw[0]), int(wrist_hw[1]), image_size, image_size,
    )
    return agent_ext, wrist_ext, agent_K, wrist_K


def _iter_hdf5_files(dataset_root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(dataset_root.rglob("*.hdf5"))


def _camera_label_for_file(path: pathlib.Path) -> str:
    match = _CAMVAR_PATH_RE.search(path.name)
    if not match:
        return "original"
    return f"camvar_{int(match.group('id')):02d}"


def _parse_camera_label_list(spec: str | None) -> set[str] | None:
    if spec is None:
        return None
    labels = {item.strip() for item in re.split(r"[,+;]", spec) if item.strip()}
    if not labels:
        raise ValueError("camera label filter must contain at least one non-empty label")
    return labels


def _select_hdf5_files(
    dataset_root: pathlib.Path,
    mode: Literal["original_only", "all_views"],
    *,
    include_camera_labels: set[str] | None = None,
    exclude_camera_labels: set[str] | None = None,
) -> list[pathlib.Path]:
    files = _iter_hdf5_files(dataset_root)
    if mode == "original_only":
        selected = [path for path in files if "_camvar_" not in path.name]
    else:
        selected = files

    if include_camera_labels is not None:
        selected = [path for path in selected if _camera_label_for_file(path) in include_camera_labels]

    if exclude_camera_labels is not None:
        selected = [path for path in selected if _camera_label_for_file(path) not in exclude_camera_labels]

    return selected


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


def _create_dataset(
    repo_id: str,
    fps: int,
    image_size: int,
    *,
    image_writer_threads: int = 10,
    image_writer_processes: int = 5,
) -> LeRobotDataset:
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
            "agent_intrinsic": {
                "dtype": "float32",
                "shape": (3, 3),
                "names": ["row", "col"],
            },
            "wrist_intrinsic": {
                "dtype": "float32",
                "shape": (3, 3),
                "names": ["row", "col"],
            },
        },
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
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
    include_camera_labels: str | None = None,
    exclude_camera_labels: str | None = None,
    max_files: int = 0,
    max_episodes_per_file: int = 0,
    fps: int = 10,
    image_size: int = 256,
    num_shards: int = 1,
    shard_index: int = 0,
    stage_files_to_local: bool = False,
    push_to_hub: bool = False,
    image_writer_threads: int = 10,
    image_writer_processes: int = 5,
):
    dataset_root_path = pathlib.Path(dataset_root).expanduser().resolve()
    if not dataset_root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root_path}")

    include_label_set = _parse_camera_label_list(include_camera_labels)
    exclude_label_set = _parse_camera_label_list(exclude_camera_labels)
    if include_label_set is not None and exclude_label_set is not None:
        overlap = include_label_set & exclude_label_set
        if overlap:
            raise ValueError(f"camera label filters overlap: {sorted(overlap)}")

    selected_files = _select_hdf5_files(
        dataset_root_path,
        mode,
        include_camera_labels=include_label_set,
        exclude_camera_labels=exclude_label_set,
    )
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
    dataset = _create_dataset(
        output_repo_id,
        fps=fps,
        image_size=image_size,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )

    total_episodes = 0
    total_frames = 0

    print(
        f"[convert] repo_id={output_repo_id} mode={mode} shard_index={shard_index}/{num_shards} "
        f"selected_files={len(selected_files)} include_camera_labels={sorted(include_label_set) if include_label_set else None} "
        f"exclude_camera_labels={sorted(exclude_label_set) if exclude_label_set else None}"
    )

    for hdf5_path in selected_files:
        with h5py.File(hdf5_path, "r") as h5_file:
            data_group = h5_file["data"]
            problem_info = _load_json_attr(data_group.attrs, "problem_info")
            task = _normalize_task(problem_info)

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

                frame_count = len(actions)
                if frame_count == 0:
                    continue
                if any(len(array) != frame_count for array in (agent_images, wrist_images, ee_states, gripper_states)):
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

                agent_ext, wrist_ext, agent_K, wrist_K = _read_episode_camera_params(
                    obs_group, frame_count, image_size
                )

                for frame_index in range(frame_count):
                    dataset.add_frame(
                        {
                            "image": _preprocess_image(agent_images[frame_index], image_size),
                            "wrist_image": _preprocess_image(wrist_images[frame_index], image_size),
                            "state": state[frame_index],
                            "actions": actions[frame_index],
                            "agent_extrinsic": agent_ext[frame_index],
                            "wrist_extrinsic": wrist_ext[frame_index],
                            "agent_intrinsic": agent_K,
                            "wrist_intrinsic": wrist_K,
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

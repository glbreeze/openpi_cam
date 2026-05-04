"""Mirror of RoboTwin official `convert_aloha_data_to_lerobot_robotwin.py`."""

from __future__ import annotations

import dataclasses
import fnmatch
import json
import os
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
    ]
    cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

    features = {
        "observation.state": {"dtype": "float32", "shape": (len(motors),), "names": [motors]},
        "action": {"dtype": "float32", "shape": (len(motors),), "names": [motors]},
    }
    if has_velocity:
        features["observation.velocity"] = {"dtype": "float32", "shape": (len(motors),), "names": [motors]}
    if has_effort:
        features["observation.effort"] = {"dtype": "float32", "shape": (len(motors),), "names": [motors]}
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4
        if uncompressed:
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2

            imgs_array = [cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) for data in ep[f"/observations/images/{camera}"]]
            imgs_array = np.array(imgs_array)
        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(ep_path: Path):
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:])
        action = torch.from_numpy(ep["/action"][:])
        velocity = torch.from_numpy(ep["/observations/qvel"][:]) if "/observations/qvel" in ep else None
        effort = torch.from_numpy(ep["/observations/effort"][:]) if "/observations/effort" in ep else None
        imgs_per_cam = load_raw_images_per_camera(ep, ["cam_high", "cam_left_wrist", "cam_right_wrist"])

    return imgs_per_cam, state, action, velocity, effort


def populate_dataset(dataset: LeRobotDataset, hdf5_files: list[Path], episodes: list[int] | None = None) -> LeRobotDataset:
    if episodes is None:
        episodes = list(range(len(hdf5_files)))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]
        imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path)

        with (ep_path.parent / "instructions.json").open("r", encoding="utf-8") as f_instr:
            instructions = json.load(f_instr)["instructions"]
        instruction = np.random.choice(instructions)

        for i in range(state.shape[0]):
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "task": instruction,
            }
            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]
            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]
            dataset.add_frame(frame)
        dataset.save_episode()

    return dataset


def port_aloha(
    raw_dir: Path,
    repo_id: str,
    *,
    episodes: list[int] | None = None,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    hdf5_files = []
    for root, _, files in os.walk(raw_dir):
        hdf5_files.extend(Path(root) / filename for filename in fnmatch.filter(files, "*.hdf5"))
    hdf5_files = sorted(hdf5_files)
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 episodes found under {raw_dir}")

    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        has_effort=has_effort(hdf5_files),
        has_velocity=has_velocity(hdf5_files),
        dataset_config=dataset_config,
    )
    populate_dataset(dataset, hdf5_files, episodes=episodes)


if __name__ == "__main__":
    tyro.cli(port_aloha)

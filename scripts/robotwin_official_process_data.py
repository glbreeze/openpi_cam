"""Mirror of RoboTwin official `policy/pi0/scripts/process_data.py`.

This preserves the official RoboTwin pi0 conversion layout so downstream
conversion and training match the baseline expectations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


def load_hdf5(dataset_path: Path):
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist at {dataset_path}")

    with h5py.File(dataset_path, "r") as root:
        left_gripper = root["/joint_action/left_gripper"][()]
        left_arm = root["/joint_action/left_arm"][()]
        right_gripper = root["/joint_action/right_gripper"][()]
        right_arm = root["/joint_action/right_arm"][()]
        image_dict = {}
        for cam_name in root["/observation/"]:
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, image_dict


def images_encoding(imgs: list[np.ndarray]):
    encoded = []
    max_len = 0
    for img in imgs:
        success, encoded_image = cv2.imencode(".jpg", img)
        if not success:
            raise ValueError("Failed to encode image as JPEG")
        jpeg_data = encoded_image.tobytes()
        encoded.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    return [item.ljust(max_len, b"\0") for item in encoded], max_len


def data_transform(path: Path, episode_num: int, save_path: Path):
    save_path.mkdir(parents=True, exist_ok=True)

    for i in range(episode_num):
        instruction_path = path / "instructions" / f"episode{i}.json"
        with instruction_path.open("r", encoding="utf-8") as f_instr:
            instructions = json.load(f_instr)["seen"]

        episode_dir = save_path / f"episode_{i}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        with (episode_dir / "instructions.json").open("w", encoding="utf-8") as f:
            json.dump({"instructions": instructions}, f, indent=2)

        left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, image_dict = load_hdf5(
            path / "data" / f"episode{i}.hdf5"
        )

        qpos = []
        actions = []
        cam_high = []
        cam_right_wrist = []
        cam_left_wrist = []
        left_arm_dim = []
        right_arm_dim = []

        for j in range(left_gripper_all.shape[0]):
            left_gripper = left_gripper_all[j]
            left_arm = left_arm_all[j]
            right_gripper = right_gripper_all[j]
            right_arm = right_arm_all[j]
            state = np.array([*left_arm.tolist(), left_gripper, *right_arm.tolist(), right_gripper], dtype=np.float32)

            if j != left_gripper_all.shape[0] - 1:
                qpos.append(state)
                head = cv2.imdecode(np.frombuffer(image_dict["head_camera"][j], np.uint8), cv2.IMREAD_COLOR)
                right = cv2.imdecode(np.frombuffer(image_dict["right_camera"][j], np.uint8), cv2.IMREAD_COLOR)
                left = cv2.imdecode(np.frombuffer(image_dict["left_camera"][j], np.uint8), cv2.IMREAD_COLOR)
                cam_high.append(cv2.resize(head, (640, 480)))
                cam_right_wrist.append(cv2.resize(right, (640, 480)))
                cam_left_wrist.append(cv2.resize(left, (640, 480)))

            if j != 0:
                actions.append(state)
                left_arm_dim.append(left_arm.shape[0])
                right_arm_dim.append(right_arm.shape[0])

        with h5py.File(episode_dir / f"episode_{i}.hdf5", "w") as f:
            f.create_dataset("action", data=np.array(actions))
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.array(qpos))
            obs.create_dataset("left_arm_dim", data=np.array(left_arm_dim))
            obs.create_dataset("right_arm_dim", data=np.array(right_arm_dim))
            image = obs.create_group("images")
            cam_high_enc, len_high = images_encoding(cam_high)
            cam_right_enc, len_right = images_encoding(cam_right_wrist)
            cam_left_enc, len_left = images_encoding(cam_left_wrist)
            image.create_dataset("cam_high", data=cam_high_enc, dtype=f"S{len_high}")
            image.create_dataset("cam_right_wrist", data=cam_right_enc, dtype=f"S{len_right}")
            image.create_dataset("cam_left_wrist", data=cam_left_enc, dtype=f"S{len_left}")

        print(f"process {i} success")


def main(task_name: str, setting: str, expert_data_num: int, raw_root: Path, processed_root: Path):
    load_dir = raw_root / task_name / setting
    if not load_dir.is_dir():
        raise FileNotFoundError(f"Missing raw RoboTwin task directory: {load_dir}")

    target_dir = processed_root / f"{task_name}-{setting}-{expert_data_num}"
    print(f"read data from path: {load_dir}")
    print(f"write processed episodes to: {target_dir}")
    data_transform(load_dir, expert_data_num, target_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("setting", type=str)
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    args = parser.parse_args()
    main(args.task_name, args.setting, args.expert_data_num, args.raw_root, args.processed_root)

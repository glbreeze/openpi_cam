"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from scipy.spatial.transform import Rotation
import tensorflow_datasets as tfds
import tyro

RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]  # For simplicity we will combine multiple Libero datasets into one training dataset


def transform_state_action_to_camera(state, action, T_wc):
    """
    Convert state and action from world frame to camera frame.
    state  : [x,y,z,rx,ry,rz, g1,g2]
    action : [dx,dy,dz, drx,dry,drz, g]
    T_wc   : camera extrinsic (world <- camera)
    """
    state = np.asarray(state)
    action = np.asarray(action)

    T_cw = np.linalg.inv(T_wc)
    R_cw = T_cw[:3, :3]

    # --- transform state position ---
    p_w = state[:3]
    p_c = R_cw @ (p_w - T_wc[:3, 3])

    # --- transform state orientation ---
    R_we = Rotation.from_rotvec(state[3:6]).as_matrix()
    R_ce = R_cw @ R_we
    rotvec_c = Rotation.from_matrix(R_ce).as_rotvec()

    state_cam = np.concatenate([p_c, rotvec_c, state[6:]])

    # --- transform action deltas ---
    action_cam = action.copy()
    action_cam[:3] = R_cw @ action[:3]

    delta_R_world = Rotation.from_rotvec(action[3:6]).as_matrix()  # action_cam[3:6] = R_cw @ action[3:6]
    delta_R_cam = R_cw @ delta_R_world @ R_cw.T
    action_cam[3:6] = Rotation.from_matrix(delta_R_cam).as_rotvec()

    return state_cam.astype(np.float32), action_cam.astype(np.float32)


def _resolve_raw_dataset_names(raw_dataset_names: str | None) -> list[str]:
    if raw_dataset_names is None:
        return list(RAW_DATASET_NAMES)
    names = [name.strip() for name in raw_dataset_names.split(",") if name.strip()]
    if not names:
        raise ValueError("raw_dataset_names must contain at least one dataset name.")
    return names


def main(
    data_dir: str,
    repo_name: str = "glbreeze/libero_cam",
    raw_dataset_names: str | None = None,
    include_cam_params: bool = False,
    transform_state_actions_to_camera: bool = False,
    *,
    push_to_hub: bool = False,
):
    raw_dataset_names = _resolve_raw_dataset_names(raw_dataset_names)
    include_cam_params = include_cam_params or ("cam" in repo_name)
    if transform_state_actions_to_camera and not include_cam_params:
        raise ValueError("transform_state_actions_to_camera=True requires include_cam_params=True.")

    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    features = {
        "image": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "image",
            "shape": (256, 256, 3),
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
    }

    if include_cam_params:
        features.update(
            {
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
            }
        )

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=10,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for raw_dataset_name in raw_dataset_names:
        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        for episode in raw_dataset:
            for step in episode["steps"].as_numpy_iterator():
                state_trans, action_trans = step["observation"]["state"], step["action"]

                if transform_state_actions_to_camera:
                    agent_extrinsic = step["observation"]["agent_extrinsic"]
                    state_trans, action_trans = transform_state_action_to_camera(
                        state=state_trans, action=action_trans, T_wc=agent_extrinsic
                    )

                frame = {
                    "image": step["observation"]["image"],
                    "wrist_image": step["observation"]["wrist_image"],
                    "state": state_trans,
                    "actions": action_trans,
                    "task": step["language_instruction"].decode(),
                }
                if include_cam_params:
                    frame["agent_extrinsic"] = step["observation"]["agent_extrinsic"]
                    frame["wrist_extrinsic"] = step["observation"]["wrist_extrinsic"]
                    frame["agent_intrinsic"] = step["observation"]["agent_intrinsic"]
                    frame["wrist_intrinsic"] = step["observation"]["wrist_intrinsic"]

                dataset.add_frame(frame)
            dataset.save_episode()

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)

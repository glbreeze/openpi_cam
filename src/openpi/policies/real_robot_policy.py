import dataclasses
import functools
import pathlib

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _scalar_index(value) -> int:
    arr = np.asarray(value)
    if arr.shape == ():
        return int(arr.item())
    if arr.size == 1:
        return int(arr.reshape(-1)[0])
    raise ValueError(f"Expected scalar episode/frame index, got shape {arr.shape}")


@functools.lru_cache(maxsize=8)
def _load_pi3_scene_intrinsics(calibration_root: str) -> dict[str, np.ndarray]:
    import yaml

    path = pathlib.Path(calibration_root) / "intrinsics.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Real-robot intrinsics file not found: {path}")
    with path.open() as f:
        raw = yaml.safe_load(f)

    intrinsics = {}
    for camera_name, camera in raw["cameras"].items():
        intrinsics[camera_name] = np.asarray(
            [
                [camera["fx"], 0.0, camera["cx"]],
                [0.0, camera["fy"], camera["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    return intrinsics


@functools.lru_cache(maxsize=8)
def _load_pi3_scene_extrinsics(calibration_root: str) -> dict[tuple[int, int, str], np.ndarray]:
    import pandas as pd

    path = pathlib.Path(calibration_root) / "extrinsics_per_frame.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Real-robot extrinsics file not found: {path}")

    df = pd.read_parquet(path, columns=["episode_index", "frame_index", "camera_name", "T_base_camera"])
    extrinsics = {}
    for row in df.itertuples(index=False):
        extrinsics[(int(row.episode_index), int(row.frame_index), str(row.camera_name))] = np.asarray(
            row.T_base_camera,
            dtype=np.float32,
        ).reshape(4, 4)
    return extrinsics


@dataclasses.dataclass(frozen=True)
class Pi3SceneCalibrationLoader(transforms.DataTransformFn):
    """Attach real-robot camera intrinsics/extrinsics from pi3_scene_capture files."""

    calibration_root: str
    base_camera_name: str = "context_left"
    wrist_camera_name: str = "wrist_right"

    def __call__(self, data: dict) -> dict:
        episode_index = _scalar_index(data["episode_index"])
        frame_index = _scalar_index(data["frame_index"])
        intrinsics = _load_pi3_scene_intrinsics(self.calibration_root)
        extrinsics = _load_pi3_scene_extrinsics(self.calibration_root)

        base_key = (episode_index, frame_index, self.base_camera_name)
        wrist_key = (episode_index, frame_index, self.wrist_camera_name)
        if base_key not in extrinsics:
            raise KeyError(f"Missing base camera extrinsic for {base_key} under {self.calibration_root}")
        if wrist_key not in extrinsics:
            raise KeyError(f"Missing wrist camera extrinsic for {wrist_key} under {self.calibration_root}")

        data = dict(data)
        data["agent_intrinsic"] = intrinsics[self.base_camera_name]
        data["wrist_intrinsic"] = intrinsics[self.wrist_camera_name]
        # T_base_camera is camera-to-base, i.e. camera pose in the robot-base world frame.
        data["agent_extrinsic"] = extrinsics[base_key]
        data["wrist_extrinsic"] = extrinsics[wrist_key]
        return data


@dataclasses.dataclass(frozen=True)
class RealRobotUR5Inputs(transforms.DataTransformFn):
    """Map the UR5 real-robot dataset into the standard pi0 image/state format."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/base_image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        for key in (
            "agent_intrinsic",
            "wrist_intrinsic",
            "agent_extrinsic",
            "wrist_extrinsic",
            "pi3x_target_xy",
            "pi3x_target_logz",
            "pi3x_target_conf",
            "point_target_xy",
            "point_target_logz",
            "point_target_conf",
            "point_target_source",
        ):
            if key in data:
                inputs[key] = np.asarray(data[key])

        return inputs


@dataclasses.dataclass(frozen=True)
class RealRobotUR5Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7], dtype=np.float32)}

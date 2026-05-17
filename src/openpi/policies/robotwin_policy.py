import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies import aloha_policy


def make_robotwin_example() -> dict:
    """Creates a random input example for the Robotwin policy."""
    return {
        "state": np.ones((14,), dtype=np.float32),
        "images": {
            "head_camera": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "left_camera": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "right_camera": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "beat the block with the hammer",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[-1] == 3:
        image = np.transpose(image, (2, 0, 1))
    return image


def _get_first(data: dict, *keys: str):
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"None of the keys were found: {keys}")


def _maybe_get_first(data: dict, *keys: str):
    for key in keys:
        if key in data:
            return data[key]
    return None


@dataclasses.dataclass(frozen=True)
class RobotwinInputs(transforms.DataTransformFn):
    """Map Robotwin observations to the standard pi0 input format.

    Supports either:
    - direct LeRobot-style flat keys, e.g. `observation.images.head_camera`, `observation.state`, `action`
    - official RoboTwin-converted keys, e.g. `observation.images.cam_high`
    - a simpler inference-style structure with `images`, `state`, and optional `actions` / `prompt`
    """

    model_type: _model.ModelType
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        if "images" in data:
            raw_images = data["images"]
            # Eval clients use multiple naming conventions: RoboTwin native
            # (head_camera / left_camera / right_camera) and pi0 / aloha
            # (cam_high / cam_left_wrist / cam_right_wrist). Accept either.
            source_images = {
                "head_camera": raw_images.get("head_camera", raw_images.get("cam_high")),
                "left_camera": raw_images.get("left_camera", raw_images.get("cam_left_wrist")),
                "right_camera": raw_images.get("right_camera", raw_images.get("cam_right_wrist")),
            }
            missing = [k for k, v in source_images.items() if v is None]
            if missing:
                raise KeyError(
                    f"RobotwinInputs: 'images' dict missing for {missing}; got keys {list(raw_images)}"
                )
            state = np.asarray(_get_first(data, "state", "observation.state", "observation/state"), dtype=np.float32)
            actions = _maybe_get_first(data, "actions", "action")
            prompt = _maybe_get_first(data, "prompt", "task")
        else:
            source_images = {
                "head_camera": _get_first(
                    data,
                    "observation.images.cam_high",
                    "observation.images.head_camera",
                    "observation/image",
                    "high_image",
                ),
                "left_camera": _get_first(
                    data,
                    "observation.images.cam_left_wrist",
                    "observation.images.left_camera",
                    "observation/wrist_image_left",
                    "left_wrist_image",
                ),
                "right_camera": _get_first(
                    data,
                    "observation.images.cam_right_wrist",
                    "observation.images.right_camera",
                    "observation/wrist_image_right",
                    "right_wrist_image",
                ),
            }
            state = np.asarray(_get_first(data, "observation.state", "observation/state", "state"), dtype=np.float32)
            actions = _maybe_get_first(data, "action", "actions")
            prompt = _maybe_get_first(data, "prompt", "task")

        aloha_like = {
            "state": state,
            "images": {
                "cam_high": _parse_image(source_images["head_camera"]),
                "cam_left_wrist": _parse_image(source_images["left_camera"]),
                "cam_right_wrist": _parse_image(source_images["right_camera"]),
            },
        }
        if actions is not None:
            aloha_like["actions"] = np.asarray(actions, dtype=np.float32)
        if prompt is not None:
            aloha_like["prompt"] = prompt

        return aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)(aloha_like)


@dataclasses.dataclass(frozen=True)
class RobotwinOutputs(transforms.DataTransformFn):
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        return aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)(data)


def _adjust_K_for_openpi_image_flip(K) -> np.ndarray:
    """Identity pass-through for RoboTwin/Sapien.

    Sapien renders in OpenCV (y-down) convention natively; the converter and
    eval client both pass images through without flipping, so K stays the
    natural OpenCV K with positive fx/fy. Kept as a named identity function
    in case we add per-cam K adjustments later.
    """
    return np.asarray(K, dtype=np.float32).copy()


@dataclasses.dataclass(frozen=True)
class RobotwinCamInputs(transforms.DataTransformFn):
    """Cam-aware Robotwin inputs. Same image/state path as `RobotwinInputs`, plus
    plumbs per-camera extrinsics + intrinsics for the cam-aware Pi0 (PRoPE+ray
    encoder) recipe.

    Expected dataset keys (after the data-config repack):
        observation/cam_high_extrinsic, observation/cam_high_intrinsic
        observation/cam_left_wrist_extrinsic, observation/cam_left_wrist_intrinsic
        observation/cam_right_wrist_extrinsic, observation/cam_right_wrist_intrinsic
    Mapping (RoboTwin → openpi model field):
        cam_high       -> agent_*
        cam_left_wrist -> wrist_*
        cam_right_wrist-> right_wrist_*
    Extrinsics are stored as camera-to-world (T_wc) in OpenCV camera frame by the
    converter, so no MuJoCo-style frame fix is needed. Intrinsics are passed
    through in natural Sapien/OpenCV orientation with positive fx/fy.
    """

    model_type: _model.ModelType
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        out = RobotwinInputs(model_type=self.model_type, adapt_to_pi=self.adapt_to_pi)(data)

        cam_field_pairs = [
            ("cam_high", "agent"),
            ("cam_left_wrist", "wrist"),
            ("cam_right_wrist", "right_wrist"),
        ]
        for src_cam, dst_cam in cam_field_pairs:
            for ext_key in (f"observation.{src_cam}_extrinsic", f"observation/{src_cam}_extrinsic"):
                if ext_key in data:
                    out[f"{dst_cam}_extrinsic"] = np.asarray(data[ext_key], dtype=np.float32)
                    break
            for intr_key in (f"observation.{src_cam}_intrinsic", f"observation/{src_cam}_intrinsic"):
                if intr_key in data:
                    out[f"{dst_cam}_intrinsic"] = _adjust_K_for_openpi_image_flip(data[intr_key])
                    break

        # Carry geometry-distillation targets through (Pi3X / GT mixed dual-loss).
        for key in (
            "pi3x_target_xy", "pi3x_target_logz", "pi3x_target_conf",
            "point_target_xy", "point_target_logz", "point_target_conf",
            "point_target_source",
        ):
            if key in data:
                out[key] = data[key]

        return out

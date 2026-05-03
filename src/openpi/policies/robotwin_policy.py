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
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
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
    - a simpler inference-style structure with `images`, `state`, and optional `actions` / `prompt`
    """

    model_type: _model.ModelType
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        if "images" in data:
            source_images = data["images"]
            state = np.asarray(_get_first(data, "state", "observation.state", "observation/state"), dtype=np.float32)
            actions = _maybe_get_first(data, "actions", "action")
            prompt = _maybe_get_first(data, "prompt")
        else:
            source_images = {
                "head_camera": _get_first(
                    data,
                    "observation.images.head_camera",
                    "observation/image",
                    "high_image",
                ),
                "left_camera": _get_first(
                    data,
                    "observation.images.left_camera",
                    "observation/wrist_image_left",
                    "left_wrist_image",
                ),
                "right_camera": _get_first(
                    data,
                    "observation.images.right_camera",
                    "observation/wrist_image_right",
                    "right_wrist_image",
                ),
            }
            state = np.asarray(_get_first(data, "observation.state", "observation/state", "state"), dtype=np.float32)
            actions = _maybe_get_first(data, "action", "actions")
            prompt = _maybe_get_first(data, "prompt")

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

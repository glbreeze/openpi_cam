"""RoboCasa365 (PandaOmron) policy transforms for openpi.

The upstream robocasa LeRobot conversion (PandaOmron_modality.json) stores:

* observation.state (16-d): [base_pos(3), base_quat(4), eef_pos_rel(3),
  eef_quat_rel(4), gripper_qpos(2)]
* action (12-d): [base_motion(4), control_mode(1), eef_pos(3), eef_rot(3),
  gripper_close(1)] — note that base + control_mode come FIRST in the array,
  then the eef block, then gripper. The Pi0-pretrain-friendly Franka subset
  is at dims [5:12] (eef_pos, eef_rot, gripper_close).
* 3 cams at 256x256: robot0_agentview_left/right, robot0_eye_in_hand.

Image direction (verified empirically against `OpenDrawer/lerobot/videos/.../episode_000000.mp4`):

* The LeRobot v2.1 video frames decode **right-side-up** by pixel convention
  (row 0 is the top of the image, ceiling/wall on top, floor on bottom).
  This is opposite to what I initially expected from reading
  `convert_hdf5_lerobot.py` — the conversion path apparently produces an
  already-y-down frame (likely via a flip during `extract_trajectory` or
  during MP4 encode).
* `RoboCasaGymEnv.get_basic_observation` applies `img[::-1, :, :]` to its
  buffer before returning, and the result is also right-side-up. So the
  gym wrapper output and the LeRobot stored frames **agree** in
  orientation. The eval client should pass through unchanged.
* No openpi-side `[::-1, ::-1]` flip exists in the current
  `preprocess_observation` (JAX) or `preprocess_observation_pytorch` —
  only stale comments in `libero_policy.py`. So neither training nor
  inference re-orients the image after it reaches the model.

K + extrinsic conventions:

* The image is in standard image-pixel convention (y-down). K stays
  natural (positive fx, fy, cx=cy=W/2) — no `fx → -fx` adjustment.
  Mirrors the `RobotwinInputs` pass-through K path.
* MuJoCo's `sim.data.cam_xmat` is in OpenGL camera frame (x-right, y-up,
  z-back); the extrinsic still needs the OpenGL→OpenCV swap so that
  `K @ T_wc^{-1} @ X_world` projects to the right pixel. We reuse
  `libero_policy._mujoco_to_opencv_extrinsic` for that (negates columns
  1 and 2 of the rotation block).
"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies import libero_policy

# Dims into the 12-d action stored under "action" in the LeRobot row.
ACTION_BASE_MOTION = slice(0, 4)
ACTION_CONTROL_MODE = slice(4, 5)
ACTION_EEF_POS = slice(5, 8)
ACTION_EEF_ROT = slice(8, 11)
ACTION_GRIPPER = slice(11, 12)
ACTION_DIM = 12

# Dims into the 16-d state stored under "observation.state".
STATE_BASE_POS = slice(0, 3)
STATE_BASE_QUAT = slice(3, 7)
STATE_EEF_POS = slice(7, 10)
STATE_EEF_QUAT = slice(10, 14)
STATE_GRIPPER = slice(14, 16)
STATE_DIM = 16


def make_robocasa_example() -> dict:
    """Random inference-style input for smoke-testing."""
    return {
        "state": np.zeros((STATE_DIM,), dtype=np.float32),
        "images": {
            "agentview_left": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "agentview_right": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "eye_in_hand": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "open the top drawer",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
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


def _passthrough_intrinsic(K) -> np.ndarray:
    """Identity K pass-through for RoboCasa.

    The LeRobot stored frames decode right-side-up (verified empirically) and
    the current openpi pipeline does not apply any image flip. So K stays the
    natural OpenGL-derived matrix (positive fx, fy, cx=cy=W/2). Same pattern
    as `robotwin_policy._adjust_K_for_openpi_image_flip` (kept as a named
    helper in case a per-cam adjustment is needed later).
    """
    return np.asarray(K, dtype=np.float32).copy()


@dataclasses.dataclass(frozen=True)
class RobocasaInputs(transforms.DataTransformFn):
    """RoboCasa365 PandaOmron -> Pi0 input format.

    Accepts both the LeRobot dot-form keys (`observation.images.robot0_*`,
    `observation.state`, `action`) used during training and a simpler
    inference-style structure (`images`, `state`, `actions`, `prompt`).

    Image-to-slot mapping:

        agentview_left -> base_0_rgb         (agent slot, always valid)
        eye_in_hand    -> left_wrist_0_rgb   (wrist slot, always valid)
        agentview_right-> right_wrist_0_rgb  (second exo, optional for v0)

    RoboCasa365 datasets carry all three cameras. RoboCasa v0 human50 carries
    only agentview_left + eye_in_hand, so the right_wrist slot is padded and
    masked out just like LIBERO's missing right wrist camera.

    State and action are zero-padded out to the model's configured dims.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        if "images" in data:
            raw_images = data["images"]
            source_images = {
                "agentview_left": raw_images.get(
                    "agentview_left",
                    raw_images.get("robot0_agentview_left"),
                ),
                "agentview_right": raw_images.get(
                    "agentview_right",
                    raw_images.get("robot0_agentview_right"),
                ),
                "eye_in_hand": raw_images.get(
                    "eye_in_hand",
                    raw_images.get("robot0_eye_in_hand"),
                ),
            }
            missing = [k for k in ("agentview_left", "eye_in_hand") if source_images[k] is None]
            if missing:
                raise KeyError(f"RobocasaInputs: 'images' dict missing for {missing}; got keys {list(raw_images)}")
            state = np.asarray(_get_first(data, "state", "observation.state", "observation/state"), dtype=np.float32)
            actions = _maybe_get_first(data, "actions", "action")
            prompt = _maybe_get_first(data, "prompt", "task", "annotation.human.task_description")
        else:
            source_images = {
                "agentview_left": _get_first(
                    data,
                    "observation.images.robot0_agentview_left",
                    "observation/agentview_left_image",
                    "video.robot0_agentview_left",
                ),
                "agentview_right": _maybe_get_first(
                    data,
                    "observation.images.robot0_agentview_right",
                    "observation/agentview_right_image",
                    "video.robot0_agentview_right",
                ),
                "eye_in_hand": _get_first(
                    data,
                    "observation.images.robot0_eye_in_hand",
                    "observation/eye_in_hand_image",
                    "video.robot0_eye_in_hand",
                ),
            }
            state = np.asarray(_get_first(data, "observation.state", "observation/state", "state"), dtype=np.float32)
            actions = _maybe_get_first(data, "action", "actions")
            prompt = _maybe_get_first(data, "prompt", "task", "annotation.human.task_description")

        base_image = _parse_image(source_images["agentview_left"])
        wrist_image = _parse_image(source_images["eye_in_hand"])
        has_right_image = source_images["agentview_right"] is not None
        right_image = _parse_image(source_images["agentview_right"]) if has_right_image else np.zeros_like(base_image)
        right_mask = np.True_ if has_right_image or self.model_type == _model.ModelType.PI0_FAST else np.False_

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": right_mask,
            },
        }

        if actions is not None:
            inputs["actions"] = np.asarray(actions, dtype=np.float32)
        if prompt is not None:
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class RobocasaOutputs(transforms.DataTransformFn):
    """Slice the first ACTION_DIM (=12) dims off the model's padded action chunk.

    Returns the raw 12-d action in the LeRobot storage order:
        [base_motion(4), control_mode(1), eef_pos(3), eef_rot(3), gripper(1)].

    The eval client is responsible for re-packing this into the gym wrapper's
    dict-shaped action via the keys defined in `PandaOmronKeyConverter.unmap_action`.
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}


@dataclasses.dataclass(frozen=True)
class RobocasaCamInputs(transforms.DataTransformFn):
    """Camera-aware variant of `RobocasaInputs` for the Pi3X cam-aware Pi0 recipe.

    Plumbs per-camera intrinsics (K) and camera-to-world extrinsics (T_wc) for
    all three cams. K is passed through unchanged (right-side-up image, no
    flip in the openpi pipeline — see module docstring). T_wc is the
    OpenGL→OpenCV camera-frame swap from `libero_policy._mujoco_to_opencv_extrinsic`
    (negate columns 1 and 2 of the rotation block).

    Expected dataset keys (post repack into `observation/*` form by the data
    config):

        observation/agentview_left_extrinsic   (4, 4) MuJoCo cam-to-world
        observation/agentview_left_intrinsic   (3, 3) natural OpenGL K
        observation/agentview_right_extrinsic
        observation/agentview_right_intrinsic
        observation/eye_in_hand_extrinsic
        observation/eye_in_hand_intrinsic

    Camera slot mapping (matches `RobocasaInputs`):

        agentview_left  -> agent_*
        eye_in_hand     -> wrist_*
        agentview_right -> right_wrist_*
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        out = RobocasaInputs(model_type=self.model_type)(data)

        cam_field_pairs = [
            ("agentview_left", "agent"),
            ("eye_in_hand", "wrist"),
            ("agentview_right", "right_wrist"),
        ]
        for src_cam, dst_cam in cam_field_pairs:
            for ext_key in (
                f"observation.{src_cam}_extrinsic",
                f"observation/{src_cam}_extrinsic",
            ):
                if ext_key in data:
                    out[f"{dst_cam}_extrinsic"] = libero_policy._mujoco_to_opencv_extrinsic(data[ext_key])  # noqa: SLF001
                    break
            for intr_key in (
                f"observation.{src_cam}_intrinsic",
                f"observation/{src_cam}_intrinsic",
            ):
                if intr_key in data:
                    out[f"{dst_cam}_intrinsic"] = _passthrough_intrinsic(data[intr_key])
                    break

        # Pass-through Pi3X / GT geometry-distillation targets.
        for key in (
            "pi3x_target_xy",
            "pi3x_target_logz",
            "pi3x_target_conf",
            "point_target_xy",
            "point_target_logz",
            "point_target_conf",
            "point_target_source",
        ):
            if key in data:
                out[key] = data[key]

        return out

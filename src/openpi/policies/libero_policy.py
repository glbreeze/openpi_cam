import dataclasses
from typing import Literal

import einops
import numpy as np
from scipy.spatial.transform import Rotation

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


Frame = Literal["world", "agent_camera"]
ActionFrame = Frame
StateFrame = Frame


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    leading_shape = rotvec.shape[:-1]
    return Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_matrix().reshape(*leading_shape, 3, 3)


def _matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    leading_shape = matrix.shape[:-2]
    return Rotation.from_matrix(matrix.reshape(-1, 3, 3)).as_rotvec().reshape(*leading_shape, 3)


def _state_world_to_agent_camera(state: np.ndarray, agent_extrinsic: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    agent_extrinsic = np.asarray(agent_extrinsic, dtype=np.float32)
    rotation_wc = agent_extrinsic[:3, :3]
    rotation_cw = rotation_wc.T
    translation_wc = agent_extrinsic[:3, 3]

    transformed = state.copy()
    transformed[..., :3] = np.einsum("ij,...j->...i", rotation_cw, state[..., :3] - translation_wc)

    rotation_we = _rotvec_to_matrix(state[..., 3:6])
    rotation_ce = np.einsum("ij,...jk->...ik", rotation_cw, rotation_we)
    transformed[..., 3:6] = _matrix_to_rotvec(rotation_ce)
    return transformed.astype(np.float32)


def _actions_world_to_agent_camera(actions: np.ndarray, agent_extrinsic: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    agent_extrinsic = np.asarray(agent_extrinsic, dtype=np.float32)
    rotation_wc = agent_extrinsic[:3, :3]
    rotation_cw = rotation_wc.T

    transformed = actions.copy()
    transformed[..., :3] = np.einsum("ij,...j->...i", rotation_cw, actions[..., :3])

    rotation_delta_w = _rotvec_to_matrix(actions[..., 3:6])
    rotation_delta_c = np.einsum("ij,...jk,kl->...il", rotation_cw, rotation_delta_w, rotation_wc)
    transformed[..., 3:6] = _matrix_to_rotvec(rotation_delta_c)
    return transformed.astype(np.float32)


def _actions_agent_camera_to_world(actions: np.ndarray, agent_extrinsic: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    agent_extrinsic = np.asarray(agent_extrinsic, dtype=np.float32)
    rotation_wc = agent_extrinsic[:3, :3]
    rotation_cw = rotation_wc.T

    transformed = actions.copy()
    transformed[..., :3] = np.einsum("ij,...j->...i", rotation_wc, actions[..., :3])

    rotation_delta_c = _rotvec_to_matrix(actions[..., 3:6])
    rotation_delta_w = np.einsum("ij,...jk,kl->...il", rotation_wc, rotation_delta_c, rotation_cw)
    transformed[..., 3:6] = _matrix_to_rotvec(rotation_delta_w)
    return transformed.astype(np.float32)


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType
    state_frame: StateFrame = "world"
    action_frame: ActionFrame = "world"
    preconverted_state_frame: bool = False
    preconverted_action_frame: bool = False

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        state = np.asarray(data["observation/state"])
        agent_extrinsic = data.get("observation/agent_extrinsic")

        has_actions = "actions" in data
        should_convert_state = not (self.preconverted_state_frame and has_actions)

        if self.state_frame == "agent_camera":
            if agent_extrinsic is None:
                raise ValueError("state_frame='agent_camera' requires observation/agent_extrinsic.")
            if should_convert_state:
                state = _state_world_to_agent_camera(state, agent_extrinsic)
        elif self.state_frame != "world":
            raise ValueError(f"Unsupported LIBERO state_frame: {self.state_frame}")

        if self.action_frame == "agent_camera":
            if agent_extrinsic is None:
                raise ValueError("action_frame='agent_camera' requires observation/agent_extrinsic.")
        elif self.action_frame != "world":
            raise ValueError(f"Unsupported LIBERO action_frame: {self.action_frame}")

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # ---- processing possible camera param ----
        if "observation/agent_extrinsic" in data:
            inputs["agent_extrinsic"] = data["observation/agent_extrinsic"]

        if "observation/wrist_extrinsic" in data:
            inputs["wrist_extrinsic"] = data["observation/wrist_extrinsic"]

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if has_actions:
            actions = np.asarray(data["actions"])
            if self.action_frame == "agent_camera" and not self.preconverted_action_frame:
                actions = _actions_world_to_agent_camera(actions, agent_extrinsic)
            inputs["actions"] = actions

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    action_frame: ActionFrame = "world"

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        actions = np.asarray(data["actions"][:, :7])
        if self.action_frame == "agent_camera":
            if "agent_extrinsic" not in data:
                raise ValueError("action_frame='agent_camera' outputs require agent_extrinsic.")
            actions = _actions_agent_camera_to_world(actions, data["agent_extrinsic"])
        elif self.action_frame != "world":
            raise ValueError(f"Unsupported LIBERO action_frame: {self.action_frame}")
        return {"actions": actions}

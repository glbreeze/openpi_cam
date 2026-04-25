import dataclasses
import logging
import pathlib

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

logger = logging.getLogger("openpi.libero_policy")


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


def _adjust_K_for_openpi_image_flip(K) -> np.ndarray:
    """Absorb openpi's `_preprocess_image` `[::-1, ::-1]` flip into K.

    openpi flips the LIBERO image on both axes before feeding it to the model:
      - `flipud` maps opengl-y-up onto standard y-down pixel convention; this
        is already consistent with positive fy, so cy/fy stay unchanged.
      - `fliplr` mirrors x; with a centered principal point (cx = W/2) the
        correct absorption is `fx -> -fx`, which flips ray directions in x
        and inverts the u-coordinate of the projection.

    Must be paired with `_mujoco_to_opencv_extrinsic` on the matching camera
    extrinsic so that `K @ viewmat @ X_w` projects into the flipped image
    consistently.

    Inputs may be float ndarray or torch tensor. Returns the same type.
    """
    K_out = np.asarray(K).copy()
    K_out[..., 0, 0] = -K_out[..., 0, 0]
    return K_out.astype(np.float32)


def _mujoco_to_opencv_extrinsic(T) -> np.ndarray:
    """Convert a (4, 4) camera-to-world extrinsic from MuJoCo to OpenCV camera-frame convention.

    MuJoCo's ``cam_xmat`` stores the camera axes in world coords as (x-right,
    y-up, z-back) — the OpenGL convention. Our intrinsic K (positive fx/fy,
    cx=W/2, cy=H/2, plus the `fx -> -fx` fliplr adjustment) is a standard
    OpenCV pinhole matrix that assumes an (x-right, y-down, z-forward) camera
    frame. To pair them cleanly we redefine the camera frame by
    right-multiplying the C2W matrix by diag(1, -1, -1, 1) — this negates the
    camera's y and z axes without moving the camera in the world:

        T_wc_opencv = T_wc_mujoco @ diag(1, -1, -1, 1)

    Concretely this negates columns 1 and 2 of the rotation block while
    leaving the translation column and the homogeneous row untouched. The
    result is a valid SE(3) matrix whose viewmat maps world points into the
    OpenCV camera frame used by `K` and by `ray_embed`.

    Applying this on every extrinsic that is paired with an image that has
    been fed through openpi's `_preprocess_image` (`[::-1, ::-1]`) is what
    makes `K @ viewmat @ X_w` land on the correct pixel in the stored (flipped)
    image.
    """
    T_out = np.asarray(T).copy()
    T_out[..., :3, 1] = -T_out[..., :3, 1]
    T_out[..., :3, 2] = -T_out[..., :3, 2]
    return T_out.astype(np.float32)


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

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
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
            inputs["agent_extrinsic"] = _mujoco_to_opencv_extrinsic(data["observation/agent_extrinsic"])

        if "observation/wrist_extrinsic" in data:
            inputs["wrist_extrinsic"] = _mujoco_to_opencv_extrinsic(data["observation/wrist_extrinsic"])

        if "observation/agent_intrinsic" in data:
            inputs["agent_intrinsic"] = _adjust_K_for_openpi_image_flip(data["observation/agent_intrinsic"])

        if "observation/wrist_intrinsic" in data:
            inputs["wrist_intrinsic"] = _adjust_K_for_openpi_image_flip(data["observation/wrist_intrinsic"])

        # ---- Pi3X distillation targets, when cached upstream by Pi3xLiberoTargetLoader ----
        for key in ("pi3x_target_xy", "pi3x_target_logz", "pi3x_target_conf"):
            if key in data:
                inputs[key] = data[key]

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class Pi3xLiberoTargetLoader(transforms.DataTransformFn):
    """Inject pre-computed Pi3X patch-level distillation targets into the data dict.

    Reads `episode_index` / `frame_index` from the LeRobot row, looks up
    `{root}/{cam}/episode_{NNNNNN}.npz` for each cam in `cam_to_npz_subdir`, slices
    the row at `frame_index`, and stacks across cams to produce the V views consumed
    by the auxiliary point head loss.

    The cam list ordering must match the openpi image insertion order
    (`base_0_rgb`, `left_wrist_0_rgb`, ...). Padded views (e.g. LIBERO's
    `right_wrist_0_rgb`) have no teacher target and are excluded — the loss site
    slices `pred[:, :V]` to match.
    """

    root: str
    cam_to_npz_subdir: tuple[tuple[str, str], ...] = (
        ("base", "agent"),
        ("left_wrist", "wrist"),
    )

    def __call__(self, data: dict) -> dict:
        episode_index = int(np.asarray(data["episode_index"]).item())
        frame_index = int(np.asarray(data["frame_index"]).item())

        root = pathlib.Path(self.root).expanduser()
        xy_views, logz_views, conf_views = [], [], []
        for _, subdir in self.cam_to_npz_subdir:
            npz_path = root / subdir / f"episode_{episode_index:06d}.npz"
            with np.load(npz_path, mmap_mode="r") as f:
                xy_views.append(np.asarray(f["xy"][frame_index], dtype=np.float32))
                logz_views.append(np.asarray(f["log_z"][frame_index], dtype=np.float32))
                conf_views.append(np.asarray(f["conf"][frame_index], dtype=np.float32))

        data["pi3x_target_xy"] = np.stack(xy_views, axis=0)  # (V, 16, 16, 2)
        data["pi3x_target_logz"] = np.stack(logz_views, axis=0)  # (V, 16, 16, 1)
        data["pi3x_target_conf"] = np.stack(conf_views, axis=0)  # (V, 16, 16, 1)
        return data


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :7])}

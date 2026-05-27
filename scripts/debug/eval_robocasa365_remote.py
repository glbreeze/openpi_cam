"""Run RoboCasa365 rollouts against an openpi websocket policy server."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import gymnasium as gym
import h5py
import imageio
import numpy as np
import robocasa  # noqa: F401
import robosuite  # noqa: F401

from openpi_client import websocket_client_policy


CAM_KEYS = {
    "agentview_left": (
        "video.robot0_agentview_left",
        "robot0_agentview_left_image",
        "observation.images.robot0_agentview_left",
    ),
    "agentview_right": (
        "video.robot0_agentview_right",
        "robot0_agentview_right_image",
        "observation.images.robot0_agentview_right",
    ),
    "eye_in_hand": (
        "video.robot0_eye_in_hand",
        "robot0_eye_in_hand_image",
        "observation.images.robot0_eye_in_hand",
    ),
}
CAM_NAMES = {
    "agentview_left": "robot0_agentview_left",
    "agentview_right": "robot0_agentview_right",
    "eye_in_hand": "robot0_eye_in_hand",
}

STATE_KEYS = (
    "state.base_position",
    "state.base_rotation",
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
)
V0_STATE_KEYS = (
    "robot0_base_pos",
    "robot0_base_quat",
    "robot0_base_to_eef_pos",
    "robot0_base_to_eef_quat",
    "robot0_gripper_qpos",
)

GRIPPER_CLOSE_THRESHOLD = 0.0
BASE_MOTION_DEADZONE = 0.02
ROBOCASA_V0_ACTION_DIM = 12


def _patch_eval_action_unmap() -> None:
    """Eval-only patch for RoboCasa PandaOmron action thresholding."""
    import robocasa.wrappers.gym_wrapper as robocasa_gym_wrapper

    def unmap_action(cls, input_action):
        return {
            "robot0_right_gripper": (
                -1.0
                if input_action["action.gripper_close"] < GRIPPER_CLOSE_THRESHOLD
                else 1.0
            ),
            "robot0_right": np.concatenate(
                (
                    input_action["action.end_effector_position"],
                    input_action["action.end_effector_rotation"],
                ),
                axis=-1,
            ),
            "robot0_base": input_action["action.base_motion"][..., 0:3],
            "robot0_torso": input_action["action.base_motion"][..., 3:4],
            "robot0_base_mode": -1.0,
        }

    robocasa_gym_wrapper.PandaOmronKeyConverter.unmap_action = classmethod(unmap_action)


def _absolutize_mjcf_asset_paths(xml_str: str, source_xml_dir: str) -> str:
    """Keep read-only RoboCasa assets loadable when generated XML lives elsewhere."""
    root = ET.fromstring(xml_str)
    asset = root.find("asset")
    if asset is None:
        return xml_str

    for elem in asset.findall("mesh") + asset.findall("texture"):
        asset_file = elem.get("file")
        if asset_file is None or os.path.isabs(asset_file):
            continue
        elem.set("file", os.path.normpath(os.path.join(source_xml_dir, asset_file)))

    return ET.tostring(root, encoding="utf8").decode("utf8")


def _patch_robocasa_temp_mjcf_xml() -> None:
    """Write RoboCasa's generated object XML files to scratch instead of read-only assets."""
    import robocasa.environments.kitchen.kitchen as kitchen_mod
    import robocasa.models.objects.objects as objects_mod
    from robosuite.models.objects import MujocoXMLObject

    if getattr(kitchen_mod.MJCFObject, "_openpi_tmp_xml_patch", False):
        return

    original_cls = objects_mod.MJCFObject

    class ScratchMJCFObject(original_cls):
        _openpi_tmp_xml_patch = True

        def __init__(
            self,
            name,
            mjcf_path,
            scale=1.0,
            solimp=(0.998, 0.998, 0.001),
            solref=(0.001, 1),
            density=100,
            friction=(0.95, 0.3, 0.1),
            margin=None,
            rgba=None,
            priority=None,
        ):
            if isinstance(scale, float):
                scale = [scale, scale, scale]
            elif isinstance(scale, tuple) or isinstance(scale, list):
                assert len(scale) == 3
                scale = tuple(scale)
            else:
                raise TypeError(f"got invalid scale: {scale}")
            scale = np.array(scale)

            self.solimp = solimp
            self.solref = solref
            self.density = density
            self.friction = friction
            self.margin = margin
            self.priority = priority
            self.rgba = rgba

            xml_path = mjcf_path
            source_xml_dir = os.path.dirname(xml_path)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            xml_str = ET.tostring(root, encoding="utf8").decode("utf8")
            xml_str = self.postprocess_model_xml(xml_str)
            xml_str = _absolutize_mjcf_asset_paths(xml_str, source_xml_dir)

            tmp_root = Path(
                os.environ.get(
                    "ROBOCASA_TMP_XML_DIR",
                    os.path.join(os.environ.get("TMPDIR", "/tmp"), "robocasa_mjcf_xml"),
                )
            )
            tmp_root.mkdir(parents=True, exist_ok=True)
            time_str = str(time.time()).replace(".", "_")
            new_xml_path = tmp_root / f"{time_str}_{os.getpid()}.xml"
            new_xml_path.write_text(xml_str)

            try:
                MujocoXMLObject.__init__(
                    self,
                    fname=str(new_xml_path),
                    name=name,
                    joints=[dict(type="free", damping="0.0005")],
                    obj_type="all",
                    duplicate_collision_geoms=False,
                    scale=scale,
                )
            finally:
                new_xml_path.unlink(missing_ok=True)

    kitchen_mod.MJCFObject = ScratchMJCFObject


def _first_obs_value(obs: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in obs:
            return obs[key]
    raise KeyError(f"None of {keys} found; available keys: {sorted(obs)}")


def _maybe_obs_value(obs: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in obs:
            return obs[key]
    return None


def _resize_chw(image: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC RGB image, got {arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        if arr.shape[1] == size and arr.shape[2] == size:
            return np.ascontiguousarray(arr)
        arr = arr.transpose(1, 2, 0)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected HWC or CHW RGB image, got {arr.shape}")
    if arr.shape[0] == size and arr.shape[1] == size:
        return np.ascontiguousarray(arr.transpose(2, 0, 1))

    from PIL import Image

    resized = Image.fromarray(arr.astype(np.uint8)).resize((size, size), Image.BILINEAR)
    return np.ascontiguousarray(np.asarray(resized).transpose(2, 0, 1))


def _resize_hwc(image: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC or CHW RGB image, got {arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = arr.transpose(1, 2, 0)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected HWC or CHW RGB image, got {arr.shape}")
    if arr.shape[0] == size and arr.shape[1] == size:
        return np.ascontiguousarray(arr)

    from PIL import Image

    return np.ascontiguousarray(
        np.asarray(Image.fromarray(arr.astype(np.uint8)).resize((size, size), Image.BILINEAR))
    )


def _image_orientation_mse(source_hwc: np.ndarray, candidate_hwc: np.ndarray) -> dict[str, float]:
    candidate = np.asarray(candidate_hwc, dtype=np.float32)
    source = _resize_hwc(source_hwc, int(candidate.shape[0])).astype(np.float32)
    modes = {
        "as_is": candidate,
        "vflip": candidate[::-1, :, :],
        "hflip": candidate[:, ::-1, :],
        "rot180": candidate[::-1, ::-1, :],
    }
    return {name: float(np.mean((source - value) ** 2)) for name, value in modes.items()}


def _policy_input_image_checks(demo: h5py.Group | None, policy_input: dict[str, Any]) -> dict[str, Any]:
    if demo is None:
        return {}
    checks: dict[str, Any] = {}
    mapping = {
        "agentview_left": "robot0_agentview_left_image",
        "eye_in_hand": "robot0_eye_in_hand_image",
        "agentview_right": "robot0_agentview_right_image",
    }
    images = policy_input.get("images") or {}
    for policy_cam, hdf5_obs_name in mapping.items():
        if policy_cam not in images:
            continue
        src_key = f"obs/{hdf5_obs_name}"
        if src_key not in demo:
            continue
        candidate = np.asarray(images[policy_cam]).transpose(1, 2, 0)
        mse = _image_orientation_mse(demo[src_key][0], candidate)
        checks[policy_cam] = {
            "source_shape": list(np.asarray(demo[src_key][0]).shape),
            "policy_shape": list(candidate.shape),
            "mse": mse,
            "best": min(mse, key=mse.get),
        }
    return checks


def _policy_input_camera_summary(policy_input: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for cam in ("agentview_left", "eye_in_hand", "agentview_right"):
        ext = policy_input.get(f"observation.{cam}_extrinsic")
        intr = policy_input.get(f"observation.{cam}_intrinsic")
        if ext is None and intr is None:
            continue
        cam_summary: dict[str, Any] = {}
        if ext is not None:
            ext_arr = np.asarray(ext, dtype=np.float32)
            cam_summary["extrinsic_translation"] = ext_arr[:3, 3].tolist()
            cam_summary["extrinsic_rotation_row0"] = ext_arr[0, :3].tolist()
        if intr is not None:
            intr_arr = np.asarray(intr, dtype=np.float32)
            cam_summary["intrinsic_diag"] = [float(intr_arr[0, 0]), float(intr_arr[1, 1])]
            cam_summary["intrinsic_center"] = [float(intr_arr[0, 2]), float(intr_arr[1, 2])]
        summary[cam] = cam_summary
    return summary


def _demo_image_for_policy(demo: h5py.Group, policy_cam: str, step: int, image_size: int) -> np.ndarray | None:
    mapping = {
        "agentview_left": "robot0_agentview_left_image",
        "eye_in_hand": "robot0_eye_in_hand_image",
        "agentview_right": "robot0_agentview_right_image",
    }
    hdf5_name = mapping.get(policy_cam)
    if hdf5_name is None:
        return None
    key = f"obs/{hdf5_name}"
    if key not in demo:
        return None
    idx = min(max(int(step), 0), int(demo[key].shape[0]) - 1)
    return _resize_chw(np.asarray(demo[key][idx]), image_size)


def _scale_intrinsic(K: np.ndarray, src_size: int, dst_size: int) -> np.ndarray:
    scale = float(dst_size) / float(src_size)
    out = np.asarray(K, dtype=np.float32).copy()
    out[0, 0] *= scale
    out[1, 1] *= scale
    out[0, 2] *= scale
    out[1, 2] *= scale
    return out


def _camera_matrices_from_demo_cache(
    cache_demo: h5py.Group,
    step: int,
    image_size: int,
    *,
    send_opencv_extrinsics: bool,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    mapping = {
        "agentview_left": ("K_agent", "T_wc_agent"),
        "eye_in_hand": ("K_wrist", "T_wc_wrist"),
    }
    src_size = int(cache_demo.file.attrs.get("image_size", 128))
    for policy_cam, (K_key, T_key) in mapping.items():
        if K_key not in cache_demo or T_key not in cache_demo:
            continue
        T_seq = cache_demo[T_key]
        idx = min(max(int(step), 0), int(T_seq.shape[0]) - 1)
        T_wc = np.asarray(T_seq[idx], dtype=np.float32)
        if send_opencv_extrinsics:
            T_wc = _mujoco_to_opencv_extrinsic(T_wc)
        out[f"observation.{policy_cam}_intrinsic"] = _scale_intrinsic(
            np.asarray(cache_demo[K_key], dtype=np.float32),
            src_size=src_size,
            dst_size=image_size,
        )
        out[f"observation.{policy_cam}_extrinsic"] = T_wc
    return out


def _as_prompt_string(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.item() if value.shape == () else value.tolist()
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value)


def _prompt_for_obs(obs: dict[str, Any], prompt: str, *, env=None) -> str:
    if prompt and prompt.lower() != "auto":
        return prompt
    obs_prompt = obs.get("annotation.human.task_description")
    if obs_prompt is not None:
        return _as_prompt_string(obs_prompt)
    get_ep_meta = getattr(env, "get_ep_meta", None)
    if get_ep_meta is not None:
        try:
            ep_meta = get_ep_meta()
        except Exception:
            ep_meta = None
        if isinstance(ep_meta, dict):
            for key in ("lang", "task", "description"):
                if ep_meta.get(key):
                    return _as_prompt_string(ep_meta[key])
    return "OpenDrawer"


def _obs_to_policy_input(
    obs: dict[str, Any],
    prompt: str,
    image_size: int,
    *,
    env=None,
    include_camera_matrices: bool = True,
    require_camera_matrices: bool = False,
    send_opencv_extrinsics: bool = False,
    flip_images: bool = False,
    demo_policy_images: h5py.Group | None = None,
    demo_camera_cache: h5py.Group | None = None,
    demo_step: int = 0,
) -> dict[str, Any]:
    images = {}
    for policy_key, obs_keys in CAM_KEYS.items():
        image = _maybe_obs_value(obs, obs_keys)
        demo_image = (
            _demo_image_for_policy(demo_policy_images, policy_key, demo_step, image_size)
            # Do not resurrect optional cameras from HDF5 if the live eval env
            # does not expose them. RoboCasa24 training pads right_wrist with
            # mask=False, while the raw HDF5 may still contain agentview_right.
            if demo_policy_images is not None and (policy_key != "agentview_right" or image is not None)
            else None
        )
        if demo_image is not None:
            images[policy_key] = demo_image
            continue

        if image is None:
            if policy_key == "agentview_right":
                continue
            raise KeyError(f"Missing required RoboCasa camera {policy_key}; available keys: {sorted(obs)}")
        if flip_images:
            image = np.asarray(image)[::-1, :, :]
        images[policy_key] = _resize_chw(np.asarray(image), image_size)

    policy_input = {
        "state": _obs_to_state(obs),
        "images": images,
        "prompt": _prompt_for_obs(obs, prompt, env=env),
    }
    if include_camera_matrices:
        matrices = _camera_matrices_for_policy(
            env,
            obs,
            image_size,
            send_opencv_extrinsics=send_opencv_extrinsics,
        )
        if demo_camera_cache is not None:
            matrices.update(
                _camera_matrices_from_demo_cache(
                    demo_camera_cache,
                    demo_step,
                    image_size,
                    send_opencv_extrinsics=send_opencv_extrinsics,
                )
            )
        if require_camera_matrices:
            required = [
                f"observation.{cam}_{kind}"
                for cam in ("agentview_left", "eye_in_hand")
                for kind in ("intrinsic", "extrinsic")
            ]
            missing = [key for key in required if key not in matrices]
            if missing:
                raise RuntimeError(f"Missing required camera matrices for cam-aware eval: {missing}")
        policy_input.update(matrices)
    return policy_input


def _obs_to_state(obs: dict[str, Any]) -> np.ndarray:
    if "observation.state" in obs:
        return np.asarray(obs["observation.state"], dtype=np.float32)
    if "observation/state" in obs:
        return np.asarray(obs["observation/state"], dtype=np.float32)

    if all(key in obs for key in V0_STATE_KEYS):
        return np.concatenate(
            [np.asarray(obs[key], dtype=np.float32).reshape(-1) for key in V0_STATE_KEYS],
            axis=0,
        ).astype(np.float32)

    try:
        parts = [np.asarray(obs[key], dtype=np.float32).reshape(-1) for key in STATE_KEYS]
    except KeyError as exc:
        raise KeyError(f"Missing RoboCasa state key {exc.args[0]!r}; available keys: {sorted(obs)}") from exc
    return np.concatenate(parts, axis=0).astype(np.float32)


def _inner_sim_env(env):
    candidates = []
    if env is not None:
        candidates.append(env)
        unwrapped = getattr(env, "unwrapped", None)
        if unwrapped is not None:
            candidates.append(unwrapped)
            inner = getattr(unwrapped, "env", None)
            if inner is not None:
                candidates.append(inner)
        inner = getattr(env, "env", None)
        if inner is not None:
            candidates.append(inner)

    for candidate in candidates:
        if hasattr(candidate, "sim"):
            return candidate
    return None


def _camera_id(model, name: str) -> int:
    if hasattr(model, "camera_name2id"):
        return int(model.camera_name2id(name))
    import mujoco

    raw_model = getattr(model, "_model", model)
    return int(mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_CAMERA, name))


def _camera_fovy(model, cam_id: int) -> float:
    if hasattr(model, "cam_fovy"):
        return float(model.cam_fovy[cam_id])
    return float(model._model.cam_fovy[cam_id])


def _camera_pose(sim, cam_id: int) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.asarray(sim.data.cam_xmat[cam_id], dtype=np.float32).reshape(3, 3)
    T[:3, 3] = np.asarray(sim.data.cam_xpos[cam_id], dtype=np.float32)
    return T


def _mujoco_to_opencv_extrinsic(T_wc_mj: np.ndarray) -> np.ndarray:
    """Convert MuJoCo/OpenGL camera-to-world to OpenCV camera-to-world."""
    T = np.asarray(T_wc_mj, dtype=np.float32).copy()
    T[:3, 1:3] *= -1.0
    return T


def _intrinsic_from_fovy(fovy_deg: float, src_h: int, src_w: int, dst_size: int) -> np.ndarray:
    focal = (float(src_h) / 2.0) / np.tan(np.deg2rad(float(fovy_deg)) / 2.0)
    K = np.array(
        [[focal, 0.0, float(src_w) / 2.0], [0.0, focal, float(src_h) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    K[0, 0] *= float(dst_size) / float(src_w)
    K[0, 2] *= float(dst_size) / float(src_w)
    K[1, 1] *= float(dst_size) / float(src_h)
    K[1, 2] *= float(dst_size) / float(src_h)
    return K


def _image_hw(image: np.ndarray) -> tuple[int, int]:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        return int(arr.shape[1]), int(arr.shape[2])
    return int(arr.shape[0]), int(arr.shape[1])


def _camera_matrices_for_policy(
    env,
    obs: dict[str, Any],
    image_size: int,
    *,
    send_opencv_extrinsics: bool,
) -> dict[str, np.ndarray]:
    sim_env = _inner_sim_env(env)
    if sim_env is None:
        return {}

    out = {}
    sim = sim_env.sim
    for policy_cam, obs_keys in CAM_KEYS.items():
        image = _maybe_obs_value(obs, obs_keys)
        if image is None:
            continue
        arr = np.asarray(image)
        if arr.ndim < 2:
            continue
        try:
            cam_id = _camera_id(sim.model, CAM_NAMES[policy_cam])
        except Exception:
            continue
        fovy = _camera_fovy(sim.model, cam_id)
        src_h, src_w = _image_hw(arr)
        out[f"observation.{policy_cam}_intrinsic"] = _intrinsic_from_fovy(
            fovy,
            src_h=src_h,
            src_w=src_w,
            dst_size=image_size,
        )
        T_wc = _camera_pose(sim, cam_id)
        if send_opencv_extrinsics:
            T_wc = _mujoco_to_opencv_extrinsic(T_wc)
        out[f"observation.{policy_cam}_extrinsic"] = T_wc
    return out


def _action_to_gym(action: np.ndarray, action_space) -> dict[str, np.ndarray]:
    action = np.asarray(action, dtype=np.float32)
    base_motion = action[0:4].copy()
    base_motion[np.abs(base_motion) < BASE_MOTION_DEADZONE] = 0.0
    action_dict = {
        "action.base_motion": base_motion,
        "action.control_mode": np.full((1,), -1.0, dtype=np.float32),
        "action.end_effector_position": action[5:8],
        "action.end_effector_rotation": action[8:11],
        "action.gripper_close": action[11:12],
    }

    clipped = {}
    for key, value in action_dict.items():
        space = action_space.spaces[key]
        arr = np.asarray(value, dtype=np.float32).reshape(space.shape)
        clipped[key] = np.clip(arr, space.low, space.high).astype(np.float32)
    return clipped


def _action_to_robocasa_v0(action: np.ndarray, env) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] < ROBOCASA_V0_ACTION_DIM:
        raise ValueError(f"Expected at least {ROBOCASA_V0_ACTION_DIM} action dims, got {action.shape}")

    base_motion = action[0:4].copy()
    base_motion[np.abs(base_motion) < BASE_MOTION_DEADZONE] = 0.0
    flat = np.concatenate(
        (
            action[5:8],   # eef_pos
            action[8:11],  # eef_rot
            action[11:12], # gripper
            base_motion,
            action[4:5],   # control_mode
        ),
        axis=0,
    ).astype(np.float32)

    if hasattr(env, "action_spec"):
        low, high = env.action_spec
        flat = np.clip(flat, np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32))
    return flat


def _reset_env(env, backend: str, seed: int) -> dict[str, Any]:
    if backend == "gym":
        obs, _ = env.reset(seed=seed)
        return obs

    np.random.seed(seed)
    seed_fn = getattr(env, "seed", None)
    if seed_fn is not None:
        try:
            seed_fn(seed)
        except TypeError:
            pass
    return env.reset()


def _observations_after_reset_to(env) -> dict[str, Any]:
    obs_fn = getattr(env, "_get_observations", None)
    if obs_fn is not None:
        return obs_fn()
    obs_fn = getattr(env, "get_observations", None)
    if obs_fn is not None:
        return obs_fn()
    raise RuntimeError("Unable to fetch observations after reset_to; env has no observation method.")


def _xml_array(value: Any) -> str:
    return " ".join(f"{float(x):.17g}" for x in np.asarray(value).reshape(-1))


def _find_xml_camera(root: ET.Element, name: str) -> ET.Element | None:
    for camera in root.iter("camera"):
        if camera.get("name") == name:
            return camera
    return None


def _find_xml_body(root: ET.Element, name: str | None) -> ET.Element | None:
    if not name:
        return None
    for candidate in (name, name.replace("base0_", "mobilebase0_"), name.replace("mobilebase0_", "base0_")):
        for body in root.iter("body"):
            if body.get("name") == candidate:
                return body
    return None


def _remove_xml_cameras(root: ET.Element, name: str) -> None:
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "camera" and child.get("name") == name:
                parent.remove(child)


def _apply_ep_meta_camera_configs_to_xml(xml: str, ep_meta: dict[str, Any]) -> str:
    cam_configs = ep_meta.get("cam_configs") or {}
    if not cam_configs:
        return xml

    root = ET.fromstring(xml)
    worldbody = root.find("worldbody")
    if worldbody is None:
        return xml

    for cam_name, cam_config in cam_configs.items():
        old_camera = _find_xml_camera(root, cam_name)
        old_attribs = dict(old_camera.attrib) if old_camera is not None else {}
        _remove_xml_cameras(root, cam_name)
        parent = _find_xml_body(worldbody, cam_config.get("parent_body")) or worldbody
        camera = ET.Element("camera")
        for key, value in old_attribs.items():
            if key not in {"name", "pos", "quat"}:
                camera.set(key, value)
        if camera.get("mode") is None:
            camera.set("mode", "fixed")
        camera.set("name", cam_name)
        parent.append(camera)

        camera.set("pos", _xml_array(cam_config["pos"]))
        camera.set("quat", _xml_array(cam_config["quat"]))
        for key, value in (cam_config.get("camera_attribs") or {}).items():
            camera.set(key, str(value))

    return ET.tostring(root, encoding="unicode")


def _demo_keys_from_arg(value: str, dataset_file: h5py.File) -> list[str]:
    if value:
        keys = [part.strip() for part in value.replace(":", ",").split(",") if part.strip()]
    else:
        keys = sorted(
            (key for key in dataset_file["data"].keys() if key.startswith("demo_")),
            key=lambda key: int(key.split("_")[1]),
        )
    missing = [key for key in keys if f"data/{key}" not in dataset_file]
    if missing:
        raise KeyError(f"Missing demo keys in {dataset_file.filename}: {missing}")
    return keys


def _make_demo_reset_env(dataset_path: Path, render_image_size: int):
    from robocasa.scripts.playback_dataset import get_env_metadata_from_dataset

    env_meta = get_env_metadata_from_dataset(str(dataset_path))
    env_kwargs = copy.deepcopy(env_meta["env_kwargs"])
    env_kwargs["env_name"] = env_meta["env_name"]
    # reset_to loads each demo's exact saved XML; don't require the external
    # generative-texture pack merely to construct the shell env.
    env_kwargs["generative_textures"] = None
    env_kwargs["camera_names"] = [
        "robot0_agentview_left",
        "robot0_eye_in_hand",
    ]
    env_kwargs["camera_widths"] = render_image_size
    env_kwargs["camera_heights"] = render_image_size
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = True
    env_kwargs["camera_depths"] = False
    env_kwargs["ignore_done"] = True
    return robosuite.make(**env_kwargs), env_meta


def _reset_to_demo(env, demo: h5py.Group) -> dict[str, Any]:
    ep_meta = json.loads(demo.attrs["ep_meta"])
    if hasattr(env, "set_attrs_from_ep_meta"):
        env.set_attrs_from_ep_meta(ep_meta)
    elif hasattr(env, "set_ep_meta"):
        env.set_ep_meta(ep_meta)

    env.reset()
    robosuite_version_id = int(robosuite.__version__.split(".")[1])
    if robosuite_version_id <= 3:
        from robosuite.utils.mjcf_utils import postprocess_model_xml

        xml = postprocess_model_xml(demo.attrs["model_file"])
    else:
        xml = env.edit_model_xml(demo.attrs["model_file"])
    xml = _apply_ep_meta_camera_configs_to_xml(xml, ep_meta)

    env.reset_from_xml_string(xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(np.asarray(demo["states"][0]))
    env.sim.forward()
    if hasattr(env, "update_sites"):
        env.update_sites()
    if hasattr(env, "update_state"):
        env.update_state()
    return _observations_after_reset_to(env)


def _step_env(env, backend: str, action: np.ndarray):
    if backend == "gym":
        return env.step(_action_to_gym(action, env.action_space))
    obs, reward, done, info = env.step(_action_to_robocasa_v0(action, env))
    return obs, reward, bool(done), False, info


def _apply_eval_overrides(
    action: np.ndarray,
    *,
    step: int,
    force_gripper_after: int | None,
    force_gripper_value: float,
) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).copy()
    if force_gripper_after is not None and step >= force_gripper_after:
        action[11] = np.float32(force_gripper_value)
    return action


def _maybe_success_value(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if not value:
            return None
        return bool(all(bool(v) for v in value.values()))
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value)
        if arr.size == 0:
            return None
        return bool(np.all(arr))
    return bool(value)


def _check_success(env, info: dict[str, Any]) -> bool:
    for key in ("success", "is_success", "task_success", "eval_success"):
        value = _maybe_success_value(info.get(key))
        if value is not None:
            return value

    for target in (
        env,
        getattr(env, "unwrapped", None),
        getattr(getattr(env, "unwrapped", None), "env", None),
        getattr(env, "env", None),
    ):
        if target is None:
            continue
        for method_name in ("_check_success", "check_success", "_check_successes"):
            method = getattr(target, method_name, None)
            if method is None:
                continue
            try:
                value = _maybe_success_value(method())
            except TypeError:
                continue
            if value is not None:
                return value
        for attr_name in ("success", "eval_success"):
            if hasattr(target, attr_name):
                value = _maybe_success_value(getattr(target, attr_name))
                if value is not None:
                    return value
    return False


def _task_debug_metrics(env) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    target = _inner_sim_env(env)
    if target is None:
        return metrics

    drawer = getattr(target, "drawer", None)
    if drawer is not None and hasattr(drawer, "get_door_state"):
        try:
            metrics["drawer_door_state"] = {
                str(k): float(v) for k, v in drawer.get_door_state(env=target).items()
            }
        except Exception as exc:  # pragma: no cover - debug best effort
            metrics["drawer_door_state_error"] = repr(exc)

    try:
        site_ids = target.robots[0].eef_site_id
        if isinstance(site_ids, dict) and "right" in site_ids:
            metrics["right_eef_site_pos"] = target.sim.data.site_xpos[site_ids["right"]].tolist()
    except Exception as exc:  # pragma: no cover - debug best effort
        metrics["right_eef_site_pos_error"] = repr(exc)

    sim = getattr(target, "sim", None)
    if drawer is not None and sim is not None:
        try:
            handle_pos = _find_named_object_position(sim, getattr(drawer, "handle_name", ""))
            if handle_pos is None:
                handle_pos = _find_drawer_handle_position(sim, getattr(drawer, "name", ""))
            if handle_pos is not None:
                metrics["drawer_handle_pos"] = handle_pos.tolist()
                eef_pos = metrics.get("right_eef_site_pos")
                if eef_pos is not None:
                    metrics["right_eef_to_handle_dist"] = float(
                        np.linalg.norm(np.asarray(eef_pos, dtype=np.float32) - handle_pos)
                    )
        except Exception as exc:  # pragma: no cover - debug best effort
            metrics["drawer_handle_pos_error"] = repr(exc)

    return metrics


def _model_names(model, kind: str) -> list[str]:
    names = getattr(model, f"{kind}_names", None)
    if names is None:
        return []
    return [name.decode() if isinstance(name, bytes) else str(name) for name in names]


def _find_named_object_position(sim, name: str) -> np.ndarray | None:
    if not name:
        return None
    model = sim.model
    for name_to_id, xpos in (
        (getattr(model, "site_name2id", None), getattr(sim.data, "site_xpos", None)),
        (getattr(model, "body_name2id", None), getattr(sim.data, "body_xpos", None)),
        (getattr(model, "geom_name2id", None), getattr(sim.data, "geom_xpos", None)),
    ):
        if name_to_id is None or xpos is None:
            continue
        try:
            idx = name_to_id(name)
        except Exception:
            continue
        return np.asarray(xpos[idx], dtype=np.float32)
    return None


def _find_drawer_handle_position(sim, drawer_name: str) -> np.ndarray | None:
    model = sim.model
    needles = [name.lower() for name in (drawer_name, "drawer") if name]
    candidates: list[np.ndarray] = []
    for kind, name_to_id, xpos in (
        ("site", getattr(model, "site_name2id", None), getattr(sim.data, "site_xpos", None)),
        ("body", getattr(model, "body_name2id", None), getattr(sim.data, "body_xpos", None)),
        ("geom", getattr(model, "geom_name2id", None), getattr(sim.data, "geom_xpos", None)),
    ):
        if name_to_id is None or xpos is None:
            continue
        for name in _model_names(model, kind):
            lower = name.lower()
            if "handle" not in lower:
                continue
            if needles and not any(needle in lower for needle in needles):
                continue
            try:
                candidates.append(np.asarray(xpos[name_to_id(name)], dtype=np.float32))
            except Exception:
                continue
    if not candidates:
        return None
    return np.mean(np.stack(candidates, axis=0), axis=0)


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--env", default="robocasa/OpenDrawer")
    parser.add_argument("--backend", choices=("gym", "robocasa-v0"), default="gym")
    parser.add_argument("--split", default="target")
    parser.add_argument("--prompt", default="auto")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--render-image-size",
        type=int,
        default=None,
        help="Simulator render size before policy resize. RoboCasa24 human demos were rendered at 128.",
    )
    parser.add_argument("--no-camera-matrices", action="store_true")
    parser.add_argument("--require-camera-matrices", action="store_true")
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--video-every", type=int, default=10)
    parser.add_argument("--debug-out", type=Path, default=None)
    parser.add_argument("--force-gripper-after", type=int, default=None)
    parser.add_argument("--force-gripper-value", type=float, default=1.0)
    parser.add_argument(
        "--demo-dataset",
        type=Path,
        default=None,
        help="Optional RoboCasa v0 HDF5. If set, each rollout resets to saved demo model/ep_meta/states[0].",
    )
    parser.add_argument(
        "--demo-keys",
        default="",
        help="Comma-separated demo keys for --demo-dataset. Defaults to all demos in numeric order.",
    )
    parser.add_argument(
        "--send-opencv-extrinsics",
        action="store_true",
        help=(
            "Send OpenCV-frame T_wc to the policy server. This matches legacy "
            "RoboCasa checkpoints trained with already-OpenCV LeRobot extrinsics "
            "and a server-side MuJoCo-to-OpenCV transform."
        ),
    )
    parser.add_argument(
        "--demo-policy-images",
        action="store_true",
        help="When --demo-dataset is set, feed source HDF5 demo images to the policy instead of live renders.",
    )
    parser.add_argument(
        "--demo-camera-cache",
        type=Path,
        default=None,
        help="Optional stage-1 cam matrix cache HDF5. Uses demos/<demo_key> K/T for policy camera inputs.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    if args.backend == "gym":
        _patch_eval_action_unmap()
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    render_image_size = args.render_image_size
    if render_image_size is None:
        render_image_size = 128 if args.backend == "robocasa-v0" else args.image_size
    demo_file = None
    demo_camera_cache_file = None
    demo_keys: list[str] = []
    if args.backend == "gym":
        if args.demo_dataset is not None:
            raise ValueError("--demo-dataset is only supported with --backend robocasa-v0")
        env = gym.make(args.env, enable_render=True, split=args.split)
    else:
        from robocasa.utils.env_utils import create_env

        _patch_robocasa_temp_mjcf_xml()
        task_name = args.env.removeprefix("robocasa/")
        if args.demo_dataset is not None:
            demo_file = h5py.File(args.demo_dataset, "r")
            demo_keys = _demo_keys_from_arg(args.demo_keys, demo_file)
            if args.demo_camera_cache is not None:
                demo_camera_cache_file = h5py.File(args.demo_camera_cache, "r")
            env, env_meta = _make_demo_reset_env(args.demo_dataset, render_image_size)
            if env_meta["env_name"] != task_name:
                logging.warning(
                    "--env %s differs from demo dataset env %s; using demo dataset env.",
                    task_name,
                    env_meta["env_name"],
                )
        else:
            env = create_env(
                env_name=task_name,
                robots="PandaOmron",
                camera_names=[
                    "robot0_agentview_left",
                    "robot0_eye_in_hand",
                ],
                camera_widths=render_image_size,
                camera_heights=render_image_size,
                seed=args.seed,
                render_onscreen=False,
                randomize_cameras=False,
            )

    episodes = []
    debug_records = []
    try:
        for ep in range(args.episodes):
            seed = args.seed + ep
            demo_key = None
            prompt_arg = args.prompt
            if demo_file is not None:
                demo_key = demo_keys[ep % len(demo_keys)]
                demo = demo_file[f"data/{demo_key}"]
                cache_demo = (
                    demo_camera_cache_file[f"demos/{demo_key}"]
                    if demo_camera_cache_file is not None and f"demos/{demo_key}" in demo_camera_cache_file
                    else None
                )
                obs = _reset_to_demo(env, demo)
                if args.prompt.lower() == "auto":
                    ep_meta = json.loads(demo.attrs["ep_meta"])
                    if ep_meta.get("lang"):
                        prompt_arg = str(ep_meta["lang"])
            else:
                cache_demo = None
                obs = _reset_env(env, args.backend, seed)
            episode_prompt = _prompt_for_obs(obs, prompt_arg, env=env)
            frames = []
            done = False
            success = False
            steps = 0
            policy_calls = 0
            total_reward = 0.0
            last_info: dict[str, Any] = {}

            save_video = args.video_dir is not None and args.video_every > 0 and (ep % args.video_every == 0)
            while not done and steps < args.max_steps:
                state_before = _obs_to_state(obs)
                task_debug_before = _task_debug_metrics(env)
                policy_input = _obs_to_policy_input(
                    obs,
                    prompt_arg,
                    args.image_size,
                    env=env,
                    include_camera_matrices=not args.no_camera_matrices,
                    require_camera_matrices=args.require_camera_matrices,
                    send_opencv_extrinsics=args.send_opencv_extrinsics,
                    flip_images=(args.backend == "robocasa-v0"),
                    demo_policy_images=demo if demo_file is not None and args.demo_policy_images else None,
                    demo_camera_cache=cache_demo,
                    demo_step=steps,
                )
                result = policy.infer(policy_input)
                actions = np.asarray(result["actions"], dtype=np.float32)[: args.chunk_size]
                executed_actions = []
                policy_calls += 1
                chunk_start_step = steps

                for action in actions:
                    executed_action = _apply_eval_overrides(
                        action,
                        step=steps,
                        force_gripper_after=args.force_gripper_after,
                        force_gripper_value=args.force_gripper_value,
                    )
                    executed_actions.append(executed_action)
                    if save_video:
                        frame = _maybe_obs_value(obs, CAM_KEYS["agentview_left"])
                        if frame is not None:
                            if args.backend == "robocasa-v0":
                                frame = np.asarray(frame)[::-1, :, :]
                            frames.append(np.asarray(frame))
                    obs, reward, terminated, truncated, info = _step_env(env, args.backend, executed_action)
                    steps += 1
                    total_reward += float(reward)
                    last_info = dict(info)
                    success = _check_success(env, last_info)
                    done = bool(terminated or truncated or success or steps >= args.max_steps)
                    if done:
                        break
                if args.debug_out is not None:
                    state_after = _obs_to_state(obs)
                    task_debug_after = _task_debug_metrics(env)
                    executed_actions_array = np.asarray(executed_actions, dtype=np.float32)
                    input_debug = {}
                    if demo_file is not None and chunk_start_step == 0:
                        input_debug = {
                            "policy_input_image_checks": _policy_input_image_checks(demo, policy_input),
                            "policy_input_camera_summary": _policy_input_camera_summary(policy_input),
                        }
                    debug_records.append(
                        {
                            "episode": ep,
                            "policy_call": policy_calls,
                            "chunk_start_step": chunk_start_step,
                            "chunk_end_step": steps,
                            "prompt": episode_prompt,
                            "backend": args.backend,
                            "demo_key": demo_key,
                            "sent_extrinsics": "opencv" if args.send_opencv_extrinsics else "mujoco_raw",
                            "policy_image_source": "hdf5_demo" if args.demo_policy_images else "env_render",
                            "policy_camera_source": "stage1_cache" if cache_demo is not None else "env_sim",
                            "state_before": state_before.tolist(),
                            "state_after": state_after.tolist(),
                            "state_delta": (state_after - state_before).tolist(),
                            "task_debug_before": task_debug_before,
                            "task_debug_after": task_debug_after,
                            "action_first": actions[0].tolist(),
                            "action_mean": actions.mean(axis=0).tolist(),
                            "action_std": actions.std(axis=0).tolist(),
                            "action_min": actions.min(axis=0).tolist(),
                            "action_max": actions.max(axis=0).tolist(),
                            "executed_action_first": executed_actions_array[0].tolist(),
                            "executed_action_mean": executed_actions_array.mean(axis=0).tolist(),
                            "executed_action_std": executed_actions_array.std(axis=0).tolist(),
                            "executed_action_min": executed_actions_array.min(axis=0).tolist(),
                            "executed_action_max": executed_actions_array.max(axis=0).tolist(),
                            "success_after_chunk": bool(success),
                            **input_debug,
                        }
                    )
                    args.debug_out.parent.mkdir(parents=True, exist_ok=True)
                    args.debug_out.write_text(json.dumps(debug_records, indent=2, sort_keys=True) + "\n")

            if save_video and frames:
                video_path = args.video_dir / f"episode_{ep:04d}_success_{int(success)}.mp4"
                try:
                    _write_video(video_path, frames, fps=20)
                except Exception:
                    logging.exception("Failed to write video %s; continuing rollout eval.", video_path)

            episode = {
                "episode": ep,
                "seed": seed,
                "demo_key": demo_key,
                "prompt": episode_prompt,
                        "sent_extrinsics": "opencv" if args.send_opencv_extrinsics else "mujoco_raw",
                        "policy_image_source": "hdf5_demo" if args.demo_policy_images else "env_render",
                        "policy_camera_source": "stage1_cache" if cache_demo is not None else "env_sim",
                        "success": bool(success),
                "steps": steps,
                "policy_calls": policy_calls,
                "total_reward": total_reward,
                "last_info_keys": sorted(last_info.keys()),
            }
            episodes.append(episode)
            success_rate = float(np.mean([e["success"] for e in episodes]))
            print(json.dumps({**episode, "running_success_rate": success_rate}), flush=True)

            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(
                    {
                        "env": args.env,
                        "backend": args.backend,
                        "split": args.split,
                        "prompt": args.prompt,
                        "demo_dataset": None if args.demo_dataset is None else str(args.demo_dataset),
                        "demo_keys": demo_keys,
                        "sent_extrinsics": "opencv" if args.send_opencv_extrinsics else "mujoco_raw",
                        "policy_image_source": "hdf5_demo" if args.demo_policy_images else "env_render",
                        "demo_camera_cache": None if args.demo_camera_cache is None else str(args.demo_camera_cache),
                        "episodes_requested": args.episodes,
                        "episodes_completed": len(episodes),
                        "success_rate": success_rate,
                        "episodes": episodes,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
    finally:
        if demo_file is not None:
            demo_file.close()
        if demo_camera_cache_file is not None:
            demo_camera_cache_file.close()
        close_fn = getattr(env, "close", None)
        if close_fn is not None:
            close_fn()


if __name__ == "__main__":
    main()

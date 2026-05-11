import numpy as np
import os
from pathlib import Path
import torch

from openpi_client import websocket_client_policy


CAMERA_MAP = {
    "cam_high": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}


def _homogeneous_extrinsic(extrinsic) -> np.ndarray:
    matrix = np.asarray(extrinsic, dtype=np.float32)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape == (3, 4):
        bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        return np.concatenate([matrix, bottom], axis=0)
    raise ValueError(f"Expected camera extrinsic shape (3, 4) or (4, 4), got {matrix.shape}")


def _camera_to_world_extrinsic(extrinsic_cv) -> np.ndarray:
    """Convert RoboTwin/Sapien OpenCV extrinsic_cv from world-to-camera to T_wc."""
    world_to_camera = _homogeneous_extrinsic(extrinsic_cv)
    camera_to_world = np.eye(4, dtype=np.float32)
    rotation = world_to_camera[:3, :3]
    translation = world_to_camera[:3, 3]
    camera_to_world[:3, :3] = rotation.T
    camera_to_world[:3, 3] = -rotation.T @ translation
    return camera_to_world


def _resize_image_to_square_chw(image, target_hw: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Match convert_robotwin_cam_to_lerobot.py's direct square resize."""
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB image shape (H, W, 3), got {arr.shape}")
    src_hw = (int(arr.shape[0]), int(arr.shape[1]))
    chw = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float() / 255.0
    if src_hw != (target_hw, target_hw):
        chw = torch.nn.functional.interpolate(
            chw[None],
            size=(target_hw, target_hw),
            mode="bilinear",
            align_corners=False,
        )[0]
    chw_uint8 = (chw * 255.0).clamp(0, 255).to(torch.uint8).numpy()
    return chw_uint8, src_hw


def _scale_intrinsic_to_square(K, src_hw: tuple[int, int], target_hw: int) -> np.ndarray:
    src_h, src_w = src_hw
    out = np.asarray(K, dtype=np.float32).copy()
    out[0, 0] *= float(target_hw) / float(src_w)
    out[0, 2] *= float(target_hw) / float(src_w)
    out[1, 1] *= float(target_hw) / float(src_h)
    out[1, 2] *= float(target_hw) / float(src_h)
    return out


class OpenPIRemotePolicy:
    def __init__(self, host: str, port: int, pi0_step: int = 50):
        self.policy = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.pi0_step = pi0_step
        self.image_size = int(os.environ.get("OPENPI_REMOTE_IMAGE_SIZE", "224"))
        self.instruction = None
        self._legacy_flip_images = os.environ.get("OPENPI_REMOTE_LEGACY_IMAGE_FLIP", "0") == "1"
        self._dump_dir = os.environ.get("OPENPI_REMOTE_DUMP_OBS_DIR")
        self._dump_limit = int(os.environ.get("OPENPI_REMOTE_DUMP_LIMIT", "0"))
        self._dump_count = 0

    def set_language(self, instruction: str) -> None:
        self.instruction = instruction

    def _dump_observation(self, policy_obs: dict) -> None:
        if not self._dump_dir or self._dump_count >= self._dump_limit:
            return
        try:
            from PIL import Image

            dump_dir = Path(self._dump_dir)
            dump_dir.mkdir(parents=True, exist_ok=True)
            idx = self._dump_count
            for cam_name, image in policy_obs["images"].items():
                arr = np.asarray(image)
                if np.issubdtype(arr.dtype, np.floating):
                    arr = (255 * arr).clip(0, 255).astype(np.uint8)
                if arr.ndim == 3 and arr.shape[0] == 3:
                    arr = np.transpose(arr, (1, 2, 0))
                Image.fromarray(arr.astype(np.uint8)).save(dump_dir / f"{idx:03d}_{cam_name}.png")
            np.savez_compressed(
                dump_dir / f"{idx:03d}_meta.npz",
                state=policy_obs["state"],
                prompt=np.asarray(policy_obs.get("prompt", ""), dtype=object),
                cam_high_intrinsic=policy_obs.get("observation.cam_high_intrinsic"),
                cam_left_wrist_intrinsic=policy_obs.get("observation.cam_left_wrist_intrinsic"),
                cam_right_wrist_intrinsic=policy_obs.get("observation.cam_right_wrist_intrinsic"),
                cam_high_extrinsic=policy_obs.get("observation.cam_high_extrinsic"),
                cam_left_wrist_extrinsic=policy_obs.get("observation.cam_left_wrist_extrinsic"),
                cam_right_wrist_extrinsic=policy_obs.get("observation.cam_right_wrist_extrinsic"),
            )
            self._dump_count += 1
        except Exception as exc:
            print(f"[openpi_remote] failed to dump observation: {exc}", flush=True)
            self._dump_count = self._dump_limit

    def _policy_image(self, image) -> np.ndarray:
        arr = np.asarray(image)
        if self._legacy_flip_images:
            arr = np.ascontiguousarray(arr[::-1, ::-1])
        return arr

    def make_observation(self, observation: dict) -> dict:
        obs = observation["observation"]
        state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)

        policy_obs = {
            "state": state,
            "images": {},
            "prompt": self.instruction,
        }

        for openpi_cam, robotwin_cam in CAMERA_MAP.items():
            cam_obs = obs[robotwin_cam]
            image, src_hw = _resize_image_to_square_chw(
                self._policy_image(cam_obs["rgb"]),
                self.image_size,
            )
            policy_obs["images"][robotwin_cam] = image
            policy_obs[f"observation.{openpi_cam}_intrinsic"] = np.asarray(
                _scale_intrinsic_to_square(cam_obs["intrinsic_cv"], src_hw, self.image_size),
                dtype=np.float32,
            )
            policy_obs[f"observation.{openpi_cam}_extrinsic"] = _camera_to_world_extrinsic(
                cam_obs["extrinsic_cv"]
            )

        self._dump_observation(policy_obs)
        return policy_obs

    def get_action_chunk(self, observation: dict) -> np.ndarray:
        if self.instruction is None:
            raise RuntimeError("Instruction must be set before inference.")
        result = self.policy.infer(self.make_observation(observation))
        actions = np.asarray(result["actions"], dtype=np.float32)[: self.pi0_step]
        if self._dump_dir and 0 < self._dump_count <= self._dump_limit:
            idx = self._dump_count - 1
            dump_dir = Path(self._dump_dir)
            np.save(dump_dir / f"{idx:03d}_actions.npy", actions)
            stats = {
                "shape": actions.shape,
                "min": float(np.min(actions)),
                "max": float(np.max(actions)),
                "mean": float(np.mean(actions)),
                "std": float(np.std(actions)),
                "first": actions[0].tolist(),
                "last": actions[-1].tolist(),
            }
            (dump_dir / f"{idx:03d}_actions.txt").write_text(str(stats), encoding="utf-8")
        return actions

    def reset(self) -> None:
        self.instruction = None


def get_model(usr_args):
    host = usr_args.get("server_host", "127.0.0.1")
    port = int(usr_args.get("server_port", usr_args.get("port", 8000)))
    pi0_step = int(usr_args.get("pi0_step", 50))
    return OpenPIRemotePolicy(host=host, port=port, pi0_step=pi0_step)


def eval(TASK_ENV, model, observation):
    if model.instruction is None:
        model.set_language(TASK_ENV.get_instruction())

    actions = model.get_action_chunk(observation)
    for action in actions:
        TASK_ENV.take_action(action)


def reset_model(model):
    model.reset()

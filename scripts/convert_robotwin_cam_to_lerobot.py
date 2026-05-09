"""Convert raw RoboTwin2.0 HDF5 (with depth + cam params) to a cam-aware LeRobot v3.

This is the cam-aware sibling of `convert_aloha_data_to_lerobot_robotwin_official.py`.
The main differences:
  - 14-D state and 14-D action are derived from `/joint_action/{left,right}_{arm,gripper}`
    matching the official converter's convention.
  - Per-frame extrinsics + intrinsics are written for each of the three cameras,
    matching the LIBERO cam-aware schema (`<cam>_extrinsic`, `<cam>_intrinsic`).
  - Images are square-resized to `--image-size` (default 224) before storage; K
    is scaled accordingly.

Expected raw layout (produced by `bash collect_data.sh <task> <task_config>` with
`data_type.depth: true` and `data_type.qpos: true` in the task config):

    <raw_dir>/data/episode<i>.hdf5
    <raw_dir>/instructions/episode<i>.json

Per-episode HDF5 keys we read:
    /joint_action/{left_arm, left_gripper, right_arm, right_gripper}
    /observation/<cam>/rgb               (T, H, W, 3) uint8
    /observation/<cam>/intrinsic_cv      (T, 3, 3) or (3, 3) float
    /observation/<cam>/extrinsic_cv      (T, 4, 4) or (4, 4) float
    /observation/<cam>/cam2world_gl      (T, 4, 4) optional, used for log/debug

Sapien `extrinsic_cv` is world->camera. We invert it once to write camera-to-world
(`T_wc`, OpenCV camera frame), so the dataset matches the LIBERO convention.

Output cam name mapping (RoboTwin -> openpi pi0 / aloha):
    head_camera   -> cam_high
    left_camera   -> cam_left_wrist
    right_camera  -> cam_right_wrist
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm

logger = logging.getLogger("convert_robotwin_cam_to_lerobot")

DEFAULT_CAM_MAP = {
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}


def _resize_image(img: np.ndarray, target_hw: int) -> np.ndarray:
    """Square-resize an HxWx3 uint8 image to target_hw x target_hw via torch bilinear.

    Applies a 180-degree (`[::-1, ::-1]`) flip first to match the openpi
    cam-aware convention used by `convert_libero_hdf5_to_lerobot._preprocess_image`.
    The matching `fx -> -fx` half of the flip is applied at training time by
    `RobotwinCamInputs._adjust_K_for_openpi_image_flip` and at Pi3X-cache time
    by `cache_pi3x_targets._adjust_K_openpi`. K stored in the parquet stays
    fx-positive (i.e. natural-orientation) — same as LIBERO.
    """
    flipped = np.ascontiguousarray(img[::-1, ::-1])
    chw = torch.from_numpy(flipped).permute(2, 0, 1).float() / 255.0
    chw = torch.nn.functional.interpolate(chw[None], size=(target_hw, target_hw), mode="bilinear", align_corners=False)[0]
    chw = (chw * 255.0).clamp(0, 255).to(torch.uint8).numpy()
    return chw


def _scale_K_to_square(K: np.ndarray, src_hw: tuple[int, int], target_hw: int) -> np.ndarray:
    src_h, src_w = src_hw
    out = np.asarray(K, dtype=np.float32).copy()
    out[0, 0] *= float(target_hw) / float(src_w)
    out[0, 2] *= float(target_hw) / float(src_w)
    out[1, 1] *= float(target_hw) / float(src_h)
    out[1, 2] *= float(target_hw) / float(src_h)
    return out


def _maybe_per_frame(arr: np.ndarray, frame_idx: int) -> np.ndarray:
    """Return arr[frame_idx] if T-leading else arr.

    Handles K (3,3), Tcw (3,4) or (4,4), per-frame stacks and static.
    """
    arr = np.asarray(arr)
    if arr.ndim >= 3 and arr.shape[1:] in {(3, 3), (3, 4), (4, 4)}:
        return arr[frame_idx]
    return arr


def _world_from_camera(extrinsic_cv: np.ndarray) -> np.ndarray:
    """Invert Sapien's world->camera extrinsic to get camera-to-world (T_wc).

    RoboTwin writes `extrinsic_cv` as a (3,4) matrix: [R | t] with R world->cam,
    t the cam-frame translation of the world origin. We invert to OpenCV camera-to-world.
    """
    T = np.asarray(extrinsic_cv, dtype=np.float32)
    if T.shape == (4, 4):
        R, t = T[:3, :3], T[:3, 3]
    elif T.shape == (3, 4):
        R, t = T[:3, :3], T[:3, 3]
    else:
        raise ValueError(f"unsupported extrinsic shape {T.shape}")
    Twc = np.eye(4, dtype=np.float32)
    Twc[:3, :3] = R.T
    Twc[:3, 3] = -R.T @ t
    return Twc


def _decode_rgb_seq(raw: np.ndarray) -> np.ndarray:
    """RoboTwin stores RGB as JPEG byte strings per frame; decode to (T, H, W, 3) uint8 RGB."""
    raw = np.asarray(raw)
    out = []
    for buf in raw:
        if isinstance(buf, bytes):
            arr = np.frombuffer(buf.rstrip(b"\x00"), dtype=np.uint8)
        else:
            arr = np.asarray(buf, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("cv2.imdecode returned None for an rgb frame")
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return np.stack(out, axis=0)


def _build_state_action(ep: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    """RoboTwin already writes `/joint_action/vector` as (T, 14)
    [left_arm(6) | left_gripper(1) | right_arm(6) | right_gripper(1)].
    Use it directly; action[t] = state[t+1]."""
    if "/joint_action/vector" in ep:
        state = np.asarray(ep["/joint_action/vector"][:], dtype=np.float32)
    else:
        la = ep["/joint_action/left_arm"][:]
        lg = ep["/joint_action/left_gripper"][:]
        ra = ep["/joint_action/right_arm"][:]
        rg = ep["/joint_action/right_gripper"][:]
        T = la.shape[0]
        state = np.empty((T, 14), dtype=np.float32)
        state[:, :6] = la
        state[:, 6] = lg
        state[:, 7:13] = ra
        state[:, 13] = rg
    actions = np.empty_like(state)
    actions[:-1] = state[1:]
    actions[-1] = state[-1]
    return state, actions


def _create_dataset(repo_id: str, image_hw: int, use_videos: bool = True) -> LeRobotDataset:
    motors = [
        "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll",
        "left_wrist_angle", "left_wrist_rotate", "left_gripper",
        "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll",
        "right_wrist_angle", "right_wrist_rotate", "right_gripper",
    ]
    cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

    features: dict[str, dict] = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": [motors]},
        "action": {"dtype": "float32", "shape": (14,), "names": [motors]},
    }
    image_dtype = "video" if use_videos else "image"
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": image_dtype,
            "shape": (3, image_hw, image_hw),
            "names": ["channels", "height", "width"],
        }
        features[f"observation.{cam}_extrinsic"] = {
            "dtype": "float32",
            "shape": (4, 4),
            "names": [["rows"], ["cols"]],
        }
        features[f"observation.{cam}_intrinsic"] = {
            "dtype": "float32",
            "shape": (3, 3),
            "names": [["rows"], ["cols"]],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=25,
        robot_type="aloha",
        features=features,
        use_videos=use_videos,
    )


def _iter_episodes(raw_dir: Path) -> list[Path]:
    return sorted((raw_dir / "data").glob("episode*.hdf5"))


def _load_instruction(raw_dir: Path, episode_index: int) -> str:
    instruction_path = raw_dir / "instructions" / f"episode{episode_index}.json"
    with instruction_path.open("r", encoding="utf-8") as f_instr:
        payload = json.load(f_instr)
    instructions = payload.get("instructions") or payload.get("seen") or []
    if not instructions:
        raise ValueError(f"No instructions found in {instruction_path}")
    return str(np.random.choice(instructions))


def _convert_episode(
    dataset: LeRobotDataset,
    ep_path: Path,
    instruction: str,
    *,
    image_hw: int,
    cam_map: dict[str, str],
):
    with h5py.File(ep_path, "r") as ep:
        state, actions = _build_state_action(ep)
        T = state.shape[0]

        # Pre-scan camera dims for K scaling (depth/rgb share H,W).
        cam_meta: dict[str, dict] = {}
        for sapien_cam, out_cam in cam_map.items():
            cam_grp = ep[f"/observation/{sapien_cam}"]
            rgb_seq = _decode_rgb_seq(cam_grp["rgb"][()])  # (T, H, W, 3) uint8
            K_full = cam_grp["intrinsic_cv"][()]
            E_full = cam_grp["extrinsic_cv"][()]  # world->cam (3,4)
            src_hw = (int(rgb_seq.shape[1]), int(rgb_seq.shape[2]))
            cam_meta[out_cam] = {
                "rgb": rgb_seq,
                "K": K_full,
                "E": E_full,
                "src_hw": src_hw,
            }

        for t in range(T):
            frame: dict = {
                "observation.state": torch.from_numpy(state[t]),
                "action": torch.from_numpy(actions[t]),
                "task": instruction,
            }
            for out_cam, meta in cam_meta.items():
                rgb = meta["rgb"][t]
                rgb_resized = _resize_image(rgb, image_hw)
                K = _maybe_per_frame(meta["K"], t)
                E = _maybe_per_frame(meta["E"], t)
                K_scaled = _scale_K_to_square(K, meta["src_hw"], image_hw)
                Twc = _world_from_camera(E)
                frame[f"observation.images.{out_cam}"] = rgb_resized
                frame[f"observation.{out_cam}_intrinsic"] = K_scaled.astype(np.float32)
                frame[f"observation.{out_cam}_extrinsic"] = Twc.astype(np.float32)
            dataset.add_frame(frame)
        dataset.save_episode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, help="RoboTwin/data/<task>/<task_config>")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--cam-map",
        default=",".join(f"{k}:{v}" for k, v in DEFAULT_CAM_MAP.items()),
    )
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--no-videos", action="store_true", help="Store frames as image PNG/JPEG instead of mp4 video.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    cam_map: dict[str, str] = {}
    for token in args.cam_map.split(","):
        sap, out = token.split(":", 1)
        cam_map[sap.strip()] = out.strip()

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    files = _iter_episodes(raw_dir)
    if args.max_episodes > 0:
        files = files[: args.max_episodes]
    if not files:
        raise FileNotFoundError(f"No episode HDF5 files under {raw_dir}/data")

    logger.info("converting %d episodes from %s to repo %s", len(files), raw_dir, args.repo_id)

    dataset = _create_dataset(args.repo_id, args.image_size, use_videos=not args.no_videos)
    for ep_idx, ep_path in enumerate(tqdm.tqdm(files)):
        try:
            episode_number = int("".join(c for c in ep_path.stem if c.isdigit()))
        except ValueError:
            episode_number = ep_idx
        instruction = _load_instruction(raw_dir, episode_number)
        _convert_episode(dataset, ep_path, instruction, image_hw=args.image_size, cam_map=cam_map)

    logger.info("converted to %s", HF_LEROBOT_HOME / args.repo_id)


if __name__ == "__main__":
    main()

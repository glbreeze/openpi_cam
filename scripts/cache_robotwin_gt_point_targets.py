"""Cache Sapien-GT dense point targets for RoboTwin2.0 point-head distillation.

This is the RoboTwin analog of `scripts/cache_libero_gt_point_targets.py`.
Output cache layout matches `MixedPointTargetLoader` / `DualPointTargetLoader`:

    {output_root}/{cam}/episode_{NNNNNN}.npz
        xy    : (T, R, R, 2)   pre-z-multiplication direction (matches PointHead)
        log_z : (T, R, R, 1)   log depth (meters)
        conf  : (T, R, R, 1)   conf logits (pre-sigmoid; ±10 from depth validity)

Preprocessing follows openpi's image pipeline:
  1. Read 480x640 (or whatever the raw H,W is) Sapien depth (mm float64) +
     intrinsic_cv (3,3) + extrinsic_cv (4,4) per frame from the raw RoboTwin
     HDF5 produced by `bash collect_data.sh ... <task_config_with_depth>`.
  2. Convert depth from mm -> m (Sapien's `get_depth()` writes `(-z) * 1000`).
  3. Apply `[::-1, ::-1]` flip on depth (matches `_preprocess_image`).
  4. Resize 480x640 -> target_resolution x target_resolution (nearest, square).
  5. Scale K by target_res / src_hw and apply `fx -> -fx` (openpi flip).
  6. Project pixel grid through the intrinsic to (x_dir, y_dir, z), pool to
     `output_resolution` if requested, write fp16 npz.

Cam name mapping (default):
  head_camera     -> cam_high
  left_camera     -> cam_left_wrist
  right_camera    -> cam_right_wrist

Usage:
    python scripts/cache_robotwin_gt_point_targets.py \\
        --raw-dir /path/to/RoboTwin/data/<task>/<task_config> \\
        --output-root ~/.cache/openpi/gt_point_targets_224/<repo_id> \\
        --output-resolution 224
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("cache_robotwin_gt_point_targets")


DEFAULT_CAM_MAP = {
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}


def _adjust_and_scale_k(K: np.ndarray, src_hw: tuple[int, int], target_hw: int) -> np.ndarray:
    """Scale a Sapien-OpenCV K to a target square resolution.

    Sapien depth is rendered at `src_hw = (H, W)` (e.g. 240x320) — we square-resize
    to `target_hw` and scale K accordingly. **No** fx-negation: the converter and
    eval client both keep images in natural Sapien (OpenCV y-down) orientation.
    """
    src_h, src_w = src_hw
    sx = float(target_hw) / float(src_w)
    sy = float(target_hw) / float(src_h)
    out = np.asarray(K, dtype=np.float32).copy()
    out[0, 0] = out[0, 0] * sx
    out[0, 2] = out[0, 2] * sx
    out[1, 1] = out[1, 1] * sy
    out[1, 2] = out[1, 2] * sy
    return out


def _resize_depth(depth: np.ndarray, target_hw: int) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected depth (H, W), got {depth.shape}")
    depth_t = torch.from_numpy(depth[None, None])
    if depth_t.shape[-1] != target_hw or depth_t.shape[-2] != target_hw:
        depth_t = F.interpolate(depth_t, size=(target_hw, target_hw), mode="nearest")
    return depth_t[0, 0].numpy()


def _depth_to_targets(depth: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    xy = np.stack([(u - cx) / fx, (v - cy) / fy], axis=-1).astype(np.float32)
    valid = np.isfinite(depth) & (depth > 1e-5)
    safe_depth = np.where(valid, depth, 1.0).astype(np.float32)
    log_z = np.log(safe_depth)[..., None].astype(np.float32)
    conf = np.where(valid[..., None], 10.0, -10.0).astype(np.float32)
    return xy, log_z, conf


def _pool_targets(
    xy: np.ndarray, log_z: np.ndarray, conf: np.ndarray, output_resolution: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if output_resolution == xy.shape[0]:
        return xy, log_z, conf
    if xy.shape[0] % output_resolution != 0:
        raise ValueError(f"Cannot pool {xy.shape[0]} -> {output_resolution}")
    k = xy.shape[0] // output_resolution

    def pool(arr):
        t = torch.from_numpy(arr).permute(2, 0, 1)[None]
        return F.avg_pool2d(t, kernel_size=k, stride=k)[0].permute(1, 2, 0).numpy()

    return pool(xy), pool(log_z), pool(conf)


def _process_episode_cam(
    depth_seq: np.ndarray,
    K: np.ndarray,
    *,
    target_resolution: int,
    output_resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """depth_seq: (T, H, W) in mm. K: (3, 3) OpenCV intrinsic at (H, W).
    Returns xy, log_z, conf as fp16 (T, R, R, *) with R = output_resolution."""
    if depth_seq.ndim == 4 and depth_seq.shape[-1] == 1:
        depth_seq = depth_seq[..., 0]
    if depth_seq.ndim != 3:
        raise ValueError(f"Expected depth (T, H, W), got {depth_seq.shape}")

    src_hw = (int(depth_seq.shape[1]), int(depth_seq.shape[2]))
    K_scaled = _adjust_and_scale_k(K, src_hw, target_resolution)

    xy_list, log_z_list, conf_list = [], [], []
    for depth_mm in depth_seq:
        depth_m = depth_mm.astype(np.float32) / 1000.0  # Sapien `get_depth` => mm
        # No [::-1, ::-1] flip: depth stays in natural Sapien (y-down) orientation
        # to match the converter's parquet image and the eval client's input.
        depth_m = _resize_depth(depth_m, target_resolution)
        xy, log_z, conf = _depth_to_targets(depth_m, K_scaled)
        xy, log_z, conf = _pool_targets(xy, log_z, conf, output_resolution)
        xy_list.append(xy.astype(np.float16))
        log_z_list.append(log_z.astype(np.float16))
        conf_list.append(conf.astype(np.float16))

    return (
        np.stack(xy_list, axis=0),
        np.stack(log_z_list, axis=0),
        np.stack(conf_list, axis=0),
    )


def _read_per_frame_K(obs_cam: h5py.Group) -> np.ndarray:
    """RoboTwin's cameras.get_config() writes intrinsic_cv per frame. Episodes are
    static cam intrinsics, so the per-frame stack should be constant. We use frame 0."""
    K = obs_cam["intrinsic_cv"][()]
    if K.ndim == 3:
        K = K[0]
    return np.asarray(K, dtype=np.float32)


def _iter_episode_files(raw_dir: Path) -> list[Path]:
    files = sorted((raw_dir / "data").glob("episode*.hdf5"))
    if not files:
        files = sorted(raw_dir.rglob("episode*.hdf5"))
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, help="RoboTwin/data/<task>/<task_config>")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--target-resolution", type=int, default=224)
    parser.add_argument("--output-resolution", type=int, default=224, choices=(16, 224))
    parser.add_argument(
        "--cam-map",
        default=",".join(f"{k}:{v}" for k, v in DEFAULT_CAM_MAP.items()),
        help="Comma-separated `<sapien_cam>:<out_cam>` pairs; defaults match the RoboTwin->openpi convention.",
    )
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--start-episode-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    cam_map: dict[str, str] = {}
    for token in args.cam_map.split(","):
        sap, out = token.split(":", 1)
        cam_map[sap.strip()] = out.strip()

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    files = _iter_episode_files(raw_dir)
    if args.max_episodes > 0:
        files = files[: args.max_episodes]
    if not files:
        raise FileNotFoundError(f"No episode HDF5 files found under {raw_dir}")
    logger.info("found %d episodes under %s", len(files), raw_dir)

    for offset, path in enumerate(files):
        ep_idx = int(args.start_episode_index) + offset
        out_paths = {out_cam: output_root / out_cam / f"episode_{ep_idx:06d}.npz" for out_cam in cam_map.values()}
        if args.resume and all(p.exists() for p in out_paths.values()):
            logger.info("[skip] episode_%06d already cached", ep_idx)
            continue
        for p in out_paths.values():
            p.parent.mkdir(parents=True, exist_ok=True)

        logger.info("processing %s -> episode_%06d", path.name, ep_idx)
        with h5py.File(path, "r") as f:
            obs = f["/observation"]
            for sapien_cam, out_cam in cam_map.items():
                if sapien_cam not in obs:
                    raise KeyError(f"{path}: missing /observation/{sapien_cam}")
                cam_grp = obs[sapien_cam]
                if "depth" not in cam_grp:
                    raise KeyError(
                        f"{path}: /observation/{sapien_cam}/depth missing -- re-collect with data_type.depth=true"
                    )
                depth_seq = cam_grp["depth"][()]
                K = _read_per_frame_K(cam_grp)
                xy, log_z, conf = _process_episode_cam(
                    depth_seq,
                    K,
                    target_resolution=args.target_resolution,
                    output_resolution=args.output_resolution,
                )
                np.savez_compressed(out_paths[out_cam], xy=xy, log_z=log_z, conf=conf)


if __name__ == "__main__":
    main()

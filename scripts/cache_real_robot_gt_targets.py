"""Cache UR5 real-robot ground-truth depth as openpi distillation targets.

This script converts the extracted raw RealSense depth sidecar into the same
per-episode `.npz` target contract that openpi's auxiliary point-head loss
already consumes:

    xy    : (T, R, R, 2)  camera-frame normalized rays before z-multiplication
    log_z : (T, R, R, 1)  log depth in meters
    conf  : (T, R, R, 1)  confidence logits (+10 valid, -10 invalid)

For the current UR5 real-robot distillation configs, set `R=224` so the cache
matches `aux_point_head.output_resolution=224` without changing model code.

Input raw sidecar layout:
    {dataset_root}/pi3_scene_capture/depth_raw/chunk-XXX/
        observation.images.context_left_depth/episode_000000.npz
        observation.images.wrist_right_depth/episode_000000.npz

Each input `.npz` contains:
    depth      : (T, 480, 640) uint16 millimeters
    valid_mask : (T,) bool      frame-level sync validity

Output layout (chosen to match the existing UR5 loader/config wiring):
    {output_root}/agent/episode_000000.npz
    {output_root}/right_wrist/episode_000000.npz

This lets the current stage1/stage2 real-robot distillation configs reuse the
GT cache via `pi3x_targets_root` without modifying the model.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

logger = logging.getLogger("cache_real_robot_gt_targets")

CAMERA_SPECS = (
    {
        "depth_dir": "observation.images.context_left_depth",
        "intrinsics_key": "context_left",
        "output_subdir": "agent",
    },
    {
        "depth_dir": "observation.images.wrist_right_depth",
        "intrinsics_key": "wrist_right",
        "output_subdir": "right_wrist",
    },
)


def _load_intrinsics(intrinsics_yaml: Path) -> dict[str, np.ndarray]:
    with intrinsics_yaml.open() as f:
        payload = yaml.safe_load(f)

    out: dict[str, np.ndarray] = {}
    for cam_name, cam_data in payload["cameras"].items():
        K = np.array(
            [
                [cam_data["fx"], 0.0, cam_data["cx"]],
                [0.0, cam_data["fy"], cam_data["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        out[cam_name] = K
    return out


def _scaled_intrinsics(K: np.ndarray, src_hw: tuple[int, int], target_hw: int) -> np.ndarray:
    src_h, src_w = src_hw
    scale_x = target_hw / src_w
    scale_y = target_hw / src_h

    K_out = K.copy()
    K_out[0, 0] *= scale_x
    K_out[0, 2] *= scale_x
    K_out[1, 1] *= scale_y
    K_out[1, 2] *= scale_y
    return K_out


def _compute_xy_grid(K_scaled: np.ndarray, target_hw: int, device: torch.device) -> torch.Tensor:
    fx = float(K_scaled[0, 0])
    fy = float(K_scaled[1, 1])
    cx = float(K_scaled[0, 2])
    cy = float(K_scaled[1, 2])

    u = torch.arange(target_hw, dtype=torch.float32, device=device)
    v = torch.arange(target_hw, dtype=torch.float32, device=device)
    grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")

    x = (grid_u - cx) / fx
    y = (grid_v - cy) / fy
    return torch.stack([x, y], dim=-1)  # (R, R, 2)


def _load_depth_episode(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path) as f:
        depth_mm = np.asarray(f["depth"], dtype=np.uint16)
        valid_mask = np.asarray(f["valid_mask"], dtype=bool)
    return depth_mm, valid_mask


def _process_episode(
    depth_npz: Path,
    out_path: Path,
    xy_grid: torch.Tensor,
    *,
    target_hw: int,
    batch_size: int,
    device: torch.device,
) -> None:
    depth_mm, valid_mask = _load_depth_episode(depth_npz)
    T, src_h, src_w = depth_mm.shape
    if (src_h, src_w) != (480, 640):
        raise RuntimeError(f"{depth_npz}: expected depth shape (*, 480, 640), got {depth_mm.shape}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    xy_buf = np.empty((T, target_hw, target_hw, 2), dtype=np.float16)
    logz_buf = np.empty((T, target_hw, target_hw, 1), dtype=np.float16)
    conf_buf = np.empty((T, target_hw, target_hw, 1), dtype=np.float16)

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        depth_chunk = torch.from_numpy(depth_mm[start:end].astype(np.float32)).to(device) / 1000.0
        frame_valid = torch.from_numpy(valid_mask[start:end]).to(device=device, dtype=torch.bool)

        # Per-pixel validity: synced frame and non-zero sensor return.
        valid_px = (depth_chunk > 0.0) & frame_valid[:, None, None]

        depth_resized = F.interpolate(depth_chunk[:, None], size=(target_hw, target_hw), mode="nearest").squeeze(1)
        valid_resized = F.interpolate(
            valid_px[:, None].to(torch.float32),
            size=(target_hw, target_hw),
            mode="nearest",
        ).squeeze(1) > 0.5

        safe_depth = torch.where(valid_resized, depth_resized, torch.ones_like(depth_resized))
        log_z = torch.log(safe_depth)
        conf = torch.where(valid_resized, torch.full_like(safe_depth, 10.0), torch.full_like(safe_depth, -10.0))

        xy = xy_grid.expand(end - start, -1, -1, -1)

        xy_buf[start:end] = xy.cpu().numpy().astype(np.float16)
        logz_buf[start:end, ..., 0] = log_z.cpu().numpy().astype(np.float16)
        conf_buf[start:end, ..., 0] = conf.cpu().numpy().astype(np.float16)

    np.savez_compressed(out_path, xy=xy_buf, log_z=logz_buf, conf=conf_buf)


def _select_episodes(paths: list[Path], episode_range: str | None) -> list[Path]:
    if not episode_range:
        return paths
    if ":" in episode_range:
        lo, hi = episode_range.split(":")
        lo_i = int(lo) if lo else 0
        hi_i = int(hi) if hi else len(paths)
        return paths[lo_i:hi_i]

    wanted = {int(x) for x in episode_range.split(",") if x.strip()}
    out = []
    for p in paths:
        idx = int(p.stem.split("_")[-1])
        if idx in wanted:
            out.append(p)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/scratch/yz11445/real_robot_data/ur5_lab_test_tube_camera_shifts"),
        help="Local UR5 dataset root containing pi3_scene_capture/depth_raw and meta/info.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/scratch/yz11445/openpi_cache/pi3x_targets_224/ur5_lab_test_tube_camera_shifts"),
        help="Root for the cached GT targets. Subdirs 'agent' and 'right_wrist' are created here.",
    )
    parser.add_argument("--target-hw", type=int, default=224, choices=[224], help="Output spatial resolution.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--episode-range",
        type=str,
        default=None,
        help="Either 'lo:hi' python slice or 'i,j,k' explicit ids. Default: all episodes.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Recompute episodes even if the output file already exists.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    depth_root = dataset_root / "pi3_scene_capture" / "depth_raw"
    intrinsics_yaml = dataset_root / "pi3_scene_capture" / "intrinsics.yaml"
    info_json = dataset_root / "meta" / "info.json"

    if not depth_root.exists():
        raise SystemExit(f"Depth root not found: {depth_root}")
    if not intrinsics_yaml.exists():
        raise SystemExit(f"Intrinsics yaml not found: {intrinsics_yaml}")
    if not info_json.exists():
        raise SystemExit(f"Dataset info not found: {info_json}")

    with info_json.open() as f:
        dataset_info = json.load(f)
    chunk_size = int(dataset_info["chunks_size"])

    intrinsics = _load_intrinsics(intrinsics_yaml)
    device = torch.device(args.device)

    xy_grids: dict[str, torch.Tensor] = {}
    for spec in CAMERA_SPECS:
        K_scaled = _scaled_intrinsics(intrinsics[spec["intrinsics_key"]], src_hw=(480, 640), target_hw=args.target_hw)
        xy_grids[spec["output_subdir"]] = _compute_xy_grid(K_scaled, args.target_hw, device)

    # Episode selection is driven by the context_left view, then mirrored to wrist_right.
    context_left_dir = depth_root / "chunk-000" / "observation.images.context_left_depth"
    episode_paths = sorted(context_left_dir.glob("episode_*.npz"))
    episode_paths = _select_episodes(episode_paths, args.episode_range)
    if not episode_paths:
        raise SystemExit(f"No episodes found under {context_left_dir}")

    start_time = time.time()
    processed = 0
    for context_left_npz in episode_paths:
        episode_idx = int(context_left_npz.stem.split("_")[-1])
        chunk_dir = depth_root / f"chunk-{episode_idx // chunk_size:03d}"

        for spec in CAMERA_SPECS:
            depth_npz = chunk_dir / spec["depth_dir"] / f"episode_{episode_idx:06d}.npz"
            out_path = output_root / spec["output_subdir"] / f"episode_{episode_idx:06d}.npz"
            if out_path.exists() and not args.no_skip_existing:
                continue

            logger.info("episode=%06d view=%s -> %s", episode_idx, spec["depth_dir"], out_path)
            _process_episode(
                depth_npz,
                out_path,
                xy_grids[spec["output_subdir"]],
                target_hw=args.target_hw,
                batch_size=args.batch_size,
                device=device,
            )
        processed += 1

    elapsed = time.time() - start_time
    logger.info("Done. processed=%d episodes elapsed=%.1fs output_root=%s", processed, elapsed, output_root)


if __name__ == "__main__":
    main()

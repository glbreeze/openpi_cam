#!/usr/bin/env python3
"""Rebuild LeRobot depth MP4s from raw sidecar depth arrays.

This restores the local `videos/..._depth` tree expected by LeRobot dataset
metadata while keeping the GT cache independent under `openpi_cache`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


DEFAULT_VIEWS = (
    "context_front_depth",
    "context_left_depth",
    "context_top_depth",
    "wrist_right_depth",
)


@dataclass(frozen=True)
class EncodeTask:
    raw_npz: Path
    output_mp4: Path
    fps: int
    max_depth_m: float
    overwrite: bool


def _episode_sort_key(path: Path) -> int:
    stem = path.stem
    if stem.startswith("episode_"):
        return int(stem.split("_", 1)[1])
    return -1


def _depth_frame_to_rgb(depth_frame_mm: np.ndarray, valid: bool, max_depth_m: float) -> np.ndarray:
    if not valid:
        gray = np.zeros(depth_frame_mm.shape, dtype=np.uint8)
    else:
        depth_m = depth_frame_mm.astype(np.float32) / 1000.0
        gray = np.clip(depth_m / max_depth_m, 0.0, 1.0)
        gray = (gray * 255.0 + 0.5).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def _encode_one(task: EncodeTask) -> str:
    if task.output_mp4.exists() and not task.overwrite:
        return f"skip {task.output_mp4}"

    task.output_mp4.parent.mkdir(parents=True, exist_ok=True)
    with np.load(task.raw_npz) as data:
        depth = data["depth"]
        valid_mask = data["valid_mask"] if "valid_mask" in data else None
        writer = imageio.get_writer(
            task.output_mp4,
            fps=task.fps,
            codec="libx264",
            macro_block_size=None,
        )
        try:
            for idx, depth_frame in enumerate(depth):
                valid = True if valid_mask is None else bool(valid_mask[idx])
                writer.append_data(_depth_frame_to_rgb(depth_frame, valid, task.max_depth_m))
        finally:
            writer.close()
    return f"wrote {task.output_mp4}"


def _parse_episode_range(text: str | None) -> tuple[int, int] | None:
    if text is None:
        return None
    start_str, end_str = text.split(":", 1)
    return int(start_str), int(end_str)


def _iter_tasks(
    dataset_root: Path,
    videos_root: Path,
    views: tuple[str, ...],
    fps: int,
    max_depth_m: float,
    overwrite: bool,
    episode_range: tuple[int, int] | None,
) -> list[EncodeTask]:
    depth_root = dataset_root / "pi3_scene_capture" / "depth_raw" / "chunk-000"
    start, end = episode_range if episode_range is not None else (None, None)
    tasks: list[EncodeTask] = []
    for view in views:
        raw_dir = depth_root / f"observation.images.{view}"
        if not raw_dir.is_dir():
            raise FileNotFoundError(f"Missing raw depth directory: {raw_dir}")
        out_dir = videos_root / f"observation.images.{view}"
        for raw_npz in sorted(raw_dir.glob("episode_*.npz"), key=_episode_sort_key):
            ep_idx = _episode_sort_key(raw_npz)
            if start is not None and ep_idx < start:
                continue
            if end is not None and ep_idx >= end:
                continue
            tasks.append(
                EncodeTask(
                    raw_npz=raw_npz,
                    output_mp4=out_dir / f"{raw_npz.stem}.mp4",
                    fps=fps,
                    max_depth_m=max_depth_m,
                    overwrite=overwrite,
                )
            )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/scratch/yz11445/real_robot_data/ur5_lab_test_tube_camera_shifts"),
    )
    parser.add_argument(
        "--videos-root",
        type=Path,
        default=None,
        help="Override output videos root. Defaults to <dataset-root>/videos/chunk-000",
    )
    parser.add_argument("--views", nargs="+", default=list(DEFAULT_VIEWS), choices=list(DEFAULT_VIEWS))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-depth-m", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--episode-range", type=str, default=None, help="Half-open range, e.g. 0:2")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")

    episode_range = _parse_episode_range(args.episode_range)
    views = tuple(args.views)
    videos_root = args.videos_root or (args.dataset_root / "videos" / "chunk-000")
    tasks = _iter_tasks(
        args.dataset_root,
        videos_root,
        views,
        args.fps,
        args.max_depth_m,
        args.overwrite,
        episode_range,
    )
    logging.info("dataset root: %s", args.dataset_root)
    logging.info("videos root: %s", videos_root)
    logging.info("views: %s", views)
    logging.info("tasks: %d", len(tasks))
    if not tasks:
        return

    completed = 0
    if args.workers <= 1:
        for task in tasks:
            completed += 1
            msg = _encode_one(task)
            if completed <= 5 or completed % 25 == 0 or completed == len(tasks):
                logging.info("[%d/%d] %s", completed, len(tasks), msg)
        return

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_encode_one, task) for task in tasks]
        for fut in concurrent.futures.as_completed(futures):
            completed += 1
            msg = fut.result()
            if completed <= 5 or completed % 25 == 0 or completed == len(tasks):
                logging.info("[%d/%d] %s", completed, len(tasks), msg)


if __name__ == "__main__":
    main()

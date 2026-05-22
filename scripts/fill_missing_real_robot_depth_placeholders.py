#!/usr/bin/env python3
"""Fill missing LeRobot depth MP4 paths with a valid placeholder video.

This is enough to satisfy the LeRobot dataset file-existence assertion for
training paths that only decode RGB streams. GT supervision remains separate.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


DEPTH_VIEWS = (
    "context_front_depth",
    "context_left_depth",
    "context_top_depth",
    "wrist_right_depth",
)


def _ensure_placeholder(path: Path, fps: int) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=None)
    try:
        for _ in range(8):
            writer.append_data(frame)
    finally:
        writer.close()
    return path


def _episode_paths(meta_dir: Path) -> list[str]:
    episodes_jsonl = meta_dir / "episodes.jsonl"
    names: list[str] = []
    with episodes_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            episode_index = int(json.loads(line)["episode_index"])
            names.append(f"episode_{episode_index:06d}.mp4")
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/scratch/yz11445/real_robot_data/ur5_lab_test_tube_camera_shifts"),
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")

    meta_dir = args.dataset_root / "meta"
    videos_root = args.dataset_root / "videos" / "chunk-000"
    placeholder = _ensure_placeholder(Path("/scratch/yz11445/tmp/ur5_depth_placeholder.mp4"), args.fps)
    episode_names = _episode_paths(meta_dir)

    created = 0
    for view in DEPTH_VIEWS:
        view_dir = videos_root / f"observation.images.{view}"
        view_dir.mkdir(parents=True, exist_ok=True)
        for episode_name in episode_names:
            dst = view_dir / episode_name
            if dst.exists():
                continue
            shutil.copy2(placeholder, dst)
            created += 1
        logging.info("%s complete: %d/%d present", view, len(list(view_dir.glob("episode_*.mp4"))), len(episode_names))

    logging.info("created placeholders: %d", created)


if __name__ == "__main__":
    main()

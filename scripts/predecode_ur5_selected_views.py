#!/usr/bin/env python3
"""Predecode the two UR5 RGB training views into a local PNG frame cache.

This preserves the decoded frame content while removing repeated runtime video
decode from training. It does not change episode order, prompts, actions, or
Pi3X targets. It only materializes the exact decoded RGB frames once.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import pathlib
from typing import Iterable

import numpy as np
from PIL import Image
import torchvision.io


DEFAULT_INPUT_ROOT = pathlib.Path("/scratch/yz11445/real_robot_data/ur5_lab_test_tube_camera_shifts/videos")
DEFAULT_OUTPUT_ROOT = pathlib.Path("/scratch/yz11445/real_robot_data/ur5_lab_test_tube_camera_shifts_predecoded")
DEFAULT_VIEWS = (
    "observation.images.context_left_rgb",
    "observation.images.wrist_right_rgb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=pathlib.Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--view", action="append", dest="views", help="Video key to predecode. Repeatable.")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild already-decoded episodes instead of skipping them.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit episode indices to decode, e.g. --episodes 0 1 2",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log progress after every N completed videos.",
    )
    return parser.parse_args()


def _iter_video_files(input_root: pathlib.Path, views: Iterable[str], episodes: set[int] | None) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for view in views:
        view_dir = input_root / "chunk-000" / view
        if not view_dir.is_dir():
            raise FileNotFoundError(f"Missing input view directory: {view_dir}")
        for video_path in sorted(view_dir.glob("episode_*.mp4")):
            if episodes is not None:
                episode_idx = int(video_path.stem.split("_")[-1])
                if episode_idx not in episodes:
                    continue
            files.append(video_path)
    return files


def _episode_output_dir(output_root: pathlib.Path, input_root: pathlib.Path, video_path: pathlib.Path) -> pathlib.Path:
    rel = video_path.relative_to(input_root)
    return (output_root / rel).with_suffix("")


def _is_complete(output_dir: pathlib.Path) -> bool:
    meta_path = output_dir / "meta.json"
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return False
    expected = int(meta.get("num_frames", -1))
    if expected < 0:
        return False
    actual = len(list(output_dir.glob("*.png")))
    return actual == expected


def _decode_one(job: tuple[str, str, str, bool]) -> tuple[str, int, float]:
    video_path_str, input_root_str, output_root_str, overwrite = job
    video_path = pathlib.Path(video_path_str)
    input_root = pathlib.Path(input_root_str)
    output_root = pathlib.Path(output_root_str)
    output_dir = _episode_output_dir(output_root, input_root, video_path)

    if not overwrite and _is_complete(output_dir):
        meta = json.loads((output_dir / "meta.json").read_text())
        return str(video_path), int(meta["num_frames"]), float(meta.get("fps", 0.0))

    output_dir.mkdir(parents=True, exist_ok=True)

    frames, _, info = torchvision.io.read_video(str(video_path), pts_unit="sec", output_format="TCHW")
    fps = float(info.get("video_fps", 0.0))
    num_frames = int(frames.shape[0])

    for idx, frame in enumerate(frames):
        frame_np = frame.permute(1, 2, 0).contiguous().cpu().numpy()
        Image.fromarray(np.asarray(frame_np, dtype=np.uint8), mode="RGB").save(output_dir / f"{idx:06d}.png")

    meta = {
        "source_video": str(video_path),
        "num_frames": num_frames,
        "fps": fps,
        "height": int(frames.shape[2]) if num_frames else 0,
        "width": int(frames.shape[3]) if num_frames else 0,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    return str(video_path), num_frames, fps


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    views = tuple(args.views) if args.views else DEFAULT_VIEWS
    episodes = set(args.episodes) if args.episodes is not None else None

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    video_files = _iter_video_files(input_root, views, episodes)
    if not video_files:
        raise FileNotFoundError("No matching UR5 video files found to predecode.")

    jobs = [(str(path), str(input_root), str(output_root), args.overwrite) for path in video_files]
    logging.info(
        "Predecoding %d videos from %s to %s with views=%s workers=%d overwrite=%s",
        len(video_files),
        input_root,
        output_root,
        views,
        args.workers,
        args.overwrite,
    )

    completed = 0
    total_frames = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_decode_one, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            video_path, num_frames, fps = future.result()
            completed += 1
            total_frames += num_frames
            if completed == 1 or completed % args.log_every == 0 or completed == len(jobs):
                logging.info(
                    "Completed %d/%d videos | last=%s frames=%d fps=%.2f | total_frames=%d",
                    completed,
                    len(jobs),
                    video_path,
                    num_frames,
                    fps,
                    total_frames,
                )

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "views": list(views),
        "videos_processed": len(video_files),
        "total_frames": total_frames,
        "workers": args.workers,
    }
    (output_root / "predecode_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    logging.info("Finished predecoding. Summary written to %s", output_root / "predecode_summary.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a memory-mappable array cache for the two UR5 RGB training views.

The cache mirrors the LeRobot video path layout, but stores each episode/view as
`*.npy` with shape `[T, C, H, W]` and dtype `uint8`. Training can then slice the
exact frame directly by `frame_index` without mp4 seek/decode in the hot path.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import pathlib
from typing import Iterable

import av
import numpy as np


DEFAULT_DATASET_ROOT = pathlib.Path("/scratch/yz11445/real_robot_data/ur5_lab_test_tube_camera_shifts")
DEFAULT_CACHE_ROOT = pathlib.Path("/scratch/yz11445/real_robot_data/ur5_lab_test_tube_camera_shifts_array_cache")
DEFAULT_VIEWS = (
    "observation.images.context_left_rgb",
    "observation.images.wrist_right_rgb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=pathlib.Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cache-root", type=pathlib.Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--view", action="append", dest="views", help="Video key to cache. Repeatable.")
    parser.add_argument("--workers", type=int, default=4, help="Number of decode threads.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild arrays that already exist.")
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit episode indices to cache, e.g. --episodes 0 1 2",
    )
    parser.add_argument("--log-every", type=int, default=10, help="Log progress after every N completed videos.")
    return parser.parse_args()


def _iter_video_files(dataset_root: pathlib.Path, views: Iterable[str], episodes: set[int] | None) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    videos_root = dataset_root / "videos"
    for view in views:
        view_glob_root = videos_root / "chunk-000" / view
        if not view_glob_root.is_dir():
            raise FileNotFoundError(f"Missing input view directory: {view_glob_root}")
        for video_path in sorted(view_glob_root.glob("episode_*.mp4")):
            if episodes is not None:
                episode_idx = int(video_path.stem.split("_")[-1])
                if episode_idx not in episodes:
                    continue
            files.append(video_path)
    return files


def _cache_array_path(cache_root: pathlib.Path, dataset_root: pathlib.Path, video_path: pathlib.Path) -> pathlib.Path:
    rel = video_path.relative_to(dataset_root)
    return (cache_root / rel).with_suffix(".npy")


def _cache_meta_path(cache_root: pathlib.Path, dataset_root: pathlib.Path, video_path: pathlib.Path) -> pathlib.Path:
    rel = video_path.relative_to(dataset_root)
    return (cache_root / rel).with_suffix(".json")


def _is_complete(array_path: pathlib.Path, meta_path: pathlib.Path) -> bool:
    return array_path.is_file() and meta_path.is_file()


def _probe_video(video_path: pathlib.Path) -> tuple[int, int, int, float]:
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate) if stream.average_rate is not None else 0.0
        width = int(stream.codec_context.width)
        height = int(stream.codec_context.height)
        num_frames = int(stream.frames or 0)

    if num_frames <= 0:
        num_frames = 0
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for _ in container.decode(stream):
                num_frames += 1

    if num_frames <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Unable to probe video shape for {video_path}")
    return num_frames, height, width, fps


def _decode_video_to_memmap(video_path: pathlib.Path, array_path: pathlib.Path) -> tuple[tuple[int, ...], float]:
    num_frames, height, width, fps = _probe_video(video_path)
    frames = np.lib.format.open_memmap(
        array_path,
        mode="w+",
        dtype=np.uint8,
        shape=(num_frames, 3, height, width),
    )

    write_index = 0
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            rgb = frame.to_ndarray(format="rgb24")
            frames[write_index] = np.transpose(rgb, (2, 0, 1))
            write_index += 1

    if write_index != num_frames:
        raise ValueError(
            f"Decoded frame count mismatch for {video_path}: expected {num_frames}, wrote {write_index}"
        )
    frames.flush()
    return (num_frames, 3, height, width), fps


def _build_one(job: tuple[str, str, str, bool]) -> tuple[str, tuple[int, ...], float]:
    video_path_str, dataset_root_str, cache_root_str, overwrite = job
    video_path = pathlib.Path(video_path_str)
    dataset_root = pathlib.Path(dataset_root_str)
    cache_root = pathlib.Path(cache_root_str)
    array_path = _cache_array_path(cache_root, dataset_root, video_path)
    meta_path = _cache_meta_path(cache_root, dataset_root, video_path)

    if not overwrite and _is_complete(array_path, meta_path):
        meta = json.loads(meta_path.read_text())
        return str(video_path), tuple(meta["shape"]), float(meta.get("fps", 0.0))

    array_path.parent.mkdir(parents=True, exist_ok=True)
    shape, fps = _decode_video_to_memmap(video_path, array_path)

    meta = {
        "source_video": str(video_path),
        "shape": list(shape),
        "dtype": "uint8",
        "fps": fps,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return str(video_path), tuple(shape), fps


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    dataset_root = args.dataset_root.resolve()
    cache_root = args.cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    views = tuple(args.views) if args.views else DEFAULT_VIEWS
    episodes = set(args.episodes) if args.episodes is not None else None
    video_files = _iter_video_files(dataset_root, views, episodes)
    if not video_files:
        raise FileNotFoundError("No matching UR5 video files found to cache.")

    jobs = [(str(path), str(dataset_root), str(cache_root), args.overwrite) for path in video_files]
    logging.info(
        "Building UR5 video array cache for %d videos from %s to %s with views=%s workers=%d overwrite=%s",
        len(video_files),
        dataset_root,
        cache_root,
        views,
        args.workers,
        args.overwrite,
    )

    completed = 0
    total_frames = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_build_one, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            video_path, shape, fps = future.result()
            completed += 1
            total_frames += int(shape[0])
            if completed == 1 or completed % args.log_every == 0 or completed == len(jobs):
                logging.info(
                    "Completed %d/%d videos | last=%s shape=%s fps=%.2f | total_frames=%d",
                    completed,
                    len(jobs),
                    video_path,
                    shape,
                    fps,
                    total_frames,
                )

    summary = {
        "dataset_root": str(dataset_root),
        "cache_root": str(cache_root),
        "views": list(views),
        "videos_processed": len(video_files),
        "total_frames": total_frames,
        "workers": args.workers,
    }
    (cache_root / "cache_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    logging.info("Finished building cache. Summary written to %s", cache_root / "cache_summary.json")


if __name__ == "__main__":
    main()

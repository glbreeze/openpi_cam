#!/usr/bin/env python3
"""Predecode LeRobot video streams into memory-mappable uint8 numpy arrays.

The training loader looks for a sibling cache named ``{dataset_root}_array_cache``
and mirrors LeRobot video paths with ``.npy`` files. Each array is stored as
``(T, C, H, W)`` uint8 RGB so training can index frames without decoding MP4s.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import logging
import pathlib
import tempfile
from collections.abc import Sequence

import av
import numpy as np

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - pandas fallback is only for lean envs.
    pq = None


LOGGER = logging.getLogger("cache_lerobot_video_arrays")
DEFAULT_VIDEO_KEYS = (
    "observation.images.context_left_rgb",
    "observation.images.wrist_right_rgb",
)


@dataclasses.dataclass(frozen=True)
class VideoJob:
    dataset_root: pathlib.Path
    cache_root: pathlib.Path
    video_key: str
    video_path: pathlib.Path
    output_path: pathlib.Path
    metadata_path: pathlib.Path
    episode_index: int
    expected_frames: int
    expected_shape_chw: tuple[int, int, int]
    fps: float
    overwrite: bool


def _load_json(path: pathlib.Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def _format_dataset_path(template: str, episode_index: int, chunks_size: int, video_key: str | None = None) -> str:
    values = {
        "episode_chunk": _episode_chunk(episode_index, chunks_size),
        "episode_index": episode_index,
    }
    if video_key is not None:
        values["video_key"] = video_key
    return template.format(**values)


def _parquet_num_rows(path: pathlib.Path) -> int:
    if pq is not None:
        return pq.ParquetFile(path).metadata.num_rows

    import pandas as pd

    return len(pd.read_parquet(path, columns=[]))


def _decode_video_to_memmap(
    video_path: pathlib.Path,
    output_path: pathlib.Path,
    expected_frames: int,
    expected_shape_chw: tuple[int, int, int],
) -> tuple[pathlib.Path, tuple[int, int, int, int]]:
    channels, height, width = expected_shape_chw
    if channels != 3:
        raise ValueError(f"{video_path}: only 3-channel RGB videos are supported, got {expected_shape_chw}")

    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.stem}.",
        suffix=".npy",
        delete=False,
    ) as f:
        tmp_path = pathlib.Path(f.name)

    try:
        arr = np.lib.format.open_memmap(
            tmp_path,
            mode="w+",
            dtype=np.uint8,
            shape=(expected_frames, channels, height, width),
        )
        decoded = 0
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                if decoded >= expected_frames:
                    raise ValueError(f"{video_path}: decoded more than expected {expected_frames} frames")
                rgb = frame.to_ndarray(format="rgb24")
                if rgb.shape != (height, width, channels):
                    raise ValueError(
                        f"{video_path}: decoded frame shape {rgb.shape}, expected {(height, width, channels)}"
                    )
                arr[decoded] = np.transpose(rgb, (2, 0, 1))
                decoded += 1
        arr.flush()
        del arr
        if decoded != expected_frames:
            raise ValueError(f"{video_path}: decoded {decoded} frames, parquet has {expected_frames} rows")
        return tmp_path, (decoded, channels, height, width)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _atomic_write_json(path: pathlib.Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".json",
        delete=False,
    ) as f:
        tmp_path = pathlib.Path(f.name)
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    try:
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _process_job(job: VideoJob) -> dict:
    if job.output_path.exists() and job.metadata_path.exists() and not job.overwrite:
        arr = np.load(job.output_path, mmap_mode="r")
        return {
            "status": "skipped",
            "video_key": job.video_key,
            "episode_index": job.episode_index,
            "frames": int(arr.shape[0]),
            "output_path": str(job.output_path),
        }

    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path, shape = _decode_video_to_memmap(
        job.video_path,
        job.output_path,
        job.expected_frames,
        job.expected_shape_chw,
    )
    tmp_path.replace(job.output_path)
    _atomic_write_json(
        job.metadata_path,
        {
            "dtype": "uint8",
            "episode_index": job.episode_index,
            "fps": job.fps,
            "shape": list(shape),
            "source_video": str(job.video_path),
            "video_key": job.video_key,
        },
    )
    return {
        "status": "processed",
        "video_key": job.video_key,
        "episode_index": job.episode_index,
        "frames": int(shape[0]),
        "output_path": str(job.output_path),
    }


def _build_jobs(
    dataset_root: pathlib.Path,
    cache_root: pathlib.Path,
    video_keys: Sequence[str],
    episodes: Sequence[int] | None,
    max_episodes: int | None,
    overwrite: bool,
) -> tuple[list[VideoJob], dict]:
    info = _load_json(dataset_root / "meta" / "info.json")
    features = info["features"]
    chunks_size = int(info.get("chunks_size", 1000))
    data_path_template = info["data_path"]
    video_path_template = info["video_path"]
    fps = float(info["fps"])

    for video_key in video_keys:
        feature = features.get(video_key)
        if feature is None:
            raise KeyError(f"Missing video key in meta/info.json: {video_key}")
        if feature.get("dtype") != "video":
            raise TypeError(f"Feature is not a video: {video_key}")

    if episodes is None:
        episode_indices = list(range(int(info["total_episodes"])))
    else:
        episode_indices = list(episodes)
    if max_episodes is not None:
        episode_indices = episode_indices[:max_episodes]

    frame_counts: dict[int, int] = {}
    jobs: list[VideoJob] = []
    for ep_idx in episode_indices:
        parquet_path = dataset_root / _format_dataset_path(data_path_template, ep_idx, chunks_size)
        if not parquet_path.exists():
            raise FileNotFoundError(parquet_path)
        frame_counts[ep_idx] = _parquet_num_rows(parquet_path)

        for video_key in video_keys:
            rel_video_path = pathlib.Path(_format_dataset_path(video_path_template, ep_idx, chunks_size, video_key))
            video_path = dataset_root / rel_video_path
            if not video_path.exists():
                raise FileNotFoundError(video_path)
            output_path = (cache_root / rel_video_path).with_suffix(".npy")
            jobs.append(
                VideoJob(
                    dataset_root=dataset_root,
                    cache_root=cache_root,
                    video_key=video_key,
                    video_path=video_path,
                    output_path=output_path,
                    metadata_path=output_path.with_suffix(".json"),
                    episode_index=ep_idx,
                    expected_frames=frame_counts[ep_idx],
                    expected_shape_chw=tuple(features[video_key]["shape"]),
                    fps=fps,
                    overwrite=overwrite,
                )
            )

    return jobs, info


def _parse_episode_list(value: str) -> list[int]:
    episodes: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start_s, end_s = part.split(":", 1)
            episodes.extend(range(int(start_s), int(end_s)))
        else:
            episodes.append(int(part))
    return episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=pathlib.Path, required=True)
    parser.add_argument(
        "--cache-root",
        type=pathlib.Path,
        default=None,
        help="Defaults to a sibling named {dataset_root.name}_array_cache.",
    )
    parser.add_argument("--video-key", action="append", dest="video_keys", default=None)
    parser.add_argument("--episodes", type=_parse_episode_list, default=None, help="Comma list/ranges, e.g. 0,3,10:20.")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    dataset_root = args.dataset_root.expanduser().resolve()
    cache_root = args.cache_root
    if cache_root is None:
        cache_root = dataset_root.parent / f"{dataset_root.name}_array_cache"
    else:
        cache_root = cache_root.expanduser().resolve()

    video_keys = tuple(args.video_keys or DEFAULT_VIDEO_KEYS)
    jobs, _ = _build_jobs(
        dataset_root=dataset_root,
        cache_root=cache_root,
        video_keys=video_keys,
        episodes=args.episodes,
        max_episodes=args.max_episodes,
        overwrite=args.overwrite,
    )

    LOGGER.info("dataset_root=%s", dataset_root)
    LOGGER.info("cache_root=%s", cache_root)
    LOGGER.info("video_keys=%s", ", ".join(video_keys))
    LOGGER.info("jobs=%d", len(jobs))
    if args.dry_run:
        for job in jobs[:10]:
            LOGGER.info("dry-run job: %s -> %s", job.video_path, job.output_path)
        return

    processed = 0
    skipped = 0
    total_frames = 0
    if args.workers <= 1:
        results_iter = (_process_job(job) for job in jobs)
    else:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.workers)
        results_iter = (future.result() for future in concurrent.futures.as_completed(
            [executor.submit(_process_job, job) for job in jobs]
        ))

    try:
        for idx, result in enumerate(results_iter, start=1):
            processed += result["status"] == "processed"
            skipped += result["status"] == "skipped"
            total_frames += int(result["frames"])
            if idx == 1 or idx % args.log_every == 0 or idx == len(jobs):
                LOGGER.info(
                    "done %d/%d processed=%d skipped=%d last=%s ep=%06d frames=%d",
                    idx,
                    len(jobs),
                    processed,
                    skipped,
                    result["video_key"],
                    result["episode_index"],
                    result["frames"],
                )
    finally:
        if args.workers > 1:
            executor.shutdown(wait=True, cancel_futures=True)

    _atomic_write_json(
        cache_root / "cache_summary.json",
        {
            "cache_root": str(cache_root),
            "dataset_root": str(dataset_root),
            "total_frames": total_frames,
            "videos_processed": processed,
            "videos_skipped": skipped,
            "views": list(video_keys),
            "workers": args.workers,
        },
    )
    LOGGER.info("cache complete: %s", cache_root)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge local RoboTwin LeRobot v2.1 repos and GT point-target caches.

The converter currently emits one LeRobot repo per RoboTwin task. This script
builds a small joint-training repo by rewriting episode/task/global frame
indices while preserving the per-frame payload columns.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


CAM_DIRS = {
    "cam_high": "agent",
    "cam_left_wrist": "wrist",
    "cam_right_wrist": "right_wrist",
}

REQUIRED_CAM_COLUMNS = tuple(
    f"observation.{cam}_{suffix}"
    for cam in ("cam_high", "cam_left_wrist", "cam_right_wrist")
    for suffix in ("extrinsic", "intrinsic")
)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def _offset_stats(stats_row: dict, *, episode_offset: int, task_offset: int, index_offset: int) -> dict:
    row = json.loads(json.dumps(stats_row))
    row["episode_index"] = int(row["episode_index"]) + episode_offset
    stats = row.get("stats", {})

    if "episode_index" in stats:
        stats["episode_index"].update(
            {
                "min": [row["episode_index"]],
                "max": [row["episode_index"]],
                "mean": [float(row["episode_index"])],
            }
        )
    if "task_index" in stats:
        old_task = int(stats["task_index"]["min"][0])
        new_task = old_task + task_offset
        stats["task_index"].update({"min": [new_task], "max": [new_task], "mean": [float(new_task)]})
    if "index" in stats:
        for key in ("min", "max", "mean"):
            stats["index"][key] = [stats["index"][key][0] + index_offset]
    return row


def _replace_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _symlink_or_copy(src: Path, dst: Path, *, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def _video_keys(info: dict) -> list[str]:
    return [key for key, feature in info.get("features", {}).items() if feature.get("dtype") == "video"]


def merge_repos(input_roots: list[Path], output_root: Path) -> None:
    _replace_existing(output_root)
    (output_root / "data" / "chunk-000").mkdir(parents=True)
    (output_root / "meta").mkdir(parents=True)

    merged_episodes: list[dict] = []
    merged_episode_stats: list[dict] = []
    merged_tasks: list[dict] = []
    total_frames = 0
    total_episodes = 0
    total_tasks = 0
    base_info: dict | None = None

    for input_root in input_roots:
        meta_root = input_root / "meta"
        info = json.loads((meta_root / "info.json").read_text())
        if base_info is None:
            base_info = info

        episodes = _read_jsonl(meta_root / "episodes.jsonl")
        episode_stats = _read_jsonl(meta_root / "episodes_stats.jsonl")
        tasks = _read_jsonl(meta_root / "tasks.jsonl")

        task_offset = total_tasks
        episode_offset = total_episodes
        index_offset = total_frames
        video_keys = _video_keys(info)

        for task in tasks:
            merged_tasks.append({"task_index": int(task["task_index"]) + task_offset, "task": task["task"]})

        for episode in episodes:
            old_episode = int(episode["episode_index"])
            new_episode = old_episode + episode_offset
            src_parquet = input_root / "data" / "chunk-000" / f"episode_{old_episode:06d}.parquet"
            dst_parquet = output_root / "data" / "chunk-000" / f"episode_{new_episode:06d}.parquet"

            table = pq.read_table(src_parquet)
            missing_cam_columns = [col for col in REQUIRED_CAM_COLUMNS if col not in table.column_names]
            if missing_cam_columns:
                raise KeyError(f"{src_parquet} is missing required cam-aware columns: {missing_cam_columns}")
            num_rows = table.num_rows
            table = table.set_column(
                table.schema.get_field_index("episode_index"),
                "episode_index",
                pa.array([new_episode] * num_rows, type=table.schema.field("episode_index").type),
            )
            table = table.set_column(
                table.schema.get_field_index("task_index"),
                "task_index",
                pc.add(table["task_index"], pa.scalar(task_offset, type=table.schema.field("task_index").type)),
            )
            table = table.set_column(
                table.schema.get_field_index("index"),
                "index",
                pc.add(table["index"], pa.scalar(index_offset, type=table.schema.field("index").type)),
            )
            pq.write_table(table, dst_parquet)

            for video_key in video_keys:
                video_path_template = info["video_path"]
                src_video = input_root / video_path_template.format(
                    episode_chunk=old_episode // int(info["chunks_size"]),
                    video_key=video_key,
                    episode_index=old_episode,
                )
                dst_video = output_root / video_path_template.format(
                    episode_chunk=new_episode // int(base_info["chunks_size"]),
                    video_key=video_key,
                    episode_index=new_episode,
                )
                if not src_video.is_file():
                    raise FileNotFoundError(f"Missing source video for merged episode: {src_video}")
                _symlink_or_copy(src_video.resolve(), dst_video, copy=False)

            new_episode_row = dict(episode)
            new_episode_row["episode_index"] = new_episode
            merged_episodes.append(new_episode_row)

        for stats_row in episode_stats:
            merged_episode_stats.append(
                _offset_stats(
                    stats_row,
                    episode_offset=episode_offset,
                    task_offset=task_offset,
                    index_offset=index_offset,
                )
            )

        total_frames += int(info["total_frames"])
        total_episodes += int(info["total_episodes"])
        total_tasks += int(info["total_tasks"])

    assert base_info is not None
    merged_info = dict(base_info)
    merged_info.update(
        {
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": total_tasks,
            "total_videos": total_episodes * len(_video_keys(base_info)),
            "total_chunks": 1,
            "splits": {"train": f"0:{total_episodes}"},
        }
    )
    (output_root / "meta" / "info.json").write_text(json.dumps(merged_info, indent=4) + "\n")
    _write_jsonl(output_root / "meta" / "episodes.jsonl", merged_episodes)
    _write_jsonl(output_root / "meta" / "episodes_stats.jsonl", merged_episode_stats)
    _write_jsonl(output_root / "meta" / "tasks.jsonl", merged_tasks)


def merge_gt_caches(input_roots: list[Path], output_root: Path, *, copy: bool) -> None:
    _replace_existing(output_root)
    episode_offset = 0
    for input_root in input_roots:
        counts = []
        for src_cam, dst_cam in CAM_DIRS.items():
            src_dir = input_root / src_cam
            files = sorted(src_dir.glob("episode_*.npz"))
            if not files:
                raise FileNotFoundError(f"No GT cache files found under {src_dir}")
            counts.append(len(files))
            for local_idx, src in enumerate(files):
                dst = output_root / dst_cam / f"episode_{episode_offset + local_idx:06d}.npz"
                _symlink_or_copy(src.resolve(), dst, copy=copy)
        if len(set(counts)) != 1:
            raise ValueError(f"Mismatched camera counts for {input_root}: {counts}")
        episode_offset += counts[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-repo", action="append", required=True, help="Local LeRobot repo root.")
    parser.add_argument("--output-repo", required=True, help="Output LeRobot repo root.")
    parser.add_argument("--input-gt", action="append", required=True, help="Input GT cache root.")
    parser.add_argument("--output-gt", required=True, help="Output GT cache root.")
    parser.add_argument("--copy-gt", action="store_true", help="Copy GT npz files instead of symlinking them.")
    args = parser.parse_args()

    input_repos = [Path(p).expanduser().resolve() for p in args.input_repo]
    input_gts = [Path(p).expanduser().resolve() for p in args.input_gt]
    if len(input_repos) != len(input_gts):
        raise ValueError("--input-repo and --input-gt must be provided the same number of times")

    merge_repos(input_repos, Path(args.output_repo).expanduser().resolve())
    merge_gt_caches(input_gts, Path(args.output_gt).expanduser().resolve(), copy=args.copy_gt)

    print(f"merged repo: {args.output_repo}")
    print(f"merged gt  : {args.output_gt}")


if __name__ == "__main__":
    main()

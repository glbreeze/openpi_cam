"""Merge multiple LeRobot dataset shards into one dataset.

Each shard is expected to be a complete local LeRobot dataset with embedded-image
parquet episodes and v2.1 metadata. The merged dataset rewrites episode_index,
task_index, and global index fields so training can consume it as one coherent
dataset.
"""

from __future__ import annotations

import pathlib
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tyro


def _discover_shard_roots(shards_root: pathlib.Path) -> list[pathlib.Path]:
    shard_roots = sorted(
        path
        for path in shards_root.iterdir()
        if path.is_dir() and (path / "meta" / "info.json").is_file()
    )
    if not shard_roots:
        raise ValueError(f"No shard datasets found under {shards_root}")
    return shard_roots


def _replace_column(table: pa.Table, column_name: str, array: pa.Array) -> pa.Table:
    column_index = table.schema.get_field_index(column_name)
    if column_index < 0:
        raise KeyError(f"Missing expected parquet column: {column_name}")
    return table.set_column(column_index, column_name, array)


def _rewrite_episode_table(
    table: pa.Table,
    *,
    new_episode_index: int,
    global_index_start: int,
    task_index_mapping: dict[int, int],
) -> tuple[pa.Table, int]:
    row_count = table.num_rows
    if row_count == 0:
        return table, global_index_start

    task_column = table.column("task_index").to_numpy(zero_copy_only=False)
    rewritten_task_index = np.asarray([task_index_mapping[int(value)] for value in task_column], dtype=np.int64)

    table = _replace_column(
        table,
        "episode_index",
        pa.array(np.full(row_count, new_episode_index, dtype=np.int64)),
    )
    table = _replace_column(
        table,
        "index",
        pa.array(np.arange(global_index_start, global_index_start + row_count, dtype=np.int64)),
    )
    table = _replace_column(table, "task_index", pa.array(rewritten_task_index))
    return table, global_index_start + row_count


def _copy_episode_parquet(
    source_path: pathlib.Path,
    destination_path: pathlib.Path,
    *,
    new_episode_index: int,
    global_index_start: int,
    task_index_mapping: dict[int, int],
) -> int:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(source_path)
    rewritten_table, next_global_index = _rewrite_episode_table(
        table,
        new_episode_index=new_episode_index,
        global_index_start=global_index_start,
        task_index_mapping=task_index_mapping,
    )
    pq.write_table(rewritten_table, destination_path)
    return next_global_index


def main(
    shards_root: str,
    *,
    repo_id: str,
):
    shards_root_path = pathlib.Path(shards_root).expanduser().resolve()
    if not shards_root_path.exists():
        raise FileNotFoundError(f"Shard root does not exist: {shards_root_path}")

    shard_roots = _discover_shard_roots(shards_root_path)
    first_meta = LeRobotDatasetMetadata(repo_id=f"local/{shard_roots[0].name}", root=shard_roots[0])

    output_root = HF_LEROBOT_HOME / repo_id
    if output_root.exists():
        shutil.rmtree(output_root)

    merged_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        root=output_root,
        robot_type=first_meta.robot_type,
        fps=first_meta.fps,
        features=first_meta.features,
        use_videos=False,
    )

    global_index = 0

    for shard_root in shard_roots:
        shard_meta = LeRobotDatasetMetadata(repo_id=f"local/{shard_root.name}", root=shard_root)

        if shard_meta.features != merged_meta.features:
            raise ValueError(f"Feature mismatch between shard {shard_root} and merged dataset")
        if shard_meta.fps != merged_meta.fps:
            raise ValueError(f"FPS mismatch between shard {shard_root} and merged dataset")
        if shard_meta.robot_type != merged_meta.robot_type:
            raise ValueError(f"Robot type mismatch between shard {shard_root} and merged dataset")

        task_index_mapping: dict[int, int] = {}
        for old_task_index, task in shard_meta.tasks.items():
            new_task_index = merged_meta.get_task_index(task)
            if new_task_index is None:
                merged_meta.add_task(task)
                new_task_index = merged_meta.get_task_index(task)
            task_index_mapping[int(old_task_index)] = int(new_task_index)

        for source_episode_index in sorted(shard_meta.episodes):
            episode = shard_meta.episodes[source_episode_index]
            new_episode_index = merged_meta.total_episodes

            source_path = shard_root / shard_meta.get_data_file_path(source_episode_index)
            destination_path = output_root / merged_meta.get_data_file_path(new_episode_index)
            global_index = _copy_episode_parquet(
                source_path,
                destination_path,
                new_episode_index=new_episode_index,
                global_index_start=global_index,
                task_index_mapping=task_index_mapping,
            )

            merged_meta.save_episode(
                episode_index=new_episode_index,
                episode_length=episode["length"],
                episode_tasks=episode["tasks"],
                episode_stats=shard_meta.episodes_stats[source_episode_index],
            )

            print(
                f"[merge] shard={shard_root.name} source_episode={source_episode_index} "
                f"merged_episode={new_episode_index} length={episode['length']}"
            )

    print(
        f"[merge] completed repo_id={repo_id} shards={len(shard_roots)} "
        f"episodes={merged_meta.total_episodes} frames={merged_meta.total_frames}"
    )


if __name__ == "__main__":
    tyro.cli(main)

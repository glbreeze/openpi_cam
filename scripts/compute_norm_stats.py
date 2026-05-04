"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import dataclasses

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )

    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def _robotwin_v3_row_indices(ds, task_indices: tuple[int, ...]) -> list[int] | None:
    if not task_indices:
        return None

    allowed_task_indices = set(task_indices)
    row_indices = [row_index for row_index, task_index in enumerate(ds["task_index"]) if int(task_index) in allowed_task_indices]
    if not row_indices:
        raise ValueError(f"No rows matched task_indices={task_indices} for Robotwin v3 stats.")
    return row_indices


def _robotwin_v3_episode_ranges(ds) -> dict[int, tuple[int, int]]:
    episode_ranges: dict[int, list[int]] = {}
    for row_index, episode_index in enumerate(ds["episode_index"]):
        ep_idx = int(episode_index)
        if ep_idx not in episode_ranges:
            episode_ranges[ep_idx] = [row_index, row_index + 1]
        else:
            episode_ranges[ep_idx][1] = row_index + 1
    return {episode_index: (start, end) for episode_index, (start, end) in episode_ranges.items()}


def _robotwin_v3_action_chunk(ds, row_index: int, episode_index: int, episode_ranges: dict[int, tuple[int, int]], horizon: int):
    ep_start, ep_end = episode_ranges[episode_index]
    query_indices = [max(ep_start, min(ep_end - 1, row_index + delta)) for delta in range(horizon)]
    return np.asarray(ds.select(query_indices)["action"], dtype=np.float32)


def maybe_compute_robotwin_v3_parquet_stats(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    *,
    max_frames: int | None,
) -> dict[str, normalize.NormStats] | None:
    if data_config.repo_id is None or not _data_loader.is_robotwin_lerobot_v3(data_config.repo_id):
        return None

    import datasets

    dataset_root = _data_loader.get_lerobot_dataset_root(data_config.repo_id)
    data_glob = str(dataset_root / "data" / "chunk-*" / "file-*.parquet")
    ds = datasets.load_dataset(
        "parquet",
        data_files=data_glob,
        split="train",
        columns=["observation.state", "action", "episode_index", "task_index"],
    )
    row_indices = _robotwin_v3_row_indices(ds, tuple(data_config.task_indices))
    if row_indices is None:
        total_rows = len(ds)
        selected_indices = np.arange(total_rows)
    else:
        total_rows = len(row_indices)
        selected_indices = np.asarray(row_indices)

    if max_frames is not None and max_frames < total_rows:
        rng = np.random.default_rng(0)
        selected_indices = rng.choice(selected_indices, size=max_frames, replace=False)
    selected_indices = np.sort(selected_indices)

    episode_ranges = _robotwin_v3_episode_ranges(ds)
    stats_transform = transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ]
    )
    dummy_image = np.zeros((1, 1, 3), dtype=np.uint8)
    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}

    for row_index in tqdm.tqdm(selected_indices, desc="Computing Robotwin parquet stats"):
        item = ds[int(row_index)]
        transformed = stats_transform(
            {
                "observation.state": np.asarray(item["observation.state"], dtype=np.float32),
                "action": _robotwin_v3_action_chunk(
                    ds,
                    int(row_index),
                    int(item["episode_index"]),
                    episode_ranges,
                    model_config.action_horizon,
                ),
                "observation.images.cam_high": dummy_image,
                "observation.images.cam_left_wrist": dummy_image,
                "observation.images.cam_right_wrist": dummy_image,
                "task": "",
            }
        )
        stats["state"].update(np.asarray(transformed["state"]))
        stats["actions"].update(np.asarray(transformed["actions"]))

    return {key: stat.get_statistics() for key, stat in stats.items()}


def main(
    config_name: str,
    max_frames: int | None = None,
    repo_id: str | None = None,
    batch_size: int | None = None,
):
    config = _config.get_config(config_name)
    if repo_id is not None:
        config = dataclasses.replace(config, data=dataclasses.replace(config.data, repo_id=repo_id))
    if batch_size is not None:
        config = dataclasses.replace(config, batch_size=batch_size)
    data_config = config.data.create(config.assets_dirs, config.model)
    print("--------", config.assets_dirs, "-----------")
    output_path = config.assets_dirs / data_config.repo_id

    norm_stats = maybe_compute_robotwin_v3_parquet_stats(data_config, config.model, max_frames=max_frames)
    if norm_stats is not None:
        print(f"Writing stats to: {output_path}")
        normalize.save(output_path, norm_stats)
        return

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)

"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import dataclasses
import itertools
import pathlib
import pickle

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


def _default_resume_state_path(output_path: pathlib.Path) -> pathlib.Path:
    return output_path / "norm_stats.resume.pkl"


def _resolve_output_path(config: _config.TrainConfig, data_config: _config.DataConfig) -> pathlib.Path:
    if data_config.asset_id is None and data_config.repo_id is None:
        raise ValueError("Need either data_config.asset_id or data_config.repo_id to determine norm-stats output path")

    assets_dir = config.assets_dirs
    assets_config = getattr(config.data, "assets", None)
    if assets_config is not None and assets_config.assets_dir is not None:
        assets_dir = pathlib.Path(assets_config.assets_dir)

    return assets_dir / (data_config.asset_id or data_config.repo_id)


def _save_resume_state(
    path: pathlib.Path,
    *,
    config_name: str,
    repo_id: str,
    max_frames: int | None,
    batch_size: int,
    num_batches: int,
    batches_processed: int,
    stats: dict[str, normalize.RunningStats],
) -> None:
    payload = {
        "config_name": config_name,
        "repo_id": repo_id,
        "max_frames": max_frames,
        "batch_size": batch_size,
        "num_batches": num_batches,
        "batches_processed": batches_processed,
        "stats": {key: value.state_dict() for key, value in stats.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(payload, f)
    tmp_path.replace(path)


def _load_resume_state(
    path: pathlib.Path,
    *,
    config_name: str,
    repo_id: str,
    max_frames: int | None,
    batch_size: int,
    num_batches: int,
) -> tuple[int, dict[str, normalize.RunningStats]]:
    with path.open("rb") as f:
        payload = pickle.load(f)

    expected = {
        "config_name": config_name,
        "repo_id": repo_id,
        "max_frames": max_frames,
        "batch_size": batch_size,
        "num_batches": num_batches,
    }
    for key, expected_value in expected.items():
        actual_value = payload.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"Resume state mismatch for {key}: expected {expected_value!r}, found {actual_value!r} in {path}"
            )

    batches_processed = int(payload["batches_processed"])
    stats = {key: normalize.RunningStats.from_state_dict(value) for key, value in payload["stats"].items()}
    return batches_processed, stats


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
    config_name: str,
    batch_size: int,
    resume: bool,
    resume_state_path: pathlib.Path,
    save_every_batches: int,
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
    num_batches = len(selected_indices)
    start_batch = 0
    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}

    if resume and resume_state_path.exists():
        start_batch, stats = _load_resume_state(
            resume_state_path,
            config_name=config_name,
            repo_id=data_config.repo_id,
            max_frames=max_frames,
            batch_size=batch_size,
            num_batches=num_batches,
        )
        print(f"Resuming from {resume_state_path} at batch {start_batch}/{num_batches}")
    elif resume:
        print(f"No resume state found at {resume_state_path}; starting from scratch")

    iterator = itertools.islice(selected_indices, start_batch, None)
    for batch_index, row_index in enumerate(
        tqdm.tqdm(iterator, total=num_batches, initial=start_batch, desc="Computing Robotwin parquet stats"),
        start=start_batch,
    ):
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

        completed_batches = batch_index + 1
        if save_every_batches > 0 and completed_batches % save_every_batches == 0:
            _save_resume_state(
                resume_state_path,
                config_name=config_name,
                repo_id=data_config.repo_id,
                max_frames=max_frames,
                batch_size=batch_size,
                num_batches=num_batches,
                batches_processed=completed_batches,
                stats=stats,
            )

    return {key: stat.get_statistics() for key, stat in stats.items()}


def main(
    config_name: str,
    max_frames: int | None = None,
    repo_id: str | None = None,
    asset_id: str | None = None,
    batch_size: int | None = None,
    resume: bool = False,
    resume_state_path: str | None = None,
    save_every_batches: int = 100,
):
    config = _config.get_config(config_name)
    if repo_id is not None:
        config = dataclasses.replace(config, data=dataclasses.replace(config.data, repo_id=repo_id))
    if asset_id is not None:
        assets = getattr(config.data, "assets", None)
        if assets is None:
            raise ValueError(f"Config {config_name!r} does not expose an assets field for asset_id override")
        config = dataclasses.replace(
            config,
            data=dataclasses.replace(config.data, assets=dataclasses.replace(assets, asset_id=asset_id)),
        )
    if batch_size is not None:
        config = dataclasses.replace(config, batch_size=batch_size)
    data_config = config.data.create(config.assets_dirs, config.model)
    print("--------", config.assets_dirs, "-----------")
    output_path = _resolve_output_path(config, data_config)
    state_path = pathlib.Path(resume_state_path) if resume_state_path is not None else _default_resume_state_path(output_path)

    norm_stats = maybe_compute_robotwin_v3_parquet_stats(
        data_config,
        config.model,
        max_frames=max_frames,
        config_name=config_name,
        batch_size=config.batch_size,
        resume=resume,
        resume_state_path=state_path,
        save_every_batches=save_every_batches,
    )
    if norm_stats is not None:
        print(f"Writing stats to: {output_path}")
        normalize.save(output_path, norm_stats)
        state_path.unlink(missing_ok=True)
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
    start_batch = 0

    if resume and state_path.exists():
        start_batch, stats = _load_resume_state(
            state_path,
            config_name=config_name,
            repo_id=data_config.repo_id,
            max_frames=max_frames,
            batch_size=config.batch_size,
            num_batches=num_batches,
        )
        print(f"Resuming from {state_path} at batch {start_batch}/{num_batches}")
    elif resume:
        print(f"No resume state found at {state_path}; starting from scratch")

    iterator = itertools.islice(data_loader, start_batch, None)
    for batch_index, batch in enumerate(
        tqdm.tqdm(iterator, total=num_batches, initial=start_batch, desc="Computing stats"),
        start=start_batch,
    ):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

        completed_batches = batch_index + 1
        if save_every_batches > 0 and completed_batches % save_every_batches == 0:
            _save_resume_state(
                state_path,
                config_name=config_name,
                repo_id=data_config.repo_id,
                max_frames=max_frames,
                batch_size=config.batch_size,
                num_batches=num_batches,
                batches_processed=completed_batches,
                stats=stats,
            )

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    state_path.unlink(missing_ok=True)


if __name__ == "__main__":
    tyro.cli(main)

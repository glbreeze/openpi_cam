from collections.abc import Iterator, Sequence
import json
import logging
import multiprocessing
import os
import pathlib
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


def get_lerobot_dataset_root(repo_id: str) -> pathlib.Path:
    return pathlib.Path(lerobot_dataset.HF_LEROBOT_HOME) / repo_id


def _read_lerobot_info(repo_id: str) -> dict | None:
    info_path = get_lerobot_dataset_root(repo_id) / "meta" / "info.json"
    if not info_path.exists():
        return None
    return json.loads(info_path.read_text())


def is_robotwin_lerobot_v3(repo_id: str) -> bool:
    info = _read_lerobot_info(repo_id)
    if info is None:
        return False
    return info.get("codebase_version") == "v3.0" and repo_id == "lerobot/robotwin_unified"


class RobotwinLeRobotV3Dataset(torch.utils.data.Dataset):
    """Minimal LeRobot v3 reader for robotwin_unified.

    This bypasses the old v2.1-only `LeRobotDatasetMetadata` path and reads the
    v3 parquet/video structure directly from disk.
    """

    def __init__(self, repo_id: str, *, action_horizon: int, task_indices: Sequence[int] = ()):
        self._repo_id = repo_id
        self._root = get_lerobot_dataset_root(repo_id)
        self._info = _read_lerobot_info(repo_id)
        if self._info is None:
            raise FileNotFoundError(f"Missing LeRobot metadata for {repo_id} at {self._root / 'meta' / 'info.json'}")
        if self._info.get("codebase_version") != "v3.0":
            raise ValueError(f"{repo_id} is not a LeRobot v3 dataset")

        self._fps = int(self._info["fps"])
        self._action_horizon = action_horizon
        self._task_map = self._load_task_map()
        self._episode_ranges, self._episode_video_metadata = self._load_episode_metadata()
        self._hf_dataset = self._load_data_table()
        self._row_indices = self._select_row_indices(task_indices)

    def _load_task_map(self) -> dict[int, str]:
        import pyarrow.dataset as pa_dataset

        tasks_path = self._root / "meta" / "tasks.parquet"
        table = pa_dataset.dataset(tasks_path, format="parquet").to_table(columns=["task_index", "task"])
        task_indices = table.column("task_index").to_pylist()
        tasks = table.column("task").to_pylist()
        return {int(task_index): str(task) for task_index, task in zip(task_indices, tasks, strict=True)}

    def _load_episode_metadata(
        self,
    ) -> tuple[
        dict[int, tuple[int, int]],
        dict[int, dict[str, tuple[int, int, float, float]]],
    ]:
        import pyarrow.dataset as pa_dataset

        episodes_root = self._root / "meta" / "episodes"
        table = pa_dataset.dataset(episodes_root, format="parquet").to_table(
            columns=[
                "episode_index",
                "dataset_from_index",
                "dataset_to_index",
                "videos/observation.images.cam_high/chunk_index",
                "videos/observation.images.cam_high/file_index",
                "videos/observation.images.cam_high/from_timestamp",
                "videos/observation.images.cam_high/to_timestamp",
                "videos/observation.images.cam_left_wrist/chunk_index",
                "videos/observation.images.cam_left_wrist/file_index",
                "videos/observation.images.cam_left_wrist/from_timestamp",
                "videos/observation.images.cam_left_wrist/to_timestamp",
                "videos/observation.images.cam_right_wrist/chunk_index",
                "videos/observation.images.cam_right_wrist/file_index",
                "videos/observation.images.cam_right_wrist/from_timestamp",
                "videos/observation.images.cam_right_wrist/to_timestamp",
            ]
        )
        episode_indices = table.column("episode_index").to_pylist()
        from_indices = table.column("dataset_from_index").to_pylist()
        to_indices = table.column("dataset_to_index").to_pylist()
        cam_high_chunk_indices = table.column("videos/observation.images.cam_high/chunk_index").to_pylist()
        cam_high_file_indices = table.column("videos/observation.images.cam_high/file_index").to_pylist()
        cam_high_from_timestamps = table.column("videos/observation.images.cam_high/from_timestamp").to_pylist()
        cam_high_to_timestamps = table.column("videos/observation.images.cam_high/to_timestamp").to_pylist()
        cam_left_chunk_indices = table.column("videos/observation.images.cam_left_wrist/chunk_index").to_pylist()
        cam_left_file_indices = table.column("videos/observation.images.cam_left_wrist/file_index").to_pylist()
        cam_left_from_timestamps = table.column("videos/observation.images.cam_left_wrist/from_timestamp").to_pylist()
        cam_left_to_timestamps = table.column("videos/observation.images.cam_left_wrist/to_timestamp").to_pylist()
        cam_right_chunk_indices = table.column("videos/observation.images.cam_right_wrist/chunk_index").to_pylist()
        cam_right_file_indices = table.column("videos/observation.images.cam_right_wrist/file_index").to_pylist()
        cam_right_from_timestamps = table.column("videos/observation.images.cam_right_wrist/from_timestamp").to_pylist()
        cam_right_to_timestamps = table.column("videos/observation.images.cam_right_wrist/to_timestamp").to_pylist()

        episode_ranges: dict[int, tuple[int, int]] = {}
        episode_video_metadata: dict[int, dict[str, tuple[int, int, float, float]]] = {}
        for (
            episode_index,
            from_index,
            to_index,
            cam_high_chunk_index,
            cam_high_file_index,
            cam_high_from_timestamp,
            cam_high_to_timestamp,
            cam_left_chunk_index,
            cam_left_file_index,
            cam_left_from_timestamp,
            cam_left_to_timestamp,
            cam_right_chunk_index,
            cam_right_file_index,
            cam_right_from_timestamp,
            cam_right_to_timestamp,
        ) in zip(
            episode_indices,
            from_indices,
            to_indices,
            cam_high_chunk_indices,
            cam_high_file_indices,
            cam_high_from_timestamps,
            cam_high_to_timestamps,
            cam_left_chunk_indices,
            cam_left_file_indices,
            cam_left_from_timestamps,
            cam_left_to_timestamps,
            cam_right_chunk_indices,
            cam_right_file_indices,
            cam_right_from_timestamps,
            cam_right_to_timestamps,
            strict=True,
        ):
            ep_idx = int(episode_index)
            start = int(from_index)
            end = int(to_index)
            episode_ranges[ep_idx] = (start, end)
            episode_video_metadata[ep_idx] = {
                "observation.images.cam_high": (
                    int(cam_high_chunk_index),
                    int(cam_high_file_index),
                    float(cam_high_from_timestamp),
                    float(cam_high_to_timestamp),
                ),
                "observation.images.cam_left_wrist": (
                    int(cam_left_chunk_index),
                    int(cam_left_file_index),
                    float(cam_left_from_timestamp),
                    float(cam_left_to_timestamp),
                ),
                "observation.images.cam_right_wrist": (
                    int(cam_right_chunk_index),
                    int(cam_right_file_index),
                    float(cam_right_from_timestamp),
                    float(cam_right_to_timestamp),
                ),
            }

        return episode_ranges, episode_video_metadata

    def _load_data_table(self):
        import datasets

        data_glob = str(self._root / "data" / "chunk-*" / "file-*.parquet")
        return datasets.load_dataset("parquet", data_files=data_glob, split="train")

    def _select_row_indices(self, task_indices: Sequence[int]) -> list[int] | None:
        if not task_indices:
            return None

        missing_task_indices = sorted(set(task_indices) - set(self._task_map))
        if missing_task_indices:
            raise ValueError(f"task_indices not found in dataset metadata: {missing_task_indices}")

        allowed_task_indices = set(task_indices)
        row_indices = [
            row_index
            for row_index, task_index in enumerate(self._hf_dataset["task_index"])
            if int(task_index) in allowed_task_indices
        ]
        if not row_indices:
            raise ValueError(f"No rows matched task_indices={tuple(task_indices)} for {self._repo_id}")

        logging.info(
            "Filtering %s to task_indices=%s (%s frames)",
            self._repo_id,
            tuple(task_indices),
            len(row_indices),
        )
        return row_indices

    def _action_chunk(self, global_index: int, episode_index: int) -> np.ndarray:
        ep_start, ep_end = self._episode_ranges[episode_index]
        query_indices = [max(ep_start, min(ep_end - 1, global_index + delta)) for delta in range(self._action_horizon)]
        action_rows = self._hf_dataset.select(query_indices)["action"]
        return np.asarray(action_rows, dtype=np.float32)

    def _read_video_frame(self, video_key: str, chunk_index: int, file_index: int, local_index: int) -> np.ndarray:
        import imageio.v3 as iio

        rel_path = self._info["video_path"].format(
            video_key=video_key,
            chunk_index=chunk_index,
            file_index=file_index,
        )
        return iio.imread(self._root / rel_path, index=local_index)

    def _video_frame_for_episode(self, video_key: str, episode_index: int, timestamp: float) -> np.ndarray:
        chunk_index, file_index, from_timestamp, to_timestamp = self._episode_video_metadata[episode_index][video_key]
        # LeRobot v3 stores per-episode timestamps starting from zero, while the
        # episode metadata stores the video's absolute timestamp span inside the mp4.
        local_timestamp = from_timestamp + float(timestamp)
        local_index = round(local_timestamp * self._fps)
        max_index = max(0, round(to_timestamp * self._fps) - 1)
        local_index = min(local_index, max_index)
        return self._read_video_frame(video_key, chunk_index, file_index, local_index)

    def __len__(self) -> int:
        if self._row_indices is not None:
            return len(self._row_indices)
        return len(self._hf_dataset)

    def __getitem__(self, index: SupportsIndex) -> dict:
        row_index = index.__index__()
        if self._row_indices is not None:
            row_index = self._row_indices[row_index]
        item = self._hf_dataset[row_index]
        global_index = int(item["index"])
        episode_index = int(item["episode_index"])
        task_index = int(item["task_index"])
        timestamp = float(item["timestamp"])

        return {
            "observation.state": np.asarray(item["observation.state"], dtype=np.float32),
            "action": self._action_chunk(global_index, episode_index),
            "observation.images.cam_high": self._video_frame_for_episode(
                "observation.images.cam_high", episode_index, timestamp
            ),
            "observation.images.cam_left_wrist": self._video_frame_for_episode(
                "observation.images.cam_left_wrist", episode_index, timestamp
            ),
            "observation.images.cam_right_wrist": self._video_frame_for_episode(
                "observation.images.cam_right_wrist", episode_index, timestamp
            ),
            "task": self._task_map[task_index],
        }


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)
    if is_robotwin_lerobot_v3(repo_id):
        return RobotwinLeRobotV3Dataset(
            repo_id,
            action_horizon=action_horizon,
            task_indices=data_config.task_indices,
        )

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)

    episodes = None
    if data_config.task_indices:
        missing_task_indices = sorted(set(data_config.task_indices) - set(dataset_meta.tasks))
        if missing_task_indices:
            raise ValueError(f"task_indices not found in dataset metadata: {missing_task_indices}")

        allowed_tasks = {dataset_meta.tasks[int(task_index)] for task_index in data_config.task_indices}
        episodes = [
            episode_index
            for episode_index, episode in dataset_meta.episodes.items()
            if any(task in allowed_tasks for task in episode.get("tasks", []))
        ]
        if not episodes:
            raise ValueError(f"No episodes matched task_indices={tuple(data_config.task_indices)} for {repo_id}")
        logging.info(
            "Filtering %s to task_indices=%s (%s episodes)",
            repo_id,
            tuple(data_config.task_indices),
            len(episodes),
        )

    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        episodes=episodes,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats
    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)
    # import pdb;  pdb.set_trace()
    # tp = dataset.__getitem__(0)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]

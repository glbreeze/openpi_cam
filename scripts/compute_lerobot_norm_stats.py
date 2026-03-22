"""Compute normalization statistics directly from a local LeRobot dataset."""

from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
import numpy as np
import pyarrow.parquet as pq
import tyro

import openpi.shared.normalize as normalize


def _iter_parquet_files(dataset_root: Path):
    data_dir = dataset_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"LeRobot dataset data directory not found: {data_dir}")
    yield from sorted(data_dir.rglob("*.parquet"))


def main(
    repo_id: str,
    *,
    output_root: str,
    batch_size: int = 4096,
):
    dataset_root = HF_LEROBOT_HOME / repo_id
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    num_rows = 0
    num_files = 0

    for parquet_path in _iter_parquet_files(dataset_root):
        parquet_file = pq.ParquetFile(parquet_path)
        num_files += 1
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["state", "actions"]):
            values = batch.to_pydict()
            for key in stats:
                stats[key].update(np.asarray(values[key], dtype=np.float32))
            num_rows += batch.num_rows

    if num_rows < 2:
        raise ValueError(f"Need at least 2 rows to compute statistics, found {num_rows} in {dataset_root}.")

    norm_stats = {key: value.get_statistics() for key, value in stats.items()}
    output_path = Path(output_root) / repo_id
    print(f"Read {num_rows} rows from {num_files} parquet files under {dataset_root}")
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)

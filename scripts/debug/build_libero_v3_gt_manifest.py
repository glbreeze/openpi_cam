"""Build an episode manifest from a LeRobot LIBERO dataset to raw HDF5 demos."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq


def _is_noop(action: np.ndarray, prev_action: np.ndarray | None = None, threshold: float = 1e-4) -> bool:
    action = np.asarray(action)
    if prev_action is None:
        return float(np.linalg.norm(action[:-1])) < threshold
    return (
        float(np.linalg.norm(action[:-1])) < threshold
        and float(action[-1]) == float(np.asarray(prev_action)[-1])
    )


def _filter_noops(actions: np.ndarray) -> tuple[np.ndarray, list[int]]:
    kept: list[np.ndarray] = []
    kept_indices: list[int] = []
    prev_kept_action = None
    for idx, action in enumerate(np.asarray(actions, dtype=np.float32)):
        if _is_noop(action, prev_kept_action):
            continue
        kept.append(action)
        kept_indices.append(idx)
        prev_kept_action = action
    if not kept:
        return np.zeros((0, np.asarray(actions).shape[-1]), dtype=np.float32), kept_indices
    return np.stack(kept, axis=0).astype(np.float32), kept_indices


def _fingerprint_actions(actions: np.ndarray, decimals: int) -> tuple[int, str]:
    actions = np.asarray(actions, dtype=np.float32)
    rounded = np.round(actions, decimals=decimals).astype(np.float32, copy=False)
    return int(actions.shape[0]), hashlib.sha1(rounded.tobytes()).hexdigest()


def _episode_parquets(dataset_root: Path) -> list[Path]:
    paths = sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No LeRobot parquet episodes found under {dataset_root / 'data'}")
    for idx, path in enumerate(paths):
        expected = f"episode_{idx:06d}.parquet"
        if path.name != expected:
            raise ValueError(f"Non-contiguous LeRobot episode order at {path}; expected {expected}")
    return paths


def _raw_episode_index(raw_root: Path, decimals: int) -> dict[tuple[int, str], list[dict]]:
    index: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for raw_path in sorted(raw_root.rglob("*.hdf5")):
        with h5py.File(raw_path, "r") as f:
            for episode_name in sorted(f["data"].keys()):
                raw_actions = np.asarray(f[f"data/{episode_name}/actions"], dtype=np.float32)
                filtered_actions, kept_indices = _filter_noops(raw_actions)
                length, digest = _fingerprint_actions(filtered_actions, decimals)
                index[(length, digest)].append(
                    {
                        "raw_file": str(raw_path.relative_to(raw_root)),
                        "raw_episode": episode_name,
                        "length": length,
                        "kept_indices": kept_indices,
                    }
                )
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--action-decimals", type=int, default=6)
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve()
    raw_index = _raw_episode_index(raw_root, args.action_decimals)

    rows: list[dict] = []
    used_raw: set[tuple[str, str]] = set()
    for episode_index, parquet_path in enumerate(_episode_parquets(dataset_root)):
        table = pq.read_table(parquet_path, columns=["actions", "task_index"])
        data = table.to_pydict()
        actions = np.asarray(data["actions"], dtype=np.float32)
        length, digest = _fingerprint_actions(actions, args.action_decimals)
        candidates = raw_index.get((length, digest), [])
        candidates = [
            item for item in candidates if (item["raw_file"], item["raw_episode"]) not in used_raw
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one raw match for LeRobot episode {episode_index}, "
                f"got {len(candidates)} candidates"
            )
        match = candidates[0]
        used_raw.add((match["raw_file"], match["raw_episode"]))
        rows.append(
            {
                "episode_index": episode_index,
                "task_index": int(data["task_index"][0]),
                "length": length,
                "action_sha1": digest,
                **match,
            }
        )

    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    with args.output.expanduser().open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")

    print(f"manifest={args.output.expanduser()}")
    print(f"episodes={len(rows)}")
    print(f"raw_matches={len(used_raw)}")


if __name__ == "__main__":
    main()

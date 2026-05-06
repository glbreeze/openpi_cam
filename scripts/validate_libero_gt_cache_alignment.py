"""Validate aligned LIBERO GT depth HDF5 and point-target cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import h5py
import numpy as np


def _fingerprint_actions(actions: np.ndarray, decimals: int) -> tuple[int, str]:
    actions = np.asarray(actions, dtype=np.float32)
    rounded = np.round(actions, decimals=decimals).astype(np.float32, copy=False)
    return int(actions.shape[0]), hashlib.sha1(rounded.tobytes()).hexdigest()


def _read_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    for idx, row in enumerate(rows):
        if int(row["episode_index"]) != idx:
            raise ValueError(f"Manifest is not contiguous at row {idx}: {row}")
    return rows


def _read_lerobot_lengths(dataset_root: Path) -> list[int]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    lengths: list[int] = []
    with episodes_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lengths.append(int(json.loads(line)["length"]))
    if not lengths:
        raise ValueError(f"No LeRobot episodes found in {episodes_path}")
    return lengths


def _npz_array_shape(path: Path, key: str) -> tuple[int, ...]:
    member = f"{key}.npy"
    with zipfile.ZipFile(path) as zf:
        with zf.open(member) as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, _, _ = np.lib.format.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, _, _ = np.lib.format.read_array_header_2_0(f)
            else:
                shape, _, _ = np.lib.format._read_array_header(f, version)
    return tuple(int(x) for x in shape)


def _validate_depth(depth_root: Path, manifest_rows: list[dict], decimals: int):
    root = depth_root / "libero_object"
    for row in manifest_rows:
        idx = int(row["episode_index"])
        path = root / f"episode_{idx:06d}.hdf5"
        if not path.exists():
            raise FileNotFoundError(f"Missing aligned depth HDF5: {path}")
        with h5py.File(path, "r") as f:
            episode_names = list(f["data"].keys())
            if episode_names != ["demo_0"]:
                raise ValueError(f"{path} expected exactly data/demo_0, got {episode_names}")
            ep = f["data/demo_0"]
            actions = np.asarray(ep["actions"], dtype=np.float32)
            length, digest = _fingerprint_actions(actions, decimals)
            if length != int(row["length"]) or digest != row["action_sha1"]:
                raise ValueError(f"Action fingerprint mismatch in {path}")
            obs = ep["obs"]
            for key in ("agentview_depth", "eye_in_hand_depth"):
                if key not in obs:
                    raise ValueError(f"{path} missing {key}")
                if int(obs[key].shape[0]) != length:
                    raise ValueError(f"{path}/{key} length {obs[key].shape[0]} != {length}")


def _validate_cache(cache_root: Path, manifest_rows: list[dict]):
    expected_count = len(manifest_rows)
    for view in ("agent", "wrist"):
        paths = sorted((cache_root / view).glob("episode_*.npz"))
        if len(paths) != expected_count:
            raise ValueError(f"{view} cache count {len(paths)} != {expected_count}")
        for row, path in zip(manifest_rows, paths, strict=True):
            idx = int(row["episode_index"])
            expected_name = f"episode_{idx:06d}.npz"
            if path.name != expected_name:
                raise ValueError(f"{view} cache expected {expected_name}, got {path.name}")
            for key in ("xy", "log_z", "conf"):
                shape = _npz_array_shape(path, key)
                if shape[0] != int(row["length"]):
                    raise ValueError(f"{path}:{key} length {shape[0]} != {row['length']}")
    sample_indices = sorted({0, expected_count // 2, expected_count - 1})
    for idx in sample_indices:
        for view in ("agent", "wrist"):
            path = cache_root / view / f"episode_{idx:06d}.npz"
            with np.load(path) as data:
                if set(data.files) != {"xy", "log_z", "conf"}:
                    raise ValueError(f"{path} unexpected keys {data.files}")
                for key in ("xy", "log_z", "conf"):
                    if not np.isfinite(data[key]).all():
                        raise ValueError(f"{path}:{key} contains non-finite values")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--depth-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--action-decimals", type=int, default=6)
    args = parser.parse_args()

    manifest_rows = _read_manifest(args.manifest.expanduser())
    lerobot_lengths = _read_lerobot_lengths(args.dataset_root.expanduser())
    manifest_lengths = [int(row["length"]) for row in manifest_rows]
    if manifest_lengths != lerobot_lengths:
        raise ValueError("Manifest lengths do not match LeRobot episode metadata")

    _validate_depth(args.depth_root.expanduser(), manifest_rows, args.action_decimals)
    _validate_cache(args.cache_root.expanduser(), manifest_rows)

    print("gt_cache_alignment_ok")
    print(f"episodes={len(manifest_rows)}")
    print(f"frames={sum(manifest_lengths)}")
    print(f"depth_root={args.depth_root.expanduser()}")
    print(f"cache_root={args.cache_root.expanduser()}")


if __name__ == "__main__":
    main()

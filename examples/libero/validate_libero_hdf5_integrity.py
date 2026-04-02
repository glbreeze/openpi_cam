"""Validate LIBERO-Camera HDF5 files before conversion."""

from __future__ import annotations

import argparse
import concurrent.futures
import pathlib
import re
import sys

import h5py

_CAMVAR_RE = re.compile(r"_demo_(camvar_\d+)_")


def _parse_label_set(raw: str | None) -> set[str]:
    if not raw:
        return set()
    normalized = raw.replace(",", "+")
    return {part.strip() for part in normalized.split("+") if part.strip()}


def _camera_label(path: pathlib.Path) -> str:
    match = _CAMVAR_RE.search(path.name)
    return match.group(1) if match else "original"


def _check_hdf5(path: pathlib.Path) -> tuple[pathlib.Path, str] | None:
    try:
        with h5py.File(path, "r") as h5_file:
            if "data" not in h5_file:
                return path, "missing 'data' group"
            data_group = h5_file["data"]
            try:
                next(iter(data_group.keys()))
            except StopIteration:
                return path, "empty 'data' group"
        return None
    except Exception as exc:  # noqa: BLE001
        return path, f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=pathlib.Path, required=True)
    parser.add_argument("--include-camera-labels", type=str, default="")
    parser.add_argument("--exclude-camera-labels", type=str, default="")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--delete-bad", action="store_true")
    args = parser.parse_args()

    include_labels = _parse_label_set(args.include_camera_labels)
    exclude_labels = _parse_label_set(args.exclude_camera_labels)
    overlap = include_labels & exclude_labels
    if overlap:
        raise ValueError(f"camera label filters overlap: {sorted(overlap)}")

    candidates = sorted(args.dataset_root.rglob("*.hdf5"))
    selected_files = []
    for path in candidates:
        label = _camera_label(path)
        if include_labels and label not in include_labels:
            continue
        if exclude_labels and label in exclude_labels:
            continue
        selected_files.append(path)

    print(
        "[validate-hdf5]",
        f"dataset_root={args.dataset_root}",
        f"selected_files={len(selected_files)}",
        f"include_camera_labels={sorted(include_labels) if include_labels else None}",
        f"exclude_camera_labels={sorted(exclude_labels) if exclude_labels else None}",
        f"num_workers={args.num_workers}",
        f"delete_bad={args.delete_bad}",
    )

    bad_entries: list[tuple[pathlib.Path, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
        for result in executor.map(_check_hdf5, selected_files):
            if result is None:
                continue
            bad_entries.append(result)

    if args.delete_bad:
        for path, _reason in bad_entries:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    print("[validate-hdf5]", f"checked={len(selected_files)}", f"bad={len(bad_entries)}")
    for path, reason in bad_entries:
        print(f"[validate-hdf5] bad: {path}\t{reason}")

    if bad_entries:
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Surgical fix for stale RoboTwin right_wrist GT caches (180-deg rotation bug).

The pre-2026-05-12 `cache_robotwin_gt_point_targets.py` applied a 180-degree
spatial flip to Sapien depth before computing local pointmap targets; the bug
was reverted by commit `b903441` but only `agent` / `wrist` were re-cached
afterwards. `right_wrist/episode_*.npz` files dated 2026-05-09 are the stale
artifacts.

For tasks whose source HDF5s are no longer on disk, we can recover correct
targets by rotating the cached (xy, log_z, conf) tensors 180 degrees in the
spatial dimensions. Equivalence between rotation-of-cache and re-caching from
HDF5 was verified empirically on `beat_block_hammer` where both sources
exist (log_z corr=0.978 mean-abs-diff=0.006, xy corr=1.0).

Usage:
    python scripts/debug/fix_stale_robotwin_right_wrist_gt.py \
        --gt-root ~/.cache/openpi/gt_point_targets_224 \
        --tasks robotwin_handover_block_demo_clean_camaware_50 ...
        [--dry-run]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np


def fix_stale_inplace(path: Path) -> None:
    """Apply per-channel fix derived empirically from beat_block_hammer where both
    fresh-from-HDF5 and stale caches exist:

    - xy : column flip (axis=2). The stale cache has only the x-channel mirrored
           (`fx -> -fx` in the old `_adjust_and_scale_k`). y-channel is already
           correct so we must NOT row-flip.
    - log_z, conf : 180-degree rotation (axes 1 and 2). The stale cache rendered
           depth from a row+column-flipped tensor (the "180-deg flip" bug
           reverted in commit b903441).

    Per-channel match against fresh on beat_block_hammer: xy diff=0.004 corr=1.0,
    log_z diff=0.006 corr=0.978.
    """
    with np.load(path) as data:
        xy = np.asarray(data["xy"])
        logz = np.asarray(data["log_z"])
        conf = np.asarray(data["conf"])
    xy = np.ascontiguousarray(xy[:, :, ::-1, :])           # column flip only
    logz = np.ascontiguousarray(logz[:, ::-1, ::-1, :])    # 180-deg rotation
    conf = np.ascontiguousarray(conf[:, ::-1, ::-1, :])    # 180-deg rotation
    np.savez(path, xy=xy, log_z=logz, conf=conf)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=Path("~/.cache/openpi/gt_point_targets_224").expanduser(),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        required=True,
        help="Task subdir names under gt-root, e.g. robotwin_handover_block_demo_clean_camaware_50",
    )
    parser.add_argument("--cam", default="right_wrist")
    parser.add_argument("--backup-dir", type=Path, default=Path("/tmp/stale_right_wrist_backup"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for task in args.tasks:
        cam_dir = args.gt_root / task / args.cam
        if not cam_dir.exists():
            print(f"[skip] {cam_dir} does not exist")
            continue
        episodes = sorted(cam_dir.glob("episode_*.npz"))
        if not episodes:
            print(f"[skip] {cam_dir}: no episode files")
            continue

        backup = args.backup_dir / task / args.cam
        if args.dry_run:
            print(f"[dry-run] would rotate {len(episodes)} files under {cam_dir} (backup to {backup})")
            continue

        backup.mkdir(parents=True, exist_ok=True)
        for src in episodes:
            shutil.copy2(src, backup / src.name)
        print(f"[backup] copied {len(episodes)} files -> {backup}")

        for src in episodes:
            fix_stale_inplace(src)
        print(f"[fixed] {len(episodes)} files under {cam_dir}")


if __name__ == "__main__":
    main()

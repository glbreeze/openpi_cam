"""In-place 180-deg spatial rotation + xy.y sign flip on a GT-style point cache
to convert from the legacy (depth[::-1,::-1] + K with fx-negated) convention to
the new natural-Sapien (no flip, K with positive fx/fy) convention used by the
fixed RoboTwin pipeline.

The old `cache_robotwin_gt_point_targets._adjust_and_scale_k` negated fx and the
depth feed was `depth[::-1, ::-1]`. Concretely, an old cache pixel (i, j) holds
the ray of the world point that lives at natural pixel (223-i, 223-j), with the
ray's y-component sign-flipped (because `fy` was positive but the v-coordinate
was running over a y-flipped depth grid that should have been y-down already).

Conversion:
  new[i, j]              = old[223-i, 223-j]        (spatial 180-deg rotation)
  new.xy[..., 1]         = -spatial_rotated.xy[1]    (y-sign flip)
  log_z, conf            : pass through after rotation (scalar fields)
"""

from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import numpy as np


def fix_npz(path: Path):
    with np.load(path) as f:
        xy = np.asarray(f["xy"])      # (T, R, R, 2)
        log_z = np.asarray(f["log_z"])  # (T, R, R, 1)
        conf = np.asarray(f["conf"])  # (T, R, R, 1)

    # 180-deg spatial rotation: reverse both spatial dims (axes 1 and 2 for (T, H, W, C))
    xy_rot = xy[:, ::-1, ::-1, :].copy()
    log_z_rot = log_z[:, ::-1, ::-1, :].copy()
    conf_rot = conf[:, ::-1, ::-1, :].copy()

    # y-sign flip on xy[..., 1]
    xy_rot[..., 1] = -xy_rot[..., 1]

    # Write to a sibling tmp file then atomic-rename to the final path.
    # `np.savez_compressed(filename_with_npz, ...)` writes to that exact name
    # only if the name already ends in `.npz`; otherwise it appends `.npz`.
    tmp = path.with_name(path.stem + "_tmp.npz")
    if tmp.exists():
        tmp.unlink()
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, xy=xy_rot.astype(np.float16), log_z=log_z_rot.astype(np.float16), conf=conf_rot.astype(np.float16))
    shutil.move(str(tmp), str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True, help="e.g. ~/.cache/openpi/gt_point_targets_224/robotwin_<task>_demo_clean_camaware_50")
    args = parser.parse_args()
    root = Path(args.cache_root).expanduser().resolve()
    npzs = sorted(root.rglob("episode_*.npz"))
    if not npzs:
        raise SystemExit(f"No npz files under {root}")
    for p in npzs:
        fix_npz(p)
        print(f"  fixed {p.name}")
    print(f"DONE: {len(npzs)} npz files in {root}")


if __name__ == "__main__":
    main()

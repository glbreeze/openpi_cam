"""Round-3 (final) orientation fix for RoboCasa-v0 GT and Pi3X distillation caches.

Background:
  - The original cache scripts applied `depth[::-1, ::-1]` + `fx → -fx`,
    mirroring the LIBERO convention. RoboCasa actually uses **natural
    y-down + normal-x with positive fx** end-to-end.
  - Round-1 fix wrongly applied `[::-1, ::-1]` + `xy[1]` sign flip — that
    overcorrected and left the bundle upside-down with the xy field
    coincidentally aligned (because the sign-flip on xy[1] absorbed the
    extra vertical flip introduced by `[::-1, ...]` on xy[1]).
  - Round-2 then applied flipud + xy[1] sign flip, which fixed the upside-
    down log_z but desync'd xy from log_z.
  - Round-3 (this script) is the correct one. Verified empirically by
    generating a fresh cache with the correctly-patched script
    (`depth[::-1, :]` + natural K) and comparing values:
        bundle → flipud → fresh   (mean|diff| = 0.12 on log_z; xy already
                                   matches with 0.005 mean|diff|)

The transform is:
    log_z_natural = log_z_current[::-1, :]   (flipud only)
    conf_natural  = conf_current[::-1, :]    (flipud only)
    xy_natural    = xy_current               (already correct, no change)

This relies on the bundle being in the post-round-1-fix state (round-2
applied and then undone — i.e., round-2 self-inverse re-applied). If
round-2 is currently applied, run it once more first to revert.

Usage:
    uv run scripts/fix_robocasa24_cache_orientation_round3.py \\
        --bundle ~/Research/robocasa-human50 --dry-run
    uv run scripts/fix_robocasa24_cache_orientation_round3.py \\
        --bundle ~/Research/robocasa-human50
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import tqdm

logger = logging.getLogger("fix_robocasa24_cache_orientation_round3")

CAMS = ("base", "wrist")


def _round3_arrays(xy: np.ndarray, log_z: np.ndarray, conf: np.ndarray):
    """flipud on log_z and conf only; xy is already correctly oriented
    after the round-1 fix's xy[1] sign-flip absorbed the vertical flip."""
    log_z_fixed = log_z[..., ::-1, :, :].copy()
    conf_fixed = conf[..., ::-1, :, :].copy()
    xy_fixed = xy.copy()    # no change
    return xy_fixed, log_z_fixed, conf_fixed


def _process_cache(cache_root: Path, *, dry_run: bool, backup_dir: Path | None):
    if not cache_root.exists():
        logger.warning("skip — missing: %s", cache_root)
        return
    files = []
    for cam in CAMS:
        files.extend(sorted((cache_root / cam).glob("episode_*.npz")))
    if not files:
        logger.warning("skip — empty: %s", cache_root)
        return

    logger.info("processing %d files under %s (dry_run=%s)",
                len(files), cache_root, dry_run)
    for path in tqdm.tqdm(files, desc=cache_root.name):
        with np.load(path) as f:
            xy_old = f["xy"][...]
            log_z_old = f["log_z"][...]
            conf_old = f["conf"][...]
        xy_new, log_z_new, conf_new = _round3_arrays(xy_old, log_z_old, conf_old)

        if dry_run:
            continue

        if backup_dir is not None:
            rel = path.relative_to(cache_root)
            backup_path = backup_dir / cache_root.name / rel
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            if not backup_path.exists():
                shutil.copy2(path, backup_path)

        tmp = path.with_name(path.stem + ".tmp.npz")
        if tmp.exists():
            tmp.unlink()
        np.savez_compressed(tmp,
                            xy=xy_new.astype(np.float16),
                            log_z=log_z_new.astype(np.float16),
                            conf=conf_new.astype(np.float16))
        tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path,
                        default=Path("~/Research/robocasa-human50").expanduser())
    parser.add_argument("--caches", nargs="+", default=["gt", "pi3x"],
                        choices=["gt", "pi3x"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    name_to_dir = {"gt": args.bundle / "gt_point_targets",
                   "pi3x": args.bundle / "pi3x_targets"}
    for c in args.caches:
        _process_cache(name_to_dir[c], dry_run=args.dry_run,
                       backup_dir=args.backup_dir)
    logger.info("done.")


if __name__ == "__main__":
    main()

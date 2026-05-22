#!/usr/bin/env python3
"""Merge N per-task RoboCasa-v0 LeRobot datasets into one multi-task dataset.

When stage 2 runs in parallel, each worker writes a separate LeRobot
dataset (one per task) to avoid concurrent-writer corruption. This script
concatenates them into a unified multi-task dataset whose episodes carry
their original task description in `annotation.human.task_description`
(i.e., the `task` field — Pi0 is prompt-conditioned, so the task string is
what differentiates rows).

The merge is done via LeRobot's `add_frame` / `save_episode` API rather
than raw file copies, so all metadata (episode counters, task index map,
info.json totals) ends up consistent. For image-mode datasets the cost is
PNG decode + re-encode per frame; cheap (~1-3 ms/frame).

Run in the `openpi` venv (needs lerobot). Does NOT need
robosuite/robocasa/mujoco — same as stage 2.

Usage:
    uv run scripts/merge_robocasa24_lerobot.py \
        --src-prefix robocasa24/per_task_ \
        --src-suffix _human_camaware \
        --out-repo robocasa24/all24_human_camaware

The script enumerates source repos via `<src-prefix><TaskName><src-suffix>`
for each of the 24 manipulation tasks.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import torch
import tqdm
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

TASKS_24 = (
    "PnPCounterToCab", "PnPCabToCounter", "PnPCounterToSink", "PnPSinkToCounter",
    "PnPCounterToMicrowave", "PnPMicrowaveToCounter", "PnPCounterToStove",
    "PnPStoveToCounter",
    "OpenSingleDoor", "CloseSingleDoor", "OpenDoubleDoor", "CloseDoubleDoor",
    "OpenDrawer", "CloseDrawer",
    "TurnOnSinkFaucet", "TurnOffSinkFaucet", "TurnSinkSpout",
    "TurnOnStove", "TurnOffStove",
    "CoffeeSetupMug", "CoffeeServeMug", "CoffeePressButton",
    "TurnOnMicrowave", "TurnOffMicrowave",
)

logger = logging.getLogger("merge_robocasa24_lerobot")


def _detect_use_videos(features: dict) -> bool:
    """Return True if any observation.images.* feature is encoded as 'video'."""
    for k, v in features.items():
        if k.startswith("observation.images.") and v.get("dtype") == "video":
            return True
    return False


def _to_np(x):
    return x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _img_for_add_frame(x, use_videos: bool):
    """LeRobot's add_frame expects images in the same dtype/range it was
    created with. On read, it returns CHW float32 in [0,1]. We convert back
    to CHW uint8 to match the schema we used when creating (3, H, W) uint8.
    """
    arr = _to_np(x)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src-prefix", default="robocasa24/per_task_",
        help="Prefix prepended to each task name to form the source repo_id.",
    )
    parser.add_argument(
        "--src-suffix", default="_human_camaware",
        help="Suffix appended to each task name to form the source repo_id.",
    )
    parser.add_argument(
        "--out-repo", default="robocasa24/all24_human_camaware",
        help="Output unified repo_id; lands at $HF_LEROBOT_HOME/<out-repo>/",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="Subset of task names. Default: all 24.",
    )
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Write the merged dataset in image mode (PNG-per-frame). "
             "Auto-detected from sources if omitted.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    tasks = args.tasks if args.tasks else list(TASKS_24)
    unknown = [t for t in tasks if t not in TASKS_24]
    if unknown:
        raise SystemExit(f"Unknown task(s): {unknown}. Valid: {TASKS_24}")

    # Resolve source repos and check they all exist.
    src_repos: list[tuple[str, str]] = []
    for t in tasks:
        rid = f"{args.src_prefix}{t}{args.src_suffix}"
        root = HF_LEROBOT_HOME / rid
        if not root.exists():
            logger.warning("skip %s: source repo missing at %s", t, root)
            continue
        src_repos.append((t, rid))
    if not src_repos:
        raise SystemExit("No source repos found. Check --src-prefix/--src-suffix.")
    logger.info("merging %d source repos", len(src_repos))

    # Use the first source's features as the schema. The schema is identical
    # across all 24 per-task repos by construction (stage 2 creates them all
    # the same way).
    first_ds = LeRobotDataset(repo_id=src_repos[0][1], video_backend="pyav")
    features_meta = first_ds.meta.features
    use_videos = (not args.no_videos) and _detect_use_videos(features_meta)
    logger.info("output schema: %d features; use_videos=%s",
                len(features_meta), use_videos)

    out_root = HF_LEROBOT_HOME / args.out_repo
    if out_root.exists():
        logger.warning("removing existing dataset at %s", out_root)
        shutil.rmtree(out_root)

    # LeRobotDataset.create requires the schema; just pass through what we
    # read from the first source.
    out_ds = LeRobotDataset.create(
        repo_id=args.out_repo,
        fps=int(first_ds.fps),
        robot_type=str(first_ds.meta.robot_type),
        features=features_meta,
        use_videos=use_videos,
    )

    # Which keys to round-trip into add_frame. We must NOT pass LeRobot's
    # auto-managed counters (episode_index, frame_index, index, timestamp,
    # task_index); LeRobot regenerates those.
    image_keys = [k for k in features_meta if k.startswith("observation.images.")]
    other_keys = [
        k for k in features_meta
        if not k.startswith("observation.images.")
        and k not in {"task", "episode_index", "frame_index", "index", "timestamp",
                      "task_index"}
    ]

    total_demos = 0
    total_frames = 0
    for task_name, repo_id in src_repos:
        ds = LeRobotDataset(repo_id=repo_id, video_backend="pyav")
        n_ep = ds.num_episodes
        logger.info("[%s] copying %d episodes from %s", task_name, n_ep, repo_id)
        for ep_idx in tqdm.tqdm(range(n_ep), desc=task_name):
            start = int(ds.episode_data_index["from"][ep_idx])
            end = int(ds.episode_data_index["to"][ep_idx])
            for global_t in range(start, end):
                row = ds[global_t]
                frame = {
                    "task": row["task"],
                }
                for k in image_keys:
                    frame[k] = _img_for_add_frame(row[k], use_videos)
                for k in other_keys:
                    v = row[k]
                    if isinstance(v, torch.Tensor):
                        frame[k] = v
                    else:
                        frame[k] = _to_np(v)
                out_ds.add_frame(frame)
            out_ds.save_episode()
        total_demos += n_ep
        total_frames += int(ds.num_frames)
        logger.info("[%s] +%d eps / +%d frames (totals %d / %d)",
                    task_name, n_ep, ds.num_frames, total_demos, total_frames)

    logger.info("done — %d episodes, %d frames at %s",
                total_demos, total_frames, out_root)


if __name__ == "__main__":
    main()

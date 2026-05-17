#!/usr/bin/env python3
"""Decode one frame from the downloaded RoboCasa LeRobot dataset and save
each cam as a PNG. The point is to verify, against real bytes, the claim in
`robocasa_policy.py` that the LeRobot training data is upside-down by pixel
convention (which is what `convert_hdf5_lerobot.py` writes via
`traj["obs"]["..._image"]`).

Usage (run in the `robocasa365` conda env so lerobot + torchcodec are
available):

    python scripts/debug/decode_robocasa_lerobot_frame.py \
        --dataset /home/asus/Research/CamVLA/robocasa365_sim/datasets/v1.0/target/atomic/OpenDrawer/<DATE>/lerobot \
        --episode 0 --frame 0 \
        --out /tmp/robocasa_lerobot_frame_check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, help="Path to LeRobot v2.1 dataset root")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("/tmp/robocasa_lerobot_frame_check"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ds = LeRobotDataset(repo_id=str(args.dataset), root=args.dataset)
    print(f"dataset features: {list(ds.features)}")

    # Resolve global frame index for the requested (episode, frame).
    ep_start = int(ds.episode_data_index["from"][args.episode])
    ep_end = int(ds.episode_data_index["to"][args.episode])
    if not 0 <= args.frame < (ep_end - ep_start):
        raise ValueError(f"frame {args.frame} out of range for episode {args.episode} of length {ep_end - ep_start}")
    sample = ds[ep_start + args.frame]

    cams = {k: v for k, v in sample.items() if k.startswith("observation.images.")}
    print(f"\nfound {len(cams)} cam channels at episode={args.episode} frame={args.frame}")

    for k, v in cams.items():
        # LeRobot returns CHW float [0, 1] tensors by default.
        arr = np.asarray(v)
        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
        safe_k = k.replace(".", "__").replace("/", "__")
        imageio.imwrite(args.out / f"{safe_k}__as_stored.png", arr)
        flipped = arr[::-1, :, :]
        imageio.imwrite(args.out / f"{safe_k}__after_vflip.png", flipped)
        print(f"  {k:55s} shape={arr.shape} -> wrote __as_stored.png (raw) and __after_vflip.png (one [::-1, :, :])")

    print(f"\nArtifacts in {args.out.resolve()}")
    print("Empirical finding on OpenDrawer (verified by decoding episode_000000.mp4 directly):")
    print("__as_stored.png is RIGHT-SIDE-UP. So the LeRobot training data matches the gym")
    print("wrapper output (both right-side-up), and the eval client passes frames through")
    print("to the policy server with no vflip.")


if __name__ == "__main__":
    main()

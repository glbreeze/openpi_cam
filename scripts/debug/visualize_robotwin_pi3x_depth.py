#!/usr/bin/env python3
"""Visualize RoboTwin RGB, Pi3X depth, and simulator GT depth caches."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


VIEW_TO_IMAGE = {
    "agent": "observation.images.cam_high",
    "wrist": "observation.images.cam_left_wrist",
    "right_wrist": "observation.images.cam_right_wrist",
}
VIEW_TO_GT = {
    "agent": "agent",
    "wrist": "wrist",
    "right_wrist": "cam_right_wrist_unused",
}


def _load_png_cell(cell) -> np.ndarray:
    if isinstance(cell, dict) and "bytes" in cell:
        payload = cell["bytes"]
    else:
        payload = cell
    return np.asarray(Image.open(BytesIO(payload)).convert("RGB"))


def _depth_from_npz(root: Path, view: str, episode: int, frame: int) -> tuple[np.ndarray, np.ndarray]:
    path = root / view / f"episode_{episode:06d}.npz"
    with np.load(path) as data:
        logz = np.asarray(data["log_z"][frame, ..., 0], dtype=np.float32)
        conf = np.asarray(data["conf"][frame, ..., 0], dtype=np.float32)
    return np.exp(np.clip(logz, -20.0, 15.0)), conf


def _stats(depth: np.ndarray, conf: np.ndarray | None = None) -> str:
    mask = np.isfinite(depth) & (depth > 0)
    if conf is not None:
        mask &= np.isfinite(conf)
    vals = depth[mask]
    if vals.size == 0:
        return "no valid"
    q = np.percentile(vals, [1, 50, 99])
    return f"p1={q[0]:.3f}m med={q[1]:.3f}m p99={q[2]:.3f}m"


def _plot_one(
    *,
    rgb: np.ndarray,
    pi3x_depth: np.ndarray,
    pi3x_conf: np.ndarray,
    gt_depth: np.ndarray,
    gt_conf: np.ndarray,
    title: str,
    out_path: Path,
) -> dict[str, float]:
    valid = np.isfinite(gt_depth) & (gt_depth > 0) & np.isfinite(pi3x_depth) & (pi3x_depth > 0)
    # GT cache confidence is usually +10 everywhere; Pi3X confidence is a logit.
    valid &= np.isfinite(pi3x_conf)
    ratio = np.full_like(gt_depth, np.nan, dtype=np.float32)
    ratio[valid] = pi3x_depth[valid] / gt_depth[valid]
    abs_log_err = np.full_like(gt_depth, np.nan, dtype=np.float32)
    abs_log_err[valid] = np.abs(np.log(pi3x_depth[valid]) - np.log(gt_depth[valid]))

    gt_vals = gt_depth[valid]
    pi_vals = pi3x_depth[valid]
    ratio_vals = ratio[valid]
    log_err_vals = abs_log_err[valid]
    metrics = {
        "valid_pixels": int(valid.sum()),
        "gt_median_m": float(np.median(gt_vals)),
        "pi3x_median_m": float(np.median(pi_vals)),
        "ratio_median": float(np.median(ratio_vals)),
        "ratio_p10": float(np.percentile(ratio_vals, 10)),
        "ratio_p90": float(np.percentile(ratio_vals, 90)),
        "abs_log_err_median": float(np.median(log_err_vals)),
    }

    vmax = float(np.percentile(np.concatenate([gt_vals, pi_vals]), 98))
    vmax = max(vmax, 1e-3)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB")
    im = axes[0, 1].imshow(pi3x_depth, vmin=0, vmax=vmax, cmap="magma")
    axes[0, 1].set_title("Pi3X depth (m)\n" + _stats(pi3x_depth, pi3x_conf))
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046)
    im = axes[0, 2].imshow(gt_depth, vmin=0, vmax=vmax, cmap="magma")
    axes[0, 2].set_title("GT depth (m)\n" + _stats(gt_depth, gt_conf))
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046)
    im = axes[1, 0].imshow(np.log(np.clip(pi3x_depth, 1e-6, None)), cmap="viridis")
    axes[1, 0].set_title("Pi3X log depth")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046)
    im = axes[1, 1].imshow(np.log(np.clip(gt_depth, 1e-6, None)), cmap="viridis")
    axes[1, 1].set_title("GT log depth")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046)
    im = axes[1, 2].imshow(np.clip(ratio, 0, 20), vmin=0, vmax=20, cmap="coolwarm")
    axes[1, 2].set_title(
        "Pi3X / GT depth ratio\n"
        f"med={metrics['ratio_median']:.2f}, p10={metrics['ratio_p10']:.2f}, p90={metrics['ratio_p90']:.2f}"
    )
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/scratch/yp2841/geometry-vla/robotwin/handover_block_demo_clean_camaware_50"))
    parser.add_argument("--pi3x-root", type=Path, default=Path("/scratch/yp2841/geometry-vla/.cache/openpi/pi3x_targets_224/robotwin_handover_block_demo_clean_camaware_50"))
    parser.add_argument("--gt-root", type=Path, default=Path("/scratch/yp2841/geometry-vla/.cache/openpi/gt_point_targets_224/robotwin_handover_block_demo_clean_camaware_50"))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 80, 160])
    parser.add_argument("--views", nargs="+", default=["agent", "wrist", "right_wrist"])
    parser.add_argument("--out-dir", type=Path, default=Path("log/robotwin_depth_debug"))
    args = parser.parse_args()

    parquet = args.dataset_root / "data" / "chunk-000" / f"episode_{args.episode:06d}.parquet"
    df = pd.read_parquet(parquet)
    rows = []
    for frame in args.frames:
        for view in args.views:
            image_key = VIEW_TO_IMAGE[view]
            rgb = _load_png_cell(df.iloc[frame][image_key])
            pi3x_depth, pi3x_conf = _depth_from_npz(args.pi3x_root, view, args.episode, frame)
            gt_view = VIEW_TO_GT[view]
            gt_depth, gt_conf = _depth_from_npz(args.gt_root, gt_view, args.episode, frame)
            out_path = args.out_dir / f"episode_{args.episode:06d}_frame_{frame:04d}_{view}.png"
            metrics = _plot_one(
                rgb=rgb,
                pi3x_depth=pi3x_depth,
                pi3x_conf=pi3x_conf,
                gt_depth=gt_depth,
                gt_conf=gt_conf,
                title=f"RoboTwin handover_block ep={args.episode} frame={frame} view={view}",
                out_path=out_path,
            )
            row = {"episode": args.episode, "frame": frame, "view": view, "path": str(out_path), **metrics}
            rows.append(row)
            print(row)
    pd.DataFrame(rows).to_csv(args.out_dir / f"episode_{args.episode:06d}_depth_metrics.csv", index=False)


if __name__ == "__main__":
    main()

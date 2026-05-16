#!/usr/bin/env python3
"""Quantify whether Pi3X targets carry usable structure on a dataset.

For each (view, episode), compute:
  - spatial std of log_z (after per-frame mean subtraction; uniform field => 0)
  - Pearson correlation between Pi3X log_z and GT log_z (after per-frame demeaning)
  - scale-invariant log-depth RMSE (Pi3X vs GT after per-frame mean alignment)
  - mean Pi3X conf (sigmoid of logit)
  - fraction of pixels above a conf threshold

These tell us whether the Pi3X teacher is collapsing to a uniform prior on
clean RoboTwin scenes (low spatial std, low correlation with GT) vs. carrying
real geometry signal that the scale-invariant distill loss can use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _per_frame_stats(logz_a: np.ndarray, logz_b: np.ndarray) -> dict[str, float]:
    """Compare two (T, H, W) log-depth tensors after per-frame mean-centering."""
    a = logz_a.astype(np.float32)
    b = logz_b.astype(np.float32)
    a_flat = a.reshape(a.shape[0], -1)
    b_flat = b.reshape(b.shape[0], -1)
    a_c = a_flat - a_flat.mean(axis=1, keepdims=True)
    b_c = b_flat - b_flat.mean(axis=1, keepdims=True)
    a_std = a_c.std(axis=1)
    b_std = b_c.std(axis=1)
    eps = 1e-6
    corr = (a_c * b_c).mean(axis=1) / (a_std * b_std + eps)
    rmse = np.sqrt(((a_c - b_c) ** 2).mean(axis=1))
    return {
        "pi3x_logz_spatial_std": float(a_std.mean()),
        "gt_logz_spatial_std": float(b_std.mean()),
        "pearson_corr_per_frame_mean": float(corr.mean()),
        "pearson_corr_per_frame_p10": float(np.percentile(corr, 10)),
        "pearson_corr_per_frame_p90": float(np.percentile(corr, 90)),
        "scale_invariant_log_rmse": float(rmse.mean()),
    }


def _conf_stats(conf_logit: np.ndarray) -> dict[str, float]:
    conf = 1.0 / (1.0 + np.exp(-conf_logit.astype(np.float32)))
    return {
        "pi3x_conf_mean": float(conf.mean()),
        "pi3x_conf_frac_above_0.5": float((conf > 0.5).mean()),
        "pi3x_conf_frac_above_0.8": float((conf > 0.8).mean()),
    }


def analyze_episode(pi3x_path: Path, gt_path: Path) -> dict[str, float]:
    with np.load(pi3x_path) as p:
        pi3x_logz = np.asarray(p["log_z"][..., 0])
        pi3x_conf_logit = np.asarray(p["conf"][..., 0])
        pi3x_xy = np.asarray(p["xy"])
    with np.load(gt_path) as g:
        gt_logz = np.asarray(g["log_z"][..., 0])
        gt_xy = np.asarray(g["xy"])

    out = {}
    out.update(_per_frame_stats(pi3x_logz, gt_logz))
    out.update(_conf_stats(pi3x_conf_logit))
    # xy direction comparison (mean cosine between rays).
    pi3x_dir = pi3x_xy.astype(np.float32)
    gt_dir = gt_xy.astype(np.float32)
    cos = (pi3x_dir * gt_dir).sum(axis=-1) / (
        np.linalg.norm(pi3x_dir, axis=-1) * np.linalg.norm(gt_dir, axis=-1) + 1e-6
    )
    out["xy_cosine_mean"] = float(cos.mean())
    out["pi3x_logz_global_mean"] = float(pi3x_logz.mean())
    out["gt_logz_global_mean"] = float(gt_logz.mean())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi3x-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--views", nargs="+", default=["agent", "wrist"])
    parser.add_argument("--gt-view-map", nargs="*", default=[],
                        help="Optional remap, e.g. right_wrist=cam_right_wrist_unused")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--label", type=str, default="")
    args = parser.parse_args()

    remap = {}
    for kv in args.gt_view_map:
        k, v = kv.split("=")
        remap[k] = v

    rows = []
    for view in args.views:
        pi3x_view_dir = args.pi3x_root / view
        gt_view = remap.get(view, view)
        gt_view_dir = args.gt_root / gt_view
        if not pi3x_view_dir.exists() or not gt_view_dir.exists():
            print(f"[skip] view={view}: missing {pi3x_view_dir} or {gt_view_dir}")
            continue
        episodes = sorted(pi3x_view_dir.glob("episode_*.npz"))[: args.episodes]
        for pi3x_path in episodes:
            gt_path = gt_view_dir / pi3x_path.name
            if not gt_path.exists():
                continue
            stats = analyze_episode(pi3x_path, gt_path)
            stats["view"] = view
            stats["episode"] = pi3x_path.stem
            stats["label"] = args.label
            rows.append(stats)

    table = pd.DataFrame(rows)
    cols = ["label", "view", "episode"] + [c for c in table.columns if c not in {"label", "view", "episode"}]
    table = table[cols]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print("== Aggregate by view ==")
    agg = table.drop(columns=["episode"]).groupby(["label", "view"]).mean(numeric_only=True)
    print(agg.to_string(float_format=lambda x: f"{x:.4f}"))

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out_csv, index=False)
        print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()

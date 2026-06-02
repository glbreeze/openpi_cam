#!/usr/bin/env python3
"""Create appendix figures for the RoboTwin handover_block Pi3X failure case."""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


VIEWS = ("agent", "wrist", "right_wrist")
VIEW_TO_IMAGE = {
    "agent": "observation.images.cam_high",
    "wrist": "observation.images.cam_left_wrist",
    "right_wrist": "observation.images.cam_right_wrist",
}
VIEW_LABEL = {
    "agent": "overhead",
    "wrist": "left wrist",
    "right_wrist": "right wrist",
}


def _load_png_cell(cell) -> np.ndarray:
    payload = cell["bytes"] if isinstance(cell, dict) and "bytes" in cell else cell
    return np.asarray(Image.open(BytesIO(payload)).convert("RGB"))


def _robust_normalize_logz(logz: np.ndarray) -> np.ndarray:
    """Scale-invariant per-frame log-depth normalization."""
    flat = logz.reshape(logz.shape[0], -1)
    med = np.nanmedian(flat, axis=1)[:, None, None]
    p10 = np.nanpercentile(flat, 10, axis=1)[:, None, None]
    p90 = np.nanpercentile(flat, 90, axis=1)[:, None, None]
    scale = np.maximum(p90 - p10, 1e-4)
    return np.clip((logz - med) / scale, -3.0, 3.0)


def _frame_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    af = a.reshape(a.shape[0], -1)
    bf = b.reshape(b.shape[0], -1)
    af = af - af.mean(axis=1, keepdims=True)
    bf = bf - bf.mean(axis=1, keepdims=True)
    denom = np.sqrt((af * af).sum(axis=1) * (bf * bf).sum(axis=1))
    return np.divide((af * bf).sum(axis=1), denom, out=np.zeros(a.shape[0], dtype=np.float32), where=denom > 1e-6)


def _load_logz(root: Path, view: str, episode: int, stride: int) -> np.ndarray:
    path = root / view / f"episode_{episode:06d}.npz"
    with np.load(path) as data:
        arr = np.asarray(data["log_z"][..., 0], dtype=np.float32)
    return arr[:, ::stride, ::stride]


def _scan_episode(pi3x_root: Path, gt_root: Path, episode: int, stride: int) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for view in VIEWS:
        pi_logz = _load_logz(pi3x_root, view, episode, stride)
        gt_logz = _load_logz(gt_root, view, episode, stride)
        n = min(pi_logz.shape[0], gt_logz.shape[0])
        pi_logz = pi_logz[:n]
        gt_logz = gt_logz[:n]

        pi_rel = _robust_normalize_logz(pi_logz)
        gt_rel = _robust_normalize_logz(gt_logz)
        si_mae = np.mean(np.abs(pi_rel - gt_rel), axis=(1, 2))
        corr = _frame_corr(pi_rel, gt_rel)

        pi_med_depth = np.exp(np.nanmedian(pi_logz.reshape(n, -1), axis=1))
        gt_med_depth = np.exp(np.nanmedian(gt_logz.reshape(n, -1), axis=1))
        ratio = pi_med_depth / np.maximum(gt_med_depth, 1e-6)
        abs_log_ratio = np.abs(np.log(np.maximum(ratio, 1e-6)))

        for frame in range(n):
            rows.append(
                {
                    "episode": int(episode),
                    "frame": int(frame),
                    "view": view,
                    "si_mae": float(si_mae[frame]),
                    "corr": float(corr[frame]),
                    "median_depth_ratio": float(ratio[frame]),
                    "abs_log_median_depth_ratio": float(abs_log_ratio[frame]),
                    "pi3x_median_m": float(pi_med_depth[frame]),
                    "gt_median_m": float(gt_med_depth[frame]),
                }
            )
    return rows


def _load_one(root: Path, view: str, episode: int, frame: int) -> np.ndarray:
    path = root / view / f"episode_{episode:06d}.npz"
    with np.load(path) as data:
        return np.asarray(data["log_z"][frame, ..., 0], dtype=np.float32)


def _plot_worst_case(
    df: pd.DataFrame,
    dataset_root: Path,
    pi3x_root: Path,
    gt_root: Path,
    episode: int,
    frame: int,
    out_path: Path,
) -> None:
    parquet = dataset_root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    table = pd.read_parquet(parquet)

    fig, axes = plt.subplots(len(VIEWS), 4, figsize=(15.5, 9.5), constrained_layout=True)
    for row_idx, view in enumerate(VIEWS):
        rgb = _load_png_cell(table.iloc[frame][VIEW_TO_IMAGE[view]])
        pi_logz = _load_one(pi3x_root, view, episode, frame)
        gt_logz = _load_one(gt_root, view, episode, frame)
        pi_rel = _robust_normalize_logz(pi_logz[None])[0]
        gt_rel = _robust_normalize_logz(gt_logz[None])[0]
        err = np.abs(pi_rel - gt_rel)

        row = df[(df.episode == episode) & (df.frame == frame) & (df.view == view)].iloc[0]
        title_suffix = f"ratio={row.median_depth_ratio:.1f}x\nSI-MAE={row.si_mae:.2f}, corr={row['corr']:.2f}"
        depth = np.exp(np.clip(np.stack([gt_logz, pi_logz]), -20.0, 15.0))
        metric_vmin = np.log(max(float(np.percentile(depth, 1)), 1e-3))
        metric_vmax = np.log(max(float(np.percentile(depth, 99)), 1e-3))

        axes[row_idx, 0].imshow(rgb)
        axes[row_idx, 0].set_title(f"{VIEW_LABEL[view]} RGB", fontsize=9)
        im = axes[row_idx, 1].imshow(gt_logz, cmap="magma", vmin=metric_vmin, vmax=metric_vmax)
        axes[row_idx, 1].set_title(f"GT log-depth\nmedian z={row.gt_median_m:.3f}m", fontsize=9)
        fig.colorbar(im, ax=axes[row_idx, 1], fraction=0.046)
        im = axes[row_idx, 2].imshow(pi_logz, cmap="magma", vmin=metric_vmin, vmax=metric_vmax)
        axes[row_idx, 2].set_title(f"Pi3X raw log-z\nmedian z={row.pi3x_median_m:.3f}m", fontsize=9)
        fig.colorbar(im, ax=axes[row_idx, 2], fraction=0.046)
        im = axes[row_idx, 3].imshow(err, cmap="magma", vmin=0, vmax=2.5)
        axes[row_idx, 3].set_title("relative-shape error\n" + title_suffix, fontsize=9)
        fig.colorbar(im, ax=axes[row_idx, 3], fraction=0.046)

    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(
        f"RoboTwin handover_block: automatically selected Pi3X/GT mismatch "
        f"(episode {episode}, frame {frame})",
        fontsize=14,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_distribution(df: pd.DataFrame, selected: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    colors = {"agent": "#4c78a8", "wrist": "#f58518", "right_wrist": "#e45756"}
    df = df.copy()
    df["log10_ratio"] = np.log10(np.maximum(df["median_depth_ratio"].to_numpy(), 1e-6))
    bins = np.linspace(float(df["log10_ratio"].quantile(0.005)), float(df["log10_ratio"].quantile(0.995)), 70)
    for view in VIEWS:
        vals = df[df.view == view]["log10_ratio"].to_numpy()
        ax.hist(vals, bins=bins, alpha=0.48, label=VIEW_LABEL[view], color=colors[view])
    sel_rows = df[(df.episode == selected["episode"]) & (df.frame == selected["frame"])]
    for _, row in sel_rows.iterrows():
        ax.axvline(row.log10_ratio, color=colors[row["view"]], linewidth=2.0, linestyle="--")
    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("log10(Pi3X median depth / simulator GT median depth)")
    ax.set_ylabel("Frame/view count")
    ax.set_title("Handover block raw Pi3X depth-scale mismatch distribution")
    ax.legend(frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _write_appendix_tex(summary: dict, out_path: Path, figure_path: Path, hist_path: Path) -> None:
    rows = summary["selected_view_rows"]
    worst = max(rows, key=lambda x: x["abs_log_median_depth_ratio"])
    mean_mae = sum(r["si_mae"] for r in rows) / len(rows)
    mean_ratio = sum(r["abs_log_median_depth_ratio"] for r in rows) / len(rows)
    text = rf"""
\paragraph{{RoboTwin depth-teacher failure case.}}
RoboTwin contains visually clean, texture-sparse simulator scenes.  This makes it
a useful stress test for geometry-aware policies that rely on a monocular
foundation model teacher.  In the \texttt{{handover\_block}} task we therefore
compare the Pi3X teacher cache against simulator ground-truth depth, using the
same camera views and frame indices used by policy training.  Because the
student is trained on the raw Pi3X \texttt{{log\_z}} targets, we select the
frame with the largest mean absolute log-ratio between the Pi3X median depth and
the simulator-GT median depth across the three cameras.  We also report a
scale-invariant shape error, computed after robustly normalizing each log-depth
map within the frame, to separate global scale mismatch from relative geometry.

Figure~\ref{{fig:robotwin-pi3x-worst}} shows the automatically selected
example: episode {summary["episode"]}, frame {summary["frame"]}.  The error is
most severe in the {VIEW_LABEL[worst["view"]]} camera: Pi3X predicts a median
depth of {worst["pi3x_median_m"]:.3f}m while the simulator GT median is
{worst["gt_median_m"]:.3f}m, a {worst["median_depth_ratio"]:.1f}$\times$
mismatch.  Averaged over the three cameras, the selected frame has mean
absolute log-ratio {mean_ratio:.2f} and scale-invariant MAE {mean_mae:.2f}.
Thus the auxiliary target is geometrically miscalibrated in the clean simulator
domain even when the RGB image is unambiguous and ground-truth depth is
available.

This explains the negative transfer observed for Pi3X-distilled RoboTwin
policies: the auxiliary target can inject a confident but geometrically wrong
prior in the clean simulator domain.  Replacing the monocular teacher with
simulator ground-truth depth removes this source of target noise, which is why
the GT-only variant improves success rate even though the Pi3X-distilled variant
can have lower training losses.

\begin{{figure}}[t]
  \centering
  \includegraphics[width=\linewidth]{{{figure_path.as_posix()}}}
  \caption{{Worst automatically selected Pi3X/GT depth mismatch in RoboTwin
  \texttt{{handover\_block}}.  GT and Pi3X log-depth panels use the same color
  scale per view.  The right column shows the remaining relative-shape error
  after per-frame robust log-depth normalization.}}
  \label{{fig:robotwin-pi3x-worst}}
\end{{figure}}

\begin{{figure}}[t]
  \centering
  \includegraphics[width=0.72\linewidth]{{{hist_path.as_posix()}}}
  \caption{{Distribution of raw median-depth scale mismatch across the
  handover-block cache.  Zero means Pi3X and simulator GT have the same median
  depth.  Dashed vertical lines mark the three views in the selected failure
  frame.}}
  \label{{fig:robotwin-pi3x-error-hist}}
\end{{figure}}
""".strip()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/scratch/yp2841/geometry-vla/robotwin/handover_block_demo_clean_camaware_50"))
    parser.add_argument("--pi3x-root", type=Path, default=Path("/scratch/yp2841/geometry-vla/.cache/openpi/pi3x_targets_224/robotwin_handover_block_demo_clean_camaware_50"))
    parser.add_argument("--gt-root", type=Path, default=Path("/scratch/yp2841/geometry-vla/.cache/openpi/gt_point_targets_224/robotwin_handover_block_demo_clean_camaware_50"))
    parser.add_argument("--out-dir", type=Path, default=Path("paper_assets/robotwin_handover_pi3x_appendix"))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--spatial-stride", type=int, default=4)
    args = parser.parse_args()

    all_rows: list[dict[str, float]] = []
    for episode in range(args.episodes):
        all_rows.extend(_scan_episode(args.pi3x_root, args.gt_root, episode, args.spatial_stride))
        print(f"scanned episode {episode:06d}", flush=True)

    df = pd.DataFrame(all_rows)
    grouped = (
        df.groupby(["episode", "frame"], as_index=False)
        .agg(
            mean_abs_log_median_depth_ratio=("abs_log_median_depth_ratio", "mean"),
            max_abs_log_median_depth_ratio=("abs_log_median_depth_ratio", "max"),
            mean_si_mae=("si_mae", "mean"),
            max_si_mae=("si_mae", "max"),
            min_corr=("corr", "min"),
        )
        .sort_values(["mean_abs_log_median_depth_ratio", "max_abs_log_median_depth_ratio"], ascending=False)
    )
    selected = grouped.iloc[0].to_dict()
    episode = int(selected["episode"])
    frame = int(selected["frame"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "handover_block_pi3x_gt_depth_errors_by_view.csv", index=False)
    grouped.head(50).to_csv(args.out_dir / "handover_block_pi3x_gt_depth_top_frames.csv", index=False)

    fig_path = args.out_dir / f"handover_block_pi3x_gt_worst_ep{episode:06d}_frame{frame:04d}.png"
    hist_path = args.out_dir / "handover_block_pi3x_gt_error_histogram.png"
    _plot_worst_case(df, args.dataset_root, args.pi3x_root, args.gt_root, episode, frame, fig_path)
    _plot_distribution(df, {"episode": episode, "frame": frame}, hist_path)

    selected_rows = (
        df[(df.episode == episode) & (df.frame == frame)]
        .sort_values("si_mae", ascending=False)
        .to_dict(orient="records")
    )
    summary = {
        "episode": episode,
        "frame": frame,
        "selection": selected,
        "selected_view_rows": selected_rows,
        "figure": str(fig_path),
        "histogram": str(hist_path),
        "num_frame_view_pairs": int(len(df)),
        "episodes_scanned": int(args.episodes),
        "spatial_stride": int(args.spatial_stride),
    }
    (args.out_dir / "handover_block_pi3x_gt_appendix_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_appendix_tex(
        summary,
        args.out_dir / "appendix_robotwin_pi3x_failure_case.tex",
        fig_path,
        hist_path,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
